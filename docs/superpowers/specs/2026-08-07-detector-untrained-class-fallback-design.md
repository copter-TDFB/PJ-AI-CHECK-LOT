# Design: skip the YOLO detector when a class has zero trained boxes

**Date:** 2026-08-07
**Status:** approved, pending implementation plan

## Problem

`RegionDetector.crop_all()` (`pipeline/detector.py`) is supposed to fall back to a
full-image heuristic crop when a packaging class has no YOLO-trained boxes of its
own (documented design for `30_sachet`, added 2026-06-30, which deliberately ships
with no detector training data — see `config/packagings/30_sachet.yaml`).

In practice it does not. `_yolo_crop_all()` matches boxes by `{image_class}_` prefix;
when no class in the model starts with that prefix, it falls through to an
`else` branch that trusts **every box detected anywhere in the photo, regardless
of class** (written 2026-05-30, for `capsule_box`, whose own class has no
underscore suffix — a real, different scenario). For a class like `30_sachet`
that has *zero* trained boxes at all, this means: if any other trained class's
box fires anywhere in the photo, the pipeline crops to that box instead of the
intended full image.

### Confirmed root cause

Reproduced directly against `IMG_20260803_141639.jpg` (the triggering photo):
`RegionDetector.crop_all(img_bytes, "30_sachet")` returns
`bbox=[416, 2085, 1219, 2451]`, `class_name='grade_bag_product'` — a box trained
for the unrelated `grade_bag` packaging class. This bbox is pixel-identical to
what production's `/predict` returned for the same photo, confirming the
production detector exhibits the same behavior. `grade_bag` and `30_sachet` are
the same matcha product line with visually similar product-name label layouts,
which is why `grade_bag_product` false-fires on `30_sachet` photos specifically.

Swept all 22 real (non-`aug_`) training photos in `images/30_sachet/` through
`crop_all()`: **16/22 (73%) fall through cleanly to the full-image heuristic**
(`bbox=None`); **6/22 (27%) get hijacked** — 5 by `grade_bag_product`, 1 by
`print_sticker_back_lot_exp`. This is not a regression — it has been silently
present since the class was added (2026-06-30); most submitted photos simply
didn't happen to trigger a foreign-class detection. A prior bug fix for this
class (`ac41fae`, 2026-07-02) validated "no regression" by diffing OCR output
before/after a regex change — it never asserted the crop region itself was
correct, so a photo already silently returning `lot_number: null` due to this
crop bug would pass that check trivially.

## Goal

`crop_all()` must never crop to a box that doesn't provably belong to the
requested `image_class`. When a class has no trained boxes at all, skip YOLO
inference entirely (not just discard its output) and go straight to the
full-image heuristic — both for correctness and to avoid paying for a wasted
inference pass.

## Design

### Core logic change (`pipeline/detector.py`)

Add a cheap upfront check (dict lookup only, no inference) that decides whether
a class has any legitimate claim to a box in the currently loaded model:

```python
def _has_trained_boxes(self, image_class: str) -> bool:
    prefix = f"{image_class}_"
    return any(
        n.startswith(prefix) or n == image_class
        for n in self._model.names.values()
    )
```

`crop_all()` gates YOLO inference on this check:

```python
if self._model is not None and self._has_trained_boxes(image_class):
    results = self._yolo_crop_all(img_np, image_class)
    if results:
        return results
    logger.warning("YOLO: detected 0 boxes for class '%s' — heuristic fallback", image_class)
elif self._model is not None:
    logger.info(
        "YOLO: class '%s' has no trained boxes at all — skipping detector, heuristic fallback",
        image_class,
    )
return [self._heuristic_crop(img_np, image_class)]
```

Inside `_yolo_crop_all()`, delete the `else: use every box` branch. It becomes:

```python
prefix = f"{image_class}_"
target_cls_ids = {
    cid for cid, name in class_names.items()
    if name.startswith(prefix) or name == image_class
}
if not target_cls_ids:
    return []
matched_indices = [i for i, c in enumerate(cls_ids) if c in target_cls_ids]
```

(The exact-match arm keeps `capsule_box` working — now via an explicit, provable
match instead of an accidental "trust everything" fallback.)

### Behavior per class (no regressions)

| class | today | after fix |
|---|---|---|
| `back_label` / `grade_bag` / `retail_sachet` / `container_label` / `print_sticker_back` | prefix match exists → uses own boxes | unchanged |
| `capsule_box` | exact-name match, reached via the loose "use every box" path | unchanged behavior, now via an explicit exact-match check |
| `30_sachet` | "use every box" → can steal e.g. `grade_bag_product` | `_has_trained_boxes` is `False` → skips YOLO entirely → full image, deterministically, every time |

This change lives entirely inside `RegionDetector`/`crop_all()`, which is the
single shared entry point for all three `PipelineRunner` detection modes
(`single`, `cross_check`, `multi_field`) — no caller changes needed, and no
config/schema/YAML changes.

### Error handling / edge cases

- `self._model is None` (no `detector.pt` / load failed) — unaffected, already
  skips straight to heuristic.
- A class with a `_CROP_RULES` entry but also `_has_trained_boxes() is False` —
  not a real case today (the 3 rule-based heuristics — `back_label`, `grade_bag`,
  `retail_sachet` — all have real trained boxes), but if it ever occurred, the
  rule-based crop would still apply correctly through the same fallback path.
- No API/schema/YAML changes, so no migration or wizard-facing risk.

### Testing (`tests/test_detector.py`)

- New test iterating all 22 real (non-`aug_`) `images/30_sachet/*.jpg`: assert
  `crop_all(bytes, "30_sachet")` always returns exactly one detection with
  `bbox is None` (full image). This pins down the 6 photos that are hijacked
  today, turning them into a regression guard.
- New minimal regression test for `capsule_box` using a real sample photo
  (`images/capsule_box/`) asserting it still returns a crop tied to the
  `capsule_box` class (not `bbox=None`) — this class currently has zero
  detector-level test coverage despite depending on the branch being changed.
- Existing tests (`test_unknown_class_returns_full_image`,
  `test_crop_returns_bytes`, `test_crop_smaller_than_original`,
  `test_bbox_valid`, `test_real_images_per_class`) are expected to stay green
  unchanged.

## Out of scope

- No change to `_CROP_RULES`, `_crop_container_label`, or any other heuristic
  crop function.
- No change to `PackagingConfig`/registry/schema/YAML — this was considered
  (an explicit `skip_detector` flag) and rejected in favor of auto-detecting
  from the live model's own class list, so future detector-free classes are
  covered automatically with no config change required.
- Writing a `bug_fix.md` post-mortem entry is a follow-up after the fix lands
  and is validated, not part of this design.
