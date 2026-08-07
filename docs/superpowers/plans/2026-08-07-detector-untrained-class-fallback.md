# Detector Untrained-Class Fallback Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `RegionDetector.crop_all()` from cropping to a box that belongs to
some *other* packaging class when the requested class has no trained boxes of
its own (currently affects `30_sachet`, silently dropping its lot number on
~27% of real photos).

**Architecture:** `RegionDetector` gains a cheap upfront check —
`_has_trained_boxes(image_class)` — that inspects the loaded YOLO model's own
class-name list (no inference) to decide whether this class has *any*
legitimate claim to a box (prefix match, e.g. `back_label_lot`, or exact match,
e.g. `capsule_box`). `crop_all()` only runs YOLO inference when that's true;
otherwise it skips straight to the existing full-image heuristic. Inside
`_yolo_crop_all()`, the old "no prefix match → trust every box in the photo"
fallback is deleted — matching becomes prefix-OR-exact, full stop.

**Tech Stack:** Python 3.11, ultralytics YOLOv8 (`pipeline/detector.py`), pytest.

## Global Constraints

- No `print()` — use the module `logger` (project convention, `CLAUDE.md`).
- `pytest` is not on PATH — invoke as `python -m pytest`.
- Read/write any file containing Thai text with `encoding='utf-8'`.
- Only touch `pipeline/detector.py` and `tests/test_detector.py` — no
  config/schema/YAML/registry changes (see design doc's "Out of scope").

---

### Task 1: Fix `RegionDetector` untrained-class fallback + regression coverage

**Files:**
- Modify: `pipeline/detector.py:49-125` (`crop_all`, `_yolo_crop_all`; add new method `_has_trained_boxes`)
- Test: `tests/test_detector.py`

**Interfaces:**
- Produces: `RegionDetector._has_trained_boxes(self, image_class: str) -> bool` —
  new private method, used only internally by `crop_all`. No other task or
  caller depends on it.
- No changes to any public signature (`crop_all`, `crop`) — same inputs/outputs
  as before, only the internal box-selection logic changes.

- [ ] **Step 1: Write the failing regression test for the `30_sachet` hijack bug**

Add to `tests/test_detector.py` (new test, place after `test_real_images_per_class`):

```python
def test_30_sachet_never_crops_to_foreign_class_box(detector):
    """30_sachet has zero trained YOLO boxes of its own — crop_all() must
    always fall through to the full-image heuristic (bbox=None), never
    accept a box trained for a different packaging class (e.g. grade_bag)."""
    img_dir = pathlib.Path("images") / "30_sachet"
    samples = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("aug_")]
    assert samples, "expected real (non-aug_) 30_sachet training photos"

    offenders = []
    for img_path in samples:
        results = detector.crop_all(img_path.read_bytes(), "30_sachet")
        assert len(results) == 1
        if results[0].bbox is not None:
            offenders.append((img_path.name, results[0].bbox, results[0].class_name))

    assert not offenders, f"crop_all hijacked by a foreign box: {offenders}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detector.py::test_30_sachet_never_crops_to_foreign_class_box -v`

Expected: **FAIL** — `offenders` is non-empty (today ~6 of the 22 real photos
return a non-`None` bbox belonging to `grade_bag_product` or
`print_sticker_back_lot_exp`).

- [ ] **Step 3: Write the `capsule_box` baseline safety-net test**

`capsule_box` currently has zero detector-level test coverage even though it
depends on the branch this task is about to change (exact-class-name match).
Add to `tests/test_detector.py`, right after the test from Step 1:

```python
def test_capsule_box_still_crops_to_its_own_class(detector):
    """capsule_box has no '_lot' suffix — its class name matches image_class
    exactly. This must keep working after the 'use every box' fallback is
    removed (it now goes through an explicit exact-match check instead)."""
    img_path = pathlib.Path("images/capsule_box/1000010638.jpg")
    assert img_path.exists(), f"missing fixture photo: {img_path}"

    results = detector.crop_all(img_path.read_bytes(), "capsule_box")

    assert len(results) == 1
    assert results[0].bbox is not None
    assert results[0].class_name == "capsule_box"
```

- [ ] **Step 4: Run test to verify it currently passes (baseline)**

Run: `python -m pytest tests/test_detector.py::test_capsule_box_still_crops_to_its_own_class -v`

Expected: **PASS** (this confirms today's behavior for `capsule_box` before any
code changes, so if Step 8 breaks it, the regression is attributable to this
task's implementation, not a pre-existing issue).

- [ ] **Step 5: Implement `_has_trained_boxes` and gate YOLO inference on it**

In `pipeline/detector.py`, add a new method on `RegionDetector` (place it right
after `crop_all`, before `crop`):

```python
def _has_trained_boxes(self, image_class: str) -> bool:
    """True if the loaded model has at least one class that legitimately
    belongs to image_class — either '{image_class}_*' (e.g. back_label_lot)
    or an exact match (e.g. capsule_box, which has no underscore suffix)."""
    prefix = f"{image_class}_"
    return any(
        name.startswith(prefix) or name == image_class
        for name in self._model.names.values()
    )
```

Replace the body of `crop_all` (currently lines 57-67):

```python
def crop_all(self, image_bytes: bytes, image_class: str) -> list[DetectionResult]:
    """
    Crop ทุก region ที่ detect ได้สำหรับ image_class นั้น
    เรียงจากบนลงล่าง (top y ก่อน) ตามลำดับการอ่าน

    Returns:
        list[DetectionResult] อย่างน้อย 1 รายการ (fallback ถ้าไม่เจอ)
    """
    img_np = bytes_to_numpy(image_bytes)
    logger.info("Detecting lot regions — class=%s size=%dx%d",
                image_class, img_np.shape[1], img_np.shape[0])

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

- [ ] **Step 6: Remove the "use every box" fallback in `_yolo_crop_all`**

In `pipeline/detector.py`, inside `_yolo_crop_all` (currently lines 93-104),
replace:

```python
        # จับทุก YOLO class ที่ขึ้นต้นด้วย "{image_class}_"
        prefix = f"{image_class}_"
        target_cls_ids = {cid for cid, name in class_names.items() if name.startswith(prefix)}
        if target_cls_ids:
            matched_indices = [i for i, c in enumerate(cls_ids) if c in target_cls_ids]
            logger.info("YOLO: target classes for '%s' → %s", image_class, sorted(target_cls_ids))
        else:
            logger.warning("YOLO: ไม่มี class ที่ขึ้นต้นด้วย '%s' — ใช้ทุก box", prefix)
            matched_indices = list(range(len(boxes)))

        if not matched_indices:
            return []
```

with:

```python
        # จับ YOLO class ที่ตรงกับ image_class จริง — prefix "{image_class}_"
        # (เช่น back_label_lot) หรือ exact match (เช่น capsule_box ไม่มี suffix)
        prefix = f"{image_class}_"
        target_cls_ids = {
            cid for cid, name in class_names.items()
            if name.startswith(prefix) or name == image_class
        }
        logger.info("YOLO: target classes for '%s' → %s", image_class, sorted(target_cls_ids))
        matched_indices = [i for i, c in enumerate(cls_ids) if c in target_cls_ids]

        if not matched_indices:
            return []
```

(By construction — `crop_all` already gated on `_has_trained_boxes` — `target_cls_ids`
will never be empty here in practice, but `_yolo_crop_all` stays correct standalone.)

- [ ] **Step 7: Run both new tests to verify they pass**

Run: `python -m pytest tests/test_detector.py::test_30_sachet_never_crops_to_foreign_class_box tests/test_detector.py::test_capsule_box_still_crops_to_its_own_class -v`

Expected: **PASS** — both tests green.

- [ ] **Step 8: Run the full detector test suite to confirm no regressions**

Run: `python -m pytest tests/test_detector.py -v`

Expected: **PASS** — all existing tests (`test_crop_returns_bytes`,
`test_crop_smaller_than_original`, `test_bbox_valid`,
`test_unknown_class_returns_full_image`, `test_real_images_per_class`) plus
the 2 new tests, all green.

- [ ] **Step 9: Run the full project test suite**

Run: `python -m pytest`

Expected: same pre-existing failures as before this change and no new ones —
`tests/test_classifier.py` has 3 known pre-existing setup errors (fixture uses
`efficientnet_b0`, `pipeline/classifier.py` now uses `efficientnet_v2_s` — not
caused by this change, documented in `CLAUDE.md`). No other file should show
new failures.

- [ ] **Step 10: Commit**

```bash
git add pipeline/detector.py tests/test_detector.py
git commit -m "$(cat <<'EOF'
fix: stop detector from cropping to a foreign class's box

RegionDetector._yolo_crop_all's "no prefix match -> use every box"
fallback (written for capsule_box) let any other trained class's box
hijack the crop for classes with zero trained boxes of their own
(30_sachet), since the model can still false-fire e.g. grade_bag_product
on a visually similar photo. crop_all() now checks _has_trained_boxes()
before running YOLO at all, and _yolo_crop_all only ever trusts a
prefix or exact class-name match.

Confirmed via all 22 real 30_sachet training photos: 6/22 were silently
hijacked before this fix (5 by grade_bag_product, 1 by
print_sticker_back_lot_exp), now none are. capsule_box (previously
uncovered) verified unaffected.
EOF
)"
```

---

## Post-implementation follow-up (not part of this plan)

Once this fix is deployed and validated against prod (rebuild + redeploy —
`30_sachet` is a shipped, baked-in class per `CLAUDE.md`, not GCS-overlay-editable),
write a `bug_fix.md` post-mortem entry documenting root cause, mechanism, and
validation, following the existing `Fix N` numbering convention.
