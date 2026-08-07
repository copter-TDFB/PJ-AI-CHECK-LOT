# Design: select box/sachet by YOLO class, not by list position

**Date:** 2026-08-07
**Status:** approved, pending implementation plan

## Problem

`PipelineRunner._run_multi_region()` (`pipeline/pipeline_runner.py:55-93`) is the
`cross_check` detection-mode path used by `container_label` — the only packaging
today with `detection_mode: cross_check` (`config/packagings/container_label.yaml`).
It OCRs a "box" crop and a "sachet" crop separately and compares their lot/exp
values.

It picks which crop is "box" and which is "sachet" purely by **list position**:

```python
box    = crops[0] if len(crops) > 0 else {}
sachet = crops[1] if len(crops) > 1 else {}
```

`crops` is built by OCR'ing, in order, every `DetectionResult` returned by
`RegionDetector.crop_all(image_bytes, config.key)`. `crop_all()` only guarantees
that every returned detection's YOLO class belongs to the target set for that
packaging (`container_label_box` or `container_label_sachet`, matched by prefix
in `_yolo_crop_all()`) — it does **not** guarantee exactly one of each. It can
legitimately return 0, 1, or more detections of either class, sorted by `y1`
(top to bottom) with no regard for which specific class each one is.

### Confirmed root cause

While evaluating whether `DETECTOR_CONF` (currently 0.25, `pipeline/detector.py:18`)
could safely be lowered to 0.15, a box-vs-box confidence sweep across all 300
`images/container_label/*.jpg` found 2 photos where the lower threshold surfaces
a **second, overlapping `container_label_box` detection** (a near-duplicate of
the real box region, not a real sachet) — `container_label_box` conf 0.20-0.21,
alongside the real box at conf 0.77-0.81. No `container_label_sachet` was
detected in either photo.

Re-running the real pipeline (`pipeline_runner.run`) at conf 0.25 vs 0.15 on
these 2 photos confirmed the consequence: at 0.15, the duplicate box detection
lands in `crops[1]` and gets labeled "sachet". OCR then reads the same physical
box region a second time and produces:

- `lot_sachet` — a truncated/wrong fragment of the real `lot_box` value
  (e.g. `"CC00020001305"` vs the real `"CC0002000130533825P"`)
- `exp_sachet` — coincidentally equal to `exp_box` in both sampled photos
  (same date text, re-read)

`sheet_checker.check()`'s `sachet_match` logic (`utils/sheet_checker.py:146-158`)
only compares `exp_box == exp_sachet` — it never compares `lot_box` vs
`lot_sachet`. In the 2 sampled photos `sachet_match` still evaluates `True`
(the corrupted `lot_sachet` is silently ignored), but this is incidental: if a
future duplicate-box crop reads the date differently than the real box crop,
`sachet_match` would flip to `False` — a false "sachet doesn't match" flag on
a product that has no real sachet at all.

This is a latent bug independent of the `DETECTOR_CONF` value — it can occur
at 0.25 too, any time YOLO produces 2+ detections of the same
`container_label_box`/`container_label_sachet` class for one photo.

## Goal

`_run_multi_region()` must select the box crop and the sachet crop by their
actual YOLO class, never by position in the detections list. Duplicate
detections of the same class must resolve deterministically (highest
confidence wins) instead of accidentally occupying the other region's slot.

## Design

### 1. Carry detection confidence (`pipeline/detector.py`)

`DetectionResult` currently has no confidence field even though
`_yolo_crop_all()` already computes it (used only for a log line today):

```python
@dataclass
class DetectionResult:
    cropped_bytes: bytes
    bbox: Bbox | None
    class_name: str | None = None
    conf: float | None = None   # new
```

`_yolo_crop_all()` passes `conf=conf` when constructing each result. Heuristic
fallback constructions (`_heuristic_crop`, `_crop_by_rule`, `_crop_container_label`)
leave it at the default `None` — there is no YOLO confidence to report there.
Additive, default-valued field: no change for `_run_single_region` or
`_run_multi_field`, which don't read it.

### 2. Select by class, not position (`pipeline/pipeline_runner.py`)

```python
def _select_region(detections, target_class):
    matches = [d for d in detections if d.class_name == target_class]
    if not matches and all(d.class_name is None for d in detections):
        # Pure heuristic fallback (YOLO didn't run / found nothing) — the one
        # detection it returns is _crop_container_label's white-box locator,
        # which only ever targets the "box" region.
        matches = detections if target_class.endswith("_box") else []
    return max(matches, key=lambda d: d.conf or 0.0, default=None)


detections = self._detector.crop_all(image_bytes, config.key)
box_det    = _select_region(detections, f"{config.key}_box")
sachet_det = _select_region(detections, f"{config.key}_sachet")

selected = (box_det is not None) + (sachet_det is not None)
if selected < len(detections):
    logger.warning(
        "%s: %d detection(s) discarded (not selected as box or sachet)",
        config.key, len(detections) - selected,
    )

crops = []
for det in (box_det, sachet_det):
    if det is None:
        crops.append({})
        continue
    processed = self._preprocessor.run(det.cropped_bytes, config.key)
    ocr_res = self._ocr_engine.run(processed, config=config)
    text_ok = len(ocr_res["raw_text"].strip()) >= 8
    data_ok = bool(ocr_res["lot_number"] or ocr_res["exp_date"])
    if not text_ok and not data_ok:
        logger.warning("Preprocessed crop appears degraded — retrying with original bytes")
        ocr_res = self._ocr_engine.run(det.cropped_bytes, config=config)
    crops.append({"lot_number": ocr_res["lot_number"], "exp_date": ocr_res["exp_date"]})

box, sachet = crops[0], crops[1]
bbox = box_det.bbox if box_det else (sachet_det.bbox if sachet_det else None)
```

OCR only runs on the 2 selected detections (not on every detection returned),
which also avoids paying for Vision API calls on discarded duplicates.

### Behavior per case

| Case | Before | After |
|---|---|---|
| 1 box, 0 sachet (SKU has no sachet) | box = it, sachet = `{}` ✅ | unchanged ✅ |
| 1 box, 1 sachet (normal) | correct, by accident of ordering ✅ | unchanged ✅ |
| 2 box, 0 sachet (**confirmed bug**) | box = first, sachet = *second box* ❌ | box = highest-conf box, sachet = `{}` ✅ |
| 0 box, 1 sachet (box occluded, sachet visible) | box = *the sachet*, wrong ❌ | box = `{}`, sachet = it ✅ |
| YOLO didn't run/found nothing → 1 heuristic detection | box = it ✅ | unchanged, via the `class_name is None` fallback rule ✅ |
| 0 detections | box = sachet = `{}` ✅ | unchanged ✅ |

### Error handling / edge cases

- Discarding extra same-class detections is logged as a warning (visibility
  into how often this occurs in production, without failing the request).
- No change to `sheet_checker.py`'s `sachet_match` comparison logic itself —
  this fix prevents feeding it corrupted `lot_sachet`/`exp_sachet` in the first
  place, rather than changing how it interprets them.
- No config/YAML/schema change — `container_label.yaml`'s `sub_regions: [box, sachet]`
  is unchanged; the fix only changes how `_run_multi_region` interprets the
  detections it already receives.

### Testing (`tests/test_multi_region.py`, new file)

Pattern follows `tests/test_routing.py` (mock `detector.crop_all` to return
fabricated `DetectionResult`s, mock `ocr_engine.run` to return per-crop fake
OCR results keyed by which crop was passed in). Cases, matching the table above:

1. 1 box, 0 sachet → `lot_sachet`/`exp_sachet` are `None`
2. 1 box, 1 sachet → both populated from their respective crops
3. 2 box (different conf), 0 sachet → box resolves to the higher-conf
   detection; sachet fields stay `None` (**regression guard for the confirmed bug**)
4. 0 box, 1 sachet → box fields `None`, sachet populated
5. Single detection with `class_name=None` (heuristic fallback) → treated as box
6. 0 detections → both `{}`, `status="not_found"`

Existing tests (`test_routing.py`, `test_detector.py`) are expected to stay
green unchanged — none of them assert on `_run_multi_region`'s internal
box/sachet assignment.

## Out of scope

- Detector-level deduplication (rejected alternative: making `crop_all()`
  itself drop duplicate same-class boxes before returning). This would touch
  the shared entry point used by every packaging's `single`/`multi_field`
  paths too, not just `container_label`'s `cross_check` path. The same conf
  0.25→0.15 sweep that found this bug showed `single`/`multi_field` packagings
  already tolerate duplicate boxes fine (their extraction regexes still find
  the correct value in the combined OCR text) — there is no live problem there
  to justify the wider blast radius. YAGNI.
- Changing `DETECTOR_CONF` (0.25 → 0.15 or otherwise) is a separate decision,
  deliberately deferred until after this fix lands.
- `sub_regions` config format / `cross_check` YAML schema — unchanged.
