# Container-label box/sachet selection-by-class fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `_run_multi_region()` so it selects the "box" and "sachet" crops by their actual YOLO `class_name`, not by position in the detections list — preventing a duplicate `container_label_box` detection from being misread as the sachet.

**Architecture:** Two small, sequential changes. (1) `DetectionResult` gains a `conf` field so duplicate same-class detections can be resolved deterministically. (2) `_run_multi_region()` replaces `crops[0]`/`crops[1]` indexing with a class-name lookup (`_select_region`) that picks the highest-confidence detection per target class, with a documented fallback for the heuristic (non-YOLO) case.

**Tech Stack:** Python 3.11, pytest, ultralytics YOLO (already a dependency — no new libraries).

## Global Constraints

- No changes to `config/packagings/*.yaml`, `PackagingConfig`, or any API schema.
- No changes to `utils/sheet_checker.py`'s `sachet_match` comparison logic — this fix stops corrupted data from reaching it, it does not change how it's interpreted.
- `DetectionResult.conf` must be additive (default `None`) — `_run_single_region` and `_run_multi_field` must keep working unchanged, since they don't read it.
- Do not touch `RegionDetector.crop_all()`'s class-matching/heuristic-fallback logic itself (that's the `30_sachet`/`capsule_box` fix from `2026-08-07-detector-untrained-class-fallback-design.md` — unrelated, already shipped) — only add the `conf` field inside the existing YOLO-path construction of `DetectionResult`.
- Do not change `DETECTOR_CONF` (stays at its current default, `0.25`) — that's a separate, deferred decision per the design doc's "Out of scope" section.

---

### Task 1: Carry YOLO confidence on `DetectionResult`

**Files:**
- Modify: `pipeline/detector.py:21-25` (dataclass), `pipeline/detector.py:136-140` (`_yolo_crop_all` construction)
- Test: `tests/test_detector.py` (add one test function)

**Interfaces:**
- Produces: `DetectionResult.conf: float | None` (default `None`) — Task 2's `_select_region` reads this via `d.conf or 0.0` to break ties between same-class detections.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_detector.py` (after `test_capsule_box_still_crops_to_its_own_class`):

```python
def test_yolo_detection_carries_confidence(detector):
    """DetectionResult.conf must be populated for real YOLO detections so
    downstream code (e.g. _run_multi_region's box/sachet selection) can
    break ties between duplicate same-class detections."""
    if detector._model is None:
        pytest.skip("models/detector.pt not available")

    img_dir = pathlib.Path("images") / "container_label"
    samples = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("aug_")][:3]
    if not samples:
        pytest.skip("no real container_label training photos available")

    found_one = False
    for img_path in samples:
        for det in detector.crop_all(img_path.read_bytes(), "container_label"):
            if det.class_name is not None:
                assert det.conf is not None, f"{img_path.name}: {det.class_name} missing conf"
                assert 0.0 <= det.conf <= 1.0
                found_one = True
    assert found_one, "no YOLO detection found across samples — test is not exercising the code path"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detector.py::test_yolo_detection_carries_confidence -v`
Expected: FAIL with `AttributeError: 'DetectionResult' object has no attribute 'conf'`

- [ ] **Step 3: Implement — add the field and populate it**

In `pipeline/detector.py`, change the dataclass (lines 21-25):

```python
@dataclass
class DetectionResult:
    cropped_bytes: bytes   # JPEG bytes ของ region ที่ crop แล้ว
    bbox: Bbox | None      # [x1, y1, x2, y2] relative ต่อรูปต้นฉบับ
    class_name: str | None = None   # YOLO class of the matched box (e.g. "back_label_size")
    conf: float | None = None       # YOLO confidence of the matched box (None for heuristic fallback)
```

In `_yolo_crop_all()`, the detection-construction loop (lines 136-140) becomes:

```python
            detections.append(DetectionResult(
                cropped_bytes=_to_jpeg(cropped),
                bbox=[x1, y1, x2, y2],
                class_name=class_names.get(cls_ids[idx]),
                conf=conf,
            ))
```

(`conf` is already computed two lines above at `conf = float(boxes.conf[idx])` — this only adds it to the constructed object.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detector.py::test_yolo_detection_carries_confidence -v`
Expected: PASS (or SKIPPED if `models/detector.pt` isn't present in this environment — that's an accepted outcome, not a failure)

- [ ] **Step 5: Run the full detector test file to check for regressions**

Run: `python -m pytest tests/test_detector.py -v`
Expected: all tests PASS or SKIPPED, none newly FAIL

- [ ] **Step 6: Commit**

```bash
git add pipeline/detector.py tests/test_detector.py
git commit -m "feat: carry YOLO confidence on DetectionResult"
```

---

### Task 2: Select box/sachet by class_name in `_run_multi_region`

**Files:**
- Modify: `pipeline/pipeline_runner.py:55-93` (replace `_run_multi_region`), add a new module-level helper `_select_region` above the `PipelineRunner` class
- Test: `tests/test_multi_region.py` (new file)

**Interfaces:**
- Consumes: `DetectionResult(cropped_bytes, bbox, class_name, conf)` from Task 1.
- Produces: `_run_multi_region(image_bytes, config) -> tuple[dict, object]` — same external shape as before (`lot_box`/`lot_sachet`/`exp_box`/`exp_sachet`/`lot_number`/`exp_date`/`mfg_date`/`raw_text`/`confidence`/`product_name`/`size`/`status`, and a `bbox`) — `PipelineRunner.run()`'s dispatch to this method is unchanged, so no other caller needs to change.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multi_region.py`:

```python
"""_run_multi_region must select box/sachet by YOLO class_name, not by
position in the detections list — see
docs/superpowers/specs/2026-08-07-container-label-box-sachet-selection-design.md"""

from pipeline.detector import DetectionResult
from pipeline.packaging_registry import PackagingConfig
from pipeline.pipeline_runner import PipelineRunner


def _cfg():
    return PackagingConfig(
        key="k", display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot", "exp"], sheet_checks=["lot", "exp"],
        post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label="k", detector_yolo_prefixes=["k_box", "k_sachet"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=["box", "sachet"], detection_mode="cross_check",
    )


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def crop_all(self, image_bytes, key):
        return self._detections


class _StubPreprocessor:
    def run(self, cropped_bytes, key):
        return cropped_bytes


class _StubOcrEngine:
    def __init__(self, results_by_bytes):
        self._results_by_bytes = results_by_bytes

    def run(self, image_bytes, config=None):
        return self._results_by_bytes[image_bytes]


def _det(tag, class_name, conf):
    return DetectionResult(cropped_bytes=tag, bbox=[0, 0, 1, 1], class_name=class_name, conf=conf)


def _ocr(lot_number, exp_date):
    return {
        "raw_text": "enough text to pass the degraded-crop check",
        "lot_number": lot_number,
        "exp_date": exp_date,
    }


def _run(detections, ocr_by_bytes):
    runner = PipelineRunner(
        detector=_StubDetector(detections),
        preprocessor=_StubPreprocessor(),
        ocr_engine=_StubOcrEngine(ocr_by_bytes),
        qr_scanner=object(),
    )
    return runner._run_multi_region(b"unused_image_bytes", _cfg())


def test_one_box_zero_sachet():
    """SKU with no sachet at all — legitimate case, must stay correct."""
    box = _det(b"box", "k_box", 0.9)
    result, bbox = _run([box], {b"box": _ocr("LOTBOX", "2026-01-01")})
    assert result["lot_box"] == "LOTBOX"
    assert result["exp_box"] == "2026-01-01"
    assert result["lot_sachet"] is None
    assert result["exp_sachet"] is None
    assert bbox == [0, 0, 1, 1]


def test_one_box_one_sachet():
    """Normal case — both regions detected, one of each class."""
    box = _det(b"box", "k_box", 0.9)
    sachet = _det(b"sachet", "k_sachet", 0.8)
    result, _ = _run(
        [box, sachet],
        {b"box": _ocr("LOTBOX", "2026-01-01"), b"sachet": _ocr("LOTSACHET", "2026-01-01")},
    )
    assert result["lot_box"] == "LOTBOX"
    assert result["lot_sachet"] == "LOTSACHET"


def test_duplicate_box_does_not_become_sachet():
    """Regression guard for the confirmed bug: 2 detections of the SAME
    'k_box' class (no real sachet at all) must never populate lot_sachet/
    exp_sachet — the lower-confidence duplicate must be discarded, not
    misread as the sachet crop."""
    box_high = _det(b"box_high", "k_box", 0.81)
    box_low = _det(b"box_low", "k_box", 0.20)
    result, bbox = _run(
        [box_high, box_low],
        {
            b"box_high": _ocr("REALLOT", "2026-01-01"),
            b"box_low": _ocr("GHOSTLOT", "2026-01-01"),
        },
    )
    assert result["lot_box"] == "REALLOT"
    assert result["lot_sachet"] is None
    assert result["exp_sachet"] is None
    assert bbox == [0, 0, 1, 1]


def test_zero_box_one_sachet():
    """Box occluded/undetected but sachet visible — must not mislabel the
    sachet detection as the box."""
    sachet = _det(b"sachet", "k_sachet", 0.8)
    result, _ = _run([sachet], {b"sachet": _ocr("LOTSACHET", "2026-01-01")})
    assert result["lot_box"] is None
    assert result["exp_box"] is None
    assert result["lot_sachet"] == "LOTSACHET"


def test_heuristic_fallback_single_detection_treated_as_box():
    """YOLO didn't run / found nothing — the single heuristic detection
    (class_name=None) must still be treated as the box, as before."""
    heuristic = _det(b"full_image", None, None)
    result, _ = _run([heuristic], {b"full_image": _ocr("LOTBOX", "2026-01-01")})
    assert result["lot_box"] == "LOTBOX"
    assert result["lot_sachet"] is None


def test_zero_detections():
    result, bbox = _run([], {})
    assert result["lot_box"] is None
    assert result["lot_sachet"] is None
    assert result["status"] == "not_found"
    assert bbox is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_multi_region.py -v`
Expected: `test_duplicate_box_does_not_become_sachet` and `test_zero_box_one_sachet` FAIL (these are the two cases the current positional-indexing code gets wrong); the other 4 may PASS already since positional indexing happens to be correct in those cases — that's fine, they'll still pass after the fix and now serve as regression coverage.

- [ ] **Step 3: Implement — replace positional indexing with class-based selection**

In `pipeline/pipeline_runner.py`, add this module-level helper directly above the `PipelineRunner` class (after the existing imports/logger, before `class PipelineRunner:`):

```python
def _select_region(detections, target_class):
    """Pick the detection matching target_class exactly; if several match,
    the highest-confidence one wins. A single class_name=None detection
    (pure heuristic fallback, e.g. container_label's white-box locator) is
    treated as the box region, since that's the only region heuristics
    target today."""
    matches = [d for d in detections if d.class_name == target_class]
    if not matches and all(d.class_name is None for d in detections):
        matches = detections if target_class.endswith("_box") else []
    return max(matches, key=lambda d: d.conf or 0.0, default=None)
```

Replace the body of `_run_multi_region` (currently lines 55-93) with:

```python
    def _run_multi_region(
        self, image_bytes: bytes, config: PackagingConfig
    ) -> tuple[dict, object]:
        """Multi-crop OCR ใช้กับ container_label (กล่อง + ซอง แยก lot/date)"""
        detections = self._detector.crop_all(image_bytes, config.key)
        box_det = _select_region(detections, f"{config.key}_box")
        sachet_det = _select_region(detections, f"{config.key}_sachet")

        selected = (box_det is not None) + (sachet_det is not None)
        if selected < len(detections):
            logger.warning(
                "%s: %d detection(s) discarded (not selected as box or sachet)",
                config.key, len(detections) - selected,
            )

        crops: list[dict] = []
        for det in (box_det, sachet_det):
            if det is None:
                crops.append({})
                continue
            processed = self._preprocessor.run(det.cropped_bytes, config.key)
            ocr_res = self._ocr_engine.run(processed, config=config)
            text_ok = len(ocr_res["raw_text"].strip()) >= 8
            data_ok = bool(ocr_res["lot_number"] or ocr_res["exp_date"])
            if not text_ok and not data_ok:
                logger.warning(
                    "Preprocessed crop appears degraded — retrying with original bytes"
                )
                ocr_res = self._ocr_engine.run(det.cropped_bytes, config=config)
            crops.append({
                "lot_number": ocr_res["lot_number"],
                "exp_date":   ocr_res["exp_date"],
            })

        box, sachet = crops[0], crops[1]
        result = {
            "lot_number":   None,
            "exp_date":     None,
            "mfg_date":     None,
            "raw_text":     "",
            "confidence":   None,
            "product_name": None,
            "size":         None,
            "lot_box":      box.get("lot_number"),
            "lot_sachet":   sachet.get("lot_number"),
            "exp_box":      box.get("exp_date"),
            "exp_sachet":   sachet.get("exp_date"),
            "status":       "ok" if any(c.get("lot_number") for c in crops) else "not_found",
        }
        bbox = box_det.bbox if box_det else (sachet_det.bbox if sachet_det else None)
        return result, bbox
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_region.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest tests/test_routing.py tests/test_detector.py tests/test_multi_region.py -v`
Expected: all PASS or SKIPPED (SKIPPED only for the model-dependent tests in `test_detector.py` if `models/detector.pt` is absent), none newly FAIL

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline_runner.py tests/test_multi_region.py
git commit -m "fix: select container_label box/sachet by class_name, not list position

A duplicate container_label_box detection could previously land in the
'sachet' slot by position, causing OCR to misread the box region a second
time and fabricate lot_sachet/exp_sachet values with no real sachet present."
```

---

## Post-implementation note

This plan does not include changing `DETECTOR_CONF` — that threshold decision was deliberately deferred (see the design doc's "Out of scope"). Once both tasks are merged, the earlier confidence-sweep finding (2/300 `container_label` photos, 19/1036 photos overall) can be revisited with this bug no longer in the way.
