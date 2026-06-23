# Multi-field Detection Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `multi_field` detection mode so a packaging whose fields (lot/exp/product/size) are printed in different corners can be built through the wizard — each field gets its own YOLO crop class, OCR'd separately and assigned directly to that field.

**Architecture:** Introduce an explicit `detection_mode` config field (`single | cross_check | multi_field`) that replaces the implicit `len(sub_regions) > 1` routing. A new `PipelineRunner._run_multi_field` reuses the per-crop OCR pattern from `_run_multi_region` but assigns each crop's text to its field instead of cross-checking. Output shape equals `single` so `sheet_checker`/`message_builder`/`main` are untouched. The wizard's existing per-bbox label machinery and `dataset_publisher` class generation are reused as-is; only Step 1 UI and `detection_mode` plumbing are new.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest, Ultralytics YOLO, vanilla JS (wizard.html).

**Spec:** `docs/superpowers/specs/2026-06-14-multi-field-detection-design.md`

**Before starting:** create a feature branch — `git checkout -b feat/multi-field-detection` (repo default branch is `main`; do not commit feature work to `main`). Stage only the files named in each task's commit step (the working tree holds unrelated in-progress changes — never `git add -A`).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pipeline/packaging_registry.py` | `PackagingConfig` dataclass + YAML loader | add `detection_mode` field + load it |
| `config/packagings/*.yaml` | per-packaging config | add `detection_mode` line (6 files) |
| `pipeline/detector.py` | YOLO crop | add `class_name` to `DetectionResult` |
| `pipeline/pipeline_runner.py` | route + run pipeline | dispatch on `detection_mode`; add `_run_multi_field` |
| `utils/sheet_checker.py` | sheet cross-check | `is_container` keys on `detection_mode` |
| `main.py` | response assembly | `lot_for_message` keys on `detection_mode` |
| `api/schemas.py` | request models | `DetectionMode` enum + `PackagingCreate.detection_mode` + validation |
| `services/packaging_store.py` | draft persistence | store `detection_mode` in draft meta |
| `api/packagings.py` | wizard endpoints | pass `detection_mode`; multi_field training gate |
| `services/cloudrun_deployer.py` | draft → YAML on promote | write `detection_mode` + multi_field `detector_yolo_prefixes` |
| `web/wizard.html` | Step 1 UI | multi_field crop option + field ticks + plumbing |
| `tests/test_routing.py` | routing tests | migrate to `detection_mode` |
| `tests/test_detection_mode.py` | new unit tests | config load, `_run_multi_field`, schema, label mapping |

---

## Task 1: Add `detection_mode` to config + migrate YAMLs

**Files:**
- Modify: `pipeline/packaging_registry.py:29` (dataclass), `pipeline/packaging_registry.py:87` (loader)
- Modify: `config/packagings/container_label.yaml`, `back_label.yaml`, `grade_bag.yaml`, `capsule_box.yaml`, `retail_sachet.yaml`, `import_sticker.yaml`
- Test: `tests/test_detection_mode.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detection_mode.py`:

```python
"""Multi-field detection mode — config load, routing, schema, label mapping."""

from pipeline.packaging_registry import PackagingRegistry


def test_existing_packagings_have_expected_detection_mode():
    reg = PackagingRegistry()
    assert reg.get("container_label").detection_mode == "cross_check"
    for k in ("back_label", "grade_bag", "capsule_box", "retail_sachet"):
        assert reg.get(k).detection_mode == "single"


def test_detection_mode_defaults_to_single_when_missing(tmp_path, monkeypatch):
    # A config without detection_mode loads as "single"
    reg = PackagingRegistry()
    # import_sticker.yaml has detection_mode: single after migration
    assert reg.get("import_sticker").detection_mode == "single"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detection_mode.py -v`
Expected: FAIL — `AttributeError: 'PackagingConfig' object has no attribute 'detection_mode'`

- [ ] **Step 3: Add the dataclass field**

In `pipeline/packaging_registry.py`, change the last field of `PackagingConfig` (line 29) from:

```python
    sub_regions: list[str]           # ['box', 'sachet'] for container_label
```
to:
```python
    sub_regions: list[str]           # ['box', 'sachet'] for container_label
    detection_mode: str = "single"   # 'single' | 'cross_check' | 'multi_field'
```

(Must be last — it is the only field with a default.)

- [ ] **Step 4: Load it in the loader**

In `pipeline/packaging_registry.py`, in the `PackagingConfig(...)` construction (around line 87), add after `sub_regions=data.get("sub_regions", []),`:

```python
                detection_mode=data.get("detection_mode", "single"),
```

- [ ] **Step 5: Migrate the 6 YAML files**

Add a `detection_mode` line under the `sub_regions` line in each file.

`config/packagings/container_label.yaml` — add:
```yaml
detection_mode: cross_check
```
`config/packagings/back_label.yaml`, `grade_bag.yaml`, `capsule_box.yaml`, `retail_sachet.yaml`, `import_sticker.yaml` — add to each:
```yaml
detection_mode: single
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_detection_mode.py -v`
Expected: PASS (both tests)

- [ ] **Step 7: Commit**

```bash
git add pipeline/packaging_registry.py config/packagings/container_label.yaml config/packagings/back_label.yaml config/packagings/grade_bag.yaml config/packagings/capsule_box.yaml config/packagings/retail_sachet.yaml config/packagings/import_sticker.yaml tests/test_detection_mode.py
git commit -m "feat: add detection_mode config field + migrate packagings"
```

---

## Task 2: Carry YOLO class name on `DetectionResult`

**Files:**
- Modify: `pipeline/detector.py:21-24` (dataclass), `pipeline/detector.py:113-114` (populate)
- Test: `tests/test_detection_mode.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detection_mode.py`:

```python
from pipeline.detector import DetectionResult


def test_detection_result_has_class_name_default_none():
    d = DetectionResult(cropped_bytes=b"x", bbox=[0, 0, 1, 1])
    assert d.class_name is None
    d2 = DetectionResult(cropped_bytes=b"x", bbox=[0, 0, 1, 1], class_name="k_lot")
    assert d2.class_name == "k_lot"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detection_mode.py::test_detection_result_has_class_name_default_none -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'class_name'`

- [ ] **Step 3: Add the field**

In `pipeline/detector.py`, change `DetectionResult` (lines 21-24) to:

```python
@dataclass
class DetectionResult:
    cropped_bytes: bytes   # JPEG bytes ของ region ที่ crop แล้ว
    bbox: Bbox | None      # [x1, y1, x2, y2] relative ต่อรูปต้นฉบับ
    class_name: str | None = None   # YOLO class of the matched box (e.g. "back_label_size")
```

- [ ] **Step 4: Populate it in `_yolo_crop_all`**

In `pipeline/detector.py`, inside the loop in `_yolo_crop_all` (around lines 106-114), the code currently appends:

```python
            cropped = img[y1:y2, x1:x2]
            detections.append(DetectionResult(cropped_bytes=_to_jpeg(cropped), bbox=[x1, y1, x2, y2]))
```

Change the append to include the class name (the loop already has `idx` and `class_names`):

```python
            cropped = img[y1:y2, x1:x2]
            detections.append(DetectionResult(
                cropped_bytes=_to_jpeg(cropped),
                bbox=[x1, y1, x2, y2],
                class_name=class_names.get(cls_ids[idx]),
            ))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_detection_mode.py::test_detection_result_has_class_name_default_none -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/detector.py tests/test_detection_mode.py
git commit -m "feat: carry matched YOLO class name on DetectionResult"
```

---

## Task 3: Route on `detection_mode` + `_run_multi_field`

**Files:**
- Modify: `pipeline/pipeline_runner.py:1-6` (imports), `:31-37` (dispatch), add `_run_multi_field`
- Modify: `utils/sheet_checker.py:107`
- Modify: `main.py:188-192`
- Modify: `tests/test_routing.py` (migrate to detection_mode)
- Test: `tests/test_detection_mode.py`

- [ ] **Step 1: Write the failing test for `_run_multi_field`**

Append to `tests/test_detection_mode.py`:

```python
import pipeline.pipeline_runner as pr
from pipeline.pipeline_runner import PipelineRunner
from pipeline.packaging_registry import PackagingConfig


def _cfg_multi(key="k"):
    return PackagingConfig(
        key=key, display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot", "exp", "product", "size"],
        sheet_checks=["lot"], post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label=key, detector_yolo_prefixes=[f"{key}_lot"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=["lot", "exp", "product", "size"], detection_mode="multi_field",
    )


class _FakeDetector:
    def __init__(self, dets): self._dets = dets
    def crop_all(self, image_bytes, key): return self._dets


class _FakePre:
    def run(self, b, key): return b


class _FakeOcr:
    def run(self, b, config=None):
        return {"raw_text": b.decode(), "lot_number": None}


def test_multi_field_routes_each_crop_to_its_field(monkeypatch):
    dets = [
        DetectionResult(cropped_bytes=b"LOTTEXT", bbox=[0, 0, 1, 1], class_name="k_lot"),
        DetectionResult(cropped_bytes=b"EXPTEXT", bbox=[0, 1, 1, 2], class_name="k_exp"),
        DetectionResult(cropped_bytes=b"NAMETEXT", bbox=[0, 2, 1, 3], class_name="k_product"),
        DetectionResult(cropped_bytes=b"SIZETEXT", bbox=[0, 3, 1, 4], class_name="k_size"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: f"LOT::{t}")
    monkeypatch.setattr(pr, "find_expiry", lambda t: f"EXP::{t}")
    monkeypatch.setattr(pr, "find_product_name", lambda t: f"PROD::{t}")
    monkeypatch.setattr(pr, "find_size", lambda t: f"SIZE::{t}")

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, bbox = runner._run_multi_field(b"img", _cfg_multi())

    assert result["lot_number"] == "LOT::LOTTEXT"
    assert result["exp_date"] == "EXP::EXPTEXT"
    assert result["product_name"] == "PROD::NAMETEXT"
    assert result["size"] == "SIZE::SIZETEXT"
    assert result["lot_box"] is None      # shape == single, NOT cross_check
    assert result["status"] == "ok"
    assert bbox == [0, 0, 1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detection_mode.py::test_multi_field_routes_each_crop_to_its_field -v`
Expected: FAIL — `AttributeError: 'PipelineRunner' object has no attribute '_run_multi_field'`

- [ ] **Step 3: Add validator imports to pipeline_runner**

In `pipeline/pipeline_runner.py`, after the existing imports (line 4 `from utils.image_utils import stack_images_vertically`), add:

```python
from utils.validators import find_lot, find_expiry, find_product_name, find_size
```

- [ ] **Step 4: Add the dispatch branch**

In `pipeline/pipeline_runner.py` `run()`, replace lines 31-37:

```python
        # Multi-region only when there are 2+ physical crop locations
        # (e.g. container_label box+sachet). A lone ["lot"] is single-region —
        # see CONTEXT.md "Sub-region" / pipeline_runner routing.
        if len(config.sub_regions) > 1:
            return self._run_multi_region(image_bytes, config)

        return self._run_single_region(image_bytes, config)
```
with:
```python
        # Detection mode is the explicit router (see CONTEXT.md / spec
        # 2026-06-14-multi-field-detection). cross_check = compare same value
        # across crops; multi_field = each crop is a different field.
        if config.detection_mode == "cross_check":
            return self._run_multi_region(image_bytes, config)
        if config.detection_mode == "multi_field":
            return self._run_multi_field(image_bytes, config)
        return self._run_single_region(image_bytes, config)
```

- [ ] **Step 5: Implement `_run_multi_field`**

In `pipeline/pipeline_runner.py`, add this method after `_run_multi_region`:

```python
    def _run_multi_field(
        self, image_bytes: bytes, config: PackagingConfig
    ) -> tuple[dict, object]:
        """multi_field — each field has its own crop class {key}_{field}.
        OCR each crop separately and assign its text to that field's extractor.
        Returns the SAME dict shape as _run_single_region (no lot_box/sachet)."""
        detections = self._detector.crop_all(image_bytes, config.key)
        prefix = f"{config.key}_"
        texts: dict[str, list[str]] = {}
        for det in detections:
            cls = det.class_name or ""
            if not cls.startswith(prefix):
                continue
            field = cls[len(prefix):]
            processed = self._preprocessor.run(det.cropped_bytes, config.key)
            text = self._ocr_engine.run(processed, config=config)["raw_text"]
            texts.setdefault(field, []).append(text)

        def joined(field: str) -> str:
            return "\n".join(texts.get(field, [])).strip()

        lot_text = joined("lot")
        lot = (
            find_lot(lot_text, image_class=config.key, patterns=config.lot_patterns)
            if lot_text else None
        )
        result = {
            "lot_number":   lot,
            "exp_date":     find_expiry(joined("exp")) if texts.get("exp") else None,
            "mfg_date":     None,
            "product_name": find_product_name(joined("product")) if texts.get("product") else None,
            "size":         find_size(joined("size")) if texts.get("size") else None,
            "raw_text":     "\n".join(joined(f) for f in texts),
            "confidence":   None,
            "lot_box":      None,
            "lot_sachet":   None,
            "exp_box":      None,
            "exp_sachet":   None,
            "status":       "ok" if lot else "not_found",
        }
        bbox = detections[0].bbox if detections else None
        return result, bbox
```

- [ ] **Step 6: Run the multi_field test**

Run: `python -m pytest tests/test_detection_mode.py::test_multi_field_routes_each_crop_to_its_field -v`
Expected: PASS

- [ ] **Step 7: Update `sheet_checker` is_container**

In `utils/sheet_checker.py`, replace line 107:

```python
            is_container = len(config.sub_regions) > 1
```
with:
```python
            is_container = config.detection_mode == "cross_check"
```

- [ ] **Step 8: Update `main.py` lot_for_message**

In `main.py`, replace lines 188-192:

```python
    lot_for_message = (
        result.get("lot_box")
        if len(config.sub_regions) > 1
        else result.get("lot_number")
    )
```
with:
```python
    lot_for_message = (
        result.get("lot_box")
        if config.detection_mode == "cross_check"
        else result.get("lot_number")
    )
```

- [ ] **Step 9: Migrate `tests/test_routing.py` to detection_mode**

Replace the entire body of `tests/test_routing.py` with:

```python
"""Routing keys off detection_mode (single | cross_check | multi_field), the
explicit router that replaced the old len(sub_regions) inference. See
CONTEXT.md "Sub-region" and spec 2026-06-14-multi-field-detection."""

import pytest

from pipeline.packaging_registry import PackagingConfig
from pipeline.pipeline_runner import PipelineRunner
from utils.sheet_checker import SheetChecker


def _cfg(detection_mode="single", sub_regions=(), sheet_checks=("lot",)):
    return PackagingConfig(
        key="k", display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot"], sheet_checks=list(sheet_checks),
        post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label="k", detector_yolo_prefixes=["k_lot"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=list(sub_regions), detection_mode=detection_mode,
    )


@pytest.fixture
def runner():
    return PipelineRunner(
        detector=object(), preprocessor=object(),
        ocr_engine=object(), qr_scanner=object(),
    )


@pytest.mark.parametrize("mode, expected", [
    ("single", "single"),
    ("cross_check", "multi"),
    ("multi_field", "field"),
])
def test_run_dispatch(runner, monkeypatch, mode, expected):
    calls = []
    monkeypatch.setattr(runner, "_run_single_region",
                        lambda b, c: (calls.append("single"), ({}, None))[1])
    monkeypatch.setattr(runner, "_run_multi_region",
                        lambda b, c: (calls.append("multi"), ({}, None))[1])
    monkeypatch.setattr(runner, "_run_multi_field",
                        lambda b, c: (calls.append("field"), ({}, None))[1])
    runner.run(b"img", _cfg(detection_mode=mode))
    assert calls == [expected]


@pytest.mark.parametrize("mode, expected_lot_key", [
    ("single", "lot_number"),
    ("cross_check", "lot_box"),
    ("multi_field", "lot_number"),
])
def test_sheet_checker_lot_source(monkeypatch, mode, expected_lot_key):
    checker = SheetChecker()
    seen = {}

    def fake_find(lot, sheet_id, gid):
        seen["lot"] = lot
        return None

    monkeypatch.setattr(checker, "_find_row_by_lot", fake_find)
    checker.check(
        _cfg(detection_mode=mode),
        sheet_id="sid", gid=0,
        lot_number="LOTNUM", lot_box="LOTBOX", lot_sachet="LOTSACHET",
        exp_date="2026-01-01", exp_box="2026-01-01", exp_sachet="2026-01-01",
    )
    expected = {"lot_number": "LOTNUM", "lot_box": "LOTBOX"}[expected_lot_key]
    assert seen["lot"] == expected
```

- [ ] **Step 10: Run routing + detection tests**

Run: `python -m pytest tests/test_routing.py tests/test_detection_mode.py -v`
Expected: PASS (all)

- [ ] **Step 11: Commit**

```bash
git add pipeline/pipeline_runner.py utils/sheet_checker.py main.py tests/test_routing.py tests/test_detection_mode.py
git commit -m "feat: route pipeline on detection_mode + add _run_multi_field"
```

---

## Task 4: Schema validation + backend plumbing

**Files:**
- Modify: `api/schemas.py:5` (import), `:10-37` (enum + PackagingCreate)
- Modify: `services/packaging_store.py:55-78` (create_draft), `:144-160` (clone_draft)
- Modify: `api/packagings.py:90-96` (create_packaging), `:468-471` (training gate)
- Modify: `services/cloudrun_deployer.py:62-83` (YAML build)
- Test: `tests/test_detection_mode.py`

- [ ] **Step 1: Write the failing schema tests**

Append to `tests/test_detection_mode.py`:

```python
import pytest
from pydantic import ValidationError
from api.schemas import PackagingCreate


def test_multi_field_requires_lot_sub_region():
    with pytest.raises(ValidationError):
        PackagingCreate(key="k", display_name="K",
                        detection_mode="multi_field", sub_regions=["exp", "size"])


def test_multi_field_accepts_lot_plus_fields():
    m = PackagingCreate(key="k", display_name="K",
                        detection_mode="multi_field",
                        sub_regions=["lot", "exp", "product", "size"])
    assert m.detection_mode == "multi_field"
    assert "lot" in m.sub_regions


def test_cross_check_requires_two_sub_regions():
    with pytest.raises(ValidationError):
        PackagingCreate(key="k", display_name="K",
                        detection_mode="cross_check", sub_regions=["box"])


def test_default_detection_mode_is_single():
    m = PackagingCreate(key="k", display_name="K")
    assert m.detection_mode == "single"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_mode.py -k "multi_field_requires or cross_check_requires or default_detection" -v`
Expected: FAIL — `PackagingCreate` has no `detection_mode` / no validation

- [ ] **Step 3: Add `DetectionMode` enum + field + validator**

In `api/schemas.py`, change the import line 7:

```python
from pydantic import BaseModel, Field, field_validator
```
to:
```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

After the `PipelineType` enum (after line 12), add:

```python
class DetectionMode(str, Enum):
    single = "single"
    cross_check = "cross_check"
    multi_field = "multi_field"
```

In `PackagingCreate`, add the field after `sub_regions` (line 20):

```python
    detection_mode: DetectionMode = DetectionMode.single
```

Add this method to `PackagingCreate` (after `_validate_sub_regions`):

```python
    @model_validator(mode="after")
    def _check_detection_mode(self):
        if self.detection_mode == DetectionMode.multi_field:
            if "lot" not in self.sub_regions:
                raise ValueError("multi_field ต้องมี sub-region 'lot' (ทุก field ต้องมีกรอบของตัวเอง)")
        elif self.detection_mode == DetectionMode.cross_check:
            if len(self.sub_regions) < 2:
                raise ValueError("cross_check ต้องมีอย่างน้อย 2 sub-regions (เช่น box, sachet)")
        return self
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run: `python -m pytest tests/test_detection_mode.py -k "multi_field_requires or multi_field_accepts or cross_check_requires or default_detection" -v`
Expected: PASS

- [ ] **Step 5: Thread `detection_mode` through `create_draft`**

In `services/packaging_store.py`, change `create_draft` signature (lines 55-61) to add the param:

```python
def create_draft(
    key: str,
    display_name: str,
    description: str | None,
    pipeline: str,
    sub_regions: list[str] | None = None,
    detection_mode: str = "single",
) -> dict:
```

In the meta dict it builds (after the `"sub_regions": sub_regions or ["lot"],` line ~75), add:

```python
        "detection_mode": detection_mode,
```

In `clone_draft`'s meta dict (after `"sub_regions": active_yaml.get("sub_regions", []) or ["lot"],` line ~150), add:

```python
        "detection_mode": active_yaml.get("detection_mode", "single"),
```

- [ ] **Step 6: Pass it from `create_packaging`**

In `api/packagings.py` `create_packaging` (lines 90-96), add to the `create_draft(...)` call after `sub_regions=body.sub_regions,`:

```python
            detection_mode=body.detection_mode.value,
```

- [ ] **Step 7: Add the multi_field training gate**

In `api/packagings.py` `training_full_start`, after the labeled-count gate (after line 471), add:

```python
    if draft.get("detection_mode") == "multi_field":
        cfg = draft.get("config") or {}
        subs = draft.get("sub_regions", [])
        missing = [f for f in cfg.get("fields_extracted", []) if f not in subs]
        if missing:
            raise HTTPException(
                400,
                f"multi_field: fields {missing} have no crop sub-region — "
                "add them in step 1 or untick them in step 4",
            )
```

- [ ] **Step 8: Write `detection_mode` + multi_field prefixes in the YAML on promote**

In `services/cloudrun_deployer.py`, in the `data = {...}` dict (lines 62-83), first compute the derived values just before the dict (after line 60, the `edited = {...}` block):

```python
    sub_regions_final = draft_meta.get("sub_regions") or existing.get("sub_regions", [])
    detection_mode = draft_meta.get("detection_mode") or existing.get("detection_mode", "single")
    default_prefixes = (
        [f"{key}_{sr}" for sr in sub_regions_final]
        if detection_mode == "multi_field" and sub_regions_final
        else [f"{key}_lot"]
    )
```

Then in the `data` dict, replace the `"sub_regions": ...` line (70) and the `"detector_yolo_prefixes": ...` line (82):

```python
        "sub_regions": sub_regions_final,
```
```python
        "detector_yolo_prefixes": existing.get("detector_yolo_prefixes", default_prefixes),
```

And add a `"detection_mode"` entry after `"sub_regions"`:

```python
        "detection_mode": detection_mode,
```

- [ ] **Step 9: Write the deployer unit test**

Append to `tests/test_detection_mode.py`:

```python
def test_deployer_writes_multi_field_mode_and_prefixes(tmp_path, monkeypatch):
    import services.cloudrun_deployer as dep
    monkeypatch.setattr(dep, "_PACKAGING_DIR", tmp_path)
    draft_meta = {
        "display_name": "New Pkg",
        "pipeline": "detector_ocr",
        "sub_regions": ["lot", "exp", "product", "size"],
        "detection_mode": "multi_field",
        "config": {
            "lot_patterns": ["LOT(\\w+)"], "fields_extracted": ["lot", "exp", "product", "size"],
            "sheet_checks": ["lot"], "message_template_key": "default_full",
        },
    }
    out = dep.write_packaging_yaml("newpkg", draft_meta)
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["detection_mode"] == "multi_field"
    assert data["detector_yolo_prefixes"] == [
        "newpkg_lot", "newpkg_exp", "newpkg_product", "newpkg_size"]
```

> NOTE: confirm the public function name in `services/cloudrun_deployer.py` (the one containing the `data = {...}` block at lines 62-83). If it is not `write_packaging_yaml`, use the actual name in the test. Read the `def` above line 42 to confirm.

- [ ] **Step 10: Run Task 4 tests**

Run: `python -m pytest tests/test_detection_mode.py -v`
Expected: PASS (all)

- [ ] **Step 11: Commit**

```bash
git add api/schemas.py services/packaging_store.py api/packagings.py services/cloudrun_deployer.py tests/test_detection_mode.py
git commit -m "feat: validate + plumb detection_mode through wizard backend"
```

---

## Task 5: Wizard Step 1 UI — multi_field option

**Files:**
- Modify: `web/wizard.html` crop-picker HTML (around lines 1283-1316) and Step-1 JS (around lines 2380-2509)

> This task has no automated test harness (vanilla JS). Verify by serving the page and driving it with Playwright per CLAUDE.md ("serve `web/` and `page.evaluate('startWizard()')`").

- [ ] **Step 1: Add the third crop-picker choice (HTML)**

In `web/wizard.html`, the crop picker currently has `#sr-single` (default) and `#sr-multi` (advanced box/sachet). Add a mode selector and a `#sr-fields` block. Inside `#crop-fgroup` (after the label at line 1281), add three radio-style chips bound to `srSetMode`:

```html
                <div class="sr-mode-tabs">
                  <button type="button" class="sr-mode-tab on" data-mode="single" onclick="srSetMode('single')">ข้อมูลอยู่ใกล้กัน — กรอบเดียว</button>
                  <button type="button" class="sr-mode-tab" data-mode="multi_field" onclick="srSetMode('multi_field')">แต่ละข้อมูลอยู่คนละมุม</button>
                  <button type="button" class="sr-mode-tab" data-mode="cross_check" onclick="srSetMode('cross_check')">lot อยู่หลายที่ ต้องเทียบ</button>
                </div>
```

After the `#sr-multi` block (after line 1315), add the field-pick block:

```html
                <!-- MULTI_FIELD — each field has its own crop class -->
                <div id="sr-fields" style="display:none">
                  <div class="sr-multi-head"><span>ติ๊ก field ที่มีกรอบ/ตำแหน่งของตัวเอง — แต่ละอันจะเป็นคลาส YOLO แยก</span></div>
                  <label class="cbitem on disabled"><input type="checkbox" data-field="lot" checked disabled> lot</label>
                  <label class="cbitem"><input type="checkbox" data-field="exp"> วันหมดอายุ</label>
                  <label class="cbitem"><input type="checkbox" data-field="product"> ชื่อสินค้า</label>
                  <label class="cbitem"><input type="checkbox" data-field="size"> ขนาด</label>
                </div>
```

- [ ] **Step 2: Update the mode state machine (JS)**

In `web/wizard.html`, the existing `cropMode` var (line 2385) is `'single' | 'multi'`. Generalise it to the three modes. Replace `srSetMode` (lines 2398-2405) with:

```javascript
function srSetMode(mode) {
  cropMode = mode;   // 'single' | 'multi_field' | 'cross_check'
  document.querySelectorAll('.sr-mode-tab').forEach(t =>
    t.classList.toggle('on', t.dataset.mode === mode));
  const show = (id, on) => { const e = document.getElementById(id); if (e) e.style.display = on ? '' : 'none'; };
  show('sr-single', mode === 'single');
  show('sr-fields', mode === 'multi_field');
  show('sr-multi', mode === 'cross_check');
  if (mode === 'cross_check') srRender();
}
```

Replace `srToggleAdvanced` (lines 2407-2412) — the old advanced button now maps to cross_check:

```javascript
function srToggleAdvanced(on) {
  if (on && cropRegions.length < 2) {
    cropRegions = [{ value: 'box', custom: false }, { value: 'sachet', custom: false }];
  }
  srSetMode(on ? 'cross_check' : 'single');
}
```

- [ ] **Step 3: Add a helper to read ticked fields**

In `web/wizard.html`, near `srValues` (line 2478), add:

```javascript
function srFieldValues() {
  const out = ['lot'];
  document.querySelectorAll('#sr-fields input[data-field]').forEach(cb => {
    if (cb.checked && cb.dataset.field !== 'lot') out.push(cb.dataset.field);
  });
  return out;
}
```

- [ ] **Step 4: Build `sub_regions` + `detection_mode` in `step1Next`**

In `web/wizard.html` `step1Next` (lines 2488-2504), replace the sub_regions line and the create call. Replace line 2489:

```javascript
  const sub_regions = cropMode === 'single' ? ['lot'] : srValues();
```
with:
```javascript
  let detection_mode = cropMode;
  let sub_regions;
  if (cropMode === 'single') { sub_regions = ['lot']; }
  else if (cropMode === 'multi_field') { sub_regions = srFieldValues(); }
  else { sub_regions = srValues(); }   // cross_check
```

In the multi validation block (lines 2493-2499), change the guard `if (cropMode === 'multi')` to `if (cropMode === 'cross_check')`.

Update the `api('POST', '/api/packagings', {...})` body (line 2504) to include `detection_mode`:

```javascript
      { key, display_name, description, pipeline, sub_regions, detection_mode });
```

- [ ] **Step 5: Pre-fill Step 4 fields for multi_field**

In `web/wizard.html`, find where Step 4 renders the field checkboxes (the `#sp4 [data-group="fields"]` group). When `cropMode === 'multi_field'`, the Step-4 fields must match `sub_regions`. Add this call at the start of the function that builds/enters Step 4 (the step that contains `collectConfig`; search for where `#sp4` is shown). Add a helper and invoke it on entering Step 4:

```javascript
function prefillFieldsFromSubRegions() {
  if (cropMode !== 'multi_field') return;
  const want = new Set(srFieldValues());
  document.querySelectorAll('#sp4 [data-group="fields"] .cbitem[data-field]').forEach(el => {
    if (el.dataset.field === 'lot') return;
    el.classList.toggle('on', want.has(el.dataset.field));
  });
}
```

Call `prefillFieldsFromSubRegions()` wherever Step 4 becomes visible (e.g. inside the `goStep(4)` path or the Step-4 init). Confirm the exact entry function by searching `goStep` / `sp4` in wizard.html.

- [ ] **Step 6: Verify in browser**

Run (PowerShell, serve then drive):
```bash
python -m http.server 8090 --directory web
```
Then with Playwright (or manually): open `http://localhost:8090/wizard.html`, `page.evaluate("startWizard()")`, click the "แต่ละข้อมูลอยู่คนละมุม" tab, confirm the field checkboxes appear, tick exp/product/size, and confirm (via DevTools/network) that `POST /api/packagings` body has `detection_mode: "multi_field"` and `sub_regions: ["lot","exp","product","size"]`.
Expected: the create request carries the right mode + regions; switching tabs shows/hides the right blocks.

- [ ] **Step 7: Commit**

```bash
git add web/wizard.html
git commit -m "feat: wizard step 1 multi_field crop option + field ticks"
```

---

## Task 6: Integration test — annotation labels → YOLO classes

**Files:**
- Test: `tests/test_detection_mode.py`

This verifies the existing `dataset_publisher` machinery maps a field-labelled bbox to the correct `{key}_{field}` class — the reused path the design relies on.

- [ ] **Step 1: Write the test**

Append to `tests/test_detection_mode.py`:

```python
def test_multi_field_label_lines_map_each_field_to_its_class():
    from services.dataset_publisher import label_lines, merge_class_names
    sub_regions = ["lot", "exp", "product", "size"]
    merged = merge_class_names([], [f"k_{s}" for s in sub_regions])
    label_to_id = {s: merged.index(f"k_{s}") for s in sub_regions}

    bboxes = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "size"},
        {"x1": 0, "y1": 20, "x2": 10, "y2": 30, "label": "lot"},
    ]
    lines = label_lines(bboxes, label_to_id, 100, 100, default_label="lot")
    class_ids = [int(line.split()[0]) for line in lines]
    assert class_ids == [label_to_id["size"], label_to_id["lot"]]
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_detection_mode.py::test_multi_field_label_lines_map_each_field_to_its_class -v`
Expected: PASS (no production change needed — confirms reuse)

> If it fails on import (function names), read `services/dataset_publisher.py:32` (`merge_class_names`) and `:53` (`label_lines`) and adjust the import to the actual names.

- [ ] **Step 3: Full regression**

Run: `python -m pytest tests/test_routing.py tests/test_detection_mode.py tests/test_api_packagings.py -v`
Expected: PASS (all). If any `test_api_packagings.py` test constructs `PackagingConfig` or asserts on `sub_regions`-based routing, migrate it to `detection_mode` the same way as `test_routing.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_detection_mode.py
git commit -m "test: verify multi_field bbox labels map to YOLO field classes"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** detection_mode model (Task 1), detector class_name (Task 2), `_run_multi_field` + routing decouple + downstream-unchanged (Task 3), schema/validation/plumbing + migration + multi_field prefixes (Tasks 1 & 4), wizard Step 1 (Task 5), annotation/publisher reuse verified (Task 6), prelabel deferral (documented in spec, not implemented — intentional). All spec sections map to a task.
- **Placeholders:** none — every code step shows complete code. Two NOTEs flag function-name confirmations (`write_packaging_yaml`, `label_lines`/`merge_class_names`) the executor must verify by reading the cited line — these are real anchors, not TBDs.
- **Type consistency:** `detection_mode: str` everywhere; values `"single"|"cross_check"|"multi_field"` consistent across config, routing, schema enum, deployer, wizard. `DetectionResult.class_name` defined in Task 2, consumed in Task 3. `_run_multi_field` returns the documented single-shape dict.

## End-to-end verification (after all tasks)

1. `python -m pytest` — full suite green (except the known pre-existing `tests/test_classifier.py` 3 setup errors documented in CLAUDE.md).
2. Wizard: create draft → "แต่ละข้อมูลอยู่คนละมุม" → tick lot/exp/product/size → draft saved with `detection_mode: multi_field`.
3. Annotate ≥30 images (4 field chips appear) → Full Training → new `detector.pt` with `{key}_lot/_exp/_product/_size`; `data.yaml` gains those classes.
4. `python test_image.py <photo> <sheet_id> <gid>` → populated `lot_number/exp_date/product_name/size`, each from its own crop.
