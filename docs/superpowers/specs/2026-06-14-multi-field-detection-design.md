# Multi-field detection mode — design

## Context

The wizard can only build packagings whose detector locates **one logical region**
("lot"). At inference the detector crops every `{key}_*` YOLO class, stacks them,
runs one OCR, and regex extracts lot/exp/product/size from the combined text
(`pipeline/pipeline_runner.py` `_run_single_region`). The only alternative today is
`container_label`'s cross-check mode, inferred from `len(sub_regions) > 1`.

A new packaging is blocked: its four fields (lot, exp, product/name, size) are
printed in **different corners** of the bag. Stacking all crops and running regex on
the combined text risks mis-assigning text across fields (size regex grabbing part of
the name, etc.). The user needs each field located and read **independently**.

This is impossible now because:
1. The wizard offers no way to declare a crop per field — single mode is hardcoded to
   `['lot']`, advanced mode is for `box`/`sachet` cross-check only.
2. Routing infers cross-check from `len(sub_regions) > 1`, so any multi-region
   packaging would be wrongly routed to value-comparison (`lot == exp == name`?) and
   break.

**Goal:** add a `multi_field` detection mode where each selected field has its own
YOLO crop class `{key}_{field}`, OCR'd separately and assigned directly to that field
— no cross-field regex contamination. Unblock building the new packaging through the
wizard end-to-end (define → annotate → train).

**Out of scope (deliberate, YAGNI):**
- Mixed layouts where some fields share a crop and others are separate (e.g. lot+exp
  together, name separate). `multi_field` requires every field to own a crop (1:1).
- Prelabel mapping for `multi_field` edit-drafts (retraining). Deferred — see below.

## Core model: `detection_mode`

Replace the implicit `len(sub_regions) > 1` routing with an explicit config field.
`detection_mode` is a sub-classification of `pipeline == detector_ocr`:

| mode | packagings | `sub_regions` | runtime |
|---|---|---|---|
| `single` (default) | back_label, grade_bag, capsule_box, retail_sachet | `[]` — detector crops all `{key}_*`, stacks | 1 OCR + regex extracts all fields (current `_run_single_region`) |
| `cross_check` | container_label | `[box, sachet]` | OCR each crop, compare same value (current `_run_multi_region`) |
| `multi_field` (new) | the new packaging | `[lot, exp, product, size]` (== `fields_extracted`) | OCR each crop separately, assign its text to its own field via that field's extractor |

`multi_field` rule (MVP): `sub_regions == fields_extracted`, 1:1. Each entry becomes
YOLO class `{key}_{field}`. Each field's crop text is run through the existing
`utils/validators` extractor for that field:
- `_lot` crop → `find_lot`
- `_exp` crop → `find_expiry`
- `_product` crop → `find_product_name`
- `_size` crop → `find_size`

## Architecture changes

### 1. `pipeline/detector.py` — carry the matched class name
`DetectionResult` currently holds `cropped_bytes` + `bbox` only. `multi_field` must
know which crop is which field. Add `class_name: str | None`, populated in
`_yolo_crop_all` from `boxes.cls` (already available, currently discarded). Additive —
`single`/`cross_check` ignore it. Heuristic fallback sets `class_name = None`.

### 2. `pipeline/pipeline_runner.py` — dispatch + new runner
```python
if config.detection_mode == "cross_check": return self._run_multi_region(...)
if config.detection_mode == "multi_field": return self._run_multi_field(...)
return self._run_single_region(...)
```
`_run_multi_field` (sibling of `_run_multi_region`, reuses its per-crop OCR loop):
- group detections by `det.class_name.removeprefix(f"{key}_")`
- per field: preprocess → OCR each crop → concat text if multiple crops same field →
  run that field's extractor
- **returns the same dict shape as `_run_single_region`**: top-level `lot_number`,
  `exp_date`, `product_name`, `size`, `raw_text`, `status` (NOT `lot_box`/`lot_sachet`)

Because the output shape equals `single`, `utils/sheet_checker.py`,
`pipeline/message_builder.py`, and `main.py` need **no change**.

### 3. Routing decouple — replace 3 `len(sub_regions) > 1` checks
Switch to `detection_mode`:
- `pipeline_runner.run()` — dispatch (above)
- `utils/sheet_checker.py` `is_container` → `detection_mode == "cross_check"`
- `main.py` `lot_for_message` → `detection_mode == "cross_check"`

`multi_field` has top-level `lot_number` like `single`, so both downstream checks
treat it as non-container automatically — correct.

### 4. Schema + validation
- `pipeline/packaging_registry.py` `PackagingConfig`: add `detection_mode: str =
  "single"` (default keeps old YAML working).
- `api/schemas.py`: `PackagingCreate` / draft meta accept `detection_mode` (enum
  validated). Validation at create/promote:
  - `multi_field` → `sub_regions` non-empty, contains `lot`, equals `fields_extracted`
  - `cross_check` → `len(sub_regions) >= 2`
  - `single` → `sub_regions` empty

### 5. Migration — add `detection_mode` to existing YAMLs
`container_label` → `cross_check`; `back_label`, `grade_bag`, `capsule_box`,
`retail_sachet`, `import_sticker` → `single`. Only `container_label` had
`len > 1` today and maps cleanly to `cross_check`; nothing else relied on the old
inference.

### 6. Wizard — Step 1 crop picker + plumbing
Reuse existing annotation machinery (no annotation rewrite): the label-bar chips
already appear when `sub_regions.length > 1` (`renderLabelBar`), and
`dataset_publisher.label_lines` already maps each bbox by its own `label` field to
`{key}_{field}`. So once a `multi_field` draft has `sub_regions = [lot, exp, product,
size]`, annotation and training work as-is.

New work is only in `web/wizard.html` Step 1:
- Add a third crop-picker choice beyond single / advanced(box+sachet):
  "แต่ละข้อมูลอยู่คนละมุม" → `multi_field`. Selecting it reveals field checkboxes
  (lot required + exp/product/size).
- On selection: `sub_regions` = ticked fields, `detection_mode = "multi_field"`,
  pre-fill Step 4 `fields_extracted` to match.
- Thread `detection_mode` through `step1Next` → `POST /api/packagings` → draft meta →
  `services/cloudrun_deployer.py` (promote) → written YAML.

### Deferred: prelabel for `multi_field` edit-drafts
`services/active_learning.py` `filter_prelabel_bboxes` hardcodes `label: "prelabel"`,
discarding the YOLO class name. For a future `multi_field` **edit-draft** retrain,
prelabeled boxes would not carry the real field label, so ops must tag manually. This
does NOT affect the new packaging (manual annotation, no parent). Tracked as follow-up:
map the detector's class name → `{field}` label in prelabel.

## Testing

| level | test |
|---|---|
| unit | `_run_multi_field`: fake detections w/ `class_name` → correct field assignment; multiple crops same field → concatenated text; missing field → `None` |
| unit | `DetectionResult.class_name` populated in `_yolo_crop_all` |
| unit | dispatch picks correct branch per `detection_mode` (3 modes) |
| unit | per-field validators on isolated text (lot/exp/product/size) |
| unit | schema validation (multi_field 1:1; cross_check ≥2; single empty) |
| regression | `tests/test_routing.py` + `tests/test_api_packagings.py` pass (migrate any `len`-based asserts to `detection_mode`) |
| integration | build `multi_field` draft → annotate (mock) → `dataset_publisher` emits `{key}_{field}` classes + `label_lines` maps correctly |

Output shape equals `single`, so `sheet_checker`, `message_builder`, `test_image.py`,
`evaluate.py` need no new tests. Real retrain (≥30 labeled images) is a user-side step,
not automated.

## End-to-end verification
1. Wizard: create draft, choose "แต่ละข้อมูลอยู่คนละมุม", tick lot/exp/product/size →
   draft saved with `detection_mode: multi_field`, `sub_regions: [lot, exp, product,
   size]`.
2. Annotate ≥30 images: label-bar shows 4 field chips; tag each box.
3. Full Training: `data.yaml` gains `{key}_lot/_exp/_product/_size`; new `detector.pt`.
4. `python test_image.py <photo> <sheet_id> <gid>`: returns populated
   `lot_number/exp_date/product_name/size`, each from its own crop.
5. `pytest tests/test_routing.py tests/test_api_packagings.py` green.
