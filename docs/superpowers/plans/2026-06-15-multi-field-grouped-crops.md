# Multi-field Grouped Crops (Shared Box) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `multi_field` packaging declare that one YOLO crop holds ≥1 field (e.g. lot+exp in one box, product+size in another), OCR the shared crop once, and run each member field's extractor on that text.

**Architecture:** A `sub_regions` entry becomes a *group* — field tokens joined by `_` in canonical `FIELD_ORDER` (`"lot_exp"`). A shared `utils/field_groups.py` parses/canonicalizes/validates groups; the schema validates on create, `_run_multi_field` splits each detection's group and feeds every member field's extractor, and the wizard Step 1 gains a per-field "box number" picker that composes the groups. A single-token group (`"lot"`) is identical to today's 1:1 behaviour, so the change is fully backward compatible. `dataset_publisher` and `cloudrun_deployer` already key off `{key}_{sr}`, so composite entries flow through unchanged.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest, Ultralytics YOLOv8 (inference only here), vanilla JS + HTML wizard (`web/wizard.html`), Playwright for frontend verification.

**Design spec:** `docs/superpowers/specs/2026-06-15-multi-field-grouped-crops-design.md`

---

## File Structure

- **Create** `utils/field_groups.py` — group parsing/canonicalization/validation. One responsibility: the `sub_regions` group encoding. Imported by both `api/schemas.py` and `pipeline/pipeline_runner.py` so the encoding lives in exactly one place.
- **Create** `tests/test_field_groups.py` — unit tests for the helper + schema validation behaviour.
- **Modify** `api/schemas.py` — `PackagingCreate._check_detection_mode` calls `validate_groups` for `multi_field` and stores the canonicalized groups.
- **Modify** `pipeline/pipeline_runner.py` — `_run_multi_field` splits each detection's group into fields, feeds each field's extractor, and builds `raw_text` per-crop (no duplication).
- **Modify** `tests/test_detection_mode.py` — add grouped-crop pipeline + deployer regression tests alongside the existing 1:1 ones.
- **Modify** `web/wizard.html` — Step 1 `#sr-fields` gains a per-field box-number picker + live group preview; new `srFieldGroups()` / `srGroupChanged()` / `srPickBox()`; `step1Next` and `renderLabelBar` updated.

No changes needed in `services/dataset_publisher.py`, `services/cloudrun_deployer.py`, `utils/sheet_checker.py`, `pipeline/message_builder.py`, or `main.py` — `_run_multi_field`'s output shape is unchanged and all of these already key off `{key}_{sr}` / top-level `lot_number`.

---

## Task 1: `utils/field_groups.py` — group encoding helper

**Files:**
- Create: `utils/field_groups.py`
- Test: `tests/test_field_groups.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_field_groups.py`:

```python
"""Unit tests for multi_field group encoding (utils/field_groups)."""

import pytest

from utils.field_groups import (
    FIELD_ORDER,
    parse_group,
    canonicalize_group,
    validate_groups,
)


def test_field_order_is_the_four_known_fields():
    assert FIELD_ORDER == ["lot", "exp", "product", "size"]


def test_parse_group_splits_on_underscore():
    assert parse_group("lot_exp") == ["lot", "exp"]
    assert parse_group("lot") == ["lot"]
    assert parse_group("") == []


def test_canonicalize_sorts_by_field_order_and_dedupes():
    assert canonicalize_group("exp_lot") == "lot_exp"
    assert canonicalize_group("size_product") == "product_size"
    assert canonicalize_group("lot_lot") == "lot"


def test_validate_groups_canonicalizes_each_entry():
    assert validate_groups(["exp_lot", "size_product"]) == ["lot_exp", "product_size"]


def test_validate_groups_accepts_single_token_groups_backward_compat():
    assert validate_groups(["lot", "exp", "product", "size"]) == [
        "lot", "exp", "product", "size"]


def test_validate_groups_rejects_unknown_token():
    with pytest.raises(ValueError, match="ไม่รู้จัก"):
        validate_groups(["lot_weight"])


def test_validate_groups_rejects_field_in_two_groups():
    with pytest.raises(ValueError, match="กรอบเดียว"):
        validate_groups(["lot_exp", "exp_size"])


def test_validate_groups_rejects_union_without_lot():
    with pytest.raises(ValueError, match="lot"):
        validate_groups(["exp_product", "size"])


def test_validate_groups_rejects_empty_group():
    with pytest.raises(ValueError, match="ว่าง"):
        validate_groups(["lot", ""])


def test_validate_groups_rejects_duplicate_group_after_canonicalize():
    with pytest.raises(ValueError, match="ซ้ำ"):
        validate_groups(["lot_exp", "exp_lot"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_field_groups.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.field_groups'`

- [ ] **Step 3: Write the implementation**

Create `utils/field_groups.py`:

```python
"""Field grouping for multi_field detection mode.

A *group* is one or more field tokens that share a single YOLO crop. It is
encoded as a `sub_regions` entry — tokens joined by '_' in canonical
`FIELD_ORDER` (e.g. 'lot_exp' = lot and expiry printed in one box). A
single-token group ('lot') is the 1:1 case and behaves exactly as before.

Field tokens never contain '_', so splitting a group on '_' is unambiguous
once the '{key}_' class prefix has been stripped by the caller.
"""

FIELD_ORDER = ["lot", "exp", "product", "size"]


def parse_group(group: str) -> list[str]:
    """Split a group entry into its field tokens (drops empties)."""
    return [t for t in group.split("_") if t]


def canonicalize_group(group: str) -> str:
    """Return the group's tokens sorted by FIELD_ORDER, deduped."""
    tokens = parse_group(group)
    return "_".join(f for f in FIELD_ORDER if f in tokens)


def validate_groups(groups: list[str]) -> list[str]:
    """Validate multi_field groups and return them canonicalized.

    Rules:
    - every token must be a known field (FIELD_ORDER)
    - groups partition the fields — no field appears in two groups
    - the union of all fields must contain 'lot'
    - no empty group
    - two entries must not canonicalize to the same group

    Raises ValueError (Thai message) on any violation.
    """
    canon: list[str] = []
    seen_fields: set[str] = set()
    seen_groups: set[str] = set()
    for g in groups:
        tokens = parse_group(g)
        if not tokens:
            raise ValueError("กลุ่มว่าง — แต่ละกรอบต้องมีอย่างน้อย 1 field")
        for t in tokens:
            if t not in FIELD_ORDER:
                raise ValueError(
                    f"field ไม่รู้จัก '{t}' — ใช้ได้แค่ {', '.join(FIELD_ORDER)}")
            if t in seen_fields:
                raise ValueError(f"field '{t}' อยู่ได้กรอบเดียว — เจอซ้ำข้ามกลุ่ม")
            seen_fields.add(t)
        cg = canonicalize_group(g)
        if cg in seen_groups:
            raise ValueError(f"กลุ่มซ้ำ '{cg}' — รวมเป็นกรอบเดียว")
        seen_groups.add(cg)
        canon.append(cg)
    if "lot" not in seen_fields:
        raise ValueError("multi_field ต้องมี field 'lot' อยู่ในกรอบใดกรอบหนึ่ง")
    return canon
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_field_groups.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/field_groups.py tests/test_field_groups.py
git commit -m "feat: field_groups helper for multi_field shared-box crops"
```

---

## Task 2: Wire `validate_groups` into the create schema

**Files:**
- Modify: `api/schemas.py:46-54` (`PackagingCreate._check_detection_mode`)
- Test: `tests/test_field_groups.py` (add schema cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_field_groups.py`:

```python
from pydantic import ValidationError
from api.schemas import PackagingCreate


def test_schema_accepts_grouped_multi_field_and_canonicalizes():
    m = PackagingCreate(
        key="k", display_name="K",
        detection_mode="multi_field",
        sub_regions=["exp_lot", "size_product"],
    )
    assert m.sub_regions == ["lot_exp", "product_size"]


def test_schema_rejects_field_shared_across_groups():
    with pytest.raises(ValidationError):
        PackagingCreate(
            key="k", display_name="K",
            detection_mode="multi_field",
            sub_regions=["lot_exp", "exp_size"],
        )


def test_schema_rejects_grouped_multi_field_without_lot():
    with pytest.raises(ValidationError):
        PackagingCreate(
            key="k", display_name="K",
            detection_mode="multi_field",
            sub_regions=["exp_product", "size"],
        )


def test_schema_single_token_multi_field_still_valid():
    m = PackagingCreate(
        key="k", display_name="K",
        detection_mode="multi_field",
        sub_regions=["lot", "exp"],
    )
    assert m.sub_regions == ["lot", "exp"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_field_groups.py -k schema -q`
Expected: FAIL — `test_schema_accepts_grouped_multi_field_and_canonicalizes` fails because the current validator does not canonicalize (`sub_regions` stays `["exp_lot", "size_product"]`), and the cross-group / missing-lot cases are not rejected.

- [ ] **Step 3: Edit the implementation**

In `api/schemas.py`, add the import near the top (after the existing imports):

```python
from utils.field_groups import validate_groups
```

Replace the `multi_field` branch of `PackagingCreate._check_detection_mode` (currently lines 48-50):

```python
        if self.detection_mode == DetectionMode.multi_field:
            if "lot" not in self.sub_regions:
                raise ValueError("multi_field ต้องมี sub-region 'lot' (ทุก field ต้องมีกรอบของตัวเอง)")
```

with:

```python
        if self.detection_mode == DetectionMode.multi_field:
            self.sub_regions = validate_groups(self.sub_regions)
```

Leave the `cross_check` and `single` branches unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_field_groups.py -q`
Expected: PASS (13 passed)

Then run the existing detection-mode tests to confirm no regression:

Run: `python -m pytest tests/test_detection_mode.py -q`
Expected: PASS (the existing `test_multi_field_requires_lot_sub_region`, `test_multi_field_accepts_lot_plus_fields`, etc. still pass — `validate_groups(["lot","exp","product","size"])` returns them unchanged, and `["exp","size"]` raises ValueError → ValidationError).

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py tests/test_field_groups.py
git commit -m "feat: validate + canonicalize grouped sub_regions in PackagingCreate"
```

---

## Task 3: `_run_multi_field` grouped extraction

**Files:**
- Modify: `pipeline/pipeline_runner.py:94-135` (`_run_multi_field`)
- Test: `tests/test_detection_mode.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detection_mode.py` (the fakes `_FakeDetector`, `_FakePre`, `_FakeOcr`, `_cfg_multi` already exist in this file — reuse them; do not redefine):

```python
def _cfg_grouped(sub_regions):
    cfg = _cfg_multi()
    return cfg.__class__(**{**cfg.__dict__, "sub_regions": sub_regions})


def test_multi_field_shared_box_feeds_both_extractors(monkeypatch):
    # one crop holds lot AND exp; one crop holds product AND size
    dets = [
        DetectionResult(cropped_bytes=b"LOTEXP", bbox=[0, 0, 1, 1], class_name="k_lot_exp"),
        DetectionResult(cropped_bytes=b"PRODSIZE", bbox=[0, 1, 1, 2], class_name="k_product_size"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: f"LOT::{t}")
    monkeypatch.setattr(pr, "find_expiry", lambda t: f"EXP::{t}")
    monkeypatch.setattr(pr, "find_product_name", lambda t: f"PROD::{t}")
    monkeypatch.setattr(pr, "find_size", lambda t: f"SIZE::{t}")

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, _ = runner._run_multi_field(b"img", _cfg_grouped(["lot_exp", "product_size"]))

    assert result["lot_number"] == "LOT::LOTEXP"
    assert result["exp_date"] == "EXP::LOTEXP"        # exp read from the SAME crop as lot
    assert result["product_name"] == "PROD::PRODSIZE"
    assert result["size"] == "SIZE::PRODSIZE"
    assert result["status"] == "ok"


def test_multi_field_raw_text_not_duplicated_for_shared_crop(monkeypatch):
    dets = [
        DetectionResult(cropped_bytes=b"LOTEXP", bbox=[0, 0, 1, 1], class_name="k_lot_exp"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: "L")
    monkeypatch.setattr(pr, "find_expiry", lambda t: "E")
    monkeypatch.setattr(pr, "find_product_name", lambda t: None)
    monkeypatch.setattr(pr, "find_size", lambda t: None)

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, _ = runner._run_multi_field(b"img", _cfg_grouped(["lot_exp"]))

    # the shared crop's text appears once, not once per member field
    assert result["raw_text"].count("LOTEXP") == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_detection_mode.py -k "shared_box or raw_text_not" -q`
Expected: FAIL — current `_run_multi_field` does `field = cls[len(prefix):]` (yielding field `"lot_exp"`, which no extractor matches) so `lot_number`/`exp_date` stay `None`; and `raw_text` joins per re-keyed field.

- [ ] **Step 3: Edit the implementation**

In `pipeline/pipeline_runner.py`, add the import near the top with the other pipeline imports:

```python
from utils.field_groups import parse_group
```

Replace the body of `_run_multi_field` (lines 100-135) with:

```python
        detections = self._detector.crop_all(image_bytes, config.key)
        prefix = f"{config.key}_"
        texts: dict[str, list[str]] = {}
        raw_parts: list[str] = []
        for det in detections:
            cls = det.class_name or ""
            if not cls.startswith(prefix):
                continue
            group = cls[len(prefix):]
            processed = self._preprocessor.run(det.cropped_bytes, config.key)
            text = self._ocr_engine.run(processed, config=config)["raw_text"]
            raw_parts.append(text)
            for field in parse_group(group):
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
            "raw_text":     "\n".join(raw_parts),
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

The only behavioural changes: `parse_group(group)` fans one crop's text out to every member field, and `raw_text` is built from `raw_parts` (one entry per crop) instead of the re-keyed `texts` dict. The 1:1 case (`group == "lot"`) is unchanged because `parse_group("lot") == ["lot"]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_detection_mode.py -q`
Expected: PASS — both new tests and the existing `test_multi_field_routes_each_crop_to_its_field` (1:1) pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline_runner.py tests/test_detection_mode.py
git commit -m "feat: _run_multi_field feeds shared-box crop text to every member field"
```

---

## Task 4: Deployer regression test for grouped classes

**Files:**
- Test: `tests/test_detection_mode.py`

No production change — `services/cloudrun_deployer.write_packaging_yaml` already derives `detector_yolo_prefixes` as `{key}_{sr}` over `sub_regions`, so grouped entries yield the correct composite class names. This task locks that behaviour with a test.

- [ ] **Step 1: Write the test**

Append to `tests/test_detection_mode.py`:

```python
def test_deployer_writes_grouped_class_prefixes(tmp_path, monkeypatch):
    import services.cloudrun_deployer as dep
    monkeypatch.setattr(dep, "_PACKAGING_DIR", tmp_path)
    draft_meta = {
        "display_name": "Grouped Pkg",
        "pipeline": "detector_ocr",
        "sub_regions": ["lot_exp", "product_size"],
        "detection_mode": "multi_field",
        "config": {
            "lot_patterns": ["LOT(\\w+)"],
            "fields_extracted": ["lot", "exp", "product", "size"],
            "sheet_checks": ["lot"], "message_template_key": "default_full",
        },
    }
    out = dep.write_packaging_yaml("grouppkg", draft_meta)
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["detection_mode"] == "multi_field"
    assert data["sub_regions"] == ["lot_exp", "product_size"]
    assert data["detector_yolo_prefixes"] == ["grouppkg_lot_exp", "grouppkg_product_size"]
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_detection_mode.py -k grouped_class_prefixes -q`
Expected: PASS. If it FAILS because `write_packaging_yaml` derives prefixes differently (e.g. it splits on `_`), open `services/cloudrun_deployer.py`, confirm the derivation is `[f"{key}_{sr}" for sr in sub_regions]`, and only then adjust. Do NOT change the derivation to split groups — the composite name IS the YOLO class.

- [ ] **Step 3: Commit**

```bash
git add tests/test_detection_mode.py
git commit -m "test: deployer emits composite YOLO class for grouped sub_regions"
```

---

## Task 5: Wizard Step 1 — box-number group picker

Frontend guidance applied (frontend-design, web-interface-guidelines, ui-ux-pro-max): native `<select>` gets an explicit non-transparent `background-color` + styled `option` (Chromium-on-Windows dark-popup gotcha, see CLAUDE.md); each select has an `aria-label`; `:focus-visible` ring (never `outline:none` without replacement); disabled selects use reduced opacity + `not-allowed` cursor; the field rows are wrapped in a `<fieldset>`/`<legend>` for grouping; group membership is conveyed by number + label text + a colored dot (not color alone); a live `aria-live="polite"` preview makes the abstract grouping concrete; the preview is the one "signature" element and everything else stays quiet. No animation added (so nothing to gate on `prefers-reduced-motion`).

**Files:**
- Modify: `web/wizard.html` — CSS (near the existing `.sr-*` / `.cbitem` rules), `#sr-fields` markup (lines 1333-1339), and JS (`srSetMode`, new `srFieldGroups`/`srGroupChanged`/`srPickBox`/`renderGroupPreview`, `step1Next`, `renderLabelBar`).

- [ ] **Step 1: Add CSS for the group picker**

Find the `.annot-label-chip` block (around line 790) and add this CSS just after it (any spot inside the same `<style>` works):

```css
.sr-grp-set{border:1px solid var(--bd);border-radius:10px;padding:10px 12px;margin:0;display:flex;flex-direction:column;gap:8px}
.sr-grp-legend{font-size:11px;color:var(--t3);padding:0 6px;letter-spacing:.04em;text-transform:uppercase}
.sr-grp-row{display:flex;align-items:center;gap:10px}
.sr-grp-row .cbitem{flex:1;margin:0}
.sr-grp-box{background-color:var(--s1);color:var(--t1);border:1px solid var(--bd);border-radius:8px;padding:5px 8px;font-size:12px;min-width:88px;min-height:32px;cursor:pointer}
.sr-grp-box option{background:var(--s1);color:var(--t1)}
.sr-grp-box:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.sr-grp-box:disabled{opacity:.4;cursor:not-allowed}
.sr-grp-preview{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;min-height:26px}
.sr-grp-pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:4px 10px;border-radius:999px;color:var(--t1)}
.sr-grp-pill-dot{width:8px;height:8px;border-radius:50%;flex:none}
```

(If `--s1`, `--t1`, `--t3`, `--bd`, `--acc` are not the names used in this file, match the existing variables already used by `.annot-label-chip` and `.sr-row`.)

- [ ] **Step 2: Replace the `#sr-fields` markup**

Replace lines 1333-1339 (`<div id="sr-fields">…</div>`) with:

```html
                <!-- MULTI_FIELD — each box holds ≥1 field; same box number = shared crop -->
                <div id="sr-fields" style="display:none">
                  <div class="sr-multi-head"><span>ติ๊ก field ที่ packaging นี้มี แล้วเลือก "กรอบที่" — field ที่อยู่กรอบเลขเดียวกันจะถูกอ่านจากกรอบเดียวกัน (เช่น lot กับวันหมดอายุ พิมพ์ติดกัน)</span></div>
                  <fieldset class="sr-grp-set">
                    <legend class="sr-grp-legend">Field และกรอบ</legend>
                    <div class="sr-grp-row">
                      <label class="cbitem on disabled"><input type="checkbox" data-field="lot" checked disabled> lot</label>
                      <select class="sr-grp-box" data-box-for="lot" aria-label="กรอบของ lot" onchange="srPickBox('lot', this.value)"></select>
                    </div>
                    <div class="sr-grp-row">
                      <label class="cbitem"><input type="checkbox" data-field="exp" onchange="srGroupChanged()"> วันหมดอายุ</label>
                      <select class="sr-grp-box" data-box-for="exp" aria-label="กรอบของ วันหมดอายุ" onchange="srPickBox('exp', this.value)"></select>
                    </div>
                    <div class="sr-grp-row">
                      <label class="cbitem"><input type="checkbox" data-field="product" onchange="srGroupChanged()"> ชื่อสินค้า</label>
                      <select class="sr-grp-box" data-box-for="product" aria-label="กรอบของ ชื่อสินค้า" onchange="srPickBox('product', this.value)"></select>
                    </div>
                    <div class="sr-grp-row">
                      <label class="cbitem"><input type="checkbox" data-field="size" onchange="srGroupChanged()"> ขนาด</label>
                      <select class="sr-grp-box" data-box-for="size" aria-label="กรอบของ ขนาด" onchange="srPickBox('size', this.value)"></select>
                    </div>
                  </fieldset>
                  <div class="sr-grp-preview" id="sr-grp-preview" aria-live="polite"></div>
                </div>
```

- [ ] **Step 3: Add the group-picker JS**

In the `<script>` section, find `let cropMode = 'single';` (line ~2410) and add directly after it:

```javascript
const FIELD_ORDER = ['lot', 'exp', 'product', 'size'];
let mfBox = { lot: 1, exp: 2, product: 3, size: 4 };   // field → box number

function mfTickedFields() {
  return FIELD_ORDER.filter(f => {
    const cb = document.querySelector(`#sr-fields input[data-field="${f}"]`);
    return cb && cb.checked;
  });
}

function srGroupChanged() {
  const k = Math.max(1, mfTickedFields().length);
  FIELD_ORDER.forEach(f => {
    const sel = document.querySelector(`#sr-fields select[data-box-for="${f}"]`);
    const cb = document.querySelector(`#sr-fields input[data-field="${f}"]`);
    if (!sel || !cb) return;
    sel.disabled = !cb.checked;
    let cur = mfBox[f] || 1;
    if (cur > k) { cur = k; }
    mfBox[f] = cur;
    sel.innerHTML = Array.from({ length: k }, (_, i) => i + 1)
      .map(n => `<option value="${n}"${n === cur ? ' selected' : ''}>กรอบ ${n}</option>`)
      .join('');
  });
  renderGroupPreview();
}

function srPickBox(field, val) {
  mfBox[field] = parseInt(val, 10) || 1;
  srGroupChanged();
}

function srFieldGroups() {
  const byBox = {};
  mfTickedFields().forEach(f => { (byBox[mfBox[f]] ||= []).push(f); });
  return Object.keys(byBox).sort((a, b) => a - b)
    .map(b => FIELD_ORDER.filter(f => byBox[b].includes(f)).join('_'));
}

function prettyGroup(g) { return g.split('_').join(' + '); }

function renderGroupPreview() {
  const el = document.getElementById('sr-grp-preview');
  if (!el) return;
  el.innerHTML = srFieldGroups().map((g, i) => {
    const color = ANNOT_COLORS[i % ANNOT_COLORS.length];
    return `<span class="sr-grp-pill" style="background:${color}22;border:1px solid ${color}66">
      <span class="sr-grp-pill-dot" style="background:${color}"></span>กรอบ ${i + 1}: ${esc(prettyGroup(g))}</span>`;
  }).join('');
}
```

- [ ] **Step 4: Populate the picker when multi_field mode opens**

In `srSetMode` (line ~2423), find:

```javascript
  if (mode === 'cross_check') srRender();
```

and replace it with:

```javascript
  if (mode === 'cross_check') srRender();
  if (mode === 'multi_field') srGroupChanged();
```

- [ ] **Step 5: Use the groups in `step1Next`**

In `step1Next` (line ~2527), replace:

```javascript
  else if (cropMode === 'multi_field') { sub_regions = srFieldValues(); }
```

with:

```javascript
  else if (cropMode === 'multi_field') { sub_regions = srFieldGroups(); }
```

Then, immediately after the `cropMode === 'cross_check'` validation block (after line ~2538, the closing `}` of that `if`), add a `multi_field` guard:

```javascript
  if (cropMode === 'multi_field') {
    const union = sub_regions.join('_').split('_');
    if (!sub_regions.length || !union.includes('lot')) {
      alert('โหมดแยก field ต้องมี lot อย่างน้อย 1 กรอบ'); return;
    }
  }
```

(`srFieldValues()` is still used by `prefillFieldsFromSubRegions` to tick Step 4 fields from the ticked union — leave it unchanged.)

- [ ] **Step 6: Prettify the annotator label chips**

In `renderLabelBar` (line ~2688), replace:

```javascript
        <span class="annot-label-chip-dot"></span>${esc(r)}
```

with:

```javascript
        <span class="annot-label-chip-dot"></span>${esc(r.split('_').join(' + '))}
```

The stored bbox `label` stays the composite string (`lot_exp`); only the chip's display text changes. `annotLabelColor`, `dataset_publisher`, and `label_lines` are unaffected.

- [ ] **Step 7: Verify `srFieldGroups()` logic with Playwright**

Serve the static wizard (no backend needed for pure-logic check):

Run: `python -m http.server 8090 --directory web` (in a background shell)

Then drive it with Playwright (`mcp__plugin_playwright_playwright__*`):
1. `browser_navigate` → `http://localhost:8090/wizard.html`
2. `browser_evaluate` →
```js
() => {
  startWizard();                 // render step 1 standalone
  srSetMode('multi_field');
  // tick all four fields
  document.querySelectorAll('#sr-fields input[data-field]').forEach(cb => { cb.checked = true; });
  // group lot+exp into box 1, product+size into box 2
  mfBox = { lot: 1, exp: 1, product: 2, size: 2 };
  srGroupChanged();
  return { groups: srFieldGroups(), preview: document.getElementById('sr-grp-preview').textContent.trim() };
}
```
Expected return: `groups` = `["lot_exp", "product_size"]`; `preview` contains `กรอบ 1: lot + exp` and `กรอบ 2: product + size`.
3. `browser_take_screenshot` of `#sr-fields` for a visual sanity check (selects readable, pills colored, focus ring visible on tab).

Expected: groups assertion holds and the picker renders legibly on the dark theme.

- [ ] **Step 8: Commit**

```bash
git add web/wizard.html
git commit -m "feat: wizard Step 1 box-number group picker for multi_field shared crops"
```

---

## Task 6: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, except the known pre-existing 3 setup errors in `tests/test_classifier.py` (efficientnet_b0 vs efficientnet_v2_s fixture — documented in CLAUDE.md, not caused by this work). No other failures.

- [ ] **Step 2: Targeted re-run of the touched areas**

Run: `python -m pytest tests/test_field_groups.py tests/test_detection_mode.py tests/test_api_packagings.py tests/test_routing.py tests/test_dataset_publisher.py -q`
Expected: PASS (all green).

- [ ] **Step 3: Commit (only if any test files needed touch-ups)**

```bash
git add -A
git commit -m "test: regression sweep for multi_field grouped crops"
```

---

## Self-Review

**Spec coverage:**
- Core model (groups as composite `sub_regions`, canonical order, partition, lot-in-union, single-token backward compat) → Task 1 (`field_groups`) + Task 2 (schema). ✓
- `_run_multi_field` group split + per-crop `raw_text` → Task 3. ✓
- Schema validation (unknown token, partition, lot, dedupe) → Task 1 + Task 2. ✓
- Shared `FIELD_ORDER` constant in one place → Task 1 (`utils/field_groups.FIELD_ORDER`), mirrored in JS Task 5. ✓
- No migration (single/cross_check untouched) → no task needed; existing `test_detection_mode.py` cases re-run in Tasks 2/6 confirm. ✓
- Wizard Step 1 group picker, `fields_extracted` union prefill, chip prettify, dataset/publisher unchanged → Task 5 (prefill via existing `srFieldValues`/`prefillFieldsFromSubRegions`, untouched). ✓
- Deployer composite classes → Task 4. ✓
- Testing matrix (canonicalize, schema, pipeline grouped + single, raw_text dedup, regression) → Tasks 1-6. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type/name consistency:** `FIELD_ORDER`, `parse_group`, `canonicalize_group`, `validate_groups` used identically across Tasks 1-3. JS `srFieldGroups`/`srGroupChanged`/`srPickBox`/`mfBox`/`mfTickedFields`/`renderGroupPreview`/`prettyGroup` defined once (Task 5 Step 3) and referenced consistently in Steps 4-6. `_cfg_grouped` defined once (Task 3) and reused (no redefinition of the shared fakes). ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-multi-field-grouped-crops.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.

Which approach?
