# Multi-field grouped crops (shared-box) — design

## Context

`multi_field` detection mode (see `2026-06-14-multi-field-detection-design.md`) gives
each field its own YOLO crop class `{key}_{field}`, OCR'd separately and assigned to
that field's extractor. That spec deliberately scoped out **mixed layouts where fields
share a crop** (1 field : 1 box, strict).

A new packaging breaks that assumption: its print puts **lot + expiry in one box** and
**product + size in another**. Other new packagings keep them separate. The grouping is
a fixed property of each packaging's print layout, decided once when the class is added
— it does **not** vary per photo.

Today the wizard cannot express "these two fields share a box": Step 1 only ticks
individual fields, and `_run_multi_field` keys crops by a single field
(`field = cls.removeprefix(prefix)`).

**Goal:** let a `multi_field` packaging declare that one crop holds **≥1 field**. OCR the
shared crop once, run each member field's extractor on that same text. Unblock building
shared-box packagings through the wizard end-to-end (define → annotate → train).

**Out of scope (deliberate, YAGNI):**
- Per-photo variable grouping. Groups are fixed per packaging (the print layout is fixed).
- A field belonging to more than one group. Groups **partition** the field set.
- Touching `single` / `cross_check` modes.

## Core model: groups as composite `sub_regions` entries

Keep `sub_regions: list[str]`. Each entry is now a **group** — one or more field tokens
joined by `_`, in canonical order `[lot, exp, product, size]`:

| layout | `sub_regions` | YOLO classes |
|---|---|---|
| all separate (current 1:1) | `["lot", "exp", "product", "size"]` | `{key}_lot`, `{key}_exp`, … |
| lot+exp share, product+size share | `["lot_exp", "product_size"]` | `{key}_lot_exp`, `{key}_product_size` |
| lot+exp share, product alone, size alone | `["lot_exp", "product", "size"]` | `{key}_lot_exp`, `{key}_product`, `{key}_size` |

Rules:
- Each token ∈ `{lot, exp, product, size}` (the known extractable fields).
- **Partition:** every field appears in exactly one group; no token repeats across groups.
- The union of all groups must contain `lot`.
- Groups are canonicalized (tokens sorted by field order, deduped) so `lot_exp` and
  `exp_lot` never produce two classes.
- A single-token group (`"lot"`) splits to `["lot"]` — identical to today's 1:1
  behaviour. **Fully backward compatible.**

Field tokens have no `_` of their own, so splitting a group on `_` is unambiguous (the
`{key}_` prefix — which may itself contain `_`, e.g. `back_label_` — is stripped first
via `removeprefix`, exactly as today).

## Architecture changes

### 1. Shared constant — `FIELD_ORDER`
Add `FIELD_ORDER = ["lot", "exp", "product", "size"]` in one place (alongside the
existing field/validator wiring) and import it where canonicalization or validation
needs it. The wizard mirrors the same order in JS (it already lists these four).

### 2. `pipeline/pipeline_runner.py` — `_run_multi_field`
Generalize the grouping loop only. Today:
```python
field = cls[len(prefix):]
text = self._ocr_engine.run(processed, config=config)["raw_text"]
texts.setdefault(field, []).append(text)
```
Becomes:
```python
group = cls[len(prefix):]
text = self._ocr_engine.run(processed, config=config)["raw_text"]
for f in group.split("_"):
    texts.setdefault(f, []).append(text)
```
The result assembly (`lot = find_lot(joined("lot"))`, `exp`, `product`, `size`) is
**unchanged** — `texts` is still keyed by field. When lot+exp share a crop, that crop's
text is fed to both `find_lot` and `find_expiry`; each keeps only its own pattern.

`raw_text` is built from **per-crop** text collected once per detection (not from the
re-keyed `texts` dict), so a shared crop's text is not duplicated.

Output dict shape is unchanged (top-level `lot_number`/`exp_date`/`product_name`/`size`),
so `sheet_checker.py`, `message_builder.py`, and `main.py` need no change.

### 3. `api/schemas.py` — validation
`PackagingCreate._check_detection_mode`, `multi_field` branch:
- split each `sub_regions` entry on `_`; every token must be in `FIELD_ORDER`
- no token may repeat across groups (partition)
- the union of tokens must contain `lot`
- canonicalize each entry (sort tokens by `FIELD_ORDER`, dedupe within a group) and
  reject if two entries canonicalize to the same group

`cross_check` (≥2 sub_regions) and `single` (empty/`["lot"]`) branches unchanged.

### 4. `web/wizard.html` — Step 1 group picker + plumbing
In the `#sr-fields` (multi_field) block, each ticked field gains a **"กรอบ #"**
selector (1..N). Fields sharing a number form one group. Default: each ticked field gets
its own number → all single-token groups → identical to today.

- New `srFieldGroups()`: collect ticked fields by box number → join each group's fields
  in canonical order → return composite `sub_regions` (e.g. `["lot_exp", "product_size"]`).
- `step1Next`: `multi_field` → `sub_regions = srFieldGroups()`; validate non-empty and
  `lot` present (mirror server rules for a friendly inline error).
- Step 4 `fields_extracted` pre-fills from the **union** of all group fields.
- Annotator: `renderLabelBar` chips display a prettified group label (`lot + exp`) while
  the stored `label` stays the composite string (`lot_exp`). `dataset_publisher`,
  `label_lines`, and `annotLabelColor` are unchanged — everything still keys off
  `{key}_{sr}`.

### 5. No migration
No deployed packaging uses `multi_field` yet, so there is nothing to migrate. `single`
and `cross_check` packagings are untouched.

## Data flow (lot+exp shared example)

1. Step 1: tick lot/exp/product/size; set lot & exp to กรอบ 1, product & size to กรอบ 2
   → `sub_regions = ["lot_exp", "product_size"]`, `detection_mode = "multi_field"`.
2. Annotate ≥30 images: label-bar shows two chips `lot + exp`, `product + size`; tag
   each box with its group.
3. Full Training: `data.yaml` gains `{key}_lot_exp`, `{key}_product_size`; new
   `detector.pt`.
4. Inference: detector returns a `{key}_lot_exp` box → OCR once → `find_lot` and
   `find_expiry` both run on that text; `{key}_product_size` box → `find_product_name`
   and `find_size`. Result has populated `lot_number`/`exp_date`/`product_name`/`size`.

## Testing

| level | test |
|---|---|
| unit | canonicalize helper: `exp_lot` → `lot_exp`; dedupe within group |
| unit | schema: `["lot_exp","product_size"]` valid; reject repeated field across groups; reject unknown token; reject union without `lot`; single-token (`["lot","exp"]`) still valid |
| unit | `_run_multi_field`: detection class `{key}_lot_exp` → both `lot_number` and `exp_date` extracted from one crop; 4 fields in 2 groups → all four populated; single-token group → unchanged 1:1 result |
| unit | `raw_text` not duplicated when a crop feeds two fields |
| regression | existing 1:1 `multi_field` pipeline + schema tests stay green |
| regression | `tests/test_api_packagings.py` (incl. the `sub_regions` GET surfacing) green |

Output shape equals the prior `multi_field`, so `sheet_checker`, `message_builder`,
`test_image.py`, `evaluate.py` need no new tests. Real retrain (≥30 labeled images) is a
user-side step, not automated.

## End-to-end verification
1. Wizard: create `multi_field` draft, group lot+exp into กรอบ 1, product+size into
   กรอบ 2 → draft saved with `sub_regions: [lot_exp, product_size]`.
2. Reopen annotator → two group chips appear; tag boxes.
3. `pytest tests/test_routing.py tests/test_api_packagings.py` green.
4. Unit tests for `_run_multi_field` grouped extraction green.
