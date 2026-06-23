# Runtime Product-Aliases Editor (no retrain)

**Date:** 2026-06-23
**Status:** Approved — ready for implementation

## Problem

A packaging class that reads the product name does so via `product_aliases`
(list of `{canonical, keywords}`) — config-driven OCR text matching. Today those
aliases can only be set in the wizard's **step 4** while building/editing a draft,
and are baked into the packaging YAML at deploy time.

To add or remove a product name that an **already-deployed (active)** class reads,
the operator must currently: clone → edit-draft → re-do step 4 → full training
(~30–60 min) → deploy. That is wildly disproportionate for a change that does not
touch any model — `product_aliases` is pure text matching, exactly like
`conf_threshold`, which is already runtime-tunable without retraining (ADR 0004).

**Goal:** let the operator add/remove product names a class reads, live from the
dashboard drawer, persisted durably (survives Cloud Run revisions), no retrain,
no redeploy.

## Decision

Mirror the `conf_threshold` runtime-override pattern (ADR 0004) exactly:

- Persist the edited `product_aliases` to `config_overrides.json`
  (GCS → Drive → local fallback), in the **same per-key entry** as
  `conf_threshold`.
- `PackagingRegistry` merges the override over the YAML at load time; the YAML
  value becomes the default once an override exists.
- A new `PUT /api/packagings/{key}/product-aliases` endpoint persists the change
  then calls `main.reload_registry()` — persist-first so a storage failure
  returns 502 and changes nothing (no divergence between instance and storage).

**Storage choice:** override file (not a write-back to the packaging YAML).
Reasons: identical to the proven `conf_threshold` path, lowest risk, and keeps the
YAML as the "as-trained default". (Rejected: editing the GCS YAML via
`cloudrun_deployer` — heavier, mixes config edits into the deploy path.)

## Scope

**Scope expanded 2026-06-23 (post-implementation, after local review).** The
editor is exposed for **any active class that reads a product name** (`product`
in `fields_extracted`), not only those that already have `product_aliases`.
Rationale: the only shipped product-reading classes (`back_label`, `grade_bag`)
are exactly the ones on the hardcoded path, so the original "must already have
aliases" gate left zero usable classes.

To contain the footgun, a class currently on the hardcoded path (reads `product`
but has zero `product_aliases`) shows a **prominent red warning** in the drawer:
saving aliases there *replaces the hardcoded tea-list immediately and live* — the
operator must list every product themselves and use the `{size}` token for
size-append, or previously-matched products stop matching. The `ocr_engine`
`if aliases:` branch is unchanged; the warning makes the behavior switch
explicit at the point of edit instead of hiding it.

Backend already permits this (the `PUT` only requires `product` in
`fields_extracted`); the expansion is purely the frontend gate
(`fields_extracted.includes('product')`) plus the warning.

> Original (superseded) scope: editor only for classes with a non-empty
> `product_aliases`, excluding `back_label`/`grade_bag` outright.

## Backend design

### 1. `services/config_overrides.py`

Add `save_product_aliases(key, aliases)` parallel to `save_conf_threshold`:

```python
def save_product_aliases(key: str, aliases: list[dict]) -> dict[str, dict]:
    """Persist a product_aliases override and return the merged overrides.
    Raises on persist failure (caller must not apply the change locally)."""
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    entry = merged.setdefault(key, {})
    entry["product_aliases"] = [
        {"canonical": a["canonical"], "keywords": list(a["keywords"])}
        for a in aliases
    ]
    # ... same GCS / Drive / local write block as save_conf_threshold ...
    return merged
```

The two fields coexist in one entry, e.g.
`{"matcha_sachet": {"conf_threshold": 0.7, "product_aliases": [...]}}`.
`_validated()` already accepts any `{key: dict}` shape, so `product_aliases` as a
list inside the entry passes unchanged.

### 2. `pipeline/packaging_registry.py`

Add `_merged_product_aliases(data, override)` parallel to
`_merged_conf_threshold`:

```python
@staticmethod
def _merged_product_aliases(data: dict, override: object) -> list[dict]:
    yaml_value = data.get("product_aliases", [])
    if not isinstance(override, dict) or "product_aliases" not in override:
        return yaml_value
    ov = override["product_aliases"]
    if not isinstance(ov, list):
        logger.warning("Invalid product_aliases override for %s: %r — using YAML",
                       data.get("key"), ov)
        return yaml_value
    return ov                      # override replaces the YAML list wholesale
```

`_config_from_data` line 93 changes from
`product_aliases=data.get("product_aliases", [])` to
`product_aliases=self._merged_product_aliases(data, override)`.

### 3. `api/schemas.py`

Reuse the existing `ProductAlias` model. Add:

```python
class ProductAliasesUpdate(BaseModel):
    product_aliases: list[ProductAlias] = Field(..., min_length=1)

class ProductAliasesResponse(BaseModel):
    key: str
    product_aliases: list[ProductAlias]
    previous: list[ProductAlias]
```

Extend `PackagingResponse` with two fields the drawer needs to render the editor
(`response_model` silently strips undeclared keys — see CLAUDE.md):

```python
    product_aliases: list[ProductAlias] | None = None
    fields_extracted: list[str] | None = None
```

### 4. `api/packagings.py`

**GET `/{key}`** — in the active branch (lines 138–148), also pass
`product_aliases=cfg.product_aliases` and `fields_extracted=cfg.fields_extracted`
so the drawer knows the current aliases and whether the `size` field is on
(for the `{size}` preview).

**New `PUT /{key}/product-aliases`** (mirror of `/conf`, lines 289–319):

```python
@router.put("/{key}/product-aliases", response_model=ProductAliasesResponse)
def update_product_aliases(key: str, body: ProductAliasesUpdate):
    import main
    cfg = main.registry.get(key) if main.registry else None
    if cfg is None:
        raise HTTPException(404, f"active packaging '{key}' not found")
    if "product" not in cfg.fields_extracted:
        raise HTTPException(400, f"'{key}' does not read a product name")

    aliases = [a.model_dump() for a in body.product_aliases]
    # reject rows with empty canonical or zero keywords
    if any(not a["canonical"].strip() or not [k for k in a["keywords"] if k.strip()]
           for a in aliases):
        raise HTTPException(400, "each alias needs a canonical and >=1 keyword")

    previous = cfg.product_aliases
    from services import config_overrides
    try:
        config_overrides.save_product_aliases(key, aliases)
    except Exception as e:
        logger.exception("product_aliases override persist failed for %s", key)
        raise HTTPException(502, f"failed to persist override: {e}")
    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after product_aliases update")
        raise HTTPException(500, f"registry reload failed: {e}")
    return ProductAliasesResponse(key=key, product_aliases=aliases, previous=previous)
```

Empty list is rejected by `min_length=1` (a class with aliases must keep ≥1 — an
empty override would silently drop the class to the hardcoded tea-list fallback).

## Frontend design (`web/wizard.html`)

1. **Refactor the step-4 alias-row helpers** so they can render into any
   container, not just `#pa-rows`. `addProdAlias`/`removeProdAlias`/
   `updateProdPreview`/`updateAllProdPreviews`/`isSizeFieldOn` currently hardcode
   `#pa-rows` and the step-4 size-field selector. Parameterize the rows container
   and the "is size field on" check so the same UI works inside the drawer.

2. **Drawer card "ชื่อ Product ที่อ่าน"** — rendered in `renderDrawerBody` only
   when `pkg.product_aliases?.length` (i.e. a config-driven product-reading
   class). Pre-fills one row per existing alias, with the same `{size}` live
   preview. "Is size field on" is derived from
   `pkg.fields_extracted.includes('size')`.

3. **Save button** appears on change (like the conf editor), calls
   `PUT /{key}/product-aliases`, then `loadDashboard()` and an Undo toast that
   re-PUTs `res.previous` (mirrors `saveConfThreshold`/`undoConfThreshold`).

4. Classes without `product_aliases` (back_label/grade_bag, or non-product
   classes) show no card.

## Tests

- `tests/test_config_overrides` (or existing equivalent): `save_product_aliases`
  → `load` round-trips; writing `product_aliases` does not clobber an existing
  `conf_threshold` in the same entry, and vice-versa.
- `pipeline/packaging_registry` test: override `product_aliases` wins over YAML;
  no override → YAML value; malformed override (non-list) → falls back to YAML.
- `tests/test_api_packagings.py` (module-scoped `client`, mock `main.registry`
  cfg + `services.config_overrides.save_product_aliases` + `main.reload_registry`
  per the fixture conventions):
  - PUT on a class with `product` in `fields_extracted` → 200, response carries
    new + previous, `reload_registry` called.
  - PUT on a class without `product` → 400.
  - PUT with `[]` → 422 (schema `min_length`); row with empty canonical or no
    keyword → 400.
  - persist raises → 502, `reload_registry` NOT called.
  - GET `/{key}` for an active product-reading class returns `product_aliases` +
    `fields_extracted`.

## Out of scope

- Adding `product_aliases` to `back_label` / `grade_bag` (would switch them off
  the hardcoded path — intentionally not offered).
- `{size}` composition for `multi_field` / `cross_check` (no shipped packaging
  composes product+size there; unchanged, per the 2026-06-20 spec).
- Editing aliases on a *draft* (step 4 already does this) and fixing step-4
  prefill loss (separate known gotcha, not this feature).

## Files touched

- `services/config_overrides.py` — add `save_product_aliases`
- `pipeline/packaging_registry.py` — add `_merged_product_aliases`, wire into `_config_from_data`
- `api/schemas.py` — `ProductAliasesUpdate`, `ProductAliasesResponse`; extend `PackagingResponse`
- `api/packagings.py` — GET enrichment + new `PUT /{key}/product-aliases`
- `web/wizard.html` — parameterize alias-row helpers, drawer editor card, save + Undo
- tests as above
