# Runtime Product-Aliases Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator add/remove the product names an *active* packaging class reads, live from the dashboard drawer, with no model retrain and no redeploy.

**Architecture:** `product_aliases` is pure OCR text-matching (not a model input), so it is runtime-tunable exactly like `conf_threshold` (ADR 0004). The edited list is persisted to `config_overrides.json` (GCS → Drive → local fallback) in the same per-key entry as `conf_threshold`; `PackagingRegistry` merges the override over the YAML at load time; a new `PUT /api/packagings/{key}/product-aliases` endpoint persists-then-reloads. The frontend reuses the step-4 alias-row UI inside the drawer.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest; vanilla JS in `web/wizard.html`.

## Global Constraints

- `print()` is forbidden — use the module `logger`.
- `pytest` is not on PATH — run `python -m pytest`.
- Read/write source files as UTF-8 (files contain Thai).
- Scope: editor is offered **only** for classes that already have a non-empty `product_aliases`. `back_label`/`grade_bag` (hardcoded tea-list path) are intentionally excluded — giving them aliases would switch off their hardcoded path.
- Persist-first ordering: a storage failure must return an error and change nothing (no divergence between instance and storage).
- An active product-reading class must keep ≥1 alias — an empty override would silently drop it to the hardcoded fallback.
- `web/wizard.html` is the single source of truth; never edit the generated copies (`test wizzard/`, `dist/portable-bundle/static/`).
- `PackagingResponse` must declare every field the wizard reads — `response_model` silently strips undeclared keys.

---

### Task 1: `save_product_aliases` storage helper

**Files:**
- Modify: `services/config_overrides.py`
- Test: `tests/test_config_overrides.py`

**Interfaces:**
- Consumes: existing `load()`, `_drive_file_id()`, `_local_path()`, `gcs_store.get_store()`.
- Produces: `save_product_aliases(key: str, aliases: list[dict]) -> dict[str, dict]` — persists `{key: {..., "product_aliases": [{"canonical": str, "keywords": [str]}, ...]}}`, returns the merged overrides, raises on persist failure. Each alias dict has keys `canonical` (str) and `keywords` (list[str]).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config_overrides.py`:

```python
# ─── save_product_aliases() — local mode ─────────────────

def test_save_product_aliases_writes_and_returns_merged(local_mode):
    from services import config_overrides

    aliases = [{"canonical": "Medium {size}", "keywords": ["medium", "med"]}]
    merged = config_overrides.save_product_aliases("matcha_sachet", aliases)
    assert merged == {"matcha_sachet": {"product_aliases": aliases}}
    assert json.loads(local_mode.read_text(encoding="utf-8")) == merged


def test_save_product_aliases_coexists_with_conf(local_mode):
    from services import config_overrides

    config_overrides.save_conf_threshold("matcha_sachet", 0.7)
    aliases = [{"canonical": "Excellent", "keywords": ["excellent"]}]
    merged = config_overrides.save_product_aliases("matcha_sachet", aliases)
    assert merged["matcha_sachet"]["conf_threshold"] == 0.7
    assert merged["matcha_sachet"]["product_aliases"] == aliases
    # and the reverse: saving conf must not clobber aliases
    merged2 = config_overrides.save_conf_threshold("matcha_sachet", 0.8)
    assert merged2["matcha_sachet"]["product_aliases"] == aliases
    assert merged2["matcha_sachet"]["conf_threshold"] == 0.8


def test_save_product_aliases_normalizes_shape(local_mode):
    from services import config_overrides

    # extra keys dropped, keywords coerced to list
    merged = config_overrides.save_product_aliases(
        "x", [{"canonical": "A", "keywords": ("a", "b"), "junk": 1}])
    assert merged["x"]["product_aliases"] == [{"canonical": "A", "keywords": ["a", "b"]}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config_overrides.py -k product_aliases -v`
Expected: FAIL with `AttributeError: module 'services.config_overrides' has no attribute 'save_product_aliases'`

- [ ] **Step 3: Refactor the write block + add the new helper**

In `services/config_overrides.py`, extract the storage write (currently inline in `save_conf_threshold`, lines ~89–102) into a private helper, then add `save_product_aliases`. Replace the body of `save_conf_threshold` after building `merged` to call `_persist(merged)`:

```python
def _persist(merged: dict[str, dict]) -> None:
    """Write the merged overrides to GCS (or Drive / local fallback). Raises on failure."""
    from services import gcs_store

    store = gcs_store.get_store()
    if store is not None:
        store.write_json(_GCS_OBJECT, merged)
        return
    content = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")
    file_id = _drive_file_id()
    if file_id:
        DriveClient().update_file_content(file_id, content, mime_type="application/json")
    else:
        path = _local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.decode("utf-8"), encoding="utf-8")


def save_conf_threshold(key: str, value: float) -> dict[str, dict]:
    """Persist a conf_threshold override and return the merged overrides.

    Raises on persist failure — caller must NOT apply the change locally
    in that case, so instances never diverge from the stored overrides.
    """
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    merged.setdefault(key, {})["conf_threshold"] = float(value)
    _persist(merged)
    logger.info("Saved conf_threshold override: %s=%.2f", key, value)
    return merged


def save_product_aliases(key: str, aliases: list[dict]) -> dict[str, dict]:
    """Persist a product_aliases override and return the merged overrides.

    Raises on persist failure — caller must NOT apply the change locally
    in that case, so instances never diverge from the stored overrides.
    """
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    merged.setdefault(key, {})["product_aliases"] = [
        {"canonical": a["canonical"], "keywords": list(a["keywords"])}
        for a in aliases
    ]
    _persist(merged)
    logger.info("Saved product_aliases override: %s (%d aliases)", key, len(aliases))
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_overrides.py -v`
Expected: PASS (the new tests AND all pre-existing conf tests — the `_persist` refactor must not regress them).

- [ ] **Step 5: Commit**

```bash
git add services/config_overrides.py tests/test_config_overrides.py
git commit -m "feat: config_overrides.save_product_aliases (runtime override)"
```

---

### Task 2: Registry merges the `product_aliases` override

**Files:**
- Modify: `pipeline/packaging_registry.py:74-94` (`_config_from_data`), add `_merged_product_aliases`
- Test: `tests/test_packaging_registry.py`

**Interfaces:**
- Consumes: override dict shape `{key: {"product_aliases": [...]}}` from Task 1.
- Produces: `PackagingRegistry._merged_product_aliases(data: dict, override: object) -> list` (staticmethod) — returns the override's `product_aliases` list when present and a list, else `data.get("product_aliases", [])`. Wired into `_config_from_data` so `PackagingConfig.product_aliases` reflects the merged value.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_packaging_registry.py`:

```python
def test_merged_product_aliases_override_wins():
    from pipeline.packaging_registry import PackagingRegistry
    data = {"key": "x", "product_aliases": [{"canonical": "A", "keywords": ["a"]}]}
    override = {"product_aliases": [{"canonical": "B", "keywords": ["b"]}]}
    assert PackagingRegistry._merged_product_aliases(data, override) == override["product_aliases"]


def test_merged_product_aliases_no_override_uses_yaml():
    from pipeline.packaging_registry import PackagingRegistry
    data = {"key": "x", "product_aliases": [{"canonical": "A", "keywords": ["a"]}]}
    assert PackagingRegistry._merged_product_aliases(data, None) == data["product_aliases"]
    assert PackagingRegistry._merged_product_aliases(data, {"conf_threshold": 0.7}) == data["product_aliases"]


def test_merged_product_aliases_malformed_falls_back():
    from pipeline.packaging_registry import PackagingRegistry
    data = {"key": "x", "product_aliases": [{"canonical": "A", "keywords": ["a"]}]}
    assert PackagingRegistry._merged_product_aliases(data, {"product_aliases": "nope"}) == data["product_aliases"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_packaging_registry.py -k merged_product_aliases -v`
Expected: FAIL with `AttributeError: ... has no attribute '_merged_product_aliases'`

- [ ] **Step 3: Add the staticmethod and wire it in**

In `pipeline/packaging_registry.py`, add this staticmethod next to `_merged_conf_threshold` (after line 72):

```python
    @staticmethod
    def _merged_product_aliases(data: dict, override: object) -> list:
        yaml_value = data.get("product_aliases", [])
        if not isinstance(override, dict) or "product_aliases" not in override:
            return yaml_value
        ov = override["product_aliases"]
        if not isinstance(ov, list):
            logger.warning(
                "Invalid product_aliases override for %s: %r — using YAML value",
                data.get("key"), ov,
            )
            return yaml_value
        return ov
```

Then change line 93 inside `_config_from_data` from:

```python
            product_aliases=data.get("product_aliases", []),
```
to:
```python
            product_aliases=self._merged_product_aliases(data, override),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_packaging_registry.py -v`
Expected: PASS (new tests + all pre-existing registry tests).

- [ ] **Step 5: Commit**

```bash
git add pipeline/packaging_registry.py tests/test_packaging_registry.py
git commit -m "feat: registry merges product_aliases override over YAML"
```

---

### Task 3: Expose `product_aliases` + `fields_extracted` on GET, and add schemas

**Files:**
- Modify: `api/schemas.py` (extend `PackagingResponse`, add two new models)
- Modify: `api/packagings.py:138-148` (active branch of `get_packaging`)
- Test: `tests/test_api_packagings.py`

**Interfaces:**
- Consumes: `PackagingConfig.product_aliases` (list of dicts), `PackagingConfig.fields_extracted` (list[str]); existing `ProductAlias` model (`{canonical: str, keywords: list[str]}`).
- Produces:
  - `PackagingResponse` gains `product_aliases: list[ProductAlias] | None = None` and `fields_extracted: list[str] | None = None`.
  - `ProductAliasesUpdate(BaseModel)` with `product_aliases: list[ProductAlias] = Field(..., min_length=1)`.
  - `ProductAliasesResponse(BaseModel)` with `key: str`, `product_aliases: list[ProductAlias]`, `previous: list[ProductAlias]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_packagings.py`. First add a product-reading active fixture near the `fake_active` fixture (so it has `product` in `fields_extracted` and an existing alias):

```python
@pytest.fixture
def fake_active_product(client):
    """Temporary active YAML that reads a product name via product_aliases."""
    import yaml as _yaml
    from pathlib import Path
    key = "prod_fixture_pkg"
    yaml_path = Path(f"config/packagings/{key}.yaml")
    data = {
        "key": key, "display_name": "Prod Fixture", "pipeline": "detector_ocr",
        "conf_threshold": 0.6, "accuracy": 0.9, "gate_on_lot": True,
        "lot_short_fallback": False, "sub_regions": [],
        "lot_patterns": [r"(?i)LOT\s*([A-Z0-9]+)"],
        "fields_extracted": ["lot", "product"], "sheet_checks": ["lot", "product"],
        "post_ocr_fixes": [], "message_template_key": "default_full",
        "model_classifier_label": key, "detector_yolo_prefixes": [f"{key}_lot"],
        "product_aliases": [{"canonical": "Excellent", "keywords": ["excellent"]}],
    }
    yaml_path.write_text(_yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    from pipeline.packaging_registry import PackagingRegistry
    import main
    main.registry = PackagingRegistry()
    yield key
    for p in Path("config/packagings").glob(f"{key}.yaml*"):
        try:
            p.unlink()
        except OSError:
            pass
```

Then the GET test:

```python
def test_get_active_returns_product_aliases_and_fields(client, fake_active_product):
    g = client.get(f"/api/packagings/{fake_active_product}")
    assert g.status_code == 200, g.text
    body = g.json()
    assert body["fields_extracted"] == ["lot", "product"]
    assert body["product_aliases"] == [{"canonical": "Excellent", "keywords": ["excellent"]}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py::test_get_active_returns_product_aliases_and_fields -v`
Expected: FAIL — `body["fields_extracted"]` is `None` (response_model strips it; field not declared / not passed).

- [ ] **Step 3: Extend the schema**

In `api/schemas.py`, add the two new fields to `PackagingResponse` (after line 98, `detection_mode`):

```python
    product_aliases: list[ProductAlias] | None = None
    fields_extracted: list[str] | None = None
```

And add, after `ConfThresholdResponse` (line 117):

```python
class ProductAliasesUpdate(BaseModel):
    """Runtime edit of the product names an active class reads (no retrain)."""
    product_aliases: list[ProductAlias] = Field(..., min_length=1)


class ProductAliasesResponse(BaseModel):
    key: str
    product_aliases: list[ProductAlias]
    previous: list[ProductAlias]
```

- [ ] **Step 4: Populate them in the GET active branch**

In `api/packagings.py`, in `get_packaging` active branch (lines 138-148), add two kwargs to the `PackagingResponse(...)` call:

```python
            return PackagingResponse(
                key=cfg.key,
                display_name=cfg.display_name,
                pipeline=cfg.pipeline,
                status="active",
                image_count=_count_active_images(key),
                conf_threshold=cfg.conf_threshold,
                accuracy=cfg.accuracy,
                sub_regions=cfg.sub_regions,
                detection_mode=cfg.detection_mode,
                product_aliases=cfg.product_aliases,
                fields_extracted=cfg.fields_extracted,
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_packagings.py::test_get_active_returns_product_aliases_and_fields -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add api/schemas.py api/packagings.py tests/test_api_packagings.py
git commit -m "feat: expose product_aliases + fields_extracted on GET; add alias schemas"
```

---

### Task 4: `PUT /{key}/product-aliases` endpoint

**Files:**
- Modify: `api/packagings.py` (new route after `update_conf_threshold`, ~line 320; update imports)
- Test: `tests/test_api_packagings.py`

**Interfaces:**
- Consumes: `ProductAliasesUpdate`, `ProductAliasesResponse` (Task 3); `config_overrides.save_product_aliases` (Task 1); `main.registry`, `main.reload_registry`.
- Produces: `PUT /api/packagings/{key}/product-aliases` — 404 if not an active packaging; 400 if the class does not read a product or any alias row has an empty canonical / no non-empty keyword; 502 on persist failure (reload NOT called); 200 with `ProductAliasesResponse` on success.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api_packagings.py` (reuse `conf_overrides_env` + the `fake_active`/`fake_active_product` fixtures):

```python
# ─── PUT /{key}/product-aliases — runtime alias edit ─────

def test_put_aliases_updates_active(client, fake_active_product, conf_overrides_env):
    new = [
        {"canonical": "Excellent", "keywords": ["excellent"]},
        {"canonical": "Medium {size}", "keywords": ["medium", "med"]},
    ]
    r = client.put(f"/api/packagings/{fake_active_product}/product-aliases",
                   json={"product_aliases": new})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"] == fake_active_product
    assert body["product_aliases"] == new
    assert body["previous"] == [{"canonical": "Excellent", "keywords": ["excellent"]}]
    # GET reflects the merged override immediately
    g = client.get(f"/api/packagings/{fake_active_product}")
    assert g.json()["product_aliases"] == new


def test_put_aliases_400_when_class_has_no_product(client, fake_active, conf_overrides_env):
    # fake_active has fields_extracted == ["lot"] (no product)
    r = client.put(f"/api/packagings/{fake_active}/product-aliases",
                   json={"product_aliases": [{"canonical": "A", "keywords": ["a"]}]})
    assert r.status_code == 400


def test_put_aliases_422_when_empty_list(client, fake_active_product, conf_overrides_env):
    r = client.put(f"/api/packagings/{fake_active_product}/product-aliases",
                   json={"product_aliases": []})
    assert r.status_code == 422


def test_put_aliases_400_when_row_empty(client, fake_active_product, conf_overrides_env):
    for bad in ([{"canonical": "  ", "keywords": ["a"]}],
                [{"canonical": "A", "keywords": []}],
                [{"canonical": "A", "keywords": ["  "]}]):
        r = client.put(f"/api/packagings/{fake_active_product}/product-aliases",
                       json={"product_aliases": bad})
        assert r.status_code == 400, f"{bad} should be rejected"


def test_put_aliases_404_for_unknown_key(client, conf_overrides_env):
    r = client.put("/api/packagings/nonexistent_xyz/product-aliases",
                   json={"product_aliases": [{"canonical": "A", "keywords": ["a"]}]})
    assert r.status_code == 404


def test_put_aliases_502_when_persist_fails(client, fake_active_product, conf_overrides_env, monkeypatch):
    from services import config_overrides
    monkeypatch.setattr(config_overrides, "save_product_aliases",
                        MagicMock(side_effect=RuntimeError("drive down")))
    r = client.put(f"/api/packagings/{fake_active_product}/product-aliases",
                   json={"product_aliases": [{"canonical": "Z", "keywords": ["z"]}]})
    assert r.status_code == 502
    # nothing changed — GET still serves the YAML value
    g = client.get(f"/api/packagings/{fake_active_product}")
    assert g.json()["product_aliases"] == [{"canonical": "Excellent", "keywords": ["excellent"]}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_packagings.py -k put_aliases -v`
Expected: FAIL — 404/405 (route not defined) for the 200-path test.

- [ ] **Step 3: Add the endpoint**

In `api/packagings.py`, update the schema import to include the new models, then add the route immediately after `update_conf_threshold` (after line 319):

```python
@router.put("/{key}/product-aliases", response_model=ProductAliasesResponse)
def update_product_aliases(key: str, body: ProductAliasesUpdate):
    """Edit the product names an active class reads — no retrain (mirrors /conf).

    Persist-first: a storage failure returns 502 and changes nothing, so the
    instance never diverges from the stored overrides.
    """
    import main

    cfg = main.registry.get(key) if main.registry else None
    if cfg is None:
        raise HTTPException(404, f"active packaging '{key}' not found")
    if "product" not in cfg.fields_extracted:
        raise HTTPException(400, f"packaging '{key}' does not read a product name")

    aliases = [a.model_dump() for a in body.product_aliases]
    for a in aliases:
        if not a["canonical"].strip() or not [k for k in a["keywords"] if k.strip()]:
            raise HTTPException(400, "each alias needs a canonical and at least one keyword")

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

For the import: find the existing `from api.schemas import (...)` (or `from .schemas import ...`) block and add `ProductAliasesUpdate, ProductAliasesResponse` to it. If `ConfThresholdResponse` is already imported there, add the two names alongside it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_packagings.py -k "put_aliases or product_aliases" -v`
Expected: PASS

- [ ] **Step 5: Run the full API test module (no regressions)**

Run: `python -m pytest tests/test_api_packagings.py -v`
Expected: PASS (pre-existing `tests/test_classifier.py` 3 setup errors are unrelated and out of scope).

- [ ] **Step 6: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: PUT /{key}/product-aliases runtime alias editor endpoint"
```

---

### Task 5: Drawer editor UI (`web/wizard.html`)

**Files:**
- Modify: `web/wizard.html` — parameterize the step-4 alias-row helpers; add the drawer editor card in `renderDrawerBody`; add save/undo functions.

**Interfaces:**
- Consumes: GET `/api/packagings/{key}` now returns `product_aliases` + `fields_extracted` (Task 3); `PUT /{key}/product-aliases` (Task 4); existing `api()`, `showToast()`, `esc()`, `loadDashboard()`, and CSS classes `.pa-row`, `.pa-canonical`, `.pa-keywords`, `.pa-preview`, `.rm-btn`, `.dsec`, `.btn`.
- Produces: drawer card rendered only when `pkg.product_aliases?.length`; add/remove rows; "บันทึก" persists via PUT; Undo toast re-PUTs the previous list.

- [ ] **Step 1: Parameterize the alias-row helpers**

In `web/wizard.html` (the "Product aliases (step 4)" block, ~lines 3226-3275), refactor so the row helpers work in any container. Replace `addProdAlias` with a container-aware version + a step-4 wrapper, and make `updateProdPreview` derive "is size on" from the row's container (so the same inline `oninput` handler works in both places):

```javascript
// ─── Product aliases (shared by step 4 + drawer) ──────
function isProductFieldOn() {
  return !!document.querySelector('#sp4 [data-group="fields"] .cbitem.on[data-field="product"]');
}
function isSizeFieldOn() {
  return !!document.querySelector('#sp4 [data-group="fields"] .cbitem.on[data-field="size"]');
}
// size-on is live-DOM in step 4 (#pa-rows has no data-size-on), static in the drawer
function paSizeOn(row) {
  const rows = row.closest('.pa-rows');
  if (rows && rows.dataset.sizeOn != null) return rows.dataset.sizeOn === '1';
  return isSizeFieldOn();
}
function addProdAliasTo(containerId, canonical = '', keywords = '') {
  const c = String(canonical).replace(/"/g, '&quot;');
  const k = String(keywords).replace(/"/g, '&quot;');
  document.getElementById(containerId).insertAdjacentHTML('beforeend', `
    <div class="pa-row">
      <input type="text" class="pa-canonical" aria-label="ชื่อ Product (template)" placeholder="Medium {size}" value="${c}" oninput="updateProdPreview(this.closest('.pa-row'))">
      <input type="text" class="pa-keywords" aria-label="คำที่พบบนซอง คั่นด้วยจุลภาค" placeholder="medium" value="${k}">
      <button class="rm-btn" onclick="removeProdAlias(this)" aria-label="ลบ product นี้">×</button>
      <div class="pa-preview"></div>
    </div>`);
  updateProdPreview(document.getElementById(containerId).lastElementChild);
}
function addProdAlias(canonical = '', keywords = '') { addProdAliasTo('pa-rows', canonical, keywords); }
```

Add the class `pa-rows` to the existing step-4 container so `paSizeOn` finds it. Find `id="pa-rows"` in the step-4 markup and add `class="pa-rows"` to that element (keep the id).

In `updateProdPreview` (~line 3246), change the size-on check from `isSizeFieldOn()` to `paSizeOn(row)`:

```javascript
function updateProdPreview(row) {
  if (!row) return;
  const out = row.querySelector('.pa-preview');
  const tpl = row.querySelector('.pa-canonical').value.trim();
  if (!out) return;
  if (!tpl) { out.textContent = ''; out.className = 'pa-preview'; return; }
  if (tpl.includes('{size}')) {
    if (!paSizeOn(row)) {
      out.innerHTML = '⚠ ใช้ <code>{size}</code> แต่ยังไม่เปิด field "ขนาด"';
      out.className = 'pa-preview warn';
      return;
    }
    const resolved = tpl.replace(/\{size\}/g, PA_SAMPLE_SIZE).replace(/\s+/g, ' ').trim();
    out.textContent = `→ "${resolved}"  (ตัวอย่างขนาด ${PA_SAMPLE_SIZE})`;
  } else {
    out.textContent = `→ "${tpl}"`;
  }
  out.className = 'pa-preview';
}
```

(`removeProdAlias`, `updateAllProdPreviews`, `collectConfig`, `syncProductAliasVisibility` are unchanged — they keep operating on `#pa-rows`.)

- [ ] **Step 2: Render the drawer editor card**

In `renderDrawerBody` (`web/wizard.html` ~line 2222-2226), replace the active-config italic note so a product-reading class gets an editable aliases card. Change the `if (isActive) { ... }` block to:

```javascript
    if (isActive) {
      html += `<div class="dsec"><div class="dsec-label">Config</div>
        <div style="font-size:12px;color:var(--t3);font-style:italic">Active config อยู่ใน config/packagings/${esc(pkg.key)}.yaml — ใช้งานจริงในระบบ</div></div>`;
      if (pkg.product_aliases && pkg.product_aliases.length) {
        const sizeOn = (pkg.fields_extracted || []).includes('size') ? '1' : '0';
        html += `<div class="dsec">
          <div class="dsec-label">ชื่อ Product ที่อ่าน (แก้ได้ทันที — ไม่ต้อง retrain)</div>
          <div id="drawer-pa-rows" class="pa-rows" data-size-on="${sizeOn}"></div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <button class="btn btn-sm" onclick="addProdAliasTo('drawer-pa-rows')">+ เพิ่ม product</button>
            <button class="btn btn-primary btn-sm" onclick="saveDrawerAliases('${esc(pkg.key)}')">บันทึก product names</button>
          </div>
        </div>`;
      }
    } else {
```

After the drawer body is injected, the rows must be populated. In `openDrawer`, right after `body.innerHTML = renderDrawerBody(pkg, samplesRes);` (line 2080), add:

```javascript
    if (pkg.product_aliases && pkg.product_aliases.length) {
      pkg.product_aliases.forEach(a =>
        addProdAliasTo('drawer-pa-rows', a.canonical, (a.keywords || []).join(', ')));
    }
```

- [ ] **Step 3: Add save + undo functions**

Add near `saveConfThreshold`/`undoConfThreshold` (~line 2165):

```javascript
function collectDrawerAliases() {
  return Array.from(document.querySelectorAll('#drawer-pa-rows .pa-row')).map(row => ({
    canonical: row.querySelector('.pa-canonical').value.trim(),
    keywords: row.querySelector('.pa-keywords').value.split(',').map(s => s.trim()).filter(Boolean),
  })).filter(a => a.canonical && a.keywords.length);
}

async function saveDrawerAliases(key) {
  const aliases = collectDrawerAliases();
  if (!aliases.length) {
    showToast('ต้องมีอย่างน้อย 1 product ที่มี keyword');
    return;
  }
  try {
    const res = await api('PUT', `/api/packagings/${encodeURIComponent(key)}/product-aliases`,
      { product_aliases: aliases });
    loadDashboard();
    showToast(`บันทึก ${res.product_aliases.length} product แล้ว`, 8000, {
      label: 'Undo',
      onClick: () => undoDrawerAliases(key, res.previous),
    });
  } catch (err) {
    showToast('บันทึก product names ไม่สำเร็จ: ' + err.message);
  }
}

async function undoDrawerAliases(key, previous) {
  if (!previous || !previous.length) return;
  try {
    await api('PUT', `/api/packagings/${encodeURIComponent(key)}/product-aliases`,
      { product_aliases: previous });
    loadDashboard();
    showToast('คืนค่า product names เดิมแล้ว');
  } catch (err) {
    showToast('Undo ไม่สำเร็จ: ' + err.message);
  }
}
```

- [ ] **Step 4: Verify the step-4 flow still works (no regression)**

Serve the wizard with no backend and confirm the step-4 alias UI still renders and previews:

```bash
python -m http.server 8090 --directory web
```

Then with the agent-browser / Playwright skill: open `http://localhost:8090/wizard.html`, run `startWizard()` then `goStep(4)`, click "+ เพิ่ม", type `Medium {size}` in a `.pa-canonical`, and confirm the `.pa-preview` shows the `{size}` warning or resolved text (the `paSizeOn` path). Expected: identical behavior to before the refactor.

- [ ] **Step 5: Verify the drawer editor renders + saves**

Start the real dev server (`python -m uvicorn main:app --reload --port 8080`), serve `web/`, open the dashboard, click an active product-reading class card (one whose YAML has `product_aliases`, e.g. a matcha/grade class — NOT back_label/grade_bag which use the hardcoded list). Confirm: the "ชื่อ Product ที่อ่าน" card shows existing aliases, "+ เพิ่ม product" adds a row, "บันทึก" succeeds, an Undo toast appears, and reopening the drawer shows the saved list. If no active class currently has `product_aliases`, verify via the `prod_fixture_pkg` test path instead and note it in the commit message.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html
git commit -m "feat: drawer editor to add/remove product names on active classes"
```

---

## Self-Review notes

- **Spec coverage:** §1 storage → Task 1; §2 registry merge → Task 2; §3 schemas + §4 GET enrichment → Task 3; §4 PUT endpoint → Task 4; §5 frontend → Task 5; §6 tests are folded into Tasks 1–4 (backend) and Task 5 steps 4–5 (frontend verification, matching the repo's Playwright-based wizard convention rather than pytest for HTML).
- **Scope guard:** the editor card renders only when `pkg.product_aliases?.length`, and the PUT rejects classes without `product` in `fields_extracted` — together they keep `back_label`/`grade_bag` out, per the approved scope.
- **Type consistency:** `save_product_aliases(key, aliases: list[dict])` (Task 1) ← endpoint passes `[a.model_dump() ...]` (Task 4); `_merged_product_aliases` returns a list assigned to `PackagingConfig.product_aliases`, surfaced by GET as `product_aliases` (Task 3) and consumed by `addProdAliasTo`/`collectDrawerAliases` (Task 5). `ProductAliasesResponse.previous`/`product_aliases` are `list[ProductAlias]` ← `cfg.product_aliases` and `aliases` (list[dict], coerced by Pydantic).
- **Out of scope (unchanged):** draft step-4 editing, step-4 prefill-loss bug, `multi_field`/`cross_check` product+size composition.
```
