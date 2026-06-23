# Product Name `{size}` Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a wizard `product_aliases` `canonical` act as a free-form template where the token `{size}` is replaced by the OCR-extracted size, so one alias row covers all sizes of a product and per-row composition is explicit.

**Architecture:** Add a pure `resolve_product_template(template, size)` helper in `utils/validators.py`. In `pipeline/ocr_engine.py`, branch product-name composition: configs with `product_aliases` resolve via the template (no auto-append); configs without aliases (legacy `back_label`/`grade_bag`) keep the existing `f"{name} {size}"` auto-append. Update `web/wizard.html` to teach the `{size}` token via a rewritten hint, a template placeholder, and a live per-row preview.

**Tech Stack:** Python 3.11, pytest; vanilla HTML/CSS/JS wizard (`web/wizard.html`).

## Global Constraints

- `print()` is forbidden — use the module `logger` (not relevant to these tasks but enforced repo-wide).
- Run tests with `python -m pytest` (pytest is not on PATH).
- Source files may contain Thai — keep `encoding='utf-8'`; do not transcode.
- Type annotations on all new function signatures (PEP 8 / repo Python style).
- No commit attribution footer (attribution disabled globally).
- `web/wizard.html` is the single source of truth; the `test wizzard/` and `dist/portable-bundle/` copies regenerate automatically — do NOT edit them.
- Out of scope: `pipeline/pipeline_runner.py` (no shipped packaging composes product+size via aliases there — recorded as a known gap in the spec, not implemented).

---

### Task 1: `resolve_product_template` helper + unit tests

**Files:**
- Modify: `utils/validators.py` (add helper near `find_product_name` / `find_size`, after `_match_aliases` ~line 229)
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces: `resolve_product_template(template: str | None, size: str | None) -> str | None`
  - `{size}` not present → returns `template` unchanged (literal, e.g. `"Houjicha Powder"`).
  - `{size}` present and `size` truthy → returns `template` with every `{size}` replaced by `size`, then whitespace collapsed (`" ".join(result.split())`).
  - `{size}` present and `size` falsy → returns `None`.
  - falsy `template` → returns `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ocr.py`. First extend the import block (lines 8-14) to include the new symbol and `find_size`:

```python
from utils.validators import (
    find_expiry,
    find_lot,
    find_mfg,
    find_product_name,
    find_size,
    normalize_date,
    resolve_product_template,
)
```

Then append a new test class at the end of the file:

```python
# ─── resolve_product_template ────────────────────────────────────────────────

class TestResolveProductTemplate:
    def test_substitutes_size_token(self):
        assert resolve_product_template("Medium {size}", "40 g") == "Medium 40 g"

    def test_no_token_returns_literal(self):
        assert resolve_product_template("Houjicha Powder", "40 g") == "Houjicha Powder"

    def test_no_token_ignores_missing_size(self):
        assert resolve_product_template("Excellent", None) == "Excellent"

    def test_token_present_but_size_missing_returns_none(self):
        assert resolve_product_template("Medium {size}", None) is None

    def test_collapses_double_space(self):
        # awkward template should not leave a double space when size has its own spacing
        assert resolve_product_template("Medium  {size}", "40 g") == "Medium 40 g"

    def test_token_anywhere_and_suffix(self):
        assert resolve_product_template("Excellent {size} Powder", "200 g") == "Excellent 200 g Powder"

    def test_empty_template_returns_none(self):
        assert resolve_product_template("", "40 g") is None
        assert resolve_product_template(None, "40 g") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ocr.py::TestResolveProductTemplate -v`
Expected: FAIL at import (`ImportError: cannot import name 'resolve_product_template'`).

- [ ] **Step 3: Write minimal implementation**

In `utils/validators.py`, add directly after `_match_aliases` (after the `return None` at ~line 229, before `def find_product_name`):

```python
def resolve_product_template(template: str | None, size: str | None) -> str | None:
    """ประกอบชื่อ product จาก template — แทน token {size} ด้วยขนาดที่ OCR อ่านได้.

    - ไม่มี {size}        → คืน template ตามเดิม (เช่น 'Houjicha Powder')
    - มี {size} + มี size → แทนค่าแล้วยุบช่องว่างซ้ำ (เช่น 'Medium 40 g')
    - มี {size} + ไม่มี size → คืน None (ยืนยันไม่ได้ → product ไม่ผ่าน)
    """
    if not template:
        return None
    if "{size}" not in template:
        return template
    if not size:
        return None
    resolved = template.replace("{size}", size)
    return " ".join(resolved.split())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ocr.py::TestResolveProductTemplate -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add utils/validators.py tests/test_ocr.py
git commit -m "feat: add resolve_product_template for {size} token in product aliases"
```

---

### Task 2: `ocr_engine` branches aliases-template vs legacy-append

**Files:**
- Modify: `pipeline/ocr_engine.py:65-69`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `resolve_product_template` (Task 1), existing `find_product_name(text, aliases)` and `find_size(text)`.
- Behavior after change:
  - config WITH `product_aliases` → `product_name = resolve_product_template(matched_template, size)`.
  - config WITHOUT aliases (legacy `back_label`/`grade_bag`) → unchanged `f"{name} {size}"` when both present.

- [ ] **Step 1: Write the failing tests**

These pin the two composition paths using the public validator functions (the exact pieces `ocr_engine` wires together). Append to `tests/test_ocr.py`:

```python
# ─── product composition paths (mirrors ocr_engine branch) ───────────────────

class TestProductCompositionPaths:
    def test_aliases_path_resolves_size_token(self):
        aliases = [{"canonical": "Medium {size}", "keywords": ["medium"]}]
        text = "medium 40 g"
        name = find_product_name(text, aliases)        # -> "Medium {size}"
        size = find_size(text)                          # -> "40 g"
        assert resolve_product_template(name, size) == "Medium 40 g"

    def test_aliases_path_literal_when_no_token(self):
        aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
        text = "houjicha tea"
        name = find_product_name(text, aliases)         # -> "Houjicha Powder"
        size = find_size(text)                          # -> None (no size printed)
        assert resolve_product_template(name, size) == "Houjicha Powder"

    def test_legacy_fallback_appends_size(self):
        # no aliases -> hardcoded fallback; ocr_engine appends size as before
        name = find_product_name("houjicha tea", None)  # -> "Houjicha Powder"
        size = "40 g"
        composed = f"{name} {size}" if (name and size) else name
        assert composed == "Houjicha Powder 40 g"
```

- [ ] **Step 2: Run tests to verify current state**

Run: `python -m pytest tests/test_ocr.py::TestProductCompositionPaths -v`
Expected: PASS for the validator-level pieces (Task 1 already provides `resolve_product_template`). These tests lock the contract `ocr_engine` must honor; if any fails, the validator behavior assumed by the edit is wrong — stop and reconcile before editing `ocr_engine`.

- [ ] **Step 3: Apply the `ocr_engine` edit**

In `pipeline/ocr_engine.py`, add `resolve_product_template` to the import on line 6:

```python
from utils.validators import find_expiry, find_lot, find_mfg, find_product_name, find_size, resolve_product_template
```

Replace lines 65-69:

```python
        size = find_size(full_text) if extract_size else None
        aliases = config.product_aliases if config else None
        product_name = find_product_name(full_text, aliases) if extract_product else None
        if product_name and size:
            product_name = f"{product_name} {size}"
```

with:

```python
        size = find_size(full_text) if extract_size else None
        aliases = config.product_aliases if config else None
        product_name = find_product_name(full_text, aliases) if extract_product else None
        if product_name:
            if aliases:
                product_name = resolve_product_template(product_name, size)
            elif size:
                product_name = f"{product_name} {size}"  # legacy fallback (back_label/grade_bag) — unchanged
```

- [ ] **Step 4: Run the full OCR test suite**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: PASS (all existing tests plus Task 1 + Task 2 tests). No regressions.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ocr_engine.py tests/test_ocr.py
git commit -m "feat: ocr_engine resolves {size} template for alias-configured packagings"
```

---

### Task 3: Wizard `{size}` template hint, placeholder, and live preview

**Files:**
- Modify: `web/wizard.html` — CSS (~after line 643 `.pa-row input`), the `.pa-hint` callout block (in `#pa-card`), `addProdAlias` (~line 3217), add preview JS functions near `addProdAlias`, and the `size` field checkbox onclick (~line 1576).

**Interfaces:**
- Consumes: nothing from earlier tasks (frontend-only).
- Produces: `addProdAlias` rows now contain a `.pa-preview` element; helpers `isSizeFieldOn()`, `updateProdPreview(row)`, `updateAllProdPreviews()`.

- [ ] **Step 1: Replace the (now-incorrect) `.pa-hint` content**

The `.pa-hint` block currently tells users to type the size into `canonical` manually and split rows per size. Replace the whole `<div class="pa-hint"> ... </div>` block (the one added 2026-06-20, between the `wiz-card-desc` and the `.fgroup` in `#pa-card`) with:

```html
              <div class="pa-hint">
                💡 ช่อง <strong>"ชื่อ Product"</strong> เป็น <strong>template</strong> — พิมพ์ข้อความได้อิสระ และใส่ <code>{size}</code> ตรงไหนก็ได้เพื่อให้ระบบแทนด้วยขนาดที่ OCR อ่านเจอ · ต้องตรงกับคอลัมน์ Product Name ใน Sheet แบบทุกตัวอักษร
                <div class="pa-hint-eg">▸ <code>Medium {size}</code> → ถ้า OCR เจอ 40 g จะได้ <code>Medium 40 g</code> (แถวเดียวคุมได้ทุกขนาด)</div>
                <div class="pa-hint-eg">▸ <code>Houjicha Powder</code> → ได้ <code>Houjicha Powder</code> (ไม่ใส่ <code>{size}</code> = ไม่ต่ออะไร)</div>
                <div class="pa-hint-eg">▸ ถ้าใส่ <code>{size}</code> แต่ OCR อ่านขนาดไม่เจอ → product จะถือว่าไม่ผ่าน (เปิด field "ขนาด" ด้านบนด้วย)</div>
              </div>
```

- [ ] **Step 2: Add preview CSS**

After line 643 (`.pa-row input{...}`) add:

```css
.pa-row .pa-preview{grid-column:1 / -1;font-size:11px;color:var(--t3);font-family:'JetBrains Mono',monospace;margin-top:-2px;min-height:14px}
.pa-row .pa-preview.warn{color:var(--warn);font-family:inherit}
.pa-row .pa-preview code{background:var(--s2);padding:0 4px;border-radius:3px;color:var(--t1)}
```

- [ ] **Step 3: Add preview to `addProdAlias` and wire the input handler**

Replace the `addProdAlias` function (line 3217) body's template literal so each row includes a canonical `oninput` handler and a `.pa-preview` element:

```javascript
function addProdAlias(canonical = '', keywords = '') {
  const c = String(canonical).replace(/"/g, '&quot;');
  const k = String(keywords).replace(/"/g, '&quot;');
  document.getElementById('pa-rows').insertAdjacentHTML('beforeend', `
    <div class="pa-row">
      <input type="text" class="pa-canonical" aria-label="ชื่อ Product (template)" placeholder="Medium {size}" value="${c}" oninput="updateProdPreview(this.closest('.pa-row'))">
      <input type="text" class="pa-keywords" aria-label="คำที่พบบนซอง คั่นด้วยจุลภาค" placeholder="medium" value="${k}">
      <button class="rm-btn" onclick="removeProdAlias(this)" aria-label="ลบ product นี้">×</button>
      <div class="pa-preview"></div>
    </div>`);
  updateProdPreview(document.getElementById('pa-rows').lastElementChild);
}
```

- [ ] **Step 4: Add the preview helper functions**

Immediately after `addProdAlias` add:

```javascript
const PA_SAMPLE_SIZE = '40 g';
function isSizeFieldOn() {
  return !!document.querySelector('#sp4 [data-group="fields"] .cbitem.on[data-field="size"]');
}
function updateProdPreview(row) {
  if (!row) return;
  const out = row.querySelector('.pa-preview');
  const tpl = row.querySelector('.pa-canonical').value.trim();
  if (!out) return;
  if (!tpl) { out.textContent = ''; out.className = 'pa-preview'; return; }
  if (tpl.includes('{size}')) {
    if (!isSizeFieldOn()) {
      out.innerHTML = '⚠ ใช้ <code>{size}</code> แต่ยังไม่เปิด field "ขนาด" ด้านบน';
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
function updateAllProdPreviews() {
  document.querySelectorAll('#pa-rows .pa-row').forEach(updateProdPreview);
}
```

- [ ] **Step 5: Refresh previews when the size field is toggled**

On the `size` field checkbox (line 1576), append the refresh call to the existing `onclick`:

Change:
```html
                  <div class="cbitem" data-field="size" onclick="toggleCb(this)">
```
to:
```html
                  <div class="cbitem" data-field="size" onclick="toggleCb(this); updateAllProdPreviews()">
```

- [ ] **Step 6: Verify in the browser (no backend needed)**

Serve the wizard and drive it to the product-alias card:

```bash
python -m http.server 8090 --directory web
```

In a browser (or Playwright) at `http://localhost:8090/wizard.html`, run in the console:
```js
startWizard(); goStep(4);
document.querySelector('[data-field="product"]').classList.contains('on') || document.querySelector('[data-field="product"]').click();
```
Confirm:
- the `#pa-card` shows the new hint mentioning `{size}`;
- typing `Medium {size}` in a canonical field shows preview `→ "Medium 40 g" (ตัวอย่างขนาด 40 g)` **when** the size field is on, and the `⚠ ... ยังไม่เปิด field "ขนาด"` warning when it is off;
- typing `Houjicha Powder` shows preview `→ "Houjicha Powder"`.

Also confirm the strings exist statically:
```bash
grep -c "updateProdPreview" web/wizard.html   # expect >= 3
grep -c "pa-preview" web/wizard.html           # expect >= 4
```

- [ ] **Step 7: Commit**

```bash
git add web/wizard.html
git commit -m "feat: wizard product-name template hint + live size preview"
```

---

## Notes for the implementer

- Do not edit `test wizzard/wizard.html` or `dist/portable-bundle/static/wizard.html` — they are generated copies.
- The working tree may contain unrelated in-progress changes; stage only the files listed in each task's commit step (`git add <exact paths>`), never `git add -A`.
- `tests/test_classifier.py` has 3 known pre-existing setup errors unrelated to this work — ignore them.
