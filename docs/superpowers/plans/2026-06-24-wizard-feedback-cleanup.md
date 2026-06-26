# Wizard Feedback Standardization + Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `web/wizard.html` feedback consistent (toast for async results, themed `showConfirm` for destructive actions), plug silent failures, and remove dead code / demo hardcode / logic bugs.

**Architecture:** Single static HTML file with inline JS. All changes are surgical edits to `web/wizard.html`. Verification is a self-contained Playwright **no-backend** smoke script (`web/tests/wizard_smoke.mjs`) that serves `web/` via Python `http.server`, stubs `/api/**` with `page.route`, and asserts behavior — mostly by calling global functions via `page.evaluate`. Pure-removal units assert via `node` string checks on the file.

**Tech Stack:** Vanilla JS/HTML/CSS; Playwright 1.60 (`playwright` package, chromium already cached); Python `http.server`; Node v25 (ESM `.mjs`).

## Global Constraints

- **Scope:** `web/wizard.html` + `web/tests/wizard_smoke.mjs` ONLY. No backend changes (`api/packagings.py:603` `< 30` stays).
- **Feedback rule (verbatim):** `showToast(...)` for every async result (success AND failure); `await showConfirm(...)` for every destructive action; `alert(...)` ONLY for synchronous blocking validation.
- **Image gate (verbatim):** fresh draft = `50`; edit-draft (key ends with `__edit`) = `30`.
- **Surgical (karpathy):** every changed line traces to this plan. Do NOT reformat or "improve" adjacent code. Match existing Thai-comment style and 2-space indent.
- **Thai source:** file contains Thai; preserve UTF-8. Match elements by `onclick`/`id`, not Thai text (Playwright transit mangles Thai literals).
- **One commit per task.** Conventional-commit messages. No attribution footer.
- **Encoding:** read/write the file as UTF-8.

---

## Task 1: `showConfirm` modal + smoke-test harness

**Files:**
- Modify: `web/wizard.html` (add `showConfirm` after `showToast`, ~line 1820; add modal markup near the drawer, ~line 1719)
- Create: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Produces: global `async function showConfirm({title, body, danger})` → `Promise<boolean>` (resolves `true` on confirm, `false` on cancel/Esc/backdrop). Modal DOM ids: `#confirm-backdrop`, `#confirm-modal`, `#confirm-ok`, `#confirm-cancel`.
- Produces: `web/tests/wizard_smoke.mjs` runnable via `node web/tests/wizard_smoke.mjs`, exit code 0 = all pass, 1 = any fail. Exposes a `checks` array that later tasks append to.

- [ ] **Step 1: Write the failing test (create the harness with the first two checks)**

Create `web/tests/wizard_smoke.mjs`:

```js
// Self-contained Playwright no-backend smoke test for web/wizard.html.
// Run: node web/tests/wizard_smoke.mjs
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = resolve(__dirname, '..');         // web/
const PORT = 8090;
const ORIGIN = `http://localhost:${PORT}`;

// ── stub responses keyed by path suffix ──
function stubFor(pathname) {
  if (pathname === '/api/packagings') return [];
  if (pathname.endsWith('/samples')) return { samples: [] };
  if (/\/api\/packagings\/[^/]+$/.test(pathname)) {
    return { key: 'demo', display_name: 'Demo', status: 'draft',
             pipeline: 'detector_ocr', image_count: 0, sub_regions: ['lot'] };
  }
  return {};
}

// ── checks registry — later tasks push [name, async (page)=>{}] ──
export const checks = [];

checks.push(['showConfirm resolves true on OK', async (page) => {
  const ok = await page.evaluate(async () => {
    const p = showConfirm({ title: 't', body: 'b' });
    document.querySelector('#confirm-ok').click();
    return await p;
  });
  if (ok !== true) throw new Error(`expected true, got ${ok}`);
}]);

checks.push(['showConfirm resolves false on Cancel', async (page) => {
  const v = await page.evaluate(async () => {
    const p = showConfirm({ title: 't', body: 'b' });
    document.querySelector('#confirm-cancel').click();
    return await p;
  });
  if (v !== false) throw new Error(`expected false, got ${v}`);
}]);

// ── runner ──
async function main() {
  const server = spawn('python', ['-m', 'http.server', String(PORT), '--directory', WEB_DIR],
    { stdio: 'ignore' });
  await new Promise(r => setTimeout(r, 800));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.addInitScript((origin) => { window.API_BASE_OVERRIDE = origin; }, ORIGIN);
  await page.route('**/api/**', (route) => {
    const u = new URL(route.request().url());
    route.fulfill({ status: 200, contentType: 'application/json',
                    body: JSON.stringify(stubFor(u.pathname)) });
  });
  await page.goto(`${ORIGIN}/wizard.html`, { waitUntil: 'networkidle' });

  let failed = 0;
  for (const [name, fn] of checks) {
    try { await fn(page); console.log(`PASS  ${name}`); }
    catch (e) { failed++; console.log(`FAIL  ${name}\n      ${e.message}`); }
  }
  await browser.close();
  server.kill();
  console.log(`\n${checks.length - failed}/${checks.length} passed`);
  process.exit(failed ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL on both `showConfirm` checks with `showConfirm is not defined` (function not yet added).

- [ ] **Step 3: Add the modal markup**

In `web/wizard.html`, immediately AFTER the drawer `</aside>` (the element opened by `<aside class="drawer" id="drawer">`, closing ~line 1720), insert:

```html
<!-- CONFIRM MODAL -->
<div class="confirm-backdrop" id="confirm-backdrop">
  <div class="confirm-modal" id="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
    <div class="confirm-title" id="confirm-title">—</div>
    <div class="confirm-body" id="confirm-body"></div>
    <div class="confirm-actions">
      <button class="btn btn-ghost btn-sm" id="confirm-cancel">ยกเลิก</button>
      <button class="btn btn-primary btn-sm" id="confirm-ok">ยืนยัน</button>
    </div>
  </div>
</div>
```

Add CSS inside the existing `<style>` block (just before the closing `</style>` at ~line 1146):

```css
/* ── CONFIRM MODAL ── */
.confirm-backdrop{
  position:fixed;inset:0;z-index:120;
  background:rgba(0,0,0,.55);backdrop-filter:blur(2px);
  display:none;align-items:center;justify-content:center;
}
.confirm-backdrop.open{display:flex}
.confirm-modal{
  background:var(--s0);border:1px solid var(--bd);
  border-radius:12px;padding:22px;width:420px;max-width:92vw;
  box-shadow:0 20px 50px rgba(0,0,0,.5);
}
.confirm-title{font-family:'Playfair Display',serif;font-size:18px;font-weight:500;margin-bottom:10px}
.confirm-body{font-size:13px;color:var(--t2);line-height:1.7;margin-bottom:20px}
.confirm-actions{display:flex;gap:10px;justify-content:flex-end}
```

- [ ] **Step 4: Add the `showConfirm` function**

In `web/wizard.html`, immediately AFTER the `showToast` function (ends ~line 1820), add:

```js
function showConfirm({ title = 'ยืนยัน', body = '', danger = false } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.getElementById('confirm-backdrop');
    const ok = document.getElementById('confirm-ok');
    const cancel = document.getElementById('confirm-cancel');
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-body').innerHTML = body;
    ok.className = 'btn btn-sm ' + (danger ? 'btn-danger' : 'btn-primary');
    backdrop.classList.add('open');
    ok.focus();
    const done = (val) => {
      backdrop.classList.remove('open');
      ok.removeEventListener('click', onOk);
      cancel.removeEventListener('click', onCancel);
      backdrop.removeEventListener('click', onBackdrop);
      document.removeEventListener('keydown', onKey);
      resolve(val);
    };
    const onOk = () => done(true);
    const onCancel = () => done(false);
    const onBackdrop = (e) => { if (e.target === backdrop) done(false); };
    const onKey = (e) => { if (e.key === 'Escape') done(false); };
    ok.addEventListener('click', onOk);
    cancel.addEventListener('click', onCancel);
    backdrop.addEventListener('click', onBackdrop);
    document.addEventListener('keydown', onKey);
  });
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS on both `showConfirm` checks; `2/2 passed`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "feat(wizard): add themed showConfirm modal + smoke harness"
```

---

## Task 2: Replace native `confirm()` with `showConfirm`

**Files:**
- Modify: `web/wizard.html` — `revertDrawerAliases` (~2258), `archivePackaging` (~2511), `discardEditDraft` (~2534), `deleteDraft` (~2546), `cancelWizardStep1` (~2607), `doDeploy` (~3667)
- Modify: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: `showConfirm` from Task 1.

- [ ] **Step 1: Add the failing test**

Append to the `checks` array in `web/tests/wizard_smoke.mjs` (after the last `checks.push`):

```js
checks.push(['no native confirm() left in source', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  // showConfirm is allowed; bare confirm( is not
  const bare = (src.match(/[^w]confirm\(/g) || []);
  if (bare.length) throw new Error(`found ${bare.length} native confirm( call(s)`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `found 6 native confirm( call(s)`.

- [ ] **Step 3: Convert each call site**

`cancelWizardStep1` (~2607) — the inner `confirm` becomes awaited; make the function `async`:

```js
async function cancelWizardStep1() {
  if (curDraftKey) {
    if (!(await showConfirm({ title: 'ลบ draft ที่กำลังสร้าง?', body: 'Progress ทั้งหมดจะหาย', danger: true }))) return;
    api('DELETE', `/api/packagings/${encodeURIComponent(curDraftKey)}`).catch(() => {});
    curDraftKey = null;
  }
  showView('dashboard');
}
```

`revertDrawerAliases` (~2258) first line:

```js
async function revertDrawerAliases(key) {
  if (!(await showConfirm({ title: 'กลับไปใช้รายการมาตรฐาน?', body: 'ลบ alias ที่ตั้งไว้ แล้วกลับไปใช้รายการ product มาตรฐานของระบบ', danger: true }))) return;
```

`archivePackaging` (~2511) first line (already `async`):

```js
  if (!(await showConfirm({ title: `ปิดใช้งาน "${esc(key)}"?`, body: 'Model ยังจำ class นี้อยู่ แต่ pipeline จะส่งกลับ status="archived_class" จนกว่าจะกดเปิดใช้งานอีกครั้ง', danger: true }))) return;
```

`discardEditDraft` (~2534) first line:

```js
  if (!(await showConfirm({ title: `ทิ้ง edit draft "${esc(editKey)}"?`, body: 'Progress การแก้ไขทั้งหมด (รูปใหม่ + annotations) จะหาย — packaging เดิมไม่กระทบ', danger: true }))) return;
```

`deleteDraft` (~2546) first line:

```js
  if (!(await showConfirm({ title: `ลบ draft "${esc(key)}"?`, body: 'ดำเนินการต่อ?', danger: true }))) return;
```

`doDeploy` (~3667) — replace the `prompt`/`confirm` block:

```js
async function doDeploy() {
  const isEdit = curDraftKey && curDraftKey.endsWith('__edit');
  const parentKey = isEdit ? curDraftKey.slice(0, -'__edit'.length) : null;
  const body = isEdit
    ? `Deploy ทับ active "${esc(parentKey)}"? ตัวเก่าจะถูก backup ก่อน — Cloud Run restart ~30s`
    : 'Deploy packaging นี้ขึ้น production? Cloud Run จะ restart ~30s';
  if (!(await showConfirm({ title: 'ยืนยัน Deploy', body }))) return;
```

(Leave the rest of each function body unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `no native confirm() left in source`; all prior checks still PASS.

- [ ] **Step 5: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "refactor(wizard): replace native confirm() with themed showConfirm"
```

---

## Task 3: Plug silent failures (saveAnnotation, regex preview)

**Files:**
- Modify: `web/wizard.html` — `saveAnnotation` (~3168-3185), `updateRegex` catch (~3351)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['saveAnnotation failure toasts and does not mark labeled', async (page) => {
  const r = await page.evaluate(async () => {
    // arrange minimal annot state pointing at a stub image
    curDraftKey = 'demo';
    annot.images = [{ name: 'a.jpg', labeled: false, bbox_count: 0 }];
    annot.curIdx = 0;
    annot.bboxes = [{ x1: 1, y1: 1, x2: 9, y2: 9, label: 'lot' }];
    annot.saveTimer = null;
    return true;
  });
  if (!r) throw new Error('setup failed');
  // force the PUT to fail
  await page.route('**/annotations/**', route => route.fulfill({ status: 500, body: '{"detail":"x"}' }));
  const labeled = await page.evaluate(async () => {
    saveAnnotation();
    await new Promise(res => setTimeout(res, 400));   // wait out the 250ms debounce + fetch
    const toast = document.getElementById('toast');
    const shown = toast && toast.classList.contains('show');
    return { labeled: annot.images[0].labeled, toastShown: shown };
  });
  await page.unroute('**/annotations/**');
  if (labeled.labeled !== false) throw new Error('image was marked labeled despite save failure');
  if (!labeled.toastShown) throw new Error('no toast on save failure');
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `no toast on save failure` (current catch only `console.warn`).

- [ ] **Step 3: Fix `saveAnnotation`**

Replace the `saveAnnotation` setTimeout body (~3172-3184) so labeled state is set ONLY on success and failure toasts:

```js
  annot.saveTimer = setTimeout(async () => {
    try {
      await api('PUT', `/api/packagings/${encodeURIComponent(curDraftKey)}/annotations/${encodeURIComponent(im.name)}`, {
        bboxes: annot.bboxes.map(b => ({x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, label: b.label})),
      });
      im.labeled = annot.bboxes.length > 0;
      im.bbox_count = annot.bboxes.length;
      renderThumbStrip();
      updateProgressUI();
    } catch (err) {
      showToast('เซฟ annotation ไม่สำเร็จ — ลองวาดใหม่อีกครั้ง');
    }
  }, 250);
```

- [ ] **Step 4: Fix `updateRegex` catch**

Replace the `updateRegex` catch (~3351-3353):

```js
    } catch (err) {
      showToast('สร้าง regex preview ไม่สำเร็จ');
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `saveAnnotation failure toasts and does not mark labeled`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "fix(wizard): surface saveAnnotation + regex-preview failures via toast"
```

---

## Task 4: Add missing success feedback + unify error channel

**Files:**
- Modify: `web/wizard.html` — `deleteDraft` (~2546), `step4Next` (~3454), and error `alert(...)` → `showToast(...)` in `cloneActive` (2507), `archivePackaging` (2519), `unarchivePackaging` (2530), `discardEditDraft` (2542), `deleteDraft` (2553), `runPrelabel` (3303), `step1Next` (2822), `handleFileSelect` (2880), `step4Next` (3462), `startFullTraining` (3663), `doDeploy` deploy-success `alert(msg)` (3690) + fail (3701).
- Modify: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: `showToast`.

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['no async-flow alert() left (validation alerts allowed)', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  // these specific failure/success messages must NOT use alert anymore
  const banned = [
    'alert(`Clone', 'alert(`Archive', 'alert(`Unarchive',
    'alert(`ลบไม่สำเร็จ', 'alert(`Prelabel', 'alert(`สร้าง packaging',
    'alert(`อัพโหลดล้มเหลว', 'alert(`บันทึก config', 'alert(`Start full training',
    'alert(`Deploy failed', 'alert(msg)',
  ];
  const hit = banned.filter(b => src.includes(b));
  if (hit.length) throw new Error(`async alert() still present: ${hit.join(', ')}`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL listing the async alerts still present.

- [ ] **Step 3: Convert async `alert(...)` → `showToast(...)`**

In each `catch`/result below, replace `alert(...)` with `showToast(...)` (same message text, drop the surrounding parens style):

- `cloneActive` (2507): `showToast(\`Clone ไม่สำเร็จ: ${err.message}\`);`
- `archivePackaging` (2519): `showToast(\`Archive ไม่สำเร็จ: ${err.message}\`);`
- `unarchivePackaging` (2530): `showToast(\`Unarchive ไม่สำเร็จ: ${err.message}\`);`
- `discardEditDraft` (2542): `showToast(\`ลบไม่สำเร็จ: ${err.message}\`);`
- `deleteDraft` (2553): `showToast(\`ลบไม่สำเร็จ: ${err.message}\`);`
- `runPrelabel` (3303): `showToast(\`Prelabel ไม่สำเร็จ: ${err.message}\`);`
- `step1Next` (2822): `showToast(\`สร้าง packaging ไม่สำเร็จ: ${err.message}\`);`
- `handleFileSelect` (2880): `showToast(\`อัพโหลดล้มเหลว: ${err.message}\`);`
- `step4Next` (3462): `showToast(\`บันทึก config ไม่สำเร็จ: ${err.message}\`);`
- `startFullTraining` (3663): `showToast(\`Start full training ไม่สำเร็จ: ${err.message}\`);`
- `doDeploy` fail (3701): `showToast(\`Deploy failed: ${err.message}\`);`

Leave validation `alert(...)` in `step1Next` (2799, 2800, 2802, 2804, 2806, 2811) and `handleFileSelect`/`loadAnnotator`/`runPrelabel` guard `alert('ยังไม่ได้สร้าง draft...')` UNCHANGED — those are blocking validation.

- [ ] **Step 4: Add success toasts**

`deleteDraft` (~2546) — add a toast on success before `loadDashboard()`:

```js
async function deleteDraft(key) {
  if (!(await showConfirm({ title: `ลบ draft "${esc(key)}"?`, body: 'ดำเนินการต่อ?', danger: true }))) return;
  try {
    await api('DELETE', `/api/packagings/${encodeURIComponent(key)}`);
    closeDrawer();
    loadDashboard();
    showToast('ลบ draft แล้ว');
  } catch (err) {
    showToast(`ลบไม่สำเร็จ: ${err.message}`);
  }
}
```

`step4Next` (~3459) — add a toast on save success before `goStep(5)`:

```js
    await api('POST', `/api/packagings/${encodeURIComponent(curDraftKey)}/config`, collectConfig());
    showToast('บันทึก config แล้ว');
    goStep(5);
```

`doDeploy` deploy-success (~3680-3690) — replace the multi-line `alert(msg)` with a toast carrying the key facts (keep the existing `msg` assembly, but show via toast and keep the post-deploy navigation):

```js
    const res = await api('POST', `/api/packagings/${encodeURIComponent(curDraftKey)}/deploy`);
    const cr = res.cloud_run || {};
    const crNote = cr.triggered ? 'Cloud Run: triggered (~30s)' : `Cloud Run ข้าม: ${cr.reason || 'permission missing'}`;
    showToast(`✅ Deploy สำเร็จ — ${res.target_key || curDraftKey} · ${crNote}`, 8000);
    if (isEdit) {
      curDraftKey = null;
      showView('dashboard');
    } else {
      await loadStep5();
    }
```

(Remove the now-unused `let msg = ...` assembly lines that built the alert string.)

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `no async-flow alert() left`. All prior checks PASS.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "refactor(wizard): unify async feedback on toast + add missing success toasts"
```

---

## Task 5: Remove dead views + `promoProd`

**Files:**
- Modify: `web/wizard.html` — delete `view-staging` (~1668-1687) and `view-success` (~1689-1706)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['dead views + promoProd removed', async (page) => {
  const r = await page.evaluate(() => ({
    staging: !!document.getElementById('view-staging'),
    success: !!document.getElementById('view-success'),
    promoProd: typeof promoProd,
  }));
  if (r.staging) throw new Error('#view-staging still present');
  if (r.success) throw new Error('#view-success still present');
  if (r.promoProd !== 'undefined') throw new Error('promoProd still defined/referenced');
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `#view-staging still present`.

- [ ] **Step 3: Delete the two view blocks**

Remove the entire `<!-- ══ STAGING ... -->` block (`<div class="view" id="view-staging"> … </div>`, ~1668-1687) and the entire `<!-- ══ SUCCESS ... -->` block (`<div class="view" id="view-success"> … </div>`, ~1689-1706). `promoProd` exists only inside the staging block (the `onclick="promoProd()"` at 1682), so deleting the block removes the only reference; no function definition exists to delete.

- [ ] **Step 4: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `dead views + promoProd removed`.

- [ ] **Step 5: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "chore(wizard): remove orphaned staging/success mockup views"
```

---

## Task 6: Clear demo hardcode (step 1 inputs + step 4 lot examples)

**Files:**
- Modify: `web/wizard.html` — step-1 inputs (~1303-1311), step-4 lot rows (~1539-1550), `rx-display` (~1556)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['no demo hardcode in step1/step4 inputs', async (page) => {
  const r = await page.evaluate(() => ({
    name: document.getElementById('inp-display-name').value,
    key: document.getElementById('inp-key').value,
    desc: document.getElementById('inp-desc').value,
    lotRows: document.querySelectorAll('#lot-rows .lot-row').length,
    firstLot: document.querySelector('#lot-rows input')?.value || '',
  }));
  if (r.name || r.key || r.desc) throw new Error('step1 inputs still prefilled');
  if (r.lotRows !== 1 || r.firstLot) throw new Error('step4 lot examples still hardcoded');
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `step1 inputs still prefilled`.

- [ ] **Step 3: Empty the step-1 inputs**

Replace the three step-1 input lines (keep placeholders):

```html
<input type="text" id="inp-display-name" value="" placeholder="เช่น Capsule Box Premium">
```
```html
<input type="text" id="inp-key" value="" placeholder="เช่น capsule_box_premium" style="font-family:'JetBrains Mono',monospace">
```
```html
<textarea id="inp-desc" placeholder="อธิบายลักษณะ เช่น สีสัน, รูปร่าง, ตำแหน่งของ lot"></textarea>
```

- [ ] **Step 4: Reduce step-4 lot examples to one empty row + neutral rx-display**

Replace the three `.lot-row` blocks (~1539-1550) with a single empty row:

```html
<div class="lot-row">
  <input type="text" value="" placeholder="เลข lot ตัวอย่าง" oninput="updateRegex()">
  <button class="rm-btn" onclick="removeLot(this)">×</button>
</div>
```

Replace the `rx-display` initial content (~1556) with empty:

```html
<div class="regex-display" id="rx-display"></div>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `no demo hardcode in step1/step4 inputs`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "chore(wizard): clear demo placeholder values from step1/step4"
```

---

## Task 7: `prefillStep4FromDraft` (step-4 data-loss fix)

**Files:**
- Modify: `web/wizard.html` — add `prefillStep4FromDraft`; call it from `goStep` step-4 branch (~1887)
- Modify: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: `collectConfig`, `pickTpl`, `addProdAlias`, `syncProductAliasVisibility`, `toggleCb`, `updateRegex`.
- Produces: global `async function prefillStep4FromDraft()`.

- [ ] **Step 1: Add the failing test**

Add a stubbed draft-with-config to the route handler and a check. In `stubFor`, the single-packaging branch must return config when key is `cfgdraft`. Update `stubFor` (in `wizard_smoke.mjs`) — replace the single-packaging branch:

```js
  if (/\/api\/packagings\/[^/]+$/.test(pathname)) {
    if (pathname.endsWith('/cfgdraft')) {
      return { key: 'cfgdraft', display_name: 'Cfg', status: 'configured',
               pipeline: 'detector_ocr', image_count: 60, sub_regions: ['lot'],
               config: { lot_patterns: ['(?i)XX\\\\d+'], fields_extracted: ['lot','exp','product'],
                         sheet_checks: ['lot'], message_template_key: 'lot_exp',
                         product_aliases: [{ canonical: 'Houjicha', keywords: ['houjicha'] }] } };
    }
    return { key: 'demo', display_name: 'Demo', status: 'draft',
             pipeline: 'detector_ocr', image_count: 0, sub_regions: ['lot'] };
  }
```

Append the check:

```js
checks.push(['prefillStep4FromDraft restores saved config', async (page) => {
  const r = await page.evaluate(async () => {
    curDraftKey = 'cfgdraft';
    cropMode = 'single';
    await prefillStep4FromDraft();
    const on = (sel) => !!document.querySelector(sel);
    return {
      pattern: document.getElementById('rx-display').textContent,
      lotRows: document.querySelectorAll('#lot-rows .lot-row').length,
      product: on('#sp4 [data-group="fields"] .cbitem.on[data-field="product"]'),
      tpl: document.querySelector('#sp4 .tpl-opt.on')?.dataset.template,
      aliasRows: document.querySelectorAll('#pa-rows .pa-row').length,
    };
  });
  if (r.pattern !== '(?i)XX\\d+') throw new Error(`pattern not restored: ${r.pattern}`);
  if (r.lotRows !== 1) throw new Error('example rows not reset to 1');
  if (!r.product) throw new Error('product field not toggled on');
  if (r.tpl !== 'lot_exp') throw new Error(`template not selected: ${r.tpl}`);
  if (r.aliasRows !== 1) throw new Error(`alias rows: ${r.aliasRows}`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `prefillStep4FromDraft is not defined`.

- [ ] **Step 3: Add `prefillStep4FromDraft`**

Add immediately AFTER `prefillFieldsFromSubRegions` (~3368):

```js
async function prefillStep4FromDraft() {
  if (!curDraftKey) return;
  let pkg;
  try { pkg = await api('GET', `/api/packagings/${encodeURIComponent(curDraftKey)}`); }
  catch { return; }
  const cfg = pkg.config;
  if (!cfg) return;   // fresh draft with no saved config → keep defaults

  // Restore lot pattern directly; reset example rows to one empty (regex source
  // can't be reconstructed, and empty rows won't let updateRegex overwrite it).
  if (cfg.lot_patterns && cfg.lot_patterns.length) {
    document.getElementById('rx-display').textContent = cfg.lot_patterns[0];
  }
  document.getElementById('lot-rows').innerHTML =
    `<div class="lot-row"><input type="text" value="" placeholder="เลข lot ตัวอย่าง" oninput="updateRegex()"><button class="rm-btn" onclick="removeLot(this)">×</button></div>`;

  // Fields
  const wantFields = new Set(cfg.fields_extracted || ['lot']);
  document.querySelectorAll('#sp4 [data-group="fields"] .cbitem[data-field]').forEach(el => {
    if (el.dataset.field === 'lot') return;
    el.classList.toggle('on', wantFields.has(el.dataset.field));
  });
  // Sheet checks
  const wantSheet = new Set(cfg.sheet_checks || []);
  document.querySelectorAll('#sp4 [data-group="sheet"] .cbitem[data-field]').forEach(el => {
    el.classList.toggle('on', wantSheet.has(el.dataset.field));
  });
  // Template
  const tplKey = cfg.message_template_key;
  if (tplKey) {
    const tile = document.querySelector(`#sp4 .tpl-opt[data-template="${tplKey}"]`);
    if (tile) pickTpl(tile);
  }
  // Product aliases
  document.getElementById('pa-rows').innerHTML = '';
  (cfg.product_aliases || []).forEach(a => addProdAlias(a.canonical, (a.keywords || []).join(', ')));
  syncProductAliasVisibility();
}
```

- [ ] **Step 4: Call it from `goStep`**

In `goStep` (~1887), change the step-4 branch to prefill from the saved draft after the existing field pre-check:

```js
  if (n === 4) { prefillFieldsFromSubRegions(); prefillStep4FromDraft(); syncProductAliasVisibility(); }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `prefillStep4FromDraft restores saved config`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "fix(wizard): prefill step 4 from saved draft config (stop re-POST data loss)"
```

---

## Task 8: Complete `renderDrawerBody` step-map

**Files:**
- Modify: `web/wizard.html` — `renderDrawerBody` stepMap (~2308)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['drawer step-map covers trained/training_full', async (page) => {
  const r = await page.evaluate(() => {
    const mk = (status) => renderDrawerBody(
      { key: 'd', display_name: 'D', status, pipeline: 'detector_ocr', image_count: 60,
        conf_threshold: null, accuracy: null },
      { samples: [] });
    return { trained: mk('trained'), training: mk('training_full') };
  });
  if (!r.trained.includes('Step 5 / 5')) throw new Error('trained not step 5');
  if (!r.training.includes('Step 5 / 5')) throw new Error('training_full not step 5');
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `trained not step 5` (defaults to step 1 currently).

- [ ] **Step 3: Extend the step-map**

In `renderDrawerBody` (~2308), replace the stepMap line:

```js
    const stepMap = {draft:1, uploading:2, labeled_full:3, configured:4, training_full:5, trained:5};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `drawer step-map covers trained/training_full`.

- [ ] **Step 5: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "fix(wizard): show correct wizard step for trained/training_full drafts"
```

---

## Task 9: Image-count gate helper (fresh 50 / edit 30)

**Files:**
- Modify: `web/wizard.html` — add `imageGate()`; use in `handleFileSelect` (~2878) and `initStep2` (~1925); update count copy (~1431)
- Modify: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Produces: global `function imageGate()` → `number` (`50` for fresh, `30` for `__edit`).

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['imageGate is 50 fresh / 30 edit', async (page) => {
  const r = await page.evaluate(() => {
    curDraftKey = 'freshkey';
    const fresh = imageGate();
    curDraftKey = 'freshkey__edit';
    const edit = imageGate();
    return { fresh, edit };
  });
  if (r.fresh !== 50) throw new Error(`fresh gate ${r.fresh}`);
  if (r.edit !== 30) throw new Error(`edit gate ${r.edit}`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `imageGate is not defined`.

- [ ] **Step 3: Add the helper**

Add just before `handleFileSelect` (~2828):

```js
// Min images to unlock step 2 → step 3. Edit-drafts (__edit) reuse the parent's
// reference images at train time, so they need fewer NEW images than a fresh class.
function imageGate() { return (curDraftKey && curDraftKey.endsWith('__edit')) ? 30 : 50; }
```

- [ ] **Step 4: Use the helper at both sites**

`handleFileSelect` (~2878):

```js
    if (result.total_images >= imageGate()) document.getElementById('btn-step2-next').disabled = false;
```

`initStep2` (~1925) — change the edit-draft unlock from `newCount > 0`:

```js
    if (newCount >= imageGate()) {
      const nextBtn = document.getElementById('btn-step2-next');
      if (nextBtn) nextBtn.disabled = false;
    }
```

Update the count-placeholder copy (~1431) so it does not imply a fixed 63:

```html
<span class="cnt" id="up-cnt">0 รูป</span>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `imageGate is 50 fresh / 30 edit`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "fix(wizard): consistent image gate — 50 fresh / 30 edit-draft"
```

---

## Task 10: Remove redundant crop-mode UI

**Files:**
- Modify: `web/wizard.html` — remove the legacy advanced toggle buttons (~1341, ~1350); remove `srToggleAdvanced` if unreferenced
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['legacy crop-mode advanced toggle removed', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  if (src.includes('srToggleAdvanced')) throw new Error('srToggleAdvanced still present');
  // 3-way tabs remain
  const tabs = await page.evaluate(() => document.querySelectorAll('.sr-mode-tab').length);
  if (tabs !== 3) throw new Error(`expected 3 mode tabs, got ${tabs}`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `srToggleAdvanced still present`.

- [ ] **Step 3: Remove the two toggle buttons**

Delete the `<button type="button" class="sr-adv-toggle" onclick="srToggleAdvanced(true)">…</button>` block (~1341-1343) inside `#sr-single`, and the `<button type="button" class="sr-adv-collapse" onclick="srToggleAdvanced(false)">↩ กลับเป็นจุดเดียว</button>` (~1350) inside `#sr-multi`'s `.sr-multi-head`.

- [ ] **Step 4: Remove the now-orphaned function**

Delete the `srToggleAdvanced` function (~2703-2708). `grep -n "srToggleAdvanced" web/wizard.html` must return nothing after the button removal in Step 3.

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `legacy crop-mode advanced toggle removed`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "chore(wizard): drop redundant crop-mode toggle — 3-way tabs are the single control"
```

---

## Task 11: Dynamic class count in dashboard subtitle

**Files:**
- Modify: `web/wizard.html` — dashboard subtitle (~1228); set it in `loadDashboard` (~2020)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['dashboard subtitle count is dynamic', async (page) => {
  // stub two active packagings, reload dashboard, expect "2"
  await page.route('**/api/packagings', route => route.fulfill({ status: 200,
    contentType: 'application/json', body: JSON.stringify([
      { key: 'a', display_name: 'A', status: 'active', pipeline: 'detector_ocr', image_count: 60, accuracy: null, conf_threshold: 0.6 },
      { key: 'b', display_name: 'B', status: 'active', pipeline: 'detector_ocr', image_count: 60, accuracy: null, conf_threshold: 0.6 },
    ]) }), { times: 1 });
  const txt = await page.evaluate(async () => {
    await loadDashboard();
    return document.getElementById('dash-sub').textContent;
  });
  if (!txt.includes('2')) throw new Error(`subtitle did not reflect count: "${txt}"`);
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL — element `#dash-sub` does not exist / text hardcoded "6".

- [ ] **Step 3: Give the subtitle an id**

Replace the dashboard subtitle (~1228):

```html
<p id="dash-sub">AI model รู้จัก 6 ประเภทบรรจุภัณฑ์ — คลิกการ์ดเพื่อดูรายละเอียด</p>
```

- [ ] **Step 4: Update it from `liveCount`**

In `loadDashboard`, right after `document.querySelector('#nav-dash .nav-badge').textContent = liveCount;` (~2021), add:

```js
    const sub = document.getElementById('dash-sub');
    if (sub) sub.textContent = `AI model รู้จัก ${liveCount} ประเภทบรรจุภัณฑ์ — คลิกการ์ดเพื่อดูรายละเอียด`;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `dashboard subtitle count is dynamic`.

- [ ] **Step 6: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "fix(wizard): dashboard subtitle reflects real active class count"
```

---

## Task 12: Drop misleading HEIC label

**Files:**
- Modify: `web/wizard.html` — upload-desc (~1425)
- Modify: `web/tests/wizard_smoke.mjs`

- [ ] **Step 1: Add the failing test**

Append to `checks`:

```js
checks.push(['HEIC removed from upload label', async (page) => {
  const t = await page.evaluate(() => document.querySelector('.upload-desc')?.textContent || '');
  if (/HEIC/i.test(t)) throw new Error('HEIC still in upload label');
  if (!/JPG/i.test(t)) throw new Error('upload label lost JPG');
}]);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL `HEIC still in upload label`.

- [ ] **Step 3: Edit the label**

Replace the upload-desc line (~1425):

```html
<div class="upload-desc">รองรับ <span>JPG, PNG</span></div>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: PASS `HEIC removed from upload label`; final `13/13 passed` (or current total).

- [ ] **Step 5: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "chore(wizard): drop HEIC from upload label (accept stays image/*)"
```

---

## Self-Review

**Spec coverage:**
- Unit 1 (feedback infra) → Task 1. ✓
- Unit 2 (replace confirm) → Task 2. ✓
- Unit 3 (silent failures) → Task 3. ✓
- Unit 4 (missing success + unify error) → Task 4. ✓
- Unit 5 (dead code) → Task 5. ✓
- Unit 6 (demo hardcode) → Task 6. ✓
- Unit 7 (step4 prefill) → Task 7. ✓
- Unit 8 (stepMap) → Task 8. ✓
- Unit 9 (gating) → Task 9. ✓
- Unit 10 (crop UI) → Task 10. ✓
- Unit 11 (class count) → Task 11. ✓
- Unit 12 (HEIC) → Task 12. ✓
- Verification (approach A, no-backend Playwright) → harness in Task 1, grown per task. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test step has full assertion code. ✓

**Type/name consistency:** `showConfirm({title,body,danger})→Promise<boolean>` (Task 1) used consistently (Tasks 2,4). `imageGate()→number` defined Task 9, used same name both sites. `prefillStep4FromDraft()` defined Task 7, called in `goStep` same name. Modal ids `#confirm-ok/#confirm-cancel/#confirm-backdrop` defined Task 1, referenced in Task 1 tests only. `#dash-sub` defined+used Task 11. ✓

**Out-of-scope guard:** No backend edits; fake progress untouched; `alert()` retained for validation; `accept="image/*"` unchanged. ✓
