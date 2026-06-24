# Wizard Google Login Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate the OCR Lot Checker wizard UI behind a client-side Google Sign-In screen that only admits `@tdfb.co` Workspace accounts.

**Architecture:** A full-screen login overlay baked into `web/wizard.html` covers the app until a valid session exists. The page's existing bootstrap is wrapped in `bootWizard()` and runs only after auth. Google Identity Services (GIS) provides the sign-in button; the returned ID-token JWT is decoded and validated client-side, then an 8-hour session is persisted in `localStorage`. No backend changes.

**Tech Stack:** Vanilla JS in a single static HTML file (`web/wizard.html`), Google Identity Services (`https://accounts.google.com/gsi/client`), Playwright smoke test (`web/tests/wizard_smoke.mjs`, Node + `playwright`).

## Global Constraints

- All app code lives in the single file `web/wizard.html` — do not create new frontend files. Follow existing patterns (Thai UI copy, `var(--token)` colors, global functions).
- `print()`/`console.log` debug noise is discouraged repo-wide; keep the diff clean.
- Allow-list domain is exactly `tdfb.co`, checked via the `hd` claim AND an `@tdfb.co` email suffix AND `email_verified === true`.
- Session TTL is exactly `8 * 60 * 60 * 1000` ms. `localStorage` key is exactly `wizardAuth`.
- The OAuth client is the existing Drive client. `GOOGLE_CLIENT_ID` = the value of `DRIVE_OAUTH_CLIENT_ID` in `.env` (a `*.apps.googleusercontent.com` string; public, safe to embed).
- This is a UI gate only — no backend/API enforcement is added.
- Run the smoke test with: `node web/tests/wizard_smoke.mjs` (serves `web/` on `http://localhost:8090`). `playwright` must be installed (`npm i -D playwright` / `npx playwright install chromium` if needed).
- CSS color tokens available: `--bg #0B0C10`, `--s1 #181B22`, `--bd #2A2E3A`, `--t1 #EDEEF1`, `--t2 #8A8FA5`, `--t3 #7C8196`, `--acc #E8A020`, `--err #F87171`.

---

### Task 1: Auth core + login gate (session path, no Google yet)

Builds the overlay, session storage, claim-validation helpers, `bootWizard()` extraction, and `initAuth()` for the session/`file://` paths. The Google button itself is added in Task 2. Also adds the test harness seam (isolated-context helper + main-page seeding) so existing checks keep passing once the gate exists.

**Files:**
- Modify: `web/wizard.html` (markup after `<body>` line 1197; CSS in the `<style>` block; constants after the `API_BASE` block near line 1785; boot section at end of `<script>`)
- Test: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: nothing (first task).
- Produces (global functions other tasks rely on):
  - `decodeJwtPayload(jwt: string): object` — base64url-decodes the JWT payload segment to an object (UTF-8 safe).
  - `isAllowedClaims(payload: object): boolean` — true iff `hd === 'tdfb.co'` && `email_verified === true` && `email` ends `@tdfb.co` && `exp*1000 > Date.now()`.
  - `loadSession(): object|null` — returns the stored session if unexpired and domain-valid, else null.
  - `saveSession(claims: {email,name?,picture?}): object` — writes `{email,name,picture,exp:Date.now()+SESSION_TTL_MS}` to `localStorage`.
  - `clearSession(): void`
  - `hideGate(): void` — adds `hidden` class to `#login-gate`.
  - `bootWizard(): void` — runs the wizard bootstrap (formerly the trailing IIFE).
  - `initAuth(): void` — entry point; session→boot, `file://`→message, else (Task 2 wires the button).
  - Test helper `openWizard(page, {seed?, waitUntil?})` in the smoke file.
  - Constants: `GOOGLE_CLIENT_ID`, `ALLOWED_HD='tdfb.co'`, `SESSION_TTL_MS`, `AUTH_STORAGE_KEY='wizardAuth'`.

- [ ] **Step 1: Write the failing tests**

In `web/tests/wizard_smoke.mjs`, add the isolated-context helper immediately after the `stubFor` function definition (near line 31):

```js
// Open the wizard in a fresh isolated context (own localStorage).
// seed = a session object to pre-write under 'wizardAuth', or null.
async function openWizard(page, { seed = null, waitUntil = 'domcontentloaded' } = {}) {
  const ctx = await page.context().browser().newContext();
  const p = await ctx.newPage();
  await p.addInitScript((o) => { window.API_BASE_OVERRIDE = o; }, ORIGIN);
  if (seed) await p.addInitScript((s) => { localStorage.setItem('wizardAuth', s); }, JSON.stringify(seed));
  await p.route('https://accounts.google.com/**', (r) => r.abort());
  await p.route('**/api/**', (route) => {
    const u = new URL(route.request().url());
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(stubFor(u.pathname)) });
  });
  await p.goto(`${ORIGIN}/wizard.html`, { waitUntil });
  return { ctx, p };
}
```

Then add these three checks (anywhere in the `checks.push(...)` region):

```js
checks.push(['login gate blocks without a session', async (page) => {
  const { ctx, p } = await openWizard(page, { seed: null });
  const visible = await p.evaluate(() => {
    const g = document.getElementById('login-gate');
    return !!g && !g.classList.contains('hidden');
  });
  await ctx.close();
  if (!visible) throw new Error('gate not visible without a session');
}]);

checks.push(['login gate reveals dashboard with a valid session', async (page) => {
  const seed = { email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000 };
  const { ctx, p } = await openWizard(page, { seed });
  let hidden = false;
  try {
    await p.waitForFunction(() => {
      const g = document.getElementById('login-gate');
      return g && g.classList.contains('hidden');
    }, { timeout: 5000 });
    hidden = true;
  } catch (_) {}
  await ctx.close();
  if (!hidden) throw new Error('gate still visible with a valid session');
}]);

checks.push(['auth claim validation accepts @tdfb.co, rejects others/expired', async (page) => {
  const res = await page.evaluate(() => {
    const mk = (o) => 'x.' + btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_') + '.y';
    const future = Math.floor(Date.now() / 1000) + 3600;
    const good = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: true, exp: future })));
    const badDomain = isAllowedClaims(decodeJwtPayload(mk({ hd: 'gmail.com', email: 'a@gmail.com', email_verified: true, exp: future })));
    const unverified = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: false, exp: future })));
    const expired = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: true, exp: 1 })));
    return { good, badDomain, unverified, expired };
  });
  if (res.good !== true) throw new Error('valid @tdfb.co claims rejected');
  if (res.badDomain !== false) throw new Error('non-tdfb.co domain accepted');
  if (res.unverified !== false) throw new Error('unverified email accepted');
  if (res.expired !== false) throw new Error('expired token accepted');
}]);
```

Also seed the shared main-page so existing dashboard checks run in the authed state. In `main()`, immediately after the existing line
`await page.addInitScript((origin) => { window.API_BASE_OVERRIDE = origin; }, ORIGIN);`
add:

```js
  await page.addInitScript(() => {
    localStorage.setItem('wizardAuth', JSON.stringify({
      email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000,
    }));
  });
  await page.route('https://accounts.google.com/**', (r) => r.abort());
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node web/tests/wizard_smoke.mjs`
Expected: the three new checks FAIL — `gate not visible` / `gate still visible` / `isAllowedClaims is not defined` (no `#login-gate`, no auth functions yet).

- [ ] **Step 3: Add the login overlay markup**

In `web/wizard.html`, immediately after `<body>` (line 1197), insert:

```html
<div id="login-gate" class="login-gate" role="dialog" aria-modal="true" aria-labelledby="login-title">
  <div class="login-card">
    <div class="login-brand">🔒 OCR Lot Checker</div>
    <h1 id="login-title" class="login-h">เข้าสู่ระบบ</h1>
    <p class="login-sub">ใช้บัญชี Google ของ @tdfb.co เท่านั้น</p>
    <div id="g-signin" class="g-signin"></div>
    <div id="login-status" class="login-status" role="status" aria-live="polite"></div>
  </div>
</div>
```

- [ ] **Step 4: Add the overlay CSS**

In the `<style>` block (before `</style>`), add:

```css
.login-gate{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:var(--bg);padding:24px}
.login-gate.hidden{display:none}
.login-card{max-width:360px;width:100%;text-align:center;background:var(--s1);border:1px solid var(--bd);border-radius:16px;padding:32px 28px;box-shadow:0 20px 60px rgba(0,0,0,.45)}
.login-brand{font-size:13px;letter-spacing:.04em;color:var(--t3);margin-bottom:18px}
.login-h{font-size:22px;margin:0 0 6px;color:var(--t1)}
.login-sub{font-size:13px;color:var(--t3);margin:0 0 22px}
.g-signin{display:flex;justify-content:center;min-height:44px}
.login-status{margin-top:16px;font-size:13px;color:var(--err);min-height:18px}
```

- [ ] **Step 5: Add auth constants**

Immediately after the `API_BASE` definition block (the IIFE ending near line 1796), add:

```js
// ── Auth gate config ──
// GOOGLE_CLIENT_ID reuses the Drive OAuth client. Paste the value of
// DRIVE_OAUTH_CLIENT_ID from .env here (a public *.apps.googleusercontent.com id).
const GOOGLE_CLIENT_ID = 'PASTE_DRIVE_OAUTH_CLIENT_ID_HERE.apps.googleusercontent.com';
const ALLOWED_HD = 'tdfb.co';
const SESSION_TTL_MS = 8 * 60 * 60 * 1000;
const AUTH_STORAGE_KEY = 'wizardAuth';
```

> Implementer action: read `DRIVE_OAUTH_CLIENT_ID` from `.env` and replace the placeholder string with its exact value.

- [ ] **Step 6: Add the auth helper functions**

Add near the auth constants (still inside the main `<script>`):

```js
function decodeJwtPayload(jwt) {
  const seg = String(jwt).split('.')[1];
  const b64 = seg.replace(/-/g, '+').replace(/_/g, '/');
  const json = decodeURIComponent(
    atob(b64).split('').map((c) => '%' + c.charCodeAt(0).toString(16).padStart(2, '0')).join('')
  );
  return JSON.parse(json);
}

function isAllowedClaims(p) {
  return !!p
    && p.hd === ALLOWED_HD
    && p.email_verified === true
    && typeof p.email === 'string'
    && p.email.toLowerCase().endsWith('@' + ALLOWED_HD)
    && typeof p.exp === 'number'
    && p.exp * 1000 > Date.now();
}

function loadSession() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (!s || typeof s.exp !== 'number' || Date.now() >= s.exp) return null;
    if (typeof s.email !== 'string' || !s.email.toLowerCase().endsWith('@' + ALLOWED_HD)) return null;
    return s;
  } catch (_) { return null; }
}

function saveSession(claims) {
  const s = { email: claims.email, name: claims.name || '', picture: claims.picture || '', exp: Date.now() + SESSION_TTL_MS };
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(s));
  return s;
}

function clearSession() { localStorage.removeItem(AUTH_STORAGE_KEY); }

function hideGate() { document.getElementById('login-gate').classList.add('hidden'); }

function initAuth() {
  const status = document.getElementById('login-status');
  const session = loadSession();
  if (session) { hideGate(); bootWizard(); return; }
  if (location.protocol === 'file:') {
    status.textContent = 'เปิดผ่าน URL (https) เพื่อเข้าสู่ระบบ';
    return;
  }
  // No session over http(s): Task 2 renders the Google button here.
  status.textContent = '';
}
```

- [ ] **Step 7: Extract `bootWizard()` and make auth the entry point**

At the end of the `<script>`, replace this block:

```js
renderSteps();
srSetMode('single');
onPipelineChange();
// Bootstrap: ถ้า URL มี draft+step → resume wizard, ไม่งั้นไป dashboard
(async () => {
  const resumed = await restoreFromUrl();
  if (!resumed) loadDashboard();
})();
// Browser back/forward → sync state
window.addEventListener('popstate', () => { restoreFromUrl(); });
```

with:

```js
function bootWizard() {
  renderSteps();
  srSetMode('single');
  onPipelineChange();
  // Bootstrap: ถ้า URL มี draft+step → resume wizard, ไม่งั้นไป dashboard
  (async () => {
    const resumed = await restoreFromUrl();
    if (!resumed) loadDashboard();
  })();
}
// Browser back/forward → sync state (only meaningful once authed)
window.addEventListener('popstate', () => { if (loadSession()) restoreFromUrl(); });
// Auth gate is the entry point; bootWizard runs only after a valid session.
initAuth();
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `node web/tests/wizard_smoke.mjs`
Expected: all checks PASS, including the three new ones. The final line shows `N/N passed`.

- [ ] **Step 9: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "feat(wizard): login gate core — session, claim validation, boot gating"
```

---

### Task 2: Google Sign-In button + credential callback

Loads GIS, renders the button on the no-session path, and handles the credential callback (decode → validate → save session → reveal, or reject non-`@tdfb.co`).

**Files:**
- Modify: `web/wizard.html` (add GIS `<script>` in `<head>`; add render/callback functions; wire `initAuth` no-session path)
- Test: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: `decodeJwtPayload`, `isAllowedClaims`, `saveSession`, `hideGate`, `bootWizard`, `GOOGLE_CLIENT_ID`, `ALLOWED_HD` (Task 1).
- Produces:
  - `onGoogleCredential(resp: {credential: string}): void` — validates the JWT; on success saves session + reveals the wizard; on failure shows an error and stays gated.
  - `renderGoogleButton(status: HTMLElement): void` — renders the GIS button when the library is ready (with a brief retry), else shows a load-failure message.

- [ ] **Step 1: Write the failing test**

In `web/tests/wizard_smoke.mjs`, add:

```js
checks.push(['onGoogleCredential gates out non-tdfb.co and admits @tdfb.co', async (page) => {
  const { ctx, p } = await openWizard(page, { seed: null });
  const r = await p.evaluate(() => {
    const mk = (o) => 'x.' + btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_') + '.y';
    const future = Math.floor(Date.now() / 1000) + 3600;
    onGoogleCredential({ credential: mk({ hd: 'gmail.com', email: 'a@gmail.com', email_verified: true, exp: future }) });
    const blockedAfterBad = !document.getElementById('login-gate').classList.contains('hidden');
    const storedAfterBad = !!localStorage.getItem('wizardAuth');
    onGoogleCredential({ credential: mk({ hd: 'tdfb.co', email: 'ok@tdfb.co', email_verified: true, exp: future }) });
    const openedAfterGood = document.getElementById('login-gate').classList.contains('hidden');
    const storedAfterGood = !!localStorage.getItem('wizardAuth');
    return { blockedAfterBad, storedAfterBad, openedAfterGood, storedAfterGood };
  });
  await ctx.close();
  if (!r.blockedAfterBad) throw new Error('gate opened for non-tdfb.co account');
  if (r.storedAfterBad) throw new Error('session stored for non-tdfb.co account');
  if (!r.openedAfterGood) throw new Error('gate did not open for @tdfb.co account');
  if (!r.storedAfterGood) throw new Error('session not stored for @tdfb.co account');
}]);

checks.push(['GIS client script is present', async (page) => {
  const has = await page.evaluate(() =>
    !!document.querySelector('script[src="https://accounts.google.com/gsi/client"]'));
  if (!has) throw new Error('GIS client script tag missing');
}]);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: the two new checks FAIL — `onGoogleCredential is not defined` and `GIS client script tag missing`.

- [ ] **Step 3: Add the GIS library script**

In `web/wizard.html` `<head>` (before `</head>`), add:

```html
<script src="https://accounts.google.com/gsi/client" async defer></script>
```

- [ ] **Step 4: Add the render + callback functions**

Add to the main `<script>` (near the other auth functions from Task 1):

```js
function onGoogleCredential(resp) {
  const status = document.getElementById('login-status');
  let claims;
  try { claims = decodeJwtPayload(resp.credential); }
  catch (_) { status.textContent = 'เข้าสู่ระบบไม่สำเร็จ ลองใหม่อีกครั้ง'; return; }
  if (!isAllowedClaims(claims)) {
    status.textContent = 'เฉพาะบัญชี @tdfb.co เท่านั้น';
    try { google.accounts.id.disableAutoSelect(); } catch (_) {}
    return;
  }
  saveSession(claims);
  hideGate();
  bootWizard();
}

function renderGoogleButton(status) {
  const ready = () => window.google && google.accounts && google.accounts.id;
  const doRender = () => {
    google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: onGoogleCredential,
      hosted_domain: ALLOWED_HD,
      auto_select: false,
    });
    google.accounts.id.renderButton(document.getElementById('g-signin'),
      { type: 'standard', theme: 'filled_blue', size: 'large', text: 'signin_with', shape: 'pill' });
  };
  if (ready()) { doRender(); return; }
  let tries = 0;
  const iv = setInterval(() => {
    if (ready()) { clearInterval(iv); doRender(); }
    else if (++tries > 40) { clearInterval(iv); status.textContent = 'โหลด Google Sign-In ไม่สำเร็จ — รีเฟรชหน้าเพื่อลองใหม่'; }
  }, 100);
}
```

- [ ] **Step 5: Wire the no-session path in `initAuth`**

In `initAuth()` (from Task 1), replace the final two lines of the function:

```js
  // No session over http(s): Task 2 renders the Google button here.
  status.textContent = '';
```

with:

```js
  renderGoogleButton(status);
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: all checks PASS, including the two new ones.

- [ ] **Step 7: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "feat(wizard): Google Sign-In button + credential validation callback"
```

---

### Task 3: Topbar user chip + sign-out

Shows the signed-in email in the topbar and a sign-out control that clears the session and returns to the gate.

**Files:**
- Modify: `web/wizard.html` (topbar markup near line 1269; CSS; `renderUserChip`/`signOut`; call `renderUserChip` from `initAuth` + `onGoogleCredential`)
- Test: `web/tests/wizard_smoke.mjs`

**Interfaces:**
- Consumes: `loadSession`, `clearSession`, `onGoogleCredential`, `initAuth` (Tasks 1–2).
- Produces:
  - `renderUserChip(u: {email}): void` — fills `#tu-email` and shows `#topbar-user`.
  - `signOut(): void` — clears the session, disables auto-select, reloads.

- [ ] **Step 1: Write the failing test**

In `web/tests/wizard_smoke.mjs`, add:

```js
checks.push(['topbar shows signed-in email; sign-out returns to gate', async (page) => {
  const seed = { email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000 };
  const { ctx, p } = await openWizard(page, { seed });
  await p.waitForFunction(() => {
    const g = document.getElementById('login-gate');
    return g && g.classList.contains('hidden');
  }, { timeout: 5000 });
  const email = await p.evaluate(() => document.getElementById('tu-email')?.textContent || '');
  if (email !== 'tester@tdfb.co') { await ctx.close(); throw new Error(`chip email wrong: "${email}"`); }
  await p.click('#tu-signout');
  let backToGate = false;
  try {
    await p.waitForFunction(() => {
      const g = document.getElementById('login-gate');
      return g && !g.classList.contains('hidden');
    }, { timeout: 5000 });
    backToGate = true;
  } catch (_) {}
  const cleared = await p.evaluate(() => !localStorage.getItem('wizardAuth'));
  await ctx.close();
  if (!backToGate) throw new Error('gate did not return after sign-out');
  if (!cleared) throw new Error('session not cleared after sign-out');
}]);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node web/tests/wizard_smoke.mjs`
Expected: FAIL — `#tu-email` does not exist (chip markup not added yet).

- [ ] **Step 3: Add the topbar user chip markup**

In `web/wizard.html`, inside `#topbar`, immediately before the `topbar-add` button (the `<button class="btn btn-primary btn-sm" id="topbar-add" ...>` near line 1269), insert:

```html
    <div class="topbar-user" id="topbar-user" style="display:none">
      <span class="tu-email" id="tu-email"></span>
      <button class="btn btn-ghost btn-sm" id="tu-signout" onclick="signOut()">ออกจากระบบ</button>
    </div>
```

- [ ] **Step 4: Add the chip CSS**

In the `<style>` block, add:

```css
.topbar-user{display:flex;align-items:center;gap:10px;margin-left:8px}
.tu-email{font-size:12px;color:var(--t2);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
```

- [ ] **Step 5: Add `renderUserChip` and `signOut`**

Add to the main `<script>` (near the auth functions):

```js
function renderUserChip(u) {
  const wrap = document.getElementById('topbar-user');
  const em = document.getElementById('tu-email');
  if (!wrap || !em || !u) return;
  em.textContent = u.email || '';
  wrap.style.display = 'flex';
}

function signOut() {
  clearSession();
  try { if (window.google && google.accounts && google.accounts.id) google.accounts.id.disableAutoSelect(); } catch (_) {}
  location.reload();
}
```

- [ ] **Step 6: Call `renderUserChip` on both auth paths**

In `initAuth()`, change the session branch from:

```js
  if (session) { hideGate(); bootWizard(); return; }
```

to:

```js
  if (session) { renderUserChip(session); hideGate(); bootWizard(); return; }
```

In `onGoogleCredential()`, after `saveSession(claims);` add `renderUserChip(claims);` so the block reads:

```js
  saveSession(claims);
  renderUserChip(claims);
  hideGate();
  bootWizard();
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `node web/tests/wizard_smoke.mjs`
Expected: all checks PASS.

- [ ] **Step 8: Commit**

```bash
git add web/wizard.html web/tests/wizard_smoke.mjs
git commit -m "feat(wizard): topbar user chip + sign-out"
```

---

## Post-implementation manual steps (not code — for the operator)

1. In Google Cloud Console, open the existing Drive OAuth **Web application** client (`DRIVE_OAUTH_CLIENT_ID`). Under **Authorized JavaScript origins**, add: the production Netlify URL, `http://localhost:8090` (smoke test), and any local dev origin used to open the wizard over http. Do **not** remove the existing Authorized redirect URI `http://localhost:8765/`.
2. Confirm `GOOGLE_CLIENT_ID` in `web/wizard.html` matches `DRIVE_OAUTH_CLIENT_ID`.
3. Deploy `web/` to Netlify (push to `origin/main`) and verify: visiting the Netlify URL shows the gate; signing in with an `@tdfb.co` account reveals the dashboard; a non-`@tdfb.co` account is rejected with the Thai error.

## Self-Review

- **Spec coverage:** overlay + boot gating (Task 1), GIS button + JWT decode/validate + domain allow-list + session save (Tasks 1–2), `file://` message (Task 1), user chip + sign-out (Task 3), 8h TTL + `localStorage` key (Task 1), reuse Drive client ID + manual origins step (constants + post-impl notes), testing seam via seeded `localStorage` (all tasks). CSP note in spec requires no code today (no CSP exists). All spec sections map to a task.
- **Placeholder scan:** the only intentional fill-in is `GOOGLE_CLIENT_ID`, flagged explicitly with a read-from-`.env` instruction — not a hidden TODO. No vague "add error handling" steps; every code step shows complete code.
- **Type consistency:** `decodeJwtPayload`, `isAllowedClaims`, `loadSession`, `saveSession`, `clearSession`, `hideGate`, `bootWizard`, `initAuth`, `renderGoogleButton`, `onGoogleCredential`, `renderUserChip`, `signOut` are named identically across definition and call sites; the session object shape `{email,name,picture,exp}` and the `wizardAuth` key are consistent across tasks and tests.
