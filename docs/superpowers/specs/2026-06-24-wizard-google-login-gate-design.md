# Wizard Google Login Gate — Design

**Date:** 2026-06-24
**Status:** Approved (design phase)
**Scope:** `web/wizard.html` + `web/tests/wizard_smoke.mjs` only. No backend changes.

## Goal

Gate the OCR Lot Checker wizard UI behind a Google Sign-In screen. Only Google
accounts in the `@tdfb.co` Workspace domain may pass the gate and see the
dashboard. This is a **client-side UI gate** — it hides the admin UI from casual
/ unauthorized users; it does **not** secure the backend API.

## Explicit non-goals (stated, not built)

- No API-side enforcement. `/predict`, `/api/packagings`, deploy, training, and
  Drive-write endpoints stay publicly callable. A determined user can bypass the
  gate by reading page source or calling the API directly. This is accepted.
- No roles / multi-user accounts. Any `@tdfb.co` account has full access.
- No protection of the public `/predict` endpoint.
- The upgrade path (verify the Google ID token in FastAPI to actually protect
  the API) is documented here for the future but is **not** implemented.

## Decisions

| Decision | Choice |
|---|---|
| Approach | Google Identity Services (GIS), pure client-side |
| Allowlist | Email domain `@tdfb.co` (verified via the `hd` hosted-domain claim) |
| OAuth client | **Reuse** the existing Drive OAuth client (`DRIVE_OAUTH_CLIENT_ID`) |
| Gate scope | Everywhere — no `file://`/localhost bypass |
| Session length | Own TTL, 8 hours (`SESSION_TTL_MS`), independent of the 1h token `exp` |
| Files touched | `web/wizard.html`, `web/tests/wizard_smoke.mjs` |

## Architecture

A full-screen login overlay (`#login-gate`) is baked into `wizard.html` markup
and shown by default, so the dashboard never flashes before auth. The existing
bootstrap IIFE at the end of the `<script>` block —

```js
(async () => {
  const resumed = await restoreFromUrl();
  if (!resumed) loadDashboard();
})();
```

— is extracted into a function `bootWizard()` that runs **only after**
authentication succeeds. Google's GIS library
(`https://accounts.google.com/gsi/client`) loads from `<head>` with `async defer`.

## Components (all inside `web/wizard.html` unless noted)

1. **Login overlay markup** — a fixed, full-viewport `#login-gate` element with:
   product title, a `#g-signin` container (Google renders its button here), and a
   `#login-status` line for messages/errors.
2. **Auth config constants** — placed near `API_BASE`:
   - `GOOGLE_CLIENT_ID` — the `*.apps.googleusercontent.com` value of
     `DRIVE_OAUTH_CLIENT_ID` (public; pasted literally into the file).
   - `ALLOWED_HD = 'tdfb.co'`
   - `SESSION_TTL_MS = 8 * 60 * 60 * 1000`
   - `AUTH_STORAGE_KEY = 'wizardAuth'`
3. **Auth functions:**
   - `initAuth()` — load session; if valid → `bootWizard()`; else render the gate.
   - `onGoogleCredential(resp)` — decode + validate the JWT, save session, hide
     overlay, `bootWizard()`; on failure show an error and stay gated.
   - `decodeJwtPayload(jwt)` — base64url-decode the middle segment to an object.
   - `loadSession()` / `saveSession(claims)` / `clearSession()` — `localStorage`.
   - `bootWizard()` — the renamed existing bootstrap IIFE body.
   - `signOut()` — `clearSession()` + `google.accounts.id.disableAutoSelect()` +
     `location.reload()`.
4. **Topbar user chip** — added to `#topbar`: signed-in email (+ avatar if
   available) and a "ออกจากระบบ" (Sign out) control wired to `signOut()`.

## Data flow

1. Page load → `initAuth()`.
2. `loadSession()` reads `wizardAuth`. If present, not expired
   (`now < session.exp`), and domain valid → `bootWizard()` directly (no button).
3. Otherwise GIS is initialized:
   ```js
   google.accounts.id.initialize({
     client_id: GOOGLE_CLIENT_ID,
     callback: onGoogleCredential,
     hosted_domain: ALLOWED_HD,        // UX hint only — still verified below
     auto_select: false,
   });
   google.accounts.id.renderButton(document.getElementById('g-signin'), {...});
   ```
4. On sign-in, `onGoogleCredential(resp)` decodes `resp.credential` (an ID-token
   JWT) and **validates all of**:
   - `payload.hd === ALLOWED_HD`
   - `payload.email_verified === true`
   - `payload.email` ends with `@tdfb.co`
   - `payload.exp * 1000 > Date.now()`
5. **Pass** → `saveSession({ email, name, picture, exp: Date.now() + SESSION_TTL_MS })`,
   hide `#login-gate`, `bootWizard()`.
   **Fail** → show "เฉพาะบัญชี @tdfb.co เท่านั้น" in `#login-status`,
   `google.accounts.id.disableAutoSelect()`, stay gated.
6. Sign out → `signOut()` → overlay returns on the reload.

> The session validity window is our own `SESSION_TTL_MS`, not the token's 1-hour
> `exp`. Because this is a UI gate (not real auth), re-prompting hourly would be
> pure friction. The token `exp` is only checked at the moment of sign-in to
> reject a stale credential.

## Error handling & edge cases

- **GIS library blocked / offline** — `window.google?.accounts?.id` is absent;
  `#login-status` shows a retry message and the gate stays closed.
- **Wrong domain / unverified / decode failure** — treated as an auth failure
  with a clear message; no session is written.
- **Expired session** — `loadSession()` returns null when `now >= session.exp`;
  overlay returns on next load.
- **Corrupt `localStorage` JSON** — `loadSession()` try/catches and treats it as
  no session.
- **`file://`** — GIS cannot initialize without an http(s) origin. When
  `location.protocol === 'file:'`, `#login-status` shows
  "เปิดผ่าน URL (https) เพื่อเข้าสู่ระบบ". No bypass — the `file://` access path
  is retired (per the "gate everywhere" decision; CLAUDE.md's double-click
  workflow no longer applies to a gated build).

## OAuth client setup (one-time, manual, in Google Cloud Console)

The existing Drive **Web application** OAuth client (`DRIVE_OAUTH_CLIENT_ID`) is
reused. Add to its **Authorized JavaScript origins**:

- the production Netlify URL (e.g. `https://<site>.netlify.app`)
- `http://localhost:8090` (smoke test origin)
- any other local dev origin used to open the wizard over http

This is additive — it does not disturb the existing Authorized **redirect URIs**
(`http://localhost:8765/`) used by `scripts/generate_drive_token.py`. One OAuth
client can carry both JS origins and redirect URIs. The client ID is public and
safe to embed in `wizard.html`; no secret is exposed by this change.

## CSP note

`wizard.html` and `netlify.toml` define **no** Content-Security-Policy today, so
the remote GIS script loads without changes. If a CSP is added later it must
allow: `script-src https://accounts.google.com`,
`connect-src https://accounts.google.com`,
`frame-src https://accounts.google.com`.

## Testing

`web/tests/wizard_smoke.mjs` serves `web/` over `http://localhost:8090` and drives
the page with Playwright. To keep existing checks reaching the dashboard, the
harness seeds a valid session before navigation:

```js
await page.addInitScript((key) => {
  localStorage.setItem(key, JSON.stringify({
    email: 'tester@tdfb.co', name: 'Tester', picture: '',
    exp: Date.now() + 60 * 60 * 1000,
  }));
}, 'wizardAuth');
```

Add two new checks:

1. **Gate blocks with no session** — clear `localStorage`, reload, assert
   `#login-gate` is visible and the dashboard root is not.
2. **Gate reveals with a seeded session** — with the seeded session, assert
   `#login-gate` is hidden and `loadDashboard` ran (existing dashboard assertion).

No real Google network is involved; the gate is exercised purely through the
`localStorage` session seam.

## Risks

- **Client-side bypass** (accepted): page source and API remain reachable. If real
  protection is ever needed, implement backend ID-token verification (the
  documented upgrade path).
- **OAuth origin misconfig**: if the Netlify origin is not added to the client's
  Authorized JavaScript origins, the button silently fails to render. Mitigation:
  the overlay shows a retry/diagnostic message, and setup is documented above.
