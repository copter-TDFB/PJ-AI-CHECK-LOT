# Wizard Feedback Standardization + Cleanup — Design

**Date:** 2026-06-24
**Scope:** `web/wizard.html` only (frontend; no backend changes)
**Source:** two review rounds over `web/wizard.html` (bug/UX pass + toast/feedback-coverage pass)

## Problem

`web/wizard.html` (single static HTML, inline JS) has accumulated three classes of issues:

1. **Inconsistent feedback** — three channels used with no rule: `showToast` (themed),
   native `alert()` / `confirm()` (off-theme, jarring). Same function often uses toast on
   success but `alert` on error. The wizard flow (steps 1–5) uses `alert` exclusively; the
   drawer/dashboard uses toast.
2. **Silent failures (data-loss risk)** — `saveAnnotation` and `updateRegex` swallow errors
   to `console.warn`; the user gets no signal. A failed annotation auto-save loses work
   silently.
3. **Dead code / hardcode / logic bugs** — orphaned mockup screens, demo placeholder values
   that ship to real users, a step-4 re-POST that overwrites saved config, inconsistent
   image-count gating, an incomplete draft step-map, redundant crop-mode UI, a hardcoded
   class count, and a misleading HEIC label.

## Decisions (locked with user)

| Topic | Decision |
|---|---|
| Feedback rule | Toast-first + themed `showConfirm()` modal; `alert()` kept **only** for blocking validation |
| Image gating | Fresh draft = **50**; edit-draft (`__edit`) = **30** (matches backend guard; frontend-only) |
| HEIC | Remove "HEIC" from upload label (keep JPG/PNG); do **not** touch backend decode |
| Fake upload progress | **Out of scope** — leave 15%→100% as-is |
| Custom alert modal | **Out of scope** — `alert()` stays native for validation |
| Backend training gate | **Out of scope** — `api/packagings.py:603` `< 30` stays unchanged |
| Verification | Approach A — Playwright **no-backend** smoke (stub `fetch`/`window.api`) + manual checklist for the prod-hitting deploy flow |

## Design

### Unit 1 — Feedback infrastructure

**`showConfirm({title, body, danger})` → `Promise<boolean>`**
- New themed modal reusing the existing `.drawer-backdrop` visual language (dark surface,
  `--bd`/`--acc` tokens). Renders a title, body (supports a short HTML string), a Cancel
  button and a Confirm button (Confirm uses `btn-danger` when `danger:true`, else
  `btn-primary`).
- Resolves `true` on Confirm, `false` on Cancel / backdrop-click / `Esc`.
- Lightweight focus handling: focus the Confirm button on open; `Esc` resolves `false`.
- Single shared DOM node created lazily (mirrors `showToast`'s lazy-create pattern).

**Feedback rule (applied across the file):**
- `showToast(...)` — every async result, success **and** failure.
- `await showConfirm(...)` — every destructive / irreversible action.
- `alert(...)` — only synchronous input validation that must block progression.

### Unit 2 — Replace `confirm()` call sites (6)

All become `if (!(await showConfirm({...}))) return;` and their callers become `async`:
- `revertDrawerAliases` (2259)
- `archivePackaging` (2512) — `danger`
- `discardEditDraft` (2535) — `danger`
- `deleteDraft` (2547) — `danger`
- `cancelWizardStep1` (2611) — `danger`
- `doDeploy` (3673)

`\n\n` multi-line strings in the current prompts become `body` HTML (`<br>` / small text).

### Unit 3 — Plug silent failures

- `saveAnnotation` catch (3182): `console.warn` → `showToast('เซฟ annotation ไม่สำเร็จ — ลองวาดใหม่อีกครั้ง')`.
  On failure, **do not** mark `im.labeled = true` — keep the thumb/progress reflecting the
  real persisted state so the labeled count is honest. (Set `im.labeled`/`im.bbox_count`
  only inside the success path.)
- `updateRegex` catch (3352): `console.warn` → `showToast('สร้าง regex preview ไม่สำเร็จ')`.

### Unit 4 — Add missing success feedback; unify error channel

- `deleteDraft` success → `showToast('ลบ draft แล้ว')` (parity with `discardEditDraft`).
- `step4Next` config-save success → `showToast('บันทึก config แล้ว')` before `goStep(5)`.
- Drawer/dashboard action **errors** (`cloneActive`, `archivePackaging`, `unarchivePackaging`,
  `discardEditDraft`, `deleteDraft`, `runPrelabel`) switch from `alert(...)` →
  `showToast(...)` so success and error share one channel.
- Wizard-flow async errors that currently `alert` and genuinely block a step
  (`step1Next` create-fail, `handleFileSelect` upload-fail, `step4Next` save-fail,
  `startFullTraining`, `syncTrainedModel`, `doDeploy`) → `showToast(...)`. The final deploy
  **success** summary (currently `alert(msg)` at 3690) becomes a toast with the key facts;
  detailed fields stay available but the channel is unified.

### Unit 5 — Remove dead code

- Delete `<div id="view-staging">` (1668–1687) and `<div id="view-success">` (1689–1706) —
  never reached (`grep` shows no `showView('staging'|'success')`).
- Delete the now-orphaned `promoProd()` reference (no definition exists). No staging
  environment exists (prod-only per CLAUDE.md); the real deploy flow uses `doDeploy` →
  toast → dashboard / `loadStep5`.

### Unit 6 — Clear demo hardcode

- Step 1 inputs (1303–1311): `inp-display-name` value → empty (placeholder kept),
  `inp-key` value → empty (placeholder e.g. `new_tea_bag_box`), `inp-desc` → empty
  (placeholder kept).
- Step 4 lot-examples (1539–1550): replace the three hardcoded `TB0005…` rows with a single
  empty row. `rx-display` (1556) initial text → empty/neutral.
- Success-screen literal "New Tea Bag Box" / "7 class" copy is removed with Unit 5 (those
  views are deleted).

### Unit 7 — `prefillStep4FromDraft()` (step-4 data-loss fix)

`goStep(4)` currently re-renders step 4 from static form state; re-entering + Next re-POSTs
defaults over saved config. New behavior:
- On entering step 4 for a draft that has saved config, `GET /api/packagings/{key}` and
  populate the form from `config`:
  - set `rx-display.textContent = cfg.lot_patterns[0]` and **clear** the lot-example rows
    to a single empty row (so `updateRegex` does not overwrite the restored pattern unless
    the user types new examples).
  - toggle `[data-group="fields"]` checkboxes per `cfg.fields_extracted`.
  - toggle `[data-group="sheet"]` checkboxes per `cfg.sheet_checks`.
  - select the template tile per `cfg.message_template_key`.
  - rebuild `#pa-rows` from `cfg.product_aliases`; then `syncProductAliasVisibility()`.
- A fresh draft with no saved config keeps the (now-empty) defaults.
- Keep the existing `prefillFieldsFromSubRegions()` for the `multi_field` field pre-check;
  the new prefill runs after it and wins where config exists.

### Unit 8 — `renderDrawerBody` step-map completeness

`stepMap` (2308) `{draft:1, uploading:2, configured:4}` → add `training_full:5`,
`trained:5` (and `labeled_full:3`) so a trained draft shows the correct "Step n / 5" instead
of defaulting to 1.

### Unit 9 — Image-count gating

- `handleFileSelect` (2878): enable `btn-step2-next` when `result.total_images >= GATE`
  where `GATE = curDraftKey.endsWith('__edit') ? 30 : 50`.
- `initStep2` (1925): edit-draft enables next when `newCount >= 30` (was `> 0`).
- Update the count placeholder copy (1431) and the "อย่างน้อย 50 รูป" desc to reflect the
  fresh/edit split (e.g. edit banner notes "≥30 รูปใหม่").
- Extract the gate into a small helper (e.g. `imageGate()`) to keep the two call sites in
  sync.

### Unit 10 — Remove redundant crop-mode UI

- Remove the legacy `srToggleAdvanced` entry points: the "⚙ ขั้นสูง…" button (1341) and the
  "↩ กลับเป็นจุดเดียว" button (1350). The 3-way `.sr-mode-tabs` (`srSetMode`) is the single
  way to choose `single` / `multi_field` / `cross_check`.
- Keep `srToggleAdvanced` deletable only if no other caller remains; otherwise leave the
  function but remove its UI triggers. (Verify with grep before deleting the function.)

### Unit 11 — Dynamic class count

- Dashboard subtitle "AI model รู้จัก 6 ประเภท…" (1228) is updated from the real active
  count in `loadDashboard` (reuse `liveCount`), so it tracks reality like the nav badge
  already does.

### Unit 12 — HEIC label

- `upload-desc` (1425) "รองรับ JPG, PNG, HEIC" → "รองรับ JPG, PNG". `accept` attribute
  unchanged (`image/*`).

## Verification (Approach A)

Playwright **no-backend** smoke, serving `web/` via `python -m http.server` and stubbing the
backend (override `window.fetch` or the `api()` function) so no prod call is made:

- `showConfirm` resolves `true` on Confirm and `false` on Cancel / Esc / backdrop.
- A stubbed failing `saveAnnotation` surfaces a toast and does **not** flip the thumb to
  labeled.
- `renderDrawerBody` step-map returns the right step for `trained` / `training_full`.
- Step-2 next unlocks at 50 for a fresh key and at 30 for a `__edit` key (stub the upload
  response count).
- Crop-mode tabs switch panels; the legacy advanced buttons are gone.
- `view-staging` / `view-success` no longer exist in the DOM; `promoProd` is undefined.
- `prefillStep4FromDraft` populates fields/sheet/template/pattern/aliases from a stubbed
  draft config and clears example rows.

Manual checklist (prod-hitting, not automated): real upload, real deploy, real
clone/archive/unarchive round-trip — confirm toasts fire and `showConfirm` gates them.

Match the existing Playwright wizard gotchas in CLAUDE.md: serve with
`python -m http.server 8090 --directory web`; match elements by `onclick` attr not Thai
text; `openDrawer(key, summary)` needs the summary arg (click the real card).

## Out of scope (explicit)

- Fake upload progress bar (left as 15%→100%).
- Backend HEIC decoding.
- Custom themed `alert()` (validation stays native).
- Backend training-gate change (`api/packagings.py:603` stays `< 30`).
- Any unrelated refactor of `wizard.html` structure.

## Risks

- **`showConfirm` async conversion**: every `confirm()` caller must become `async` and be
  `await`ed; a missed `await` would let a destructive action run unconditionally. The
  Playwright tests cover the resolve-true/false paths to catch this.
- **`prefillStep4FromDraft` lot pattern**: saved drafts store the compiled regex, not the
  source examples — reconstruction is impossible, so we restore the pattern directly and
  leave examples empty. Documented as intended behavior, not a gap.
- **Single large file**: changes are spread across one 3.7k-line file; keep each unit a
  separate commit so review and rollback stay granular.
