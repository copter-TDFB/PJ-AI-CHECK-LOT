# Wizard: Back to Dashboard (2026-06-10)

## Problem

While creating a new packaging in `web/wizard.html`, the user cannot return to the
dashboard. `showView()` deliberately blocks dashboard navigation while a draft is
active, and the sidebar "Packaging" item is locked. The only exit is the Cancel
button on step 1, which deletes the draft.

## Decision

Allow returning to the dashboard from any wizard step. The draft is **always kept**
(no confirm dialog) — drafts are already auto-saved server-side at every step and
appear on the dashboard with a "Continue setup →" button.

## Changes (all in `web/wizard.html`)

1. Remove the guard in `showView()` that blocks `wizard → dashboard` when
   `curDraftKey` is set.
2. Stop adding the `.locked` class to `#nav-dash` when entering the wizard.
3. Add a "← กลับ" button (`#topbar-back`) in the topbar, before the breadcrumb,
   visible only when `currentView === 'wizard'`. Clicking it calls
   `exitWizardToDashboard()`.
4. New `exitWizardToDashboard()`: clear `curDraftKey`/`curStep`, clear the
   `#draft=…&step=…` URL hash (so refresh lands on the dashboard), call
   `showView('dashboard')`, and reload the packaging list so the draft card shows.
5. Step 1 Cancel (`cancelWizardStep1()`) keeps its current confirm + DELETE
   behaviour — the only in-wizard way to discard a draft.

No backend changes. No data loss: images, annotations, and config are persisted
on each action already.
