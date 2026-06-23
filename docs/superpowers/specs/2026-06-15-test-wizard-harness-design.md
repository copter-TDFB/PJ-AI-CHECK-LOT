# Test Wizard Harness — Design

**Date:** 2026-06-15
**Status:** Approved (pre-implementation)

## Goal

Clone the packaging wizard into an isolated test harness under `test wizzard/` that
exercises the full flow — create draft → upload images → publish dataset → generate
notebook link → click Deploy — **without ever touching the production Cloud Run
service or the production Drive dataset.**

Scope ends at "generate notebook" and a *simulated* Deploy. Actual Colab training is
out of scope.

## Non-goals

- Retraining a real model end-to-end inside the test harness (notebook link is reused
  as-is; the test does not need a runnable-trained model).
- Touching production Cloud Run in any way.
- A separate Google account (test Drive lives in the SAME account/OAuth user as prod,
  isolated to a dedicated root folder).

## Architecture

One codebase, two run profiles separated by env file and port:

| Profile | Port | Env file | Config dir | Models | Drafts | Drive |
|---|---|---|---|---|---|---|
| prod dev | 8080 | `.env` | `config/` | `models/` | `data/drafts/` | real folders |
| **test** | 8081 | `.env.test` | `data/test/config/` | `data/test/models/` | `data/test/drafts/` | `OCR-LOT-TEST/` folders |

The `test wizzard/` folder contains:
- `wizard.html` — a copy of `web/wizard.html` with an inline
  `<script>window.API_BASE_OVERRIDE='http://localhost:8081'</script>` placed **before**
  the main script block (so `API_BASE` at `web/wizard.html:1711` resolves to the test
  backend regardless of how the file is served).
- A launch script (PowerShell `run_test.ps1`) that bootstraps test dirs and starts the
  test backend on port 8081 with `.env.test`.
- `README.md` — how to run the harness.

## Isolation mechanism (env vars)

| Concern | Env var | Status | Test value |
|---|---|---|---|
| Test mode flag | `TEST_MODE` | **new** | `1` |
| Config root dir | `OCR_CONFIG_DIR` | **new** | `data/test/config` |
| Models dir | `MODELS_DIR` | existing | `data/test/models` |
| Drafts dir | `DRAFT_DIR` | existing | `data/test/drafts` |
| Crop cache | `CROP_CACHE_DIR` | existing | `data/test/crops` |
| Detector dataset folder | `DRIVE_DETECTOR_DATASET_FOLDER_ID` | existing | test folder id |
| Classifier dataset folder | `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` | existing | test folder id |
| Manifest file | `DRIVE_MANIFEST_FILE_ID` | existing | test file id |
| Config overrides file | `DRIVE_CONFIG_OVERRIDES_FILE_ID` | existing | test file id |
| OAuth user creds | `DRIVE_OAUTH_*` | existing | same as prod |

## Code changes (minimal, backward-compatible)

All three default to current behavior, so **production is unaffected** when the new env
vars are unset.

1. **`pipeline/packaging_registry.py:10`** — `_CONFIG_DIR` currently hardcoded to
   `Path(__file__).parent.parent / "config"`. Change to read `OCR_CONFIG_DIR`
   (default `"config"`).

2. **`services/cloudrun_deployer.py:25`** — `_PACKAGING_DIR` currently hardcoded to
   `Path("config/packagings")`. Change to `Path(os.getenv("OCR_CONFIG_DIR", "config")) / "packagings"`
   so registry and deployer agree on the same config root.

3. **`api/packagings.py`** deploy endpoint (`~line 660`) — gate the Cloud Run trigger:
   ```python
   if os.getenv("TEST_MODE") == "1":
       cr_result = {"triggered": False, "reason": "test mode (simulated)"}
   else:
       cr_result = cloudrun_deployer.trigger_cloud_run_revision()
   ```
   This is the single line that can reach production Cloud Run; gating it is the core
   safety guarantee.

## Test Drive setup

`scripts/setup_test_drive.py` — runs as the OAuth user (full `drive` scope, avoids the
`drive.file` blind-spot on manually-created folders). Per repo convention, inserts
`sys.path` for `services` imports and runs from repo root.

Creates:
```
OCR-LOT-TEST/                          (root)
├── data check lot/                    → DRIVE_DETECTOR_DATASET_FOLDER_ID
├── data classify check lot/           → DRIVE_CLASSIFIER_DATASET_FOLDER_ID
├── manifest.json   ({} or minimal)    → DRIVE_MANIFEST_FILE_ID
└── config_overrides.json   ({})       → DRIVE_CONFIG_OVERRIDES_FILE_ID
```

Prints every file_id ready to paste into `.env.test`. Supports `--reset` to delete and
recreate the root folder for a clean run. The combined notebook is reused via the
existing `COMBINED_NOTEBOOK_FILE_ID` — not copied.

## Bootstrap test dirs

The launch script (or a helper invoked by it) copies, on startup if missing:
- `config/` → `data/test/config/` (gives the test wizard the same 6 packagings +
  message templates as prod, to clone/edit from)
- `models/*.pt` → `data/test/models/` (so the test backend loads at startup)

`data/test/` is gitignored.

## Deploy flow in test (simulated)

```
draft
  → upload images            (lands in data/test/drafts/<key>/images/)
  → publish dataset          (REAL upload to OCR-LOT-TEST/ Drive folders)
  → generate notebook        (returns existing COMBINED_NOTEBOOK_FILE_ID link)
  → click Deploy
       hard-floor gate       (real)
       backup_artifacts      (real, into data/test/)
       write_packaging_yaml  (real, into data/test/config/packagings/)
       promote_draft_model   (real, into data/test/models/)
       reload_registry       (real, test instance)
       trigger Cloud Run     → SKIPPED (TEST_MODE) → {"triggered": false, ...}
```

## Safety guarantees

Production Cloud Run is never touched because:
- `trigger_cloud_run_revision()` is gated behind `TEST_MODE`.
- Test runs on a different port (8081) with a different config/models/drafts dir.
- Test writes only inside the `OCR-LOT-TEST/` Drive root.

Sole shared resource: the same Google Drive account + OAuth user. Test never writes
outside `OCR-LOT-TEST/`.

## Open item to verify during implementation

**CORS** — if the test wizard is served via `http.server` (cross-origin to the :8081
backend), the backend must allow that origin. Inspect the CORS middleware in `main.py`
and add the test origin if needed (test-only; do not loosen prod CORS).

## Verification checklist

- [ ] With `TEST_MODE=1`, Deploy returns `triggered: false` and the GCP Cloud Run API is
      never called (assert via log / mock).
- [ ] All writes land under `data/test/` — `config/packagings/`, `models/`, `data/drafts/`
      on disk are untouched after a full test run.
- [ ] Dataset publish lands inside `OCR-LOT-TEST/` on Drive, not the real dataset folders.
- [ ] Unset env vars → backend behaves exactly as before (prod regression guard).
- [ ] Test wizard at :8081 can list/clone the mirrored packagings.
