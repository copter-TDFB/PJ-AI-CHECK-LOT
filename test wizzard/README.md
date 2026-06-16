# Test Wizard Harness

Isolated copy of the packaging wizard for safe testing. **Never touches production
Cloud Run or the real Drive dataset.**

## One-time setup

1. Ensure the prod `.env` has `DRIVE_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN` set.
2. Create the test Drive tree and copy the printed ids:
   ```
   python scripts/setup_test_drive.py
   ```
3. Copy `.env.test.example` -> `.env.test`, paste the three `DRIVE_*` ids and the
   same `DRIVE_OAUTH_*` values as prod.

## Run

```
.\scripts\run_test_wizard.ps1
```

- Backend (TEST_MODE): http://localhost:8081
- Test wizard UI:      http://localhost:8091/wizard.html

## What is simulated

- **Deploy** writes YAML/models into `data/test/` and reloads the test registry, but
  the Cloud Run revision trigger is skipped (`{"triggered": false, "reason": "test
  mode (simulated)"}`).
- **Dataset publish** uploads real images/labels into the `OCR-LOT-TEST/` Drive
  folders only.
- **Notebook** returns the existing combined-notebook link (training is not run here).

## Safety

Production is protected by: `TEST_MODE=1` gating the trigger, port 8081, and
`data/test/*` + `OCR-LOT-TEST/` isolation. The only shared resource is the Google
account / OAuth user; the harness writes nowhere outside `OCR-LOT-TEST/`.
