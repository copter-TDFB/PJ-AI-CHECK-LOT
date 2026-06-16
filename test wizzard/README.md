# Test Wizard Harness

Isolated copy of the packaging wizard for safe testing. **Never touches production
Cloud Run or the real Drive dataset.** `wizard.html` here is frontend only — it needs
the backend (this whole project) running; the folder alone does nothing.

## Quick start for a teammate (one command)

For a **tdfb.co Workspace** member running on their own machine.

The sender must hand over the **whole project folder** including:
- `models/classifier.pt` and `models/detector.pt`
- `oauth_client.json` (the Internal OAuth client — ask the sender)

Then, from the project root in PowerShell:

```
.\scripts\setup_and_run.ps1
```

That one script: creates a virtualenv + installs deps → opens a browser for Google
consent (sign in with your tdfb.co account) → creates YOUR own `OCR-LOT-TEST` Drive
tree → writes `.env.test` → launches the backend and wizard. First run takes a few
minutes (deps + consent); later runs skip straight to launch.

Open **http://localhost:8091/wizard.html** when you see `Application startup complete`.

No GCP service-account key is needed — in `TEST_MODE` the backend boots without
Vision/Sheets credentials (the wizard flow never calls them).

## Manual setup (alternative)

1. Ensure `.env` has `DRIVE_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN` set.
2. Create the test Drive tree and copy the printed ids:
   ```
   python scripts/setup_test_drive.py
   ```
3. Copy `.env.test.example` -> `.env.test`, paste the three `DRIVE_*` ids and the
   same `DRIVE_OAUTH_*` values.
4. Run:
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
