# Runbook

Operational reference for the OCR Lot Checker Cloud Run service. For the
full request-flow architecture, read `CLAUDE.md` first — this file is about
*operating* the deployed service, not how the pipeline works internally.

- **Service:** `ocr-lot-checker`, project `pj-ai-detect-lot-no`, region `asia-southeast1`
- **Production URL:** `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`
- **Config/model store:** GCS bucket `ocr-lot-checker-config` (GCS-first as of 2026-06-22)
- **Frontend:** `web/wizard.html`, deployed to Netlify from `origin/main`

## Health check

<!-- AUTO-GENERATED: derived from main.py route decorators. -->

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check. Returns `{"status":"ok","test_mode":bool}`. Does **not** verify the model/registry loaded successfully — a 200 here does not guarantee `/predict` works (see "Known failure modes" below). |
| `/predict` | POST | Main pipeline entrypoint (multipart `file` + `sheet_id`/`sheet_gid` query params). |
| `/api/packagings/*` | various | Wizard CRUD/training/deploy API — see `api/packagings.py` for the full route list (30 endpoints as of this writing); not reproduced here to avoid drift. |

<!-- END AUTO-GENERATED -->

There is **no configured alerting/monitoring** beyond Cloud Run's own default
metrics (PLAN.md tracks this as "Phase 8 — Monitoring", not started). Checking
service health today means manually curling `/health` and reading Cloud
Logging — there is no paging/on-call setup to escalate to.

## Deployment procedure

Full command reference lives in `CLAUDE.md` § Commands — this is the condensed
step-by-step.

1. **Build** — never use `gcloud run deploy --source .` (fails against this
   project's `cloud-run-source-deploy` repo). Build to the prod image repo
   explicitly:
   ```bash
   gcloud builds submit --tag asia-southeast1-docker.pkg.dev/pj-ai-detect-lot-no/ocr-repo/ocr-lot-checker:<tag> .
   ```
2. **Deploy as a candidate, no traffic:**
   ```bash
   gcloud run deploy ocr-lot-checker --image <that-image> --region asia-southeast1 \
     --no-traffic --tag candidate \
     --set-env-vars GCS_CONFIG_BUCKET=ocr-lot-checker-config,... \
     --set-secrets DRIVE_OAUTH_CLIENT_SECRET=drive-oauth-client-secret:latest,...
   ```
   For a **code-only** change (no new env/secrets needed), a bare
   `gcloud run deploy --image <new> --no-traffic --tag candidate` **inherits**
   the live revision's env/secrets/service account and its 600s request
   timeout — you do not need to re-pass `--set-env-vars`/`--set-secrets`
   (doing so risks silently dropping a var). Read the live service's env
   first (`gcloud run services describe ocr-lot-checker --region asia-southeast1`)
   to confirm before assuming inheritance is enough.
3. **Verify the candidate** at its `candidate---ocr-lot-checker-...run.app`
   URL — hit `/health`, and ideally run a real photo through `/predict` (a
   placeholder `sheet_id` is fine; `SheetChecker.check()` catches sheet
   errors and still returns the extraction fields, just with `null` match
   flags — useful for verifying OCR/classification without a real sheet).
4. **Flip traffic:**
   ```bash
   gcloud run services update-traffic ocr-lot-checker --region asia-southeast1 --to-revisions <rev>=100
   ```

### Rollback

```bash
gcloud run services update-traffic ocr-lot-checker --region asia-southeast1 --to-revisions <previous-rev>=100
```

Previous revisions stay deployed (0% traffic) unless explicitly deleted —
rollback is just re-pointing traffic, not a redeploy. List candidates with
`gcloud run revisions list --service ocr-lot-checker --region asia-southeast1`.

## Config vs. code changes — where a fix actually needs to go

This is the single most common source of "I fixed it but prod is still
broken" confusion in this service:

| You changed... | Where it's read from in prod | What ships it |
|---|---|---|
| A **wizard-created/edited** packaging class (e.g. `print_sticker_back`) | GCS `packagings/<key>.yaml`, overlaid on top of the image | The wizard's "Deploy" button, or `cloudrun_deployer.publish_packaging_to_gcs(key)` directly |
| An **originally shipped** packaging class (e.g. `back_label`, `30_sachet`) | The image's baked `config/packagings/<key>.yaml` — **unless** it was ever published to GCS too (see below) | Rebuild the image + redeploy |
| `conf_threshold` or `product_aliases` on any class | `config_overrides.json` (GCS or Drive) — a **runtime override**, separate from the YAML | `PUT /api/packagings/{key}/conf` or `/product-aliases` — no redeploy needed |
| Model weights (`.pt`) | GCS `models/` per `manifest.json`, when `GCS_CONFIG_BUCKET` is set | `cloudrun_deployer.publish_packaging_to_gcs` (uploads changed models automatically) |

**Gotcha (hit in production 2026-07-02):** a packaging can be BOTH baked into
the image AND have a stale copy in GCS `packagings/<key>.yaml` if it was ever
published there (e.g. via the wizard's Deploy button, even if it's normally
thought of as an "originally shipped" class). If that GCS copy exists, the
registry overlay uses it **instead of** the freshly rebuilt image's version —
so rebuilding the image alone does not ship the fix. Check both:
```bash
gcloud storage cat gs://ocr-lot-checker-config/manifest.json   # is the key listed under "packagings" or "archived"?
gcloud storage cat gs://ocr-lot-checker-config/packagings/<key>.yaml   # does it match the local file you just fixed?
```
If GCS has a stale/archived entry, fix it directly (matches what the wizard's
Archive/Unarchive/Deploy buttons do internally):
```bash
GCS_CONFIG_BUCKET=ocr-lot-checker-config GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json python -c "
from services import cloudrun_deployer
cloudrun_deployer.set_archived_in_gcs('<key>', False)   # un-archive
cloudrun_deployer.publish_packaging_to_gcs('<key>')     # push the corrected YAML
"
```
Then redeploy (or force a fresh revision with `--revision-suffix <x>` if the
image/config combo already exists — Cloud Run dedupes identical
image+config and won't restart, so a running instance never re-syncs).

**Model sync only runs at process startup** (`lifespan`) — a running
revision will not pick up new GCS models or config until it restarts.

## Known failure modes / common issues

Full post-mortems (root cause, mechanism, fix, validation) live in
`bug_fix.md` — summarized here for quick triage:

| Symptom | Likely cause | Reference |
|---|---|---|
| Wrong `lot_number` that looks like a QR-decoded garbage string, on `import_sticker` | cv2 QR fallback false-positive on a bad sticker crop | `bug_fix.md` Fix 1 |
| `lot_number` missing its first 1-2 letters (e.g. `H00005...` instead of `HO0005...`) on `back_label`/`grade_bag` | Vision misreads `O`→`0` in the always-alpha lot prefix | `bug_fix.md` Fix 2 |
| `exp_date`/`exp_sachet` null on `container_label`/`retail_sachet` compact dates | Vision misreads leading `0`→`Q` in `ddmmyyyy` | `bug_fix.md` Fix 3 |
| Detector retrain produces mislabeled classes | `data.yaml` on Drive drifted from current packaging configs (append-only class list, no GC) | `bug_fix.md` Fix 4, `scripts/cleanup_detector_dataset.py` |
| `lot_number` comes back looking like a date | `_is_valid_lot()`'s date-rejection regex didn't cover the separator actually used (e.g. `-` vs `/`), or a packaging's `lot_patterns` can't skip an interleaved date line | `bug_fix.md` Fix 5 |
| A packaging class 404s / returns `unconfigured_class` unexpectedly after a deploy or restart | GCS manifest has it archived, or GCS has a stale YAML shadowing the image — see "Config vs. code changes" above | — |
| `/training/full/start` times out | Dataset publish is synchronous and sequential inside the request — scales linearly with image count. Service runs with `--timeout=600` for this reason; if it's still not enough, move publish to background rather than raising the timeout again | `CLAUDE.md` § Commands |

## Local diagnostics against prod

Read-only checks are safe to run anytime:
```bash
curl https://ocr-lot-checker-459907489982.asia-southeast1.run.app/health
curl https://ocr-lot-checker-459907489982.asia-southeast1.run.app/api/packagings
gcloud storage cat gs://ocr-lot-checker-config/manifest.json
gcloud run services describe ocr-lot-checker --region asia-southeast1
```

Writes to prod GCS config (`set_archived_in_gcs`, `publish_packaging_to_gcs`,
`config_overrides.save_*`) and any `gcloud run deploy`/`update-traffic` are
**production mutations** — confirm scope with whoever asked before running
them, even though the commands themselves are simple to run locally with
`GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json`.
