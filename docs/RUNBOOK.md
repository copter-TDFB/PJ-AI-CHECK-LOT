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

### Cleaning up test/candidate tags

**Do this every time**, right after a `--tag <name>` test deploy has served its
purpose (verification done, ticket closed, whatever the tag was for) — don't
leave it "just in case":

```bash
gcloud run services update-traffic ocr-lot-checker --region asia-southeast1 --remove-tags=<tag>
gcloud run revisions delete <revision> --region asia-southeast1 --quiet
```

Never remove `candidate` — it's the one tag every deploy reuses (see step 2
above), not a one-off.

**Why this matters (incident, 2026-08-07):** Cloud Run does not auto-delete
old revisions or tags, ever, and each revision pins a full container image by
digest in `ocr-repo`. Seven one-off test tags (`psb`, `trig`, `acc`,
`accfix`, `candidate-clf`, `kan43`, `verify` — none referenced anywhere in
code, docs, or tests) plus 108 further untagged revisions accumulated over
~2 months of normal development, ballooning `ocr-repo` to ~40GB (~$4/month
in Artifact Registry storage) before anyone noticed. Cleanup brought it down
to a single live image (~$0.03/month).

**If this recurs, the recovery procedure is:**
1. `gcloud run services describe ocr-lot-checker --region asia-southeast1 --format="value(status.traffic)"`
   — note every `revisionName`+`tag` pair. These revisions (and only these)
   must survive.
2. `gcloud run revisions list --service=ocr-lot-checker --region=asia-southeast1 --format="value(metadata.name)"`
   — delete every revision NOT in the protected set from step 1
   (`gcloud run revisions delete <name> --region=asia-southeast1 --quiet`).
3. Re-list the protected revisions' images
   (`--format="value(spec.containers[0].image)"`) to get the digest set that
   must survive.
4. `gcloud artifacts docker images list asia-southeast1-docker.pkg.dev/pj-ai-detect-lot-no/ocr-repo --format="value(DIGEST)"`
   — delete every digest NOT in the protected set
   (`gcloud artifacts docker images delete <image>@<digest> --delete-tags --quiet`).
5. Verify before AND after: `curl .../health` returns 200, and
   `status.traffic` still matches step 1 exactly.

**Gotcha:** `gcloud artifacts repositories describe/list` size (`sizeBytes`)
lags real deletions by a while — it's a periodically-refreshed aggregate, not
live. Don't judge whether cleanup worked by watching that number move; it
can take a day or more to catch up even after the underlying blobs are
already gone. To verify the real freed space immediately, compare layer
digests across the surviving images' manifests instead:
`curl -H "Authorization: Bearer $(gcloud auth print-access-token)" https://asia-southeast1-docker.pkg.dev/v2/pj-ai-detect-lot-no/ocr-repo/ocr-lot-checker/manifests/<digest>`
(`Accept: application/vnd.docker.distribution.manifest.v2+json`) and sum the
`layers[].size` for blobs not shared with a surviving image.

## Config vs. code changes — where a fix actually needs to go

This is the single most common source of "I fixed it but prod is still
broken" confusion in this service:

| You changed... | Where it's read from in prod | What ships it |
|---|---|---|
| A **wizard-created/edited** packaging class (e.g. `print_sticker_back`) | GCS `packagings/<key>.yaml`, overlaid on top of the image | The wizard's "Deploy" button, or `cloudrun_deployer.publish_packaging_to_gcs(key)` directly |
| An **originally shipped** packaging class (e.g. `back_label`, `30_sachet`) | The image's baked `config/packagings/<key>.yaml` — **unless** it was ever published to GCS too (see below) | Rebuild the image + redeploy |
| `conf_threshold` or `product_aliases` on any class | `config_overrides.json` (GCS or Drive) — a **runtime override**, separate from the YAML | `PUT /api/packagings/{key}/conf` or `/product-aliases` — no redeploy needed |
| Model weights (`.pt`) | GCS `models/` per `manifest.json`, when `GCS_CONFIG_BUCKET` is set | `cloudrun_deployer.publish_packaging_to_gcs` (uploads changed models automatically) |

**Gotcha (hit in production 2026-07-02, recurred 2026-07-06 — `bug_fix.md`
Fix 6):** a packaging can be BOTH baked into the image AND have a stale copy
in GCS `packagings/<key>.yaml` if it was ever published there (e.g. via the
wizard's Deploy button, or an ad-hoc hotfix script — even for a class that was
never created through the wizard, like `30_sachet`). If that GCS copy exists,
the registry overlay uses it **instead of** the freshly rebuilt image's
version — so rebuilding the image alone does not ship the fix. Don't assume a
class is exempt just because it was originally added via git commit; check
both:
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

**Gotcha (hit in production 2026-07-03):** the `PUT .../product-aliases` row
above is only true for flat/OR keywords. Some classes (e.g. `30_sachet`) use
AND-group keywords (`[["Medium Rich", "B Grade"]]` — all terms required) to
disambiguate near-identical products. The endpoint's validation
(`api/packagings.py` `update_product_aliases`) calls `.strip()` on every
keyword assuming it's a string, so it 500s on a nested-list entry — and the
wizard's product-alias drawer can only *write* flat OR lists to begin with
(`.split(',')` on save), silently flattening any existing AND-groups for that
class the next time you save an unrelated edit through it. If a class needs
AND-groups, skip both the wizard and the endpoint and write the override
directly:
```bash
GCS_CONFIG_BUCKET=ocr-lot-checker-config GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json python -c "
from services import config_overrides
config_overrides.save_product_aliases('<key>', [...])   # keywords may be nested lists for AND
"
```
No redeploy needed, but this bypasses the endpoint's own `main.reload_registry()`
call — a currently-warm instance keeps serving the OLD in-memory registry
until it restarts (same caveat as model sync below; a Cloud Run idle
scale-down naturally triggers this, or force one with `--revision-suffix <x>`
+ `update-traffic`). Verify with `GET /api/packagings/{key}` before assuming
it's live. Tracked in Jira KAN-31 (wizard UI + endpoint fix, not yet done).

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
| A config fix (e.g. `product_aliases`) is merged, image rebuilt and redeployed, but production behavior doesn't change at all | Both a `config_overrides.json` override AND a stray `packagings/<key>.yaml` full overlay were shadowing the fix — the overlay wins even for a class that was never wizard-created | `bug_fix.md` Fix 6 |
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
