# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OCR Lot Checker — FastAPI service that classifies a product packaging photo, locates the lot/expiry region, OCRs it (Google Cloud Vision), and cross-checks the extracted values against a Google Sheet. Deployed on Google Cloud Run; static wizard frontend on Netlify.

Production URL: `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`

Phase tracking lives in `PLAN.md`. Bug post-mortems live in `bug_fix.md`. ADRs in `docs/adr/`.

**Navigation:** a graphify knowledge graph of this repo lives in `graphify-out/` (`graph.json`, `GRAPH_REPORT.md`, `graph.html`). To locate a file/function or understand how subsystems connect, query it first instead of grepping blind: `/graphify query "<question>"` (or `& (Get-Content graphify-out/.graphify_python) -m graphify query "<question>"`). God nodes (core abstractions): `PackagingRegistry`, `GcsStore`, `DriveClient`, `find_lot`, `PipelineRunner`, `ImageClassifier`. Rebuild after large changes with `/graphify --update`. It is stale if source changed since the last build — trust the live code over the graph when they disagree. `--update` GOTCHAs: (1) `graphify-out/.graphify_python` is written UTF-8-with-BOM, so bash `$(cat .graphify_python)` → "No such file or directory" — call the interpreter by literal path (`C:/Users/copter/AppData/Roaming/uv/tools/graphifyy/Scripts/python.exe`) instead. (2) incremental detect sweeps in `web/samples/*.jpg` thumbnails + junk (`.review-test-temp/`, `--full-page`, scratchpad) — filter those out of the changed set before extraction or you pay for vision OCR on sample photos that add nothing to a code graph. (3) Gemini backend is used when `GEMINI_API_KEY` is set but its credits can be depleted (429) → fall back to subagent dispatch.

## Commands

```bash
# Local server (port 8080 to match Cloud Run)
python -m uvicorn main:app --reload --port 8080

# Run pipeline on a single image without starting the server
python test_image.py <image_path>
python test_image.py <image_path> <sheet_id> <sheet_gid>
# WARNING: test_image.py bypasses the registry — it never imports PackagingRegistry and calls
# ocr_engine.run() WITHOUT config= (single/container_label branches), so config-driven
# lot_patterns/product_aliases do NOT apply (find_lot falls back to generic _LOT_BY_CLASS).
# To test config changes on the REAL path, drive PipelineRunner with a loaded registry:
#   runner.run(img, reg.get(key))  — NOT test_image.py.

# Tests
pytest                              # all tests
pytest tests/test_integration.py    # integration tests (hits real /predict)
pytest tests/test_ocr.py -k lot     # single test by keyword

# Evaluation
python evaluate.py                  # classifier + detector accuracy per class
python eval_thresholds.py           # tune conf_threshold per class

# Wizard frontend (static, served from web/)
start "" "web\wizard.html"          # Windows

# Container build (matches the Cloud Run image)
docker build -t ocr-lot-checker .

# Deploy to Cloud Run. DO NOT use `gcloud run deploy --source .` — it pushes to the
# cloud-run-source-deploy repo and fails "Container import failed" (buildkit OCI manifest).
# Build to ocr-repo (prod's repo) then deploy --image:
gcloud builds submit --tag asia-southeast1-docker.pkg.dev/pj-ai-detect-lot-no/ocr-repo/ocr-lot-checker:<tag> .
gcloud run deploy ocr-lot-checker --image <that-image> --region asia-southeast1 --no-traffic --tag candidate \
  --set-env-vars GCS_CONFIG_BUCKET=ocr-lot-checker-config,... --set-secrets DRIVE_OAUTH_CLIENT_SECRET=drive-oauth-client-secret:latest,...
# verify candidate URL (candidate---ocr-lot-checker-...run.app), then go live:
gcloud run services update-traffic ocr-lot-checker --region asia-southeast1 --to-revisions <rev>=100
# rollback: --to-revisions <previous-rev>=100
#
# CLEANUP (mandatory once a one-off test tag has served its purpose): Cloud Run NEVER
# auto-deletes old revisions or tags, and every revision pins a full container image by
# digest in Artifact Registry (ocr-repo) — so ad-hoc `--tag <name>` test deploys left
# "just in case" accumulate forever. This happened for real: 108 dead revisions + 7 stray
# test tags (psb/trig/acc/accfix/candidate-clf/kan43/verify, none referenced anywhere in
# code/docs) piled up over ~2 months and ballooned ocr-repo to ~40GB (~$4/month) before a
# cleanup brought it to 1 image (~$0.03/month). Don't repeat this: as soon as a temporary
# tag is done being used for verification, remove BOTH the tag and its revision —
#   gcloud run services update-traffic ocr-lot-checker --region asia-southeast1 --remove-tags=<tag>
#   gcloud run revisions delete <revision> --region asia-southeast1 --quiet
# `candidate` is the one recurring exception — it's the standing staging tag reused by
# every deploy (see step above), never remove it. Full audit/recovery procedure if this
# recurs: docs/RUNBOOK.md § "Cleaning up test/candidate tags".
#
# CODE-ONLY redeploy (no config change): bare `gcloud run deploy --image <new> --no-traffic --tag candidate`
# INHERITS the live env/secrets/SA — you do NOT need to re-pass --set-env-vars/--set-secrets (which risks
# dropping a var). Read the live env first to confirm, verify the candidate /health, then update-traffic.
#
# Prod is GCS-first for models: with GCS_CONFIG_BUCKET set, the app downloads models from
# gs://.../models per manifest.json and IGNORES models baked into the image. To ship new
# models to prod you MUST publish to GCS (cloudrun_deployer.publish_packaging_to_gcs) — a
# rebuilt image alone won't change prod's models.
# Model sync runs ONLY at lifespan startup. A running revision won't pick up new GCS models
# until it restarts. Redeploying the SAME image+config does NOT create a new revision (Cloud
# Run dedupes) -> no restart -> no sync. Force one with `--revision-suffix <x>`, then
# `update-traffic --to-revisions <rev>=100` (traffic is pinned by name; a triggered/new
# revision gets 0% until you route to it). deploy_packaging's trigger now sets traffic=LATEST.
#
# REQUEST TIMEOUT: the service runs with --timeout=600 (raised from the prior 60s on
# 2026-06-26, rev 00117-jiy). 60s was too low because /training/full/start publishes the
# WHOLE dataset to Drive synchronously inside the request (api/packagings.py) — 50 images
# x ~2 Drive calls each blew past 60s and Cloud Run hard-killed the publish mid-upload.
# Like env/secrets, a bare code-only `deploy --image` carries the 600s forward (it's a
# template field inherited from the latest revision); verify with `describe ... timeoutSeconds`
# after deploy. If publishing ever grows past 600s, move the publish to a background task
# instead of raising the cap further.
```

Python 3.11. CPU-only torch is installed explicitly in the Dockerfile to keep the image small (~1.3 GB vs ~3.5 GB).

Frontend (`web/`) deploys to Netlify from `origin/main` (`netlify.toml` publish=web, no build command). Shipping wizard HTML / baked assets = push to `origin/main`; the Cloud Run image bakes `web/` too but serves only the API. Run scripts against PROD GCS locally with `GCS_CONFIG_BUCKET=ocr-lot-checker-config GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json python <script>` (the SA key has GCS access; this is how `backfill_image_count.py gcs` writes prod config).

## Architecture

The request flow through `main.py` `POST /predict`:

1. **Classifier** (`pipeline/classifier.py`) — EfficientNet-B0 fine-tune, returns `(class, confidence)`. Below `conf_threshold` → returns `low_confidence` status without running the pipeline.
2. **PackagingRegistry** (`pipeline/packaging_registry.py`) — loads per-packaging YAML from `config/packagings/*.yaml`. Each file defines `pipeline` (`detector_ocr` | `qr_scanner`), `lot_patterns` (compiled regex list), `fields_extracted`, `sheet_checks`, `sub_regions`, etc. Archived packagings use `*.yaml.archived` and are detected via `is_archived()`.
3. **PipelineRunner** (`pipeline/pipeline_runner.py`) — `pipeline == qr_scanner` → `QrScanner` only; else dispatches by `config.detection_mode`:
   - `cross_check` (`container_label`) → `_run_multi_region`: OCR box+sachet separately, compare → `lot_box`/`lot_sachet`/`exp_box`/`exp_sachet`.
   - `multi_field` → `_run_multi_field`: each `sub_regions` entry is a *group* of ≥1 field joined by `_` (e.g. `lot`, `lot_exp`) → YOLO class `{key}_{group}`; OCR each crop once, then run every member field's extractor on that text. Returns SAME shape as single (top-level `lot_number`/`exp_date`/`product_name`/`size`, no `lot_box`). LIVE EXAMPLE: `print_sticker_back` is `multi_field`. Of the 7 `detector_ocr` packagings: 5 are `single`, `container_label` is `cross_check`, `print_sticker_back` is `multi_field` (+ `import_sticker` is `qr_scanner`). NOTE: `30_sachet` (added 2026-06-30) is `single` but its key is NOT a detector class — the detector has no `30_sachet_*` boxes, so it OCRs the full image via heuristic fallback and relies on `lot_patterns` + AND-group `product_aliases`.
   - `single` (default) → single-region detector → preprocessor → OCR, all crops stacked vertically before one OCR.
4. **RegionDetector** (`pipeline/detector.py`) — YOLOv8n; `crop_all()` keeps every detected box whose YOLO class starts with `{predicted_key}_` (built from `image_class`, e.g. `back_label_` matches `back_label_lot`/`_name`/`_size`), sorts top→bottom, and returns all crops (single-region then stacks them). It does NOT read `detector_yolo_prefixes` — that field is consumed only by edit-draft prelabel (`active_learning`). Falls back to heuristics when YOLO returns nothing.
5. **Preprocessor** (`pipeline/preprocessor.py`) — **pass-through stub**: `run()` returns the crop bytes unchanged (Cloud Vision does its own enhancement; no grayscale/denoise/Otsu/deskew here). The only OpenCV image processing lives in `detector.py:_crop_container_label` — CLAHE → binary threshold → morphology → largest-contour pick — used to *locate* the white lot box for `container_label`, not to preprocess for OCR.
6. **OcrEngine** (`pipeline/ocr_engine.py`) — Google Cloud Vision. The `image_class` is passed so `utils/validators.find_lot` tries the config's `lot_patterns` first, then `_LOT_BY_CLASS[class]`, then generic patterns (`validators.py:285-300`).
7. **SheetChecker** (`utils/sheet_checker.py`) — reads the row matching the OCR'd lot in the user-supplied Google Sheet and returns `lot_match`/`exp_match`/`product_match`/`sachet_match`.
8. **build_verify_message** (`pipeline/message_builder.py`) — fills the message template (`config/message_templates/*.yaml`) keyed by `config.message_template_key`. Appends a value-driven `EXP: dd/mm/yyyy` line via `_format_exp_display` (ISO→display) when OCR read an expiry.

Models are loaded once in the FastAPI `lifespan` startup. `services/model_registry.sync()` is **GCS-first** (as of 2026-06-22): if `GCS_CONFIG_BUCKET` is set it downloads sha256-verified models from `gs://ocr-lot-checker-config/models/` per `manifest.json`; else Drive (`DRIVE_MANIFEST_FILE_ID`); else local `models/*.pt`. `PackagingRegistry` overlays per-key YAMLs from the same bucket (`packagings/<key>.yaml`, plus `archived` tombstones) on top of the baked-in image config, and `config_overrides` (conf_threshold) also lives in GCS now. Empty `GCS_CONFIG_BUCKET` → image/local fallback, so zero-env-var prod still works. Wizard Deploy publishes to GCS via `cloudrun_deployer.publish_packaging_to_gcs` (manifest written last = commit point) so changes survive Cloud Run revisions. See `services/gcs_store.py`. GOTCHA: GCS `packagings/` is *designed* to hold only wizard-created/edited classes (e.g. `print_sticker_back`), with originally shipped classes living solely in the image's `config/packagings/*.yaml` — so the GCS overlay is supposed to only override keys that exist there, and editing a shipped class's prod config is supposed to require an image redeploy, not a GCS write. **In practice this invariant can be violated**: a shipped class (e.g. `30_sachet`, added via a plain git commit, never through the wizard) was found with a stray `packagings/30_sachet.yaml` in GCS on 2026-07-06 — almost certainly written by an earlier ad-hoc hotfix — which silently shadowed the image's YAML entirely, so rebuilding+redeploying the image had **zero** effect until that stray file was deleted (see `bug_fix.md` Fix 6, `docs/RUNBOOK.md` "Config vs. code changes"). Do not assume "shipped class → no GCS overlay" — always check `gcloud storage cat gs://ocr-lot-checker-config/packagings/<key>.yaml` before concluding a config fix isn't reaching prod.

## Wizard API (`api/packagings.py`)

`/api/packagings/*` is a separate router that backs `web/wizard.html`. It lets the user add a new packaging class without code changes: upload sample images, annotate bboxes, generate regex from labelled lot strings, save a draft YAML in `data/drafts/`, then promote to `config/packagings/`. Training is single-stage: drafts label ≥30 images then run Full Training (ADR 0005-single-stage-training). Edit-drafts can `POST /{key}/training/prelabel` to auto-fill bboxes on newly-added images using the deployed detector (filtered by the parent's `detector_yolo_prefixes`). Prelabel logic lives in `services/active_learning.prelabel_remaining(key, model_path, class_prefixes=...)` — runs server-side on the active `MODEL_DETECTOR_PATH`, no Colab. ADR 0002 explains "edit active packaging via clone". ADR 0003 explains "backend writes the reference dataset directly" (Full Training publishes images/labels to Drive; data.yaml is the commit point).

As of 2026-06-15 (`docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md`), Full Training is **manual**: `/training/full/start` publishes the dataset then returns a link to ONE static combined detector+classifier Colab notebook in Drive (`COMBINED_NOTEBOOK_FILE_ID` in `api/packagings.py`, built by `scripts/build_full_training_notebook.py`). `services/notebook_generator.py` was removed. Dataset layout the notebook reads: classifier = `data classify check lot/images/<class>/`; detector `data.yaml` stores absolute `/content/drive/MyDrive/...` paths that `dataset_publisher._relativize` normalizes before folder resolution. GOTCHA: `merge_class_names` is **append-only with no GC** (numeric label ids forbid reordering), so abandoned/test-draft/renamed classes accumulate stale `names` entries forever and TEST_MODE publishes can leak junk classes into the prod dataset — `data.yaml` drifts from the live config + deployed `detector.pt`. This bit once (see `bug_fix.md` Fix 4 / spec `2026-06-26-detector-dataset-datayaml-cleanup`): the repair tool is `scripts/cleanup_detector_dataset.py` (dry-run default, `--execute`, `--verify`; remaps label ids, trashes no-config classes, rewrites `data.yaml` last). Before any detector retrain, sanity-check `data.yaml` `names`/`nc` against `models/detector.pt` names and `config/packagings/*`.

PERF (2026-06-26): `/training/full/start` publishes **synchronously inside the request** and `dataset_publisher._upload_items` loops images **sequentially** — there is no concurrency, so publish time scales linearly with image count (this is why the request timeout is 600s, not 60s — see Deploy notes). Two round-trip reductions are in place: (1) `DriveClient.upload_file` uses a single-request (simple) upload for files ≤5MB (`_SIMPLE_UPLOAD_MAX_BYTES`) and only resumable above that — resumable adds a session-init round-trip; (2) the classifier copy of each image uses Drive **server-side `DriveClient.copy_file`** of the just-uploaded detector image instead of re-uploading the bytes (falls back to a fresh upload only when the detector copy already existed/was skipped). If a class ever has enough images to approach 600s, the right fix is a bounded thread pool with **per-thread Drive clients** (googleapiclient/httplib2 is not thread-safe) or moving publish to a background task — NOT raising the timeout again.

As of 2026-06-22 (`docs/superpowers/specs/2026-06-22-train-done-sync-deploy-design.md`): after Colab finishes, the `training_full` screen shows a **"ดึง model" button** → `POST /{key}/training/full/sync` resolves all 3 Drive ids up front then downloads `full_detector.pt` + `classifier.pt` + `eval.json` (fixed Drive names; notebook writes `eval.json` LAST = done-signal) into `data/drafts/{key}/models/`, cleans partials on failure, sets status `trained` → existing Eval/Deploy screen. `/deploy` then promotes **both** detector + classifier (`promote_draft_model` returns `dict[str,Path]`) with backup+rollback on BOTH fresh and edit-draft paths (fresh path previously promoted nothing). `/training/full/done` stays 400 in prod (TEST_MODE-only sim).

Draft `status` is the wizard's resume point (`continueDraft` stepMap in `web/wizard.html`): `draft` = no images yet → step 2; `uploading` = has images → step 3 (first `save_image` bumps draft→uploading); `configured`/`training_full`/`trained` → step 4/5. An edit-draft (`{key}__edit` from clone) starts at `draft` with ZERO images of its own — parent images are display-only and full training requires ≥30 labeled images in the draft itself (`api/packagings.py` training/full/start gate).

## Adding or Editing a Packaging Class

Prefer editing the YAML in `config/packagings/<key>.yaml` over hardcoding. Required keys: `key`, `pipeline`, `lot_patterns`, `fields_extracted`, `sheet_checks`, `model_classifier_label`, `detector_yolo_prefixes`. Set `sub_regions: [box, sachet]` for multi-crop. Use `conf_threshold` (per-class override of the 0.6 default) to gate low-confidence classifier predictions. Set `gate_on_lot: false` if the packaging legitimately has no lot number (then `lot_short_fallback` may help).

The wizard card's `accuracy` is the classifier's per-class value — a MANUAL field in the YAML, no auto-pipeline from training. Fresh deploys get `accuracy: null` → card shows "—". The real held-out figure lives in Drive `data classify check lot/confusion_matrix.png` (an image, not machine-readable); the `eval.json` synced by `/training/full/sync` is the DETECTOR's (mAP/precision/recall), not classifier accuracy.

`image_count` is a YAML snapshot field (like `accuracy`), read by `GET /api/packagings` + `GET /{key}`; it falls back to `_count_active_images` (Drive on prod) only when absent. `/deploy` snapshots the dataset count into the YAML. Backfill with `scripts/backfill_image_count.py` (default = append to local YAMLs from `images/<key>/`; `gcs` mode writes into the GCS YAMLs). This removed a per-request Drive lookup (3 calls/key × N classes) that stalled the dashboard on cold Cloud Run instances — do NOT reintroduce a live count on the list/detail path.

`conf_threshold` is also user-tunable at runtime (no retrain): `PUT /api/packagings/{key}/conf` (0.50–0.95) persists an override to Drive/`data/config_overrides.json` which `PackagingRegistry(overrides=...)` merges over the YAML — the YAML value is only a default once an override exists. See ADR 0004. Registry reloads must go through `main.reload_registry()` (never `PackagingRegistry()` directly) or overrides are silently dropped.

Like `conf_threshold`, `product_aliases` is runtime-editable (no retrain): `PUT`/`DELETE /api/packagings/{key}/product-aliases` → `config_overrides.save_product_aliases`/`delete_product_aliases`, applied by `PackagingRegistry._merged_product_aliases`. DELETE reverts to the YAML/hardcoded default. Wizard drawer editor in `web/wizard.html` (shows for any active class with `product` in `fields_extracted`). GOTCHA (bit us 2026-06-30): `_merged_product_aliases` is **replace, not merge** — if a `product_aliases` override exists it FULLY SHADOWS the YAML aliases (the YAML value is ignored). A stale drawer-typed override silently overrode a corrected YAML in prod (symptom: live `/predict` product wrong while local was right). Diagnose by comparing `GET /api/packagings/{key}` `product_aliases` against the YAML; if they differ an override is shadowing. Fix: `config_overrides.delete_product_aliases(key)` (prod env: `GCS_CONFIG_BUCKET=ocr-lot-checker-config GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json`) then redeploy a fresh revision so warm instances reload. The drawer can only write FLAT string keywords (no AND groups — see below).

`config_overrides` storage splits by env: prod (`GCS_CONFIG_BUCKET` set) → GCS `config_overrides.json`; local (unset) → `DRIVE_CONFIG_OVERRIDES_FILE_ID` Drive file, else `data/config_overrides.json`. So a local `PUT /conf` or `/product-aliases` writes the Drive dev file — it does NOT affect prod (prod reads GCS).

When adding a brand-new class you also need: classifier label in the training dataset, YOLO class name with one of the `detector_yolo_prefixes`, retrained `.pt` files, updated message template if the existing four don't fit.

Product-name OCR matching is config-driven: set `product_aliases` (list of `{canonical, keywords}`) in the YAML — `find_product_name(text, aliases)` returns the `canonical` whose keyword appears in the OCR text (priority = list order; keywords matched literally via `re.escape` + word-boundary lookaround, so noisy wide crops still match). Each `keywords` entry is either a **string** (single token, OR — match any one) or a **nested list = AND** (all tokens must be present), e.g. `keywords: [["Excellent", "30 Sachets"]]`. AND-groups disambiguate same-family products via distinct tokens (live example `30_sachet`: `[["Classic Rich","500 g"]]` vs `[["Classic Rich","B Grade"]]`). Schema is `ProductAlias.keywords: list[str | list[str]]` (`api/schemas.py`) — a `list[str]`-only schema 500s `GET /api/packagings/{key}` for any class using AND-groups. Matching is case-insensitive (both text and keyword are lowercased). `canonical` MUST equal the sheet's `Product Name` (`sheet_checker` compares exact-lowercased on the lot-matched row). Empty/missing `product_aliases` → falls back to the hardcoded 9-keyword tea list in `utils/validators.py` (back_label/grade_bag stay on this). New field threads through: validators → ocr_engine/pipeline_runner → `PackagingConfig` → `schemas.PackagingConfigUpdate` → `packaging_store.clone_from_active` → `cloudrun_deployer.write_packaging_yaml`.

## Environment

Local dev reads `.env` (see `.env.example`). Key vars:

- `DRIVE_MANIFEST_FILE_ID` — empty locally (uses `models/*.pt`). NOTE: prod Cloud Run DOES set env vars (verify with `gcloud run services describe ocr-lot-checker --region asia-southeast1`): `GCS_CONFIG_BUCKET=ocr-lot-checker-config`, `DRIVE_DETECTOR_DATASET_FOLDER_ID`, `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`, `DRIVE_OAUTH_CLIENT_ID`, + secrets `DRIVE_OAUTH_CLIENT_SECRET`/`DRIVE_OAUTH_REFRESH_TOKEN`; SA `ocr-lot-checker-sa`, 2Gi/2cpu, scale 0–10. A candidate deploy MUST replicate these (read the live service first; don't trust this list blindly).
- `MODEL_CLASSIFIER_PATH` / `MODEL_DETECTOR_PATH` — local model paths.
- `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service account JSON locally; Cloud Run uses the runtime service account (`ocr-lot-checker-sa`) via ADC, no JSON key.
- Local Drive access: ADC is a user account with only `drive.file` scope — it can read/write files *created by this app* but gets 404 on files created manually in Drive web UI. Full `drive` scope is blocked by Google for user OAuth. To act as the production identity, use the SA key `gcp-key.json` (gitignored) via `GOOGLE_APPLICATION_CREDENTIALS`.
- `CONFIDENCE_THRESHOLD`, `LOG_LEVEL`.
- `MODEL_CACHE_DIR` — download cache for Drive-fetched weights (default `/tmp/models`). `OCR_LANG` — Vision language hints (default `th+en`). `REFERENCE_DETECTOR_PATH` / `REFERENCE_CLASSIFIER_PATH` — Drive My-Drive-relative dataset paths the training notebook reads (default `data check lot` / `data classify check lot`).
- `DRIVE_DETECTOR_DATASET_FOLDER_ID` / `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` — Drive folder ids of the reference dataset (`data check lot` / `data classify check lot`), shared with the SA. Required for the wizard's Full Training (dataset publish). See ADR 0003.
- `DRIVE_CONFIG_OVERRIDES_FILE_ID` — Drive file id of `config_overrides.json` (runtime tuning overrides, currently `conf_threshold` per packaging). Empty locally → falls back to `data/config_overrides.json`. See ADR 0004.
- `DRIVE_OAUTH_CLIENT_ID` / `DRIVE_OAUTH_CLIENT_SECRET` / `DRIVE_OAUTH_REFRESH_TOKEN` — OAuth user creds for Drive *writes* (dataset publish). **Service accounts have ZERO Drive storage quota** — SA-owned uploads 403 with `storageQuotaExceeded`, so `DriveClient` must act as a real Workspace user. When all three are set, `services/drive_client.py` uses them; otherwise it falls back to SA/ADC (read-only paths unaffected). Mint once via `python scripts/generate_drive_token.py` (Internal OAuth client → full `drive` scope, no Google verification, token never expires). Only `DriveClient` switches identity — `sheet_checker.py`/`cloudrun_deployer.py` keep the SA. The OAuth user must have Editor on the dataset folders. See ADR 0006-drive-dataset-write-via-oauth-user. NOTE: editing `.env` needs a dev-server restart (`--reload` re-reads code, not env).

## Wizard (`web/wizard.html`)

`web/wizard.html` is the ONLY wizard file. The TEST_MODE harness (`run_test_wizard.ps1`) and the portable bundle (`build_portable_bundle.ps1`, `dist/portable-bundle/`) were **retired 2026-06-24** — there are no longer any generated copies (`test wizzard/`, `dist/`) to keep in sync.
- **Google login gate** (client-side): a GIS gate restricts access to `@tdfb.co` (`ALLOWED_HD='tdfb.co'`). `GOOGLE_CLIENT_ID` reuses the Drive OAuth client id (`DRIVE_OAUTH_CLIENT_ID`, a public `*.apps.googleusercontent.com` id hardcoded in the page); `signOut()` clears it. The gate is purely client-side — no backend enforcement, so it gates the UI, not the API. `web/_redirects` serves `wizard.html` at `/` for manual folder deploys.
- **Targets production by default**: `API_BASE` returns the prod Cloud Run URL unless `window.API_BASE_OVERRIDE` is set (it wins if injected before the script runs). Opening the file anywhere (double-click / Netlify) hits prod; for local dev against `:8080` set `window.API_BASE_OVERRIDE` manually.
- **Step-2 image upload is chunked client-side** (2026-06-26): `handleFileSelect` splits the selection into batches of ≤20MB / ≤8 files (`chunkFilesForUpload` → `uploadBatch`) and POSTs them **sequentially**, advancing the progress bar per batch. This is REQUIRED — a single multipart POST of ~50 images exceeds **Cloud Run's 32MB request-body limit** and silently stalls at 15%. The `/{key}/images` endpoint returns cumulative `total_images`, so batching needs no backend change. Per-batch failures are isolated (one bad chunk doesn't sink the upload); only an all-failed upload throws. Don't revert to a single-POST upload.
- Card images (added 2026-06-24): count comes from Drive when local `images/<key>/` is empty (prod ships no `images/`) — `services/drive_samples.class_images` resolves `<DRIVE_CLASSIFIER_DATASET_FOLDER_ID>/images/<key>/` (folder named by `key`), TTL-cached 600s, never raises; `_count_active_images` is local-FS-first. Thumbnails are baked `web/samples/<key>/*.jpg` (regen `scripts/build_card_samples.py` → paste `BAKED_SAMPLES` into wizard.html) with a Drive-download fallback for unbaked keys. GOTCHA: `loadCardImages` fires when the key is baked OR `image_count > 0` — NOT count-only, so baked thumbnails show even when prod count is 0.
- Drawer "Sample images" are baked too (separate from card thumbs): `BAKED_DRAWER_SAMPLES` const + `web/samples/<key>/*.crop*.jpg`, built by `scripts/build_drawer_samples.py` (runs the real detector offline; mirrors `get_samples` — qr_scanner → no crops, labels from `sub_regions`, bbox scaled to the downscaled original). `openDrawer` skips the `/samples` fetch (Drive download + detector inference) when a key is baked. Re-run the script + redeploy `web/` after retraining the detector or changing dataset images.
- Active-class images are read from local FS in FOUR paths — ALL fall back to Drive when local is empty + `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` is set (prod): `_count_active_images` (count), `GET /{key}/images` (list), `GET /{key}/images/{f}` (serve, via `_drive_sample_path` temp-cache download), and drawer `GET /{key}/samples` + crop overlays (`_list_sample_files` + `_resolve_image_path` → `_drive_sample_path` → detector). Add the same fallback to any NEW active-class image read or it returns 0/empty on prod.
- In-app `TEST_MODE` flags still exist but are **dormant** (nothing drives them now): `/health` returns `test_mode`; `/training/full/done` is a TEST_MODE-only sim (stays 400 in prod).
- `OCR_CONFIG_DIR` / `DRAFT_DIR`: don't hardcode `Path("config/packagings")` or the drafts dir in `api/` — use `api/packagings._packaging_yaml_dir()` / registry `_config_dir()` so the configurable dirs are honoured.
- PowerShell 5.1 gotcha: `Get-Content -Raw` reads UTF-8 as ANSI → double-encodes Thai/symbols on rewrite. Always `Get-Content … -Raw -Encoding UTF8` when reading any Thai file (e.g. `web/wizard.html`) in a `.ps1`.
- Playwright wizard E2E: Thai string literals injected via `browser_evaluate` get mangled in transit — match elements by `onclick` attr (e.g. `archivePackaging`), not Thai text. `openDrawer(key, pkgSummary)` needs the summary arg — click the real card element instead of calling it with key only.

## Conventions specific to this repo

- `print()` is forbidden — use the module `logger`. Hooks may warn on `print` in edits.
- `pytest` is NOT on PATH — run `python -m pytest`.
- Source files contain Thai; read them with `encoding='utf-8'` (Windows default cp1252 raises `UnicodeDecodeError` on Thai bytes). Standalone scripts under `scripts/` need `sys.path.insert(0, repo_root)` to import `services`, and must run from the repo root.
- `load_dotenv()` (no arg) in a standalone script resolves `.env` from the SCRIPT'S directory, not cwd — a script outside the repo root (e.g. scratchpad) silently skips `.env`, so `DriveClient` falls back to ADC `drive.file` scope and `list_folder`/Drive reads return 0 with the SAME email (no error). Use `load_dotenv(os.path.join(repo, ".env"))` explicitly.
- `gsutil` is broken in this env (`python3.13: command not found`). Use `gcloud storage cat|cp|ls` instead.
- Printing Thai/Unicode from `python -c` to the PowerShell console raises `UnicodeEncodeError` (cp1252 charmap) — write output to a file with `encoding='utf-8'`, or set `$env:PYTHONIOENCODING='utf-8'` before the call.
- PowerShell `Out-File`/`>` adds a UTF-8 BOM that Python's `json.loads` rejects (`Unexpected UTF-8 BOM`). When a `.ps1`/`python -c` step produces JSON another tool reads, write it from Python with `Path(...).write_text(..., encoding='utf-8')` rather than piping through `Out-File`.
- Known pre-existing failure: `tests/test_classifier.py` has 3 setup errors — its fixture builds `efficientnet_b0` but `pipeline/classifier.py` now uses `efficientnet_v2_s`. Not caused by your changes; fix the fixture if touching classifier tests.
- Wizard E2E: Playwright blocks `file://` — serve with `python -m http.server 8090 --directory web` (hostname localhost makes API_BASE resolve to `http://localhost:8080`). Port 8080 is usually already taken by the dev server running with `--reload` (which auto-picks up code edits — check `/openapi.json` before assuming stale code).
- For manual wizard use you only need the backend on :8080 — `wizard.html` API_BASE resolves both `file://` and localhost to `http://localhost:8080`, so double-clicking `web/wizard.html` works. The `http.server :8090` is only needed for Playwright (which blocks `file://`).
- Preview a wizard step with NO backend: serve `web/` and `page.evaluate('startWizard()')` (or `goStep(n)`) via Playwright. `loadDashboard()` fetch errors but the step-1 form/builder render standalone — enough for visual/logic checks.
- Wizard step 4 does NOT prefill config from a saved/edit draft (only `prefillFieldsFromSubRegions` sets field checkboxes for `multi_field`). Re-entering step 4 + clicking Next re-POSTs form state, overwriting saved `lot_patterns`/`fields_extracted`/`sheet_checks`/`message_template_key`/`product_aliases` with defaults/empties — editing config via step-4 resume is lossy.
- `utils/validators.py` (find_lot / find_expiry / find_product_name / find_size) is tested in `tests/test_ocr.py` — there is no `test_validators.py`.
- Dark-theme `<select>` gotcha (`wizard.html`): native option popups render WHITE on Chromium/Windows unless you style `select option{background:var(--s1)}` AND give the select a real (non-`transparent`) `background-color` — Chromium derives popup color from it, `transparent` falls back to white. The open popup is OS-layer, so Playwright page screenshots never capture it (verify popup color by code, not screenshot).
- Date normalisation: validators return ISO `YYYY-MM-DD`; 2-digit years are accepted; separator may be `/` or `-` (e.g. `30-06-2027`, since 2026-06-30 — `_DATE_SEP`). `DATE_FALLBACK` is guarded with `(?<![\d-])`/`(?![\d-])` so dash-joined long numbers (FDA reg `11-2-02167-6-0024`, phone `02-114-3715`) are NOT misread as dates.
- The `import_sticker` class is QR-only — never route it through detector/OCR. The QR scanner uses zxing-cpp first and a sticker-crop + cv2 fallback; the cv2 fallback has caused false positives historically (see `bug_fix.md`), so order and gating matter.
- Pipeline routing keys off `config.detection_mode` (`single`|`cross_check`|`multi_field`) in 3 places that must stay in sync: `pipeline_runner.py` run(), `sheet_checker.py` is_container (`== cross_check`), `main.py` lot_for_message (`== cross_check`). `sub_regions` no longer routes — it names the crops: cross_check=`[box,sachet]`; multi_field=groups of ≥1 field joined by `_` (e.g. `[lot_exp, product_size]`) → class `{key}_{group}`. `utils/field_groups.py` (`parse_group`/`canonicalize_group`/`validate_groups`) canonicalizes (order lot,exp,product,size), enforces partition + lot-present; single-token entries = 1:1 (backward compatible). Only `cross_check` returns `lot_box`/`lot_sachet`; `single`+`multi_field` use top-level `lot_number`. See CONTEXT.md + `docs/superpowers/specs/2026-06-14-multi-field-detection-design.md` + `docs/superpowers/specs/2026-06-15-multi-field-grouped-crops-design.md`.
- `multi_field` deferred gaps: `_run_multi_field` has NO degraded-crop OCR retry (unlike `_run_multi_region`); `active_learning` prelabel hardcodes `label="prelabel"`, so retraining a multi_field edit-draft needs manual per-field box tagging.
- multi_field `fields_extracted` lists individual field tokens, but `sub_regions` entries may be composite groups (`lot_exp`) — expand with `parse_group` before any `field ⊆ sub_regions` check (the `training/full/start` gate in `api/packagings.py` did this wrong and blocked Full Training for grouped packagings).
- `PackagingResponse` (`api/schemas.py`) must declare EVERY field the wizard reads — FastAPI `response_model` silently strips undeclared keys. The annotator's label chips were invisible because GET `/api/packagings/{key}` didn't return `sub_regions`/`detection_mode` (added to the schema to fix it).
- Model `.pt.bak-*` files are training rollback snapshots — leave them alone unless explicitly cleaning up.
- `detector_yolo_prefixes` must list EVERY YOLO class the packaging owns in `detector.pt` (e.g. `back_label` → `[back_label_lot, back_label_name, back_label_size]`), not just the lot class — prelabel filters by `name.startswith(prefix)` (`active_learning.py`) and silently drops unlisted boxes (name/size/product). Inspect the real classes with `python -c "from ultralytics import YOLO; print(sorted(YOLO('models/detector.pt').names.values()))"`. Note `capsule_box`'s class is literally `capsule_box` (no `_lot` suffix), so its prefix is `[capsule_box]`. This field does NOT affect inference (see Architecture §4).
- Wizard API tests (`tests/test_api_packagings.py`): the `client` fixture is module-scoped and reloads `packaging_store` under a temp `DRAFT_DIR`; mock state with `monkeypatch.setattr` on `packaging_store.*` (e.g. `get_draft`, `list_annotation_status`), `main.registry`, and service funcs (`services.X.func`) rather than creating real drafts/models.
- The working tree often holds several in-progress features at once. Before `git add <file>`, run `git diff <file>` — staging a whole file can bundle unrelated pre-existing changes into your commit (`git add -p` is unavailable here; stage deliberately). To isolate one feature's hunks from entangled pre-existing edits in the SAME file: write a patch with the Write tool (handles Thai/UTF-8) and `git apply --cached --recount <patch>` — worktree keeps everything, index gets only the feature hunks. For a single line tangled with pre-existing edits: backup the file → `git checkout HEAD -- <file>` → re-apply that one line → `git add`+commit → restore the backup (avoids CRLF patch-apply failures).
- Check which Drive identity is active (debug 403/404 auth issues): `python -c "from dotenv import load_dotenv; load_dotenv(); from services.drive_client import DriveClient; print(DriveClient()._svc.about().get(fields='user,storageQuota').execute())"` — shows the authenticated email + quota. SA → no `user`/quota (and writes 403 with `storageQuotaExceeded`); OAuth user → real email + GB. See ADR 0006-drive-dataset-write-via-oauth-user.
- Re-minting the Drive OAuth token (`scripts/generate_drive_token.py`) listens on port **8765** (8080 is taken by the dev server); a Web-type OAuth client must list `http://localhost:8765/` in its Authorized redirect URIs *exactly* (trailing slash matters) or consent fails with `redirect_uri_mismatch`.
- `test_image.py` is NOT production-faithful: it skips the registry and omits `config=` on OcrEngine, so per-packaging `lot_patterns`/`product_aliases` are ignored (lot may return null/wrong). Production routes through `PipelineRunner.run(img, config)` (`pipeline_runner.py:51` passes `config=config`). Verify config edits via PipelineRunner, not `test_image.py`.
- `.gcloudignore` (NOT `.dockerignore`) governs `gcloud builds submit`/`--source` uploads. It KEEPS `models/` (so weights bake into the image) but must exclude `dist/`, `data/` (local drafts/models), and `models/*.bak-*` (~300MB). Verify the upload set with `gcloud meta list-files-for-upload` (~107 files expected, not 40k).
