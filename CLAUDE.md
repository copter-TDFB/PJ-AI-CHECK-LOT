# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OCR Lot Checker — FastAPI service that classifies a product packaging photo, locates the lot/expiry region, OCRs it (Google Cloud Vision), and cross-checks the extracted values against a Google Sheet. Deployed on Google Cloud Run; static wizard frontend on Netlify.

Production URL: `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`

Phase tracking lives in `PLAN.md`. Bug post-mortems live in `bug_fix.md`. ADRs in `docs/adr/`.

**Navigation:** a graphify knowledge graph of this repo lives in `graphify-out/` (`graph.json`, `GRAPH_REPORT.md`, `graph.html`). To locate a file/function or understand how subsystems connect, query it first instead of grepping blind: `/graphify query "<question>"` (or `& (Get-Content graphify-out/.graphify_python) -m graphify query "<question>"`). God nodes (core abstractions): `DriveClient`, `PackagingRegistry`, `PipelineRunner`, `find_lot`. Rebuild after large changes with `/graphify --update`. It is stale if source changed since the last build — trust the live code over the graph when they disagree.

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
```

Python 3.11. CPU-only torch is installed explicitly in the Dockerfile to keep the image small (~1.3 GB vs ~3.5 GB).

## Architecture

The request flow through `main.py` `POST /predict`:

1. **Classifier** (`pipeline/classifier.py`) — EfficientNet-B0 fine-tune, returns `(class, confidence)`. Below `conf_threshold` → returns `low_confidence` status without running the pipeline.
2. **PackagingRegistry** (`pipeline/packaging_registry.py`) — loads per-packaging YAML from `config/packagings/*.yaml`. Each file defines `pipeline` (`detector_ocr` | `qr_scanner`), `lot_patterns` (compiled regex list), `fields_extracted`, `sheet_checks`, `sub_regions`, etc. Archived packagings use `*.yaml.archived` and are detected via `is_archived()`.
3. **PipelineRunner** (`pipeline/pipeline_runner.py`) — `pipeline == qr_scanner` → `QrScanner` only; else dispatches by `config.detection_mode`:
   - `cross_check` (`container_label`) → `_run_multi_region`: OCR box+sachet separately, compare → `lot_box`/`lot_sachet`/`exp_box`/`exp_sachet`.
   - `multi_field` → `_run_multi_field`: each `sub_regions` entry is a *group* of ≥1 field joined by `_` (e.g. `lot`, `lot_exp`) → YOLO class `{key}_{group}`; OCR each crop once, then run every member field's extractor on that text. Returns SAME shape as single (top-level `lot_number`/`exp_date`/`product_name`/`size`, no `lot_box`). NOTE: the mode is fully wired but **no shipped packaging uses `multi_field` yet** — all 6 are `single` except `container_label` (`cross_check`); there's no live example to copy.
   - `single` (default) → single-region detector → preprocessor → OCR, all crops stacked vertically before one OCR.
4. **RegionDetector** (`pipeline/detector.py`) — YOLOv8n; `crop_all()` keeps every detected box whose YOLO class starts with `{predicted_key}_` (built from `image_class`, e.g. `back_label_` matches `back_label_lot`/`_name`/`_size`), sorts top→bottom, and returns all crops (single-region then stacks them). It does NOT read `detector_yolo_prefixes` — that field is consumed only by edit-draft prelabel (`active_learning`). Falls back to heuristics when YOLO returns nothing.
5. **Preprocessor** (`pipeline/preprocessor.py`) — **pass-through stub**: `run()` returns the crop bytes unchanged (Cloud Vision does its own enhancement; no grayscale/denoise/Otsu/deskew here). The only OpenCV image processing lives in `detector.py:_crop_container_label` — CLAHE → binary threshold → morphology → largest-contour pick — used to *locate* the white lot box for `container_label`, not to preprocess for OCR.
6. **OcrEngine** (`pipeline/ocr_engine.py`) — Google Cloud Vision. The `image_class` is passed so `utils/validators.find_lot` tries the config's `lot_patterns` first, then `_LOT_BY_CLASS[class]`, then generic patterns (`validators.py:285-300`).
7. **SheetChecker** (`utils/sheet_checker.py`) — reads the row matching the OCR'd lot in the user-supplied Google Sheet and returns `lot_match`/`exp_match`/`product_match`/`sachet_match`.
8. **build_verify_message** (`pipeline/message_builder.py`) — fills the message template (`config/message_templates/*.yaml`) keyed by `config.message_template_key`.

Models are loaded once in the FastAPI `lifespan` startup. `services/model_registry.sync()` returns `(classifier_path, detector_path)`: on Cloud Run it downloads from Google Drive via `DRIVE_MANIFEST_FILE_ID` (manifest.json with sha256-verified file_ids); locally it falls back to `models/classifier.pt` and `models/detector.pt`.

## Wizard API (`api/packagings.py`)

`/api/packagings/*` is a separate router that backs `web/wizard.html`. It lets the user add a new packaging class without code changes: upload sample images, annotate bboxes, generate regex from labelled lot strings, save a draft YAML in `data/drafts/`, then promote to `config/packagings/`. Training is single-stage: drafts label ≥30 images then run Full Training (ADR 0005-single-stage-training). Edit-drafts can `POST /{key}/training/prelabel` to auto-fill bboxes on newly-added images using the deployed detector (filtered by the parent's `detector_yolo_prefixes`). Prelabel logic lives in `services/active_learning.prelabel_remaining(key, model_path, class_prefixes=...)` — runs server-side on the active `MODEL_DETECTOR_PATH`, no Colab. ADR 0002 explains "edit active packaging via clone". ADR 0003 explains "backend writes the reference dataset directly" (Full Training publishes images/labels to Drive; data.yaml is the commit point).

As of 2026-06-15 (`docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md`), Full Training is **manual**: `/training/full/start` publishes the dataset then returns a link to ONE static combined detector+classifier Colab notebook in Drive (`COMBINED_NOTEBOOK_FILE_ID` in `api/packagings.py`, built by `scripts/build_full_training_notebook.py`). `services/notebook_generator.py` was removed. Dataset layout the notebook reads: classifier = `data classify check lot/images/<class>/`; detector `data.yaml` stores absolute `/content/drive/MyDrive/...` paths that `dataset_publisher._relativize` normalizes before folder resolution.

As of 2026-06-22 (`docs/superpowers/specs/2026-06-22-train-done-sync-deploy-design.md`): after Colab finishes, the `training_full` screen shows a **"ดึง model" button** → `POST /{key}/training/full/sync` resolves all 3 Drive ids up front then downloads `full_detector.pt` + `classifier.pt` + `eval.json` (fixed Drive names; notebook writes `eval.json` LAST = done-signal) into `data/drafts/{key}/models/`, cleans partials on failure, sets status `trained` → existing Eval/Deploy screen. `/deploy` then promotes **both** detector + classifier (`promote_draft_model` returns `dict[str,Path]`) with backup+rollback on BOTH fresh and edit-draft paths (fresh path previously promoted nothing). `/training/full/done` stays 400 in prod (TEST_MODE-only sim).

Draft `status` is the wizard's resume point (`continueDraft` stepMap in `web/wizard.html`): `draft` = no images yet → step 2; `uploading` = has images → step 3 (first `save_image` bumps draft→uploading); `configured`/`training_full`/`trained` → step 4/5. An edit-draft (`{key}__edit` from clone) starts at `draft` with ZERO images of its own — parent images are display-only and full training requires ≥30 labeled images in the draft itself (`api/packagings.py` training/full/start gate).

## Adding or Editing a Packaging Class

Prefer editing the YAML in `config/packagings/<key>.yaml` over hardcoding. Required keys: `key`, `pipeline`, `lot_patterns`, `fields_extracted`, `sheet_checks`, `model_classifier_label`, `detector_yolo_prefixes`. Set `sub_regions: [box, sachet]` for multi-crop. Use `conf_threshold` (per-class override of the 0.6 default) to gate low-confidence classifier predictions. Set `gate_on_lot: false` if the packaging legitimately has no lot number (then `lot_short_fallback` may help).

`conf_threshold` is also user-tunable at runtime (no retrain): `PUT /api/packagings/{key}/conf` (0.50–0.95) persists an override to Drive/`data/config_overrides.json` which `PackagingRegistry(overrides=...)` merges over the YAML — the YAML value is only a default once an override exists. See ADR 0004. Registry reloads must go through `main.reload_registry()` (never `PackagingRegistry()` directly) or overrides are silently dropped.

When adding a brand-new class you also need: classifier label in the training dataset, YOLO class name with one of the `detector_yolo_prefixes`, retrained `.pt` files, updated message template if the existing four don't fit.

Product-name OCR matching is config-driven: set `product_aliases` (list of `{canonical, keywords}`) in the YAML — `find_product_name(text, aliases)` returns the `canonical` whose keyword appears in the OCR text (priority = list order; keywords matched literally via `re.escape` + word-boundary lookaround, so noisy wide crops still match). `canonical` MUST equal the sheet's `Product Name` (`sheet_checker` compares exact-lowercased on the lot-matched row). Empty/missing `product_aliases` → falls back to the hardcoded 9-keyword tea list in `utils/validators.py` (back_label/grade_bag stay on this). New field threads through: validators → ocr_engine/pipeline_runner → `PackagingConfig` → `schemas.PackagingConfigUpdate` → `packaging_store.clone_from_active` → `cloudrun_deployer.write_packaging_yaml`.

## Environment

Local dev reads `.env` (see `.env.example`). Key vars:

- `DRIVE_MANIFEST_FILE_ID` — empty locally (uses `models/*.pt`). NOTE: as of 2026-06 production Cloud Run has NO env vars set at all — it runs entirely on image defaults (models baked into the image). Verify with `gcloud run services describe ocr-lot-checker --region asia-southeast1`.
- `MODEL_CLASSIFIER_PATH` / `MODEL_DETECTOR_PATH` — local model paths.
- `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service account JSON locally; Cloud Run uses the runtime service account (`ocr-lot-checker-sa`) via ADC, no JSON key.
- Local Drive access: ADC is a user account with only `drive.file` scope — it can read/write files *created by this app* but gets 404 on files created manually in Drive web UI. Full `drive` scope is blocked by Google for user OAuth. To act as the production identity, use the SA key `gcp-key.json` (gitignored) via `GOOGLE_APPLICATION_CREDENTIALS`.
- `CONFIDENCE_THRESHOLD`, `LOG_LEVEL`.
- `MODEL_CACHE_DIR` — download cache for Drive-fetched weights (default `/tmp/models`). `OCR_LANG` — Vision language hints (default `th+en`). `REFERENCE_DETECTOR_PATH` / `REFERENCE_CLASSIFIER_PATH` — Drive My-Drive-relative dataset paths the training notebook reads (default `data check lot` / `data classify check lot`).
- `DRIVE_DETECTOR_DATASET_FOLDER_ID` / `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` — Drive folder ids of the reference dataset (`data check lot` / `data classify check lot`), shared with the SA. Required for the wizard's Full Training (dataset publish). See ADR 0003.
- `DRIVE_CONFIG_OVERRIDES_FILE_ID` — Drive file id of `config_overrides.json` (runtime tuning overrides, currently `conf_threshold` per packaging). Empty locally → falls back to `data/config_overrides.json`. See ADR 0004.
- `DRIVE_OAUTH_CLIENT_ID` / `DRIVE_OAUTH_CLIENT_SECRET` / `DRIVE_OAUTH_REFRESH_TOKEN` — OAuth user creds for Drive *writes* (dataset publish). **Service accounts have ZERO Drive storage quota** — SA-owned uploads 403 with `storageQuotaExceeded`, so `DriveClient` must act as a real Workspace user. When all three are set, `services/drive_client.py` uses them; otherwise it falls back to SA/ADC (read-only paths unaffected). Mint once via `python scripts/generate_drive_token.py` (Internal OAuth client → full `drive` scope, no Google verification, token never expires). Only `DriveClient` switches identity — `sheet_checker.py`/`cloudrun_deployer.py` keep the SA. The OAuth user must have Editor on the dataset folders. See ADR 0006-drive-dataset-write-via-oauth-user. NOTE: editing `.env` needs a dev-server restart (`--reload` re-reads code, not env).

## Test harnesses (TEST_MODE wizard)

Two ways to run the wizard in TEST_MODE — both: backend :8081, wizard :8091, deploy is simulated (skips Cloud Run), Drive writes hit the OCR-LOT-TEST folder only:
- `scripts/run_test_wizard.ps1` — local dev. Exports `.env.test` (`TEST_MODE=1`, `OCR_CONFIG_DIR=data/test/config`, `DRAFT_DIR=data/test/drafts`, seeded from prod). Drafts live in `data/test/drafts`. This is the usual local test env (NOT the bundle).
- `dist/portable-bundle` (built by `scripts/build_portable_bundle.ps1`) — self-contained zip to hand to others: embeddable Python + deps + models + `.env` (`TEST_MODE=1`) + REAL Drive creds. Recipient unzips → `START.bat`. Holds live creds — share only with trusted people; to ship updates you MUST rebuild (it's a snapshot).
- `web/wizard.html` is the SINGLE source of truth. `test wizzard/wizard.html` and `dist/portable-bundle/static/wizard.html` are GENERATED copies (with `window.API_BASE_OVERRIDE='http://localhost:8081'` injected after `<head>`). Never edit the copies — they go stale; edit `web/` then regenerate (run_test_wizard regenerates on each start).
- `TEST_MODE=1` quirks: `/health` returns `test_mode`; wizard reads it into `IS_TEST_MODE`. `/training/full/done` is neutralized (400) in prod but in TEST_MODE writes a passing fake eval + sets status `trained` so the eval/Deploy UI appears. Step-5 status flow: preTraining → Publish dataset → `training_full` → (TEST sim button) → `trained` → Deploy.
- `OCR_CONFIG_DIR` gotcha: NEVER hardcode `Path("config/packagings")` in `api/` — it ignores `OCR_CONFIG_DIR` and silently reads the PROD config under the test harness (dropped archived cards, cloned wrong YAML). Use `api/packagings._packaging_yaml_dir()` / registry `_config_dir()`.
- PowerShell 5.1 gotcha: `Get-Content -Raw` reads UTF-8 as ANSI → double-encodes Thai/symbols on rewrite (garbles the whole wizard). Always `Get-Content … -Raw -Encoding UTF8` when copying `web/wizard.html` or any Thai file in `.ps1`.
- Playwright wizard E2E: Thai string literals injected via `browser_evaluate` get mangled in transit — match elements by `onclick` attr (e.g. `archivePackaging`), not Thai text. `openDrawer(key, pkgSummary)` needs the summary arg — click the real card element instead of calling it with key only.

## Conventions specific to this repo

- `print()` is forbidden — use the module `logger`. Hooks may warn on `print` in edits.
- `pytest` is NOT on PATH — run `python -m pytest`.
- Source files contain Thai; read them with `encoding='utf-8'` (Windows default cp1252 raises `UnicodeDecodeError` on Thai bytes). Standalone scripts under `scripts/` need `sys.path.insert(0, repo_root)` to import `services`, and must run from the repo root.
- Printing Thai/Unicode from `python -c` to the PowerShell console raises `UnicodeEncodeError` (cp1252 charmap) — write output to a file with `encoding='utf-8'`, or set `$env:PYTHONIOENCODING='utf-8'` before the call.
- PowerShell `Out-File`/`>` adds a UTF-8 BOM that Python's `json.loads` rejects (`Unexpected UTF-8 BOM`). When a `.ps1`/`python -c` step produces JSON another tool reads, write it from Python with `Path(...).write_text(..., encoding='utf-8')` rather than piping through `Out-File`.
- Known pre-existing failure: `tests/test_classifier.py` has 3 setup errors — its fixture builds `efficientnet_b0` but `pipeline/classifier.py` now uses `efficientnet_v2_s`. Not caused by your changes; fix the fixture if touching classifier tests.
- Wizard E2E: Playwright blocks `file://` — serve with `python -m http.server 8090 --directory web` (hostname localhost makes API_BASE resolve to `http://localhost:8080`). Port 8080 is usually already taken by the dev server running with `--reload` (which auto-picks up code edits — check `/openapi.json` before assuming stale code).
- Preview a wizard step with NO backend: serve `web/` and `page.evaluate('startWizard()')` (or `goStep(n)`) via Playwright. `loadDashboard()` fetch errors but the step-1 form/builder render standalone — enough for visual/logic checks.
- Wizard step 4 does NOT prefill config from a saved/edit draft (only `prefillFieldsFromSubRegions` sets field checkboxes for `multi_field`). Re-entering step 4 + clicking Next re-POSTs form state, overwriting saved `lot_patterns`/`fields_extracted`/`sheet_checks`/`message_template_key`/`product_aliases` with defaults/empties — editing config via step-4 resume is lossy.
- `utils/validators.py` (find_lot / find_expiry / find_product_name / find_size) is tested in `tests/test_ocr.py` — there is no `test_validators.py`.
- Dark-theme `<select>` gotcha (`wizard.html`): native option popups render WHITE on Chromium/Windows unless you style `select option{background:var(--s1)}` AND give the select a real (non-`transparent`) `background-color` — Chromium derives popup color from it, `transparent` falls back to white. The open popup is OS-layer, so Playwright page screenshots never capture it (verify popup color by code, not screenshot).
- Date normalisation: validators return ISO `YYYY-MM-DD`; 2-digit years are accepted.
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
