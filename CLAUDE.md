# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OCR Lot Checker — FastAPI service that classifies a product packaging photo, locates the lot/expiry region, OCRs it (Google Cloud Vision), and cross-checks the extracted values against a Google Sheet. Deployed on Google Cloud Run; static wizard frontend on Netlify.

Production URL: `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`

Phase tracking lives in `PLAN.md`. Bug post-mortems live in `bug_fix.md`. ADRs in `docs/adr/`.

## Commands

```bash
# Local server (port 8080 to match Cloud Run)
python -m uvicorn main:app --reload --port 8080

# Run pipeline on a single image without starting the server
python test_image.py <image_path>
python test_image.py <image_path> <sheet_id> <sheet_gid>

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
3. **PipelineRunner** (`pipeline/pipeline_runner.py`) — dispatches by `config.pipeline`:
   - `qr_scanner` → `QrScanner` only (used for `import_sticker`, which carries a QR).
   - `sub_regions` non-empty → multi-crop OCR (e.g. `container_label` has box + sachet, returns `lot_box`/`lot_sachet`/`exp_box`/`exp_sachet`).
   - Otherwise → single-region detector → preprocessor → OCR, with all crops stacked vertically before OCR.
4. **RegionDetector** (`pipeline/detector.py`) — YOLOv8n trained on `lot_region` boxes; filters detections by `detector_yolo_prefixes` of the predicted class; falls back to heuristics when YOLO returns nothing.
5. **Preprocessor** (`pipeline/preprocessor.py`) — grayscale → denoise → Otsu → deskew. `container_label` uses a special path: CLAHE + glare inpainting + adaptive threshold.
6. **OcrEngine** (`pipeline/ocr_engine.py`) — Google Cloud Vision. The `image_class` is passed so `utils/validators.find_lot` can pick `_LOT_BY_CLASS[class]` regex first, then fall back to generic patterns.
7. **SheetChecker** (`utils/sheet_checker.py`) — reads the row matching the OCR'd lot in the user-supplied Google Sheet and returns `lot_match`/`exp_match`/`product_match`/`sachet_match`.
8. **build_verify_message** (`pipeline/message_builder.py`) — fills the message template (`config/message_templates/*.yaml`) keyed by `config.message_template_key`.

Models are loaded once in the FastAPI `lifespan` startup. `services/model_registry.sync()` returns `(classifier_path, detector_path)`: on Cloud Run it downloads from Google Drive via `DRIVE_MANIFEST_FILE_ID` (manifest.json with sha256-verified file_ids); locally it falls back to `models/classifier.pt` and `models/detector.pt`.

## Wizard API (`api/packagings.py`)

`/api/packagings/*` is a separate router that backs `web/wizard.html`. It lets the user add a new packaging class without code changes: upload sample images, annotate bboxes, generate regex from labelled lot strings, save a draft YAML in `data/drafts/`, then promote to `config/packagings/`. Training is single-stage: drafts label ≥30 images then run Full Training (ADR 0005). Edit-drafts can `POST /{key}/training/prelabel` to auto-fill bboxes on newly-added images using the deployed detector (filtered by the parent's `detector_yolo_prefixes`). ADR 0002 explains "edit active packaging via clone". ADR 0003 explains "backend writes the reference dataset directly" (Full Training publishes images/labels to Drive; data.yaml is the commit point).

Draft `status` is the wizard's resume point (`continueDraft` stepMap in `web/wizard.html`): `draft` = no images yet → step 2; `uploading` = has images → step 3 (first `save_image` bumps draft→uploading); `configured`/`training_full`/`trained` → step 4/5. An edit-draft (`{key}__edit` from clone) starts at `draft` with ZERO images of its own — parent images are display-only and full training requires ≥30 labeled images in the draft itself (`api/packagings.py` training/full/start gate).

## Adding or Editing a Packaging Class

Prefer editing the YAML in `config/packagings/<key>.yaml` over hardcoding. Required keys: `key`, `pipeline`, `lot_patterns`, `fields_extracted`, `sheet_checks`, `model_classifier_label`, `detector_yolo_prefixes`. Set `sub_regions: [box, sachet]` for multi-crop. Use `conf_threshold` (per-class override of the 0.6 default) to gate low-confidence classifier predictions. Set `gate_on_lot: false` if the packaging legitimately has no lot number (then `lot_short_fallback` may help).

`conf_threshold` is also user-tunable at runtime (no retrain): `PUT /api/packagings/{key}/conf` (0.50–0.95) persists an override to Drive/`data/config_overrides.json` which `PackagingRegistry(overrides=...)` merges over the YAML — the YAML value is only a default once an override exists. See ADR 0004. Registry reloads must go through `main.reload_registry()` (never `PackagingRegistry()` directly) or overrides are silently dropped.

When adding a brand-new class you also need: classifier label in the training dataset, YOLO class name with one of the `detector_yolo_prefixes`, retrained `.pt` files, updated message template if the existing four don't fit.

## Environment

Local dev reads `.env` (see `.env.example`). Key vars:

- `DRIVE_MANIFEST_FILE_ID` — empty locally (uses `models/*.pt`). NOTE: as of 2026-06 production Cloud Run has NO env vars set at all — it runs entirely on image defaults (models baked into the image). Verify with `gcloud run services describe ocr-lot-checker --region asia-southeast1`.
- `MODEL_CLASSIFIER_PATH` / `MODEL_DETECTOR_PATH` — local model paths.
- `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service account JSON locally; Cloud Run uses the runtime service account (`ocr-lot-checker-sa`) via ADC, no JSON key.
- Local Drive access: ADC is a user account with only `drive.file` scope — it can read/write files *created by this app* but gets 404 on files created manually in Drive web UI. Full `drive` scope is blocked by Google for user OAuth. To act as the production identity, use the SA key `gcp-key.json` (gitignored) via `GOOGLE_APPLICATION_CREDENTIALS`.
- `CONFIDENCE_THRESHOLD`, `LOG_LEVEL`.
- `DRIVE_DETECTOR_DATASET_FOLDER_ID` / `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` — Drive folder ids of the reference dataset (`data check lot` / `data classify check lot`), shared with the SA. Required for the wizard's Full Training (dataset publish). See ADR 0003.
- `DRIVE_CONFIG_OVERRIDES_FILE_ID` — Drive file id of `config_overrides.json` (runtime tuning overrides, currently `conf_threshold` per packaging). Empty locally → falls back to `data/config_overrides.json`. See ADR 0004.

## Conventions specific to this repo

- `print()` is forbidden — use the module `logger`. Hooks may warn on `print` in edits.
- Known pre-existing failure: `tests/test_classifier.py` has 3 setup errors — its fixture builds `efficientnet_b0` but `pipeline/classifier.py` now uses `efficientnet_v2_s`. Not caused by your changes; fix the fixture if touching classifier tests.
- Wizard E2E: Playwright blocks `file://` — serve with `python -m http.server 8090 --directory web` (hostname localhost makes API_BASE resolve to `http://localhost:8080`). Port 8080 is usually already taken by the dev server running with `--reload` (which auto-picks up code edits — check `/openapi.json` before assuming stale code).
- Date normalisation: validators return ISO `YYYY-MM-DD`; 2-digit years are accepted.
- The `import_sticker` class is QR-only — never route it through detector/OCR. The QR scanner uses zxing-cpp first and a sticker-crop + cv2 fallback; the cv2 fallback has caused false positives historically (see `bug_fix.md`), so order and gating matter.
- Multi-crop classes return `lot_number=None` at the top level; downstream code uses `lot_box` for sheet lookup (see `main.py:183`).
- Model `.pt.bak-*` files are training rollback snapshots — leave them alone unless explicitly cleaning up.
