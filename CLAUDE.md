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

`/api/packagings/*` is a separate router that backs `web/wizard.html`. It lets the user add a new packaging class without code changes: upload sample images, annotate bboxes, generate regex from labelled lot strings, save a draft YAML in `data/drafts/`, then promote to `config/packagings/`. ADR 0001 explains the "wizard trains both models via reference dataset" flow; ADR 0002 explains "edit active packaging via clone".

## Adding or Editing a Packaging Class

Prefer editing the YAML in `config/packagings/<key>.yaml` over hardcoding. Required keys: `key`, `pipeline`, `lot_patterns`, `fields_extracted`, `sheet_checks`, `model_classifier_label`, `detector_yolo_prefixes`. Set `sub_regions: [box, sachet]` for multi-crop. Use `conf_threshold` (per-class override of the 0.6 default) to gate low-confidence classifier predictions. Set `gate_on_lot: false` if the packaging legitimately has no lot number (then `lot_short_fallback` may help).

When adding a brand-new class you also need: classifier label in the training dataset, YOLO class name with one of the `detector_yolo_prefixes`, retrained `.pt` files, updated message template if the existing four don't fit.

## Environment

Local dev reads `.env` (see `.env.example`). Key vars:

- `DRIVE_MANIFEST_FILE_ID` — empty locally (uses `models/*.pt`); set on Cloud Run.
- `MODEL_CLASSIFIER_PATH` / `MODEL_DETECTOR_PATH` — local model paths.
- `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service account JSON locally; Cloud Run uses the runtime service account (`ocr-lot-checker-sa`) via ADC, no JSON key.
- `CONFIDENCE_THRESHOLD`, `LOG_LEVEL`.

## Conventions specific to this repo

- `print()` is forbidden — use the module `logger`. Hooks may warn on `print` in edits.
- Date normalisation: validators return ISO `YYYY-MM-DD`; 2-digit years are accepted.
- The `import_sticker` class is QR-only — never route it through detector/OCR. The QR scanner uses zxing-cpp first and a sticker-crop + cv2 fallback; the cv2 fallback has caused false positives historically (see `bug_fix.md`), so order and gating matter.
- Multi-crop classes return `lot_number=None` at the top level; downstream code uses `lot_box` for sheet lookup (see `main.py:183`).
- Model `.pt.bak-*` files are training rollback snapshots — leave them alone unless explicitly cleaning up.
