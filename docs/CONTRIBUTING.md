# Contributing

Development guide for `pj-ocr-text-check-lot` (OCR Lot Checker). For architecture,
domain language, and operational gotchas, read `CLAUDE.md` and `CONTEXT.md` first —
this file only covers the mechanics of setting up and contributing.

## Prerequisites

- Python 3.11 (the Cloud Run image uses `python:3.11-slim`; a different local
  version can behave differently, especially around `torch`/`ultralytics`)
- `gcloud` CLI, authenticated to the `pj-ai-detect-lot-no` project (only needed
  for deploying or reading/writing prod GCS config — not for local dev)
- A Google Cloud Vision-enabled service account key (`gcp-key.json`, gitignored)
  for local OCR calls
- Optional: Docker, only if you want to build the exact production image locally

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

copy .env.example .env        # then fill in real values
# GOOGLE_APPLICATION_CREDENTIALS should point at your gcp-key.json
```

`requirements-train.txt` is separate and only needed for local model
training/evaluation scripts (`train_classifier.py`, `train_detector.py`,
`evaluate.py`) — not for running the API.

Local dev does **not** need `GCS_CONFIG_BUCKET` or `DRIVE_MANIFEST_FILE_ID` set —
leaving them empty falls back to local `models/*.pt` and `config/packagings/*.yaml`
with no network calls. See "Environment variables" below for what each one does.

## Available commands

<!-- AUTO-GENERATED: derived from CLAUDE.md § Commands and package scripts. Re-run /update-docs after adding new scripts. -->

| Command | Description |
|---|---|
| `python -m uvicorn main:app --reload --port 8080` | Local API server (port 8080 matches Cloud Run) |
| `python test_image.py <image_path>` | Run the pipeline on one image outside the server. **Not production-faithful** — bypasses `PackagingRegistry`, so config-driven `lot_patterns`/`product_aliases` don't apply. To test config changes on the real path, drive `PipelineRunner` with a loaded registry instead (see `CLAUDE.md`). |
| `python scripts/run_real_pipeline.py <image_path> [--force]` | Runs the **real** production code path (classifier → `PackagingRegistry` → `PipelineRunner`) on one image — the fix for `test_image.py`'s gap above. `--force` skips the low-confidence gate to run the pipeline anyway. |
| `pytest` | Run all tests |
| `pytest tests/test_integration.py` | Integration tests (hits a real running `/predict`) |
| `pytest tests/test_ocr.py -k lot` | Single test by keyword |
| `python evaluate.py` | Classifier + detector accuracy per class |
| `python eval_thresholds.py` | Tune `conf_threshold` per class |
| `python confusion_matrix_eval.py [--classes a,b] [--thresholds 0.5,0.6]` | Classifier confusion matrix; defaults to all active classes at 0.5/0.55/0.6. Metrics are measured on the same images used to train the classifier (no held-out set in-repo) — treat as relative (threshold-vs-threshold), not an absolute production guarantee. |
| `python scripts/detector_dataset_topup.py {dedup,prelabel,merge-all,publish,status}` | Grows the Drive detector dataset from local classifier images: dedup (dHash vs. already-published) → prelabel (run `models/detector.pt`) → human review → `publish` (Drive train/val). `publish` is dry-run by default (`--execute` to mutate Drive). |
| `python scripts/detector_annotator.py [images_dir] [labels_dir]` | Tkinter bbox annotator/validator for the topup workflow above (replaces labelImg, whose PyQt5 build is unmaintained). Auto-saves on every edit/navigation. |
| `docker build -t ocr-lot-checker .` | Build the same image Cloud Run runs |
| `start "" "web\wizard.html"` (Windows) | Open the wizard frontend locally |

<!-- END AUTO-GENERATED -->

There is no `package.json`/`Makefile`/`pyproject.toml` in this repo — the table
above is the closest thing to a script registry; it mirrors `CLAUDE.md` § Commands,
which is the source of truth if the two ever drift.

## Environment variables

<!-- AUTO-GENERATED: derived from .env.example and CLAUDE.md § Environment. -->

| Variable | Required (local) | Description |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to the SA JSON key used for Cloud Vision (and Drive fallback) locally. Cloud Run uses the runtime SA via ADC instead. |
| `MODEL_CLASSIFIER_PATH` / `MODEL_DETECTOR_PATH` | No (defaults to `models/classifier.pt` / `models/detector.pt`) | Local model weight paths, used when `DRIVE_MANIFEST_FILE_ID` and `GCS_CONFIG_BUCKET` are both unset. |
| `DRIVE_MANIFEST_FILE_ID` | No | Drive file id of the model manifest. Empty locally → local `models/*.pt`. Required on Cloud Run only if GCS isn't used. |
| `MODEL_CACHE_DIR` | No (default `/tmp/models`) | Local cache dir for Drive-downloaded weights. |
| `OCR_LANG` | No (default `th+en`) | Vision API language hints. |
| `LOG_LEVEL` | No (default `INFO`) | Python logging level. |
| `CONFIDENCE_THRESHOLD` | No (default `0.7`) | Fallback classifier confidence gate when a packaging has no per-class `conf_threshold`. |
| `REFERENCE_DETECTOR_PATH` / `REFERENCE_CLASSIFIER_PATH` | No | Drive My-Drive-relative dataset paths the training notebook reads. |
| `DRIVE_DETECTOR_DATASET_FOLDER_ID` / `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` | Only for wizard Full Training | Drive folder ids of the reference dataset, shared with the SA. |
| `DRIVE_CONFIG_OVERRIDES_FILE_ID` | No | Drive file id of `config_overrides.json` (runtime tuning, ADR 0004). Empty → local `data/config_overrides.json`. |
| `DRIVE_OAUTH_CLIENT_ID` / `DRIVE_OAUTH_CLIENT_SECRET` / `DRIVE_OAUTH_REFRESH_TOKEN` | Only for dataset publish (writes) | OAuth user credentials — service accounts have zero Drive storage quota, so writes need a real Workspace user identity (ADR 0006). Mint with `python scripts/generate_drive_token.py`. |

<!-- END AUTO-GENERATED -->

Two production-only variables are **not** in `.env.example` because they only
apply to the deployed Cloud Run service (see `CLAUDE.md` § Environment for the
full live list): `GCS_CONFIG_BUCKET` (switches config/model sync to GCS-first)
and the Drive dataset folder ids/secrets set via `--set-secrets` at deploy time.

Editing `.env` requires restarting the dev server — `--reload` re-reads code
changes, not environment variables.

## Testing

```bash
pytest                              # all tests
pytest tests/test_ocr.py -v         # validator regex tests (find_lot/find_expiry/...)
pytest tests/test_integration.py    # hits a real running /predict — start the server first
```

- `tests/test_classifier.py` has 3 known pre-existing setup errors (fixture
  builds `efficientnet_b0`, but `pipeline/classifier.py` now uses
  `efficientnet_v2_s`) — not something a normal change should fix incidentally;
  only touch it if you're specifically working on the classifier.
- There is no configured linter/formatter (no `ruff`/`black`/`flake8` config,
  no pre-commit hooks, no CI workflow in this repo) — style is enforced by
  code review against the conventions in `CLAUDE.md`, not tooling. Follow the
  existing file's style rather than reformatting wholesale.
- New validator behavior (`utils/validators.py`) should get a matching test in
  `tests/test_ocr.py` — there is no separate `test_validators.py`.
- To test a packaging-config change (`lot_patterns`, `product_aliases`, etc.)
  on the real code path, drive `PipelineRunner` with a loaded
  `PackagingRegistry` — **not** `test_image.py`, which skips the registry
  entirely (see `CLAUDE.md` § Commands for the exact pattern).

## Code conventions

- `print()` is forbidden — use the module `logger` (`logging.getLogger(__name__)`).
- Source files contain Thai text; always open with `encoding='utf-8'` — Windows'
  default `cp1252` raises `UnicodeDecodeError` on Thai bytes.
- Standalone scripts under `scripts/` need
  `sys.path.insert(0, repo_root)` to import `services`/`pipeline`/`utils`, and
  must be run from the repo root.
- Prefer editing a packaging's YAML (`config/packagings/<key>.yaml`) over
  hardcoding new logic — see `CLAUDE.md` § Adding or Editing a Packaging Class.
- Full architecture, the request flow through `main.py`, and the wizard's
  training/deploy lifecycle are documented in `CLAUDE.md` — read it before
  making non-trivial changes; it is the single most up-to-date source in the repo.

## Pull request checklist

- [ ] `pytest` passes (aside from the 3 known pre-existing `test_classifier.py`
      errors — confirm you didn't add new failures, e.g. via `git stash` + rerun)
- [ ] New/changed extraction logic (`utils/validators.py`,
      `config/packagings/*.yaml`) has a matching test in `tests/test_ocr.py`
- [ ] Config changes were verified via `PipelineRunner` + `PackagingRegistry`,
      not `test_image.py`
- [ ] No `print()` statements added; Thai-containing files opened with
      `encoding='utf-8'`
- [ ] If the change affects a **shipped** packaging class, note in the PR
      description that it needs an image rebuild + redeploy (GCS overlay only
      covers wizard-created/edited classes — see `CLAUDE.md` gotcha)
- [ ] Non-obvious bug fixes get a `bug_fix.md` entry (root cause → fix → validation)
