# Drive Dataset Direct Write — Design

Date: 2026-06-10
Status: Approved by ops (operation@tdfb.co)
Scope: Full Training data path only. Model publish + manifest update is a separate follow-up spec.

## Problem

The wizard's Full Training step bundles the new packaging's images + labels into a
ZIP (`services/training_bundle.py`), uploads it to a backend-owned Drive folder,
and relies on the generated Colab notebook to unzip and merge it with the
reference dataset at training time (ADR 0001).

Pain points:

- The reference dataset on Drive is never the complete truth — every training
  run depends on a transient addition bundle plus merge logic inside the
  notebook template.
- Addition folders accumulate on Drive and must be cleaned up manually
  (a known negative in ADR 0001).
- The merge step duplicates dataset-layout knowledge between backend and
  notebook.

## Decision

On **Full Training start**, the backend writes the new class's images and
labels **directly into the Drive reference dataset**:

- Detector: `data check lot/` (train/val images + YOLO labels + `data.yaml`)
- Classifier: `data classify check lot/{class_key}/`

The Colab notebook no longer merges anything — it mounts Drive and trains both
models straight from the reference dataset.

### Why the ADR 0001 restriction no longer applies

ADR 0001 rejected direct append because the full `drive` OAuth scope is
restricted for unverified OAuth clients. However, `services/drive_client.py`
authenticates via `google_auth_default()` — a **service account (ADC)**, not a
user OAuth client. Service accounts do not go through OAuth consent
verification. Sharing the two reference folders with the SA's email and using
the `drive` scope gives the backend direct write access. This will be recorded
as **ADR 0003** (supersedes the data-path portion of ADR 0001; the
notebook-trains-both-models portion of ADR 0001 stays in force).

## One-time setup (manual)

1. Share `data check lot` and `data classify check lot` with the service
   account email (`ocr-lot-checker-sa@...`) as **Editor**.
2. New env vars (in `.env` / Cloud Run):
   - `DRIVE_DETECTOR_DATASET_FOLDER_ID` — folder id of `data check lot`
   - `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` — folder id of `data classify check lot`
3. Change `_SCOPES` in `services/drive_client.py` from
   `https://www.googleapis.com/auth/drive.file` to
   `https://www.googleapis.com/auth/drive`.

**Known constraint:** files created by the SA inside a shared My Drive folder
count against the SA's own 15 GB quota. Acceptable for the foreseeable future;
moving the dataset to a Workspace **Shared Drive** removes the limit entirely
(not required now).

## New component: `services/dataset_publisher.py`

Replaces `training_bundle` for the **full** training path only. Public entry:

```python
def publish(key: str) -> dict:
    """Write draft's labeled images + labels into the Drive reference dataset.

    Returns summary: {images_uploaded, images_skipped, new_classes, class_ids}.
    """
```

Algorithm:

1. **Fail fast:** verify both env folder ids are set and the SA can list both
   folders; raise with a message explaining the sharing setup otherwise.
2. Read `data.yaml` from the detector folder. The train/val image directory
   layout is taken from the `train:`/`val:` paths declared inside `data.yaml`
   (label dirs follow the YOLO `images/` → `labels/` convention) — never
   hardcoded. Append the new YOLO class names
   (`{key}_{sub_region}`, ordered by the draft's `sub_regions` list) at the
   **end** of `names` if not already present. **Never reorder or remove
   existing entries** — label files reference numeric ids.
3. Compute global class ids from position in the merged `names` list.
4. Deterministic 80/20 train/val split per image: hash of the filename
   (e.g. `sha1(filename) % 10 < 8 → train`). Re-runs produce the same split.
5. For each labeled image in `data/drafts/{key}/images/` that has bboxes:
   - Filename prefixed with the packaging key: `{key}_{original_name}` to
     avoid cross-packaging collisions.
   - Upload image → `train/images/` or `val/images/` (per split).
   - Upload YOLO label txt (using **global** class ids) → matching
     `train/labels/` or `val/labels/`.
   - Upload a copy of the image (same prefixed filename) → classifier folder
     `data classify check lot/{key}/` (folder created if missing).
   - **Idempotent:** skip any upload whose target filename already exists in
     the destination folder.
6. Write `data.yaml` back **last** — it is the commit point. If the run dies
   mid-upload, the new class names were never declared, so the existing
   dataset remains valid; a retry re-uploads only the missing files.

### `DriveClient` additions

- `read_text(file_id) -> str`
- `update_file_content(file_id, content: bytes)` — overwrite an existing file
  (used for `data.yaml`)
- `list_folder(parent_id) -> list[{id, name}]` (paginated)
- `ensure_folder(name, parent_id) -> folder_id` — find-or-create

## Changes to existing code

| File | Change |
|------|--------|
| `api/packagings.py` `training_full_start` | Call `dataset_publisher.publish(key)` instead of `training_bundle.build_zip()` + zip upload. Still creates the per-run Drive folder for the notebook + trained-model outputs. |
| `services/notebook_generator.py` `build_full_notebook` | Remove bundle download + addition-merge cells. Notebook mounts Drive and trains both models directly from the reference dataset paths. Output upload (`full_detector.pt`, `full_classifier.pt` → run folder) unchanged. |
| `services/drive_client.py` | Scope change + new methods above. |
| Seed path (`training_seed_start`, `build_seed_notebook`, `training_bundle.py`) | **Unchanged.** Seed runs before images are published and its model is throwaway; the small ZIP remains the right tool. |
| `/training/full/done`, `/deploy` | Unchanged in this spec (model publish + manifest update is the follow-up spec). |

## Free behaviour: adding images to an existing class

The edit-clone flow (ADR 0002) goes through the same Full Training start. For
an existing class: its names are already in `data.yaml` (no append), new
images upload normally, previously-published images are skipped by the
idempotency rule. No extra code needed.

## Error handling

- Missing env / inaccessible folders → HTTP 500 before any upload, message
  includes the SA email to share with.
- Mid-upload failure → HTTP 500; `data.yaml` untouched; retry resumes via
  skip-existing.
- Colab training failure → no rollback needed: published labels were
  human-verified, so the dataset is still correct; only the model run failed.

## Testing

Unit tests (`tests/test_dataset_publisher.py`, DriveClient mocked):

- Global class-id mapping (new class appended after existing 9; existing ids
  untouched).
- `data.yaml` append is idempotent (publishing twice yields identical yaml).
- Deterministic split: same filename always lands in the same split; ratio
  ≈ 80/20 over a sample set.
- Label file content uses global ids and normalised coords.
- Ordering: `data.yaml` update is the final Drive write.
- Fail-fast when env vars are missing.

Manual smoke test: run `publish()` against a scratch Drive folder pair before
pointing the env vars at the real dataset.

## Out of scope (follow-up spec)

- Downloading/promoting `full_classifier.pt` (currently never promoted).
- Uploading promoted models to Drive + updating `manifest.json` so
  `model_registry.sync()` on Cloud Run picks up new weights.
- Fresh-deploy model promotion.
