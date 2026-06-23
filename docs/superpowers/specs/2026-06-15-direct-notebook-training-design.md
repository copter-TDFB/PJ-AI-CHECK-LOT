# Direct-Notebook Training (drop generated notebook)

**Date:** 2026-06-15
**Status:** Design — pending implementation plan
**Approach:** C — one combined Drive notebook (proven recipes merged) as the
source of truth; manual model sync.

## Problem

The wizard's Full Training currently *generates* a Colab notebook from Python
strings (`services/notebook_generator.py`). Its classifier recipe diverged from
the proven hand-written notebook (15 epochs single-stage vs 50 epochs two-stage
with heavy augmentation), so wizard-trained classifiers are weaker than the
notebook the user actually trusts.

The user already maintains two proven notebooks in Drive (byte-identical to the
local copies):

- Classifier — `colab_classify_training.ipynb` (file_id `1EYzV9DDE47BYnyOBtE8IwVNpZ1u9_jgy`)
- Detector — `ai_crop_lot.ipynb` (file_id `1EDiUdOwYDUWZAhQI73VGceWWZK2Te8WJ`)

The dataset publisher already writes into the same Drive folders these notebooks
read from. So generation is redundant — *if* the upload format matches what the
notebooks expect. It currently does **not** (two confirmed mismatches below).

## Decision

Drop notebook generation. A **single combined notebook** in Drive — the two
proven recipes merged (detector section then classifier section), with the
format fixes baked in — becomes the permanent source of truth. The wizard keeps
publishing the dataset, then hands the user **one** static link. The user runs it
once (`Run all`, ~90 min: detector ~60 + classifier ~30), both models save to
Drive, and the user syncs models into the running system manually
(model_registry/manifest or rebuild + redeploy). The wizard is not involved after
the hand-off.

The two original notebooks (`colab_classify_training.ipynb`, `ai_crop_lot.ipynb`)
stay untouched as personal reference; the wizard no longer uses them.

### Verified current state (Drive)

- `data check lot/data.yaml`: 12 classes, including wizard-added
  `new_tea_bag_box_*` — but `train/images` + `val/images` contain **zero**
  `new_tea_bag_box`-prefixed files (the images never landed). `train:`/`val:`
  are **absolute** paths.
- `data check lot/`: `train/{images,labels}`, `val/{images,labels}` (~353/88 → 80/20).
- `data classify check lot/`: `images/<class>/` for 8 classes, plus `models/classifier.pt`.

## Format mismatches to fix

### M1 — Classifier layout missing `images/`
- Notebook reads `data classify check lot/images/<class>/` (existing 8 classes live there).
- Publisher writes `data classify check lot/<key>/` (no `images/`).
- → Wizard-published classifier images are invisible to the notebook.
- **Fix:** publisher writes to `images/<key>/`.

### M2 — Detector `data.yaml` absolute paths break folder resolution
- `data.yaml` has `train: /content/drive/MyDrive/data check lot/train/images` (absolute).
- `dataset_publisher._resolve_dest_folders` splits the path into nested folders →
  would create `content/drive/MyDrive/...` instead of `train/images`.
- Evidence: `new_tea_bag_box_*` is declared in data.yaml but has no images anywhere
  in the detector tree.
- **Fix:** normalize `data.yaml` to **relative** paths (`train: train/images`,
  `val: val/images`). Relative resolves correctly both for the publisher and for
  YOLO in Colab (relative to the data.yaml directory).

## Changes

### Code

1. **`services/dataset_publisher.py`** — classifier destination:
   `cls_folder = ensure_folder("images", cls_root)` then `ensure_folder(key, that)`
   so images land in `images/<key>/`. (M1)

2. **`services/dataset_publisher.py`** — detector path handling: when reading
   `data.yaml`, if `train`/`val` are absolute under `/content/drive/MyDrive/<root>/`,
   normalize to relative before `_resolve_dest_folders`; and write `data.yaml` back
   with relative `train`/`val`. (M2)

3. **`api/packagings.py` `/training/full/start`** — keep `dataset_publisher.publish()`;
   remove notebook generation + upload (`build_full_notebook`, run-folder create,
   notebook upload). Return one static `colab_url` (constant: the combined
   notebook file_id). Drop `notebook_file_id` / `output_folder_id` from the draft
   `training_run` (or set null).

4. **Remove `services/notebook_generator.py` + `tests/test_notebook_generator.py`**
   (dead code once generation is gone).

5. **`web/wizard.html` step 5** — copy change: "open the training notebook, Run all
   (~90 min, detector then classifier); both models save to Drive; sync manually."
   Render the one static link instead of the generated link.

### Combined notebook (build once, store in Drive as the source of truth)

6. **Build `lot-checker-full-training.ipynb`** by merging the two proven recipes
   into one notebook, with these fixes baked in, and upload to Drive (its file_id
   becomes the constant used in change 3):
   - **Setup:** `pip install` ultralytics + classifier deps; mount Drive.
   - **Detector section** — train YOLO with the exact `ai_crop_lot.ipynb` recipe
     (yolo11s, 250 epochs, imgsz 1024, same augmentation), **but**: read the
     publisher-maintained `data check lot/data.yaml` as-is — do **not** overwrite
     it with a hardcoded class list. Save `best.pt` to Drive.
   - **Classifier section** — train EfficientNet-V2-S with the exact
     `colab_classify_training.ipynb` recipe (2-stage freeze→fine-tune, full
     augmentation, Dropout head, class-weighted loss + label smoothing, stratified
     split, OneCycleLR, confusion matrix), **but**: auto-discover classes —
     `CLASSES = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])` —
     reading `data classify check lot/images/<class>/` (matches M1). Save
     `classifier.pt` to Drive.

### Data heal (one-off, manual)

7. After M1/M2 land, re-run `publish` for the `new_tea_bag_box` draft. Idempotent:
   class names already in data.yaml are not re-appended, but the missing images
   upload to the correct `train/val` folders this time.

## Out of scope / accepted trade-offs

- **Lost automation:** `training/full/done` auto-pickup of trained models and the
  progress "notebook" step. Model sync is fully manual (user's choice).
- **Split unchanged:** 80/20 `train`/`val` via `split_for()` already happens before
  upload. Folder name stays `val` (not `test`).
- No change to the QR-only classes; classifier auto-discovery naturally includes
  whatever class folders exist under `images/`.

## Manual model-sync reference

After training:
- Classifier → `data classify check lot/models/classifier.pt`
- Detector → `data check lot/<run name>.pt` (notebook copies `best.pt` there)

To deploy: download both, replace `models/classifier.pt` + `models/detector.pt`,
then rebuild the Docker image + redeploy (production runs baked-in models, no
manifest env var set). If a manifest is later adopted, update `manifest.json`
file_ids + sha256 and set `DRIVE_MANIFEST_FILE_ID`.
