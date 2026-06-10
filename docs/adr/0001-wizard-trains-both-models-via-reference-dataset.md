# 0001 — Wizard retrains both Classifier and Detector against the Drive reference dataset

Date: 2026-06-06
Status: Accepted

## Context

When ops adds a new packaging through the wizard, the system must still recognise
all existing packagings *and* the new one. Production uses two models:

- **Classifier** (EfficientNet-V2-S) decides packaging type — knows 8 classes
  today.
- **Detector** (YOLOv11s, multi-class) finds OCR regions inside an image — knows
  9 sub-region classes today (`back_label_lot`, `container_label_box`, etc.).

Both models were trained from Google Drive reference folders that already exist:

- `MyDrive/data check lot/` — detector training set (train/val images + YOLO
  labels + `data.yaml`).
- `MyDrive/data classify check lot/` — classifier training set (one folder of
  images per class).

The Phase 3 wizard implementation trained a 1-class detector on draft images
only, which would replace the production multi-class detector with a model that
forgets the 6 existing packagings. Classifier retraining was missing entirely.

## Decision

The wizard's Full Training step retrains **both** Classifier and Detector
together. The data flow:

1. **Backend** writes the new packaging's images + YOLO labels to a new Drive
   folder it owns: `Lot Checker AI/additions/{packaging_key}/`. The backend's
   OAuth token uses the non-sensitive `drive.file` scope, so it can only see
   its own folders — it never touches the reference folders directly.
2. **Generated notebook** runs in Colab where the user is signed in with their
   own Google account and has full Drive access. The notebook mounts Drive,
   reads BOTH the reference dataset AND the addition folder, and merges them
   in `/content/data/` before training.
3. Both models train against the merged dataset and are uploaded back to the
   addition folder (where the backend can pick them up via `drive.file`).

The Seed Training step stays unchanged: a fast 1-class detector trained only on
the new packaging's hand-labeled images, used solely for pre-labeling the
remaining images of the same packaging (active learning). The seed model is
never deployed.

## Consequences

**Positive**
- Production behaviour is preserved: existing packagings keep working after
  every deploy because their reference data is still in the training set.
- One source of truth: the Drive reference folders. No staging duplication, no
  ambiguity about what each model was trained on.
- Reuses the proven training recipes from `ai_crop_lot.ipynb` and
  `colab_classify_training.ipynb` (hyperparameters, augmentations, model
  versions) instead of inventing new ones.

**Negative**
- Reference dataset Drive folder paths are hard-coded inside the notebook
  template — moving them requires regenerating the notebook.
- Full Training takes ~1-2 hours on Colab Free GPU (classifier ~30-60 min +
  detector ~30-60 min), versus ~30 min for detector-only.
- If a Full Training fails, the addition folder is left on Drive and must be
  cleaned up (either by the next run or manually).

**Alternatives rejected**
- *Direct append from backend to reference folders* — Would require the
  `drive` (full) scope, which Google classifies as restricted/sensitive and
  blocks for unverified OAuth clients. We would either need a verified custom
  OAuth client or a service-account key, both of which add deployment overhead.
- *Per-class detector models* — Would need 7+ YOLO models loaded in RAM per
  Cloud Run instance, breaking the free-tier memory budget.
- *Skip classifier; n8n supplies class param* — Pushes routing logic into n8n,
  contradicting the "ops does everything in the wizard" promise.
