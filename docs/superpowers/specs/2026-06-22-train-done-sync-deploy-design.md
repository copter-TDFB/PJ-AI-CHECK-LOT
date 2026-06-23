# Design — "Train เสร็จ → ดึง model → Deploy" button

**Date:** 2026-06-22
**Status:** Approved (design), pending implementation plan

## Problem

After the move to manual Colab Full Training (`docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md`), the post-training path is severed in production:

- The wizard's step-5 `training_full` screen (`renderStep5_training`) is a **dead-end** — it only shows an info box telling the user to "sync model เข้าระบบเอง", with **no button** to advance.
- `POST /{key}/training/full/done` returns **400 in production** (only TEST_MODE simulates completion), so the draft's `status` stays at `training_full` forever and the Eval/Deploy screen never appears.
- `POST /{key}/deploy` requires two files in `data/drafts/{key}/models/`: `eval.json` (hard-floor gate) and `full_detector.pt` (`cloudrun_deployer.promote_draft_model`). Neither ever lands locally now, so Deploy cannot run even if reached.

**Discovered gap (in scope):** `deploy_packaging` only promotes a model on the **edit-draft / overwrite path** (`parent_key is not None`). A **fresh deploy of a brand-new class writes only the YAML and never promotes any model**, so a newly-trained class would never classify/detect even after a "successful" deploy. This design closes that gap.

## Goal

Add a single button on the `training_full` screen that pulls the freshly-trained detector + classifier + eval metrics from Drive into the draft, flips the draft to `trained`, and routes the user into the **existing** Eval → Deploy flow (review metrics, then click the existing Deploy button). Detector **and** classifier are both synced and promoted so brand-new classes work end-to-end.

## Decisions (from brainstorming)

1. **Sync target:** via the existing `/deploy` mechanism (download model + eval into the draft, then the existing Deploy overwrites `models/*.pt` + restarts Cloud Run). Not manifest/`model_registry`.
2. **eval source:** the Colab notebook writes a structured `eval.json` to Drive (it already computes `metrics.box.*` on GPU). Backend does not recompute eval.
3. **Scope:** detector **and** classifier — both downloaded and promoted (new classes need a new classifier label).
4. **UX:** two steps — sync (download + `status=trained`) → review Eval screen → existing Deploy button. Not one-click sync+deploy.

## eval.json schema

`services/eval_thresholds.check_hard_floor` reads `config/eval_thresholds.yaml`:

```yaml
hard_floor:   { detector_mAP_50: 0.65, precision: 0.70, recall: 0.60 }
recommended:  { detector_mAP_50: 0.80, precision: 0.85, recall: 0.80 }
```

The notebook must write (same shape the TEST_MODE branch already produces):

```json
{
  "detector_mAP_50": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "epochs": 0,
  "imgsz": 640,
  "train_count": 0,
  "val_count": 0
}
```

## Components

### 1. Notebook (`ai_crop_lot.ipynb`, rebuilt via `scripts/build_full_training_notebook.py`)

Add a final cell to the detector section (after `metrics = best_model.val()`):

- Copy `best.pt` → Drive `data check lot/full_detector.pt` (**fixed name**, replacing the hardcoded `ai_check lot v3.pt`).
- Write Drive `data check lot/eval.json` with the schema above, sourced from `metrics.box.map50` / `metrics.box.mp` (precision) / `metrics.box.mr` (recall), plus `epochs` (from training config), `imgsz` (640), and `train_count`/`val_count` (count files under the published `train/`·`val/` image dirs).
- Classifier already saves to `data classify check lot/models/classifier.pt` — keep as-is.
- **`eval.json` is written last** so its presence is the "training finished" signal for the sync endpoint.

`build_full_training_notebook.py` injects/preserves this cell; add an assertion that the built notebook contains the eval-writing cell (regex check, mirroring the existing `iterdir()` assertion).

### 2. Backend

**New endpoint:** `POST /api/packagings/{key}/training/full/sync` (leave `/done` as-is for TEST_MODE):

1. `get_draft(key)` → 404 if missing.
2. Locate `eval.json` by name in the Drive detector dataset folder (`DRIVE_DETECTOR_DATASET_FOLDER_ID`). Not found → **409** `"ยังเทรนไม่เสร็จ — รัน Colab notebook ให้จบก่อน"`.
3. Download into `data/drafts/{key}/models/`:
   - `eval.json` (detector folder)
   - `full_detector.pt` (detector folder)
   - `full_classifier.pt` ← Drive `data classify check lot/models/classifier.pt` (`DRIVE_CLASSIFIER_DATASET_FOLDER_ID`)
4. `update_draft(key, status="trained")`.
5. Return `{ "eval": <eval_data>, "synced": true }`.

Download via `DriveClient` (search-by-name within folder → `download_file`). Any download/IO error → propagate as 500; draft stays `training_full` (status only flips after all files land).

**`services/cloudrun_deployer` changes:**

- `promote_draft_model(draft_key)` → promote **both** detector (`full_detector.pt` → `models/detector.pt`) and classifier (`full_classifier.pt` → `models/classifier.pt`). Return the promoted paths (or extend with a sibling `promote_draft_classifier`).
- `backup_artifacts` already snapshots both `detector.pt` + `classifier.pt` — reuse unchanged.
- `deploy_packaging` (`api/packagings.py`): run **backup → promote → reload** on **both** the fresh and overwrite paths whenever the draft has synced models (`full_detector.pt`/`full_classifier.pt` present), not only when `parent_key is not None`. This closes the fresh-deploy gap. Auto-rollback on failure stays (restore from backup manifest + reload).

### 3. Wizard (`web/wizard.html` `renderStep5_training`)

Add a button **"✅ Train เสร็จแล้ว — ดึง model จาก Drive"** to the `training_full` screen:

- `onclick` → `POST /api/packagings/{key}/training/full/sync`.
- On success → call `loadStep5()`; status is now `trained` → `renderStep5_eval` renders the metrics + the existing Deploy button.
- On 409 → inline message "ยังเทรนไม่เสร็จ — รอ Colab แล้วลองใหม่" (status unchanged, button re-enabled).
- Keep the existing info box; the button sits below it.

Regenerated copies (`test wizzard/wizard.html`, `dist/portable-bundle/static/wizard.html`) are produced from `web/wizard.html` — never edit directly.

## Data flow

```
Colab (GPU) → Drive: data check lot/full_detector.pt
                      data check lot/eval.json            (written last = "done" signal)
                      data classify check lot/models/classifier.pt
   │  [user clicks button in wizard]
   ▼
POST /training/full/sync → download 3 files → data/drafts/{key}/models/
                         → status = trained
   ▼
renderStep5_eval  (mAP / precision / recall + hard-floor ✓/✗)   ← user reviews
   │  [user clicks existing Deploy]
   ▼
POST /deploy → hard-floor gate → backup global models → promote detector+classifier
             → reload_registry → trigger Cloud Run revision
             (any failure → auto-rollback from backup)
```

## Error handling

| Condition | Behavior |
|-----------|----------|
| Drive missing `eval.json` | `/sync` → 409; draft stays `training_full`; button shows "ยังไม่เสร็จ" |
| Drive download / sha / IO error | `/sync` → 500; status not flipped (files land before status change) |
| hard-floor fail at deploy | Eval screen shows ✗; Deploy button disabled (existing logic) |
| deploy step fails | auto-rollback from backup manifest + `reload_registry` (existing logic) |
| Cloud Run trigger lacks IAM | non-fatal `{triggered:false,reason:...}` (existing logic) |

## Testing

- `tests/test_api_packagings.py`: mock `DriveClient` + `packaging_store`; assert `/sync` writes `eval.json` + `full_detector.pt` + `full_classifier.pt` to the temp `DRAFT_DIR/{key}/models` and sets `status=trained`; assert 409 when Drive has no `eval.json`.
- `cloudrun_deployer` unit tests (tmp dirs): `promote_draft_model` promotes both detector + classifier; fresh-path deploy backs up + promotes when synced models present.
- `scripts/build_full_training_notebook.py`: assert the built notebook contains the eval.json-writing cell (regex), alongside the existing `iterdir()` assertion.

## Out of scope

- `model_registry` / `manifest.json` path (production currently runs baked-in image models with no env vars).
- `multi_field` prelabel per-field tagging (unrelated deferred gap).
- Changing the global-retrain model (training stays whole-dataset; the synced model is global, the draft folder is only a deploy staging area).
