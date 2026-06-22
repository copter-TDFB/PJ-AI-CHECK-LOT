# Design — Colab Full Training: Resume & Skip (กันเทรนใหม่หมดเมื่อ Colab หลุด)

**Date:** 2026-06-22
**Status:** Approved (design), pending implementation plan

## Problem

Full Training is manual in Colab (one combined notebook `lot-checker-full-training.ipynb`,
built by `scripts/build_full_training_notebook.py`). It runs **Detector (YOLO, ~250 epochs,
~60 min)** then **Classifier (EfficientNet-V2-S, 2-stage, ~30 min)**; both save to Drive,
`eval.json` is written last in the detector section as the sync done-signal.

Colab Free tier disconnects often (idle ~90 min, GPU reclaim, session caps). On any
disconnect the runtime resets — `/content` is wiped — and re-running the notebook
restarts training **from scratch**, even when the detector already finished. Three failure
modes, all observed:

1. **Full runtime reset** — `/content` gone, Drive must be re-mounted.
2. **Disconnect mid-detector** — lose all detector epochs (e.g. epoch 200/250).
3. **Disconnect mid-classifier** — detector done + saved to Drive, but `Run all`
   re-trains the detector unnecessarily.

The wizard backend is **not involved during training** (it hands off a link, then later
pulls models via the "ดึง model" button). So resilience is primarily a **notebook +
build-script** problem, with one small backend touch for the reset boundary.

## Goal

Make Full Training **resumable** and **idempotent across runs**:

- **Section-level skip** — a finished section is not re-run.
- **Epoch-level resume** — a section that crashed mid-training continues from the last
  checkpoint (full resume for **both** detector and classifier), not from epoch 0.
- Checkpoints survive a runtime reset (stored on Drive).
- A genuinely new training run (new/changed dataset) must **not** wrongly skip/resume
  from stale artifacts.

## Core principle — the reset boundary

> **Pressing "Train" in the wizard = start a new run.**
> **Re-running `Run all` without pressing the button = resume the same run.**

The wizard's `POST /{key}/training/full/start` already publishes the dataset to Drive on
every "Train" click. That call becomes the **reset point**: it deletes the previous run's
markers/checkpoints. After it, the notebook is purely resumable until the next "Train"
click.

This is automatic (no `FRESH_START` toggle to remember, no dataset fingerprinting on
slow Drive listings). Trade-off: one small backend change (a delete step in the publish
flow). Decided over a notebook-only toggle (error-prone) and dataset fingerprint
(automatic but adds Drive-listing cost + notebook complexity).

## Drive layout

```
data check lot/
  full_detector.pt          ← final detector (sync reads this) — overwritten at section end
  eval.json                 ← detector done-signal (sync reads this) — written LAST in detector section
  _training/detector/        ← YOLO project= dir; last.pt/best.pt during training (this run)
data classify check lot/
  models/classifier.pt      ← final classifier (sync reads this) — overwritten at section end
  _training/classifier/
    ckpt.pt                 ← { epoch, model_state, optim_state, sched_state, best_acc, stage }
    done.flag               ← written only when the classifier fully completes
```

Baselines `full_detector.pt` and `models/classifier.pt` are **preserved** across a
publish — they are only overwritten when their section finishes. If a new run crashes
mid-training, the previous known-good model still exists on Drive (and production runs
baked-in image models anyway, so the live service is unaffected until a deploy).

## Components

### 1. Backend — publish is the reset point

`POST /api/packagings/{key}/training/full/start` (`api/packagings.py`): after
`dataset_publisher.publish()` succeeds, delete the previous run's markers/checkpoints on
Drive:

- `data check lot/eval.json` (detector done-signal)
- `data check lot/_training/detector/` (detector checkpoints)
- `data classify check lot/_training/classifier/` (incl. `ckpt.pt` + `done.flag`)

Do **not** delete `full_detector.pt` or `models/classifier.pt` (baselines; overwritten at
section end). Implement as a small helper (e.g. `dataset_publisher.reset_training_state()`
or a function in the publish module) using `DriveClient` (search-by-name within the known
folders → delete). Missing files = no-op (idempotent). A delete failure is **non-fatal**:
log a warning and continue (a stale marker only risks a wrong skip on the next run; the
publish itself — the user-visible action — must still succeed). Detector folder id from
`DRIVE_DETECTOR_DATASET_FOLDER_ID`, classifier from `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`.

### 2. Notebook — detector section

```
mount Drive (re-mount safe after reset)
if eval.json exists:                         # detector finished this run
    skip → go to classifier
elif _training/detector/<run>/weights/last.pt exists:
    model = YOLO(last.pt); model.train(resume=True)     # continue from last epoch
else:
    model = YOLO('yolo11s.pt')
    model.train(data=<published data.yaml>, project='_training/detector', name='train',
                epochs=250, imgsz=1024, <same augmentation as ai_crop_lot.ipynb>)
# on completion:
copy best.pt → full_detector.pt
metrics = model.val()
write eval.json   (LAST — its presence = detector done; schema unchanged, see below)
```

**Free-tier checkpoint durability.** Writing `last.pt` to the Drive FUSE mount every
epoch roughly doubles epoch time and risks corruption, which on free tier *increases*
disconnect probability. Instead train to fast local `/content`, and use an Ultralytics
callback (`on_fit_epoch_end`) to copy `last.pt` → Drive **every N epochs** (default
`CKPT_EVERY = 10`, configurable in a preflight cell ≈ lose ≤10 epochs). On resume after a
reset: copy the Drive `last.pt` back to `/content` first, then `resume=True`. Validate
that Ultralytics `resume=True` correctly restores epoch/optimizer/LR-scheduler from the
checkpoint during implementation.

`eval.json` schema is unchanged from `2026-06-22-train-done-sync-deploy-design.md`
(`detector_mAP_50`, `precision`, `recall`, `epochs`, `imgsz`, `train_count`, `val_count`)
so the `/sync` contract is untouched.

### 3. Notebook — classifier section (full epoch-resume)

```
if done.flag exists:                         # classifier finished this run
    skip
else:
    start_epoch, best_acc, stage = 0, 0.0, 'freeze'
    if ckpt.pt exists:
        state = torch.load(ckpt.pt)
        model/optim/sched load_state_dict(...)
        start_epoch, best_acc, stage = state['epoch']+1, state['best_acc'], state['stage']
    run the 2-stage loop (freeze → fine-tune) starting at (stage, start_epoch):
        each epoch (or every N): torch.save({epoch, model, optim, sched, best_acc, stage}) → Drive ckpt.pt
# on completion:
save classifier.pt → models/classifier.pt
write done.flag
```

The classifier is a **2-stage** recipe (freeze head → fine-tune full network), so the
checkpoint must record `stage` and the in-stage epoch, and the resume logic must
reconstruct the correct optimizer/scheduler for that stage before continuing. This is the
most intricate piece of the work. Checkpoint cadence matches the detector
(`CKPT_EVERY`, default 10) for the same free-tier durability/runtime balance. Use the
same local-then-copy-to-Drive pattern (save to `/content`, copy to Drive) to avoid FUSE
write thrash.

### 4. Build script (`scripts/build_full_training_notebook.py`)

The resilience logic is baked in at build time (same place that currently patches the
`CLASSES = ...` line and drops the `data.yaml`-overwrite cell):

- **Preflight cell** (injected before the detector section): mount Drive; define the
  Drive `_training/` paths, the done/partial detection, the skip/resume flags
  (`DET_SKIP`, `DET_RESUME`, `CLS_SKIP`, `CLS_RESUME`), and `CKPT_EVERY`.
- **Transform the detector train cell** to honor `DET_SKIP`/`DET_RESUME`, set
  `project=_training/detector`, and register the every-N-epoch Drive-copy callback.
- **Transform/wrap the classifier loop** for `CLS_SKIP`/`CLS_RESUME` + per-N-epoch
  `ckpt.pt` save and the final `done.flag` write.
- **Assertions** (mirroring the existing `iterdir()` / `full_detector.pt` checks): the
  built notebook contains the skip guards, the `resume=True` path, the checkpoint-copy
  callback/`torch.save(...ckpt.pt)`, and the `done.flag` write.

> The exact cell edits depend on the current `ai_crop_lot.ipynb` /
> `colab_classify_training.ipynb` cell shapes (the `model.train(...)` call form, the
> classifier loop structure). The implementer must read those notebooks before writing
> the transforms — regex-patching the wrong shape silently produces a broken notebook,
> which the assertions are there to catch.

## Data flow

```
[user clicks "Train" in wizard]
  POST /training/full/start → publish dataset → DELETE prior run markers/checkpoints   (reset point)
        │
        ▼  [user opens notebook, Run all]
  Detector:  eval.json? → skip ; last.pt? → resume ; else fresh   (ckpt → Drive every N epochs)
        │   on done: full_detector.pt + eval.json (written last)
        ▼
  Classifier: done.flag? → skip ; ckpt.pt? → resume(stage,epoch) ; else fresh   (ckpt → Drive every N epochs)
            on done: classifier.pt + done.flag
        │
        ▼  [Colab disconnect any time] → Run all again (no button) → resumes from the above markers
        ▼  [user clicks "ดึง model"] → existing /sync → existing Deploy   (unchanged)
```

## Error / edge cases

| Condition | Behavior |
|-----------|----------|
| Runtime reset mid-detector | Re-mount + `Run all`; `last.pt` from Drive → `resume=True` (lose ≤ `CKPT_EVERY` epochs) |
| Runtime reset after detector done, mid-classifier | `eval.json` present → skip detector; `ckpt.pt` → resume classifier |
| New training run (new data) | "Train" click wiped markers → both sections train fresh (no stale skip) |
| Drive delete fails at publish | Non-fatal: log warning, publish still succeeds (worst case: a wrong skip next run) |
| Drive checkpoint copy fails mid-run | Non-fatal per epoch: log + continue; next checkpoint interval retries |
| `CKPT_EVERY` larger than total epochs | Still saves `last.pt`/`ckpt.pt` at completion; resume degrades to fresh if nothing was flushed |

## Testing

- **Backend** (`tests/test_api_packagings.py`): mock `DriveClient` + `packaging_store`;
  assert `/training/full/start` calls the reset/delete for `eval.json`,
  `_training/detector/`, `_training/classifier/` after publish; assert a delete error is
  swallowed (publish still returns success).
- **Build script** (`scripts/build_full_training_notebook.py` self-assertions, runnable in
  CI without Colab): the built notebook contains the skip guards, `resume=True`,
  checkpoint-copy callback / `torch.save(... ckpt.pt)`, and `done.flag` write.
- **Manual Colab validation** (the real test — cannot be automated): (a) interrupt
  detector mid-run, `Run all`, confirm it resumes near the last checkpoint; (b) interrupt
  classifier mid-run, confirm detector is skipped and classifier resumes; (c) click
  "Train" again with changed data, confirm both retrain from scratch.

## Out of scope

- `/sync` and Deploy flow — unchanged; `eval.json` stays the detector done-signal and the
  `/sync` contract is untouched (`2026-06-22-train-done-sync-deploy-design.md`).
- Training recipe / hyperparameters — unchanged (same epochs, imgsz, augmentation,
  2-stage schedule). This work only adds checkpoint/resume/skip plumbing around them.
- Idle keep-alive to prevent free-tier disconnects — not reliably possible; resume is the
  answer, not prevention.
- `model_registry` / `manifest.json` path (production runs baked-in image models).
- `multi_field` prelabel per-field tagging (unrelated deferred gap).
