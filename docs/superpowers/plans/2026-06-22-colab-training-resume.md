# Colab Full Training Resume & Skip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Colab Full Training resumable — a crashed/disconnected run continues from the last checkpoint instead of retraining the detector + classifier from scratch.

**Architecture:** Three changes. (1) Backend — `POST /training/full/start` deletes the previous run's Drive markers after publishing, making the "Train" click the reset boundary. (2) Notebook build script (`scripts/build_full_training_notebook.py`) injects skip/resume/checkpoint logic into the detector + classifier cells it assembles. (3) DriveClient gains a `delete_file` method the reset needs. The notebook itself is regenerated and re-uploaded, then validated manually in Colab (cannot be CI-tested).

**Tech Stack:** Python 3.11, FastAPI, Google Drive API v3, Ultralytics YOLO, PyTorch (EfficientNet-V2-S), pytest. Notebook runs on Colab Free tier (GPU).

## Global Constraints

- `print()` is forbidden in `services/`/`api/` — use the module `logger`. (Generated **notebook** cells are exempt — Colab cells legitimately use `print`.)
- `pytest` is not on PATH — run `python -m pytest`.
- Read source files with `encoding="utf-8"` (they contain Thai).
- Reset boundary rule (verbatim from spec): **pressing "Train" = new run** (backend wipes markers); **re-running `Run all` without the button = resume** (notebook finds checkpoints).
- Drive wipe set (verbatim): delete `data check lot/eval.json`, `data check lot/_training/detector/`, `data classify check lot/_training/classifier/`. **Never** delete `full_detector.pt` or `models/classifier.pt` (baselines).
- A Drive delete failure at reset is **non-fatal** — log a warning, let publish succeed.
- `CKPT_EVERY = 10` — mirror checkpoints to Drive every 10 epochs (lose at most 10 on a crash).
- Detector done-signal stays `eval.json` (unchanged `/sync` contract). Classifier done-signal is a separate `_training/classifier/done.flag`.
- Env vars: detector Drive folder id from `DRIVE_DETECTOR_DATASET_FOLDER_ID`, classifier from `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`.

---

## File Structure

- `services/drive_client.py` — **modify**: add `delete_file(file_id)`.
- `services/dataset_publisher.py` — **modify**: add `reset_training_state(drive, det_root, cls_root)`.
- `api/packagings.py` — **modify**: call `reset_training_state` after publish in `training_full_start`; update `COMBINED_NOTEBOOK_FILE_ID` (Task 5).
- `scripts/build_full_training_notebook.py` — **modify**: refactor `main()` to expose a pure `build_notebook() -> dict`; inject detector + classifier resilience.
- `tests/test_drive_client.py` — **modify**: add `delete_file` test.
- `tests/test_dataset_publisher.py` — **modify**: add `reset_training_state` tests.
- `tests/test_api_packagings.py` — **modify**: add reset-is-called-on-start test.
- `tests/test_build_full_training_notebook.py` — **modify**: assert built notebook contains resilience code.

---

### Task 1: DriveClient.delete_file

**Files:**
- Modify: `services/drive_client.py` (add method after `find_in_folder`, ~line 221)
- Test: `tests/test_drive_client.py`

**Interfaces:**
- Produces: `DriveClient.delete_file(file_id: str) -> None` — permanently deletes a Drive file *or folder* (folder delete cascades to contents) via `files().delete()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_drive_client.py`:

```python
def test_delete_file_calls_files_delete(client_and_svc):
    client, svc = client_and_svc
    client.delete_file("fid-to-go")
    _, kwargs = svc.files.return_value.delete.call_args
    assert kwargs["fileId"] == "fid-to-go"
    svc.files.return_value.delete.return_value.execute.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drive_client.py::test_delete_file_calls_files_delete -v`
Expected: FAIL — `AttributeError: 'DriveClient' object has no attribute 'delete_file'`

- [ ] **Step 3: Write minimal implementation**

In `services/drive_client.py`, add after `find_in_folder` (the last method, ~line 221):

```python
    def delete_file(self, file_id: str) -> None:
        """Permanently delete a Drive file or folder (folder delete cascades to
        its contents). Skips trash."""
        self._svc.files().delete(fileId=file_id).execute()
        logger.info("Deleted Drive file %s", file_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drive_client.py::test_delete_file_calls_files_delete -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/drive_client.py tests/test_drive_client.py
git commit -m "feat: DriveClient.delete_file (file/folder delete)"
```

---

### Task 2: reset_training_state + wire into /training/full/start

**Files:**
- Modify: `services/dataset_publisher.py` (add function + module constants)
- Modify: `api/packagings.py` (`training_full_start`, after publish succeeds ~line 553)
- Test: `tests/test_dataset_publisher.py`, `tests/test_api_packagings.py`

**Interfaces:**
- Consumes: `DriveClient.delete_file` (Task 1), `DriveClient.find_in_folder`.
- Produces: `dataset_publisher.reset_training_state(drive, det_root: str, cls_root: str) -> dict` — deletes `eval.json` + `_training/` markers; returns `{"deleted": list[str]}`; never raises.

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/test_dataset_publisher.py`:

```python
from unittest.mock import MagicMock
from services import dataset_publisher


def test_reset_training_state_deletes_markers():
    drive = MagicMock()
    # eval.json in det, _training in det, _training in cls — all found
    drive.find_in_folder.side_effect = lambda parent, name: f"{parent}:{name}"
    out = dataset_publisher.reset_training_state(drive, "DET", "CLS")
    deleted_ids = [c.args[0] for c in drive.delete_file.call_args_list]
    assert "DET:eval.json" in deleted_ids
    assert "DET:_training" in deleted_ids
    assert "CLS:_training" in deleted_ids
    assert set(out["deleted"]) == {"eval.json", "_training", "_training"} - set()  # 3 entries
    assert len(out["deleted"]) == 3


def test_reset_training_state_skips_missing():
    drive = MagicMock()
    drive.find_in_folder.return_value = None  # nothing on Drive yet
    out = dataset_publisher.reset_training_state(drive, "DET", "CLS")
    drive.delete_file.assert_not_called()
    assert out["deleted"] == []


def test_reset_training_state_swallows_errors():
    drive = MagicMock()
    drive.find_in_folder.return_value = "some-id"
    drive.delete_file.side_effect = RuntimeError("Drive 503")
    # must NOT raise — reset is best-effort
    out = dataset_publisher.reset_training_state(drive, "DET", "CLS")
    assert out["deleted"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset_publisher.py -k reset_training_state -v`
Expected: FAIL — `AttributeError: module 'services.dataset_publisher' has no attribute 'reset_training_state'`

- [ ] **Step 3: Write minimal implementation**

In `services/dataset_publisher.py`, add near the top constants (after line 29):

```python
# Markers the notebook writes per training run. Deleting them on a new "Train"
# click is the reset boundary (see 2026-06-22-colab-training-resume spec).
_RESET_TARGETS = [("det", "eval.json"), ("det", "_training"), ("cls", "_training")]
```

And add this function (e.g. after `merge_class_names`):

```python
def reset_training_state(drive, det_root: str, cls_root: str) -> dict:
    """Delete the previous run's Drive markers/checkpoints so a fresh 'Train'
    click starts from scratch. Best-effort: a delete failure is logged and
    skipped — a stale marker only risks a wrong skip next run; it must never
    fail the (user-visible) publish. Baselines full_detector.pt /
    models/classifier.pt are intentionally NOT touched.
    """
    roots = {"det": det_root, "cls": cls_root}
    deleted: list[str] = []
    for which, name in _RESET_TARGETS:
        parent = roots[which]
        try:
            fid = drive.find_in_folder(parent, name)
            if fid:
                drive.delete_file(fid)
                deleted.append(name)
        except Exception:
            logger.warning("reset_training_state: failed to delete %s/%s", which, name, exc_info=True)
    logger.info("reset_training_state deleted: %s", deleted)
    return {"deleted": deleted}
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/test_dataset_publisher.py -k reset_training_state -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire into the endpoint**

In `api/packagings.py` `training_full_start`, immediately after `progress_store.report(key, "done")` (currently line 554, right after the publish `try/except` block) and before `packaging_store.update_draft(...)`, add:

```python
    # Reset boundary: this "Train" click starts a new run. Wipe the previous
    # run's Drive markers/checkpoints so the notebook trains fresh (not resume).
    try:
        dataset_publisher.reset_training_state(
            drive,
            det_root=os.getenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", ""),
            cls_root=os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", ""),
        )
    except Exception:
        logger.warning("reset_training_state raised for %s (non-fatal)", key, exc_info=True)
```

Verify `os` is imported at the top of `api/packagings.py`; if not, add `import os`.

- [ ] **Step 6: Write the failing endpoint test**

Add to `tests/test_api_packagings.py`:

```python
def test_training_full_start_resets_training_state(client, monkeypatch):
    from services import dataset_publisher
    import api.packagings as pkg

    monkeypatch.setattr(pkg.packaging_store, "get_draft",
                        lambda k: {"key": k, "detection_mode": "single", "sub_regions": ["lot"]})
    monkeypatch.setattr(pkg.packaging_store, "list_annotation_status",
                        lambda k: [{"labeled": True} for _ in range(30)])
    monkeypatch.setattr(pkg.packaging_store, "update_draft", lambda *a, **k: None)
    monkeypatch.setattr("services.drive_client.DriveClient", lambda: object())
    monkeypatch.setattr(dataset_publisher, "publish", lambda *a, **k: {"images_uploaded": 5})
    calls = []
    monkeypatch.setattr(dataset_publisher, "reset_training_state",
                        lambda *a, **k: calls.append(True) or {"deleted": []})

    r = client.post("/api/packagings/test_box/training/full/start")
    assert r.status_code == 200
    assert calls == [True], "reset_training_state must be called once after publish"
```

- [ ] **Step 7: Run the endpoint test to verify it passes**

Run: `python -m pytest tests/test_api_packagings.py::test_training_full_start_resets_training_state -v`
Expected: PASS

- [ ] **Step 8: Run the full affected suites**

Run: `python -m pytest tests/test_dataset_publisher.py tests/test_api_packagings.py tests/test_drive_client.py -v`
Expected: PASS (no regressions)

- [ ] **Step 9: Commit**

```bash
git add services/dataset_publisher.py api/packagings.py tests/test_dataset_publisher.py tests/test_api_packagings.py
git commit -m "feat: reset training markers on /training/full/start (resume boundary)"
```

---

### Task 3: Build script — refactor + detector resilience

**Files:**
- Modify: `scripts/build_full_training_notebook.py`
- Test: `tests/test_build_full_training_notebook.py`

**Interfaces:**
- Produces: `build_notebook() -> dict` — pure function returning the assembled notebook dict (no Drive upload), so tests can assert on cell sources. `main()` calls it, writes `OUT`, then uploads.
- The detector section emits a preflight cell defining `DET_SKIP`/`DET_RESUME`/`CKPT_EVERY`/`_copy_det_ckpt`, a train cell branching skip/resume/fresh, and val/save cells guarded by `if not DET_SKIP:`.

Context — the lifted detector cells (from `ai_crop_lot.ipynb`, after the existing `data.yaml`-drop) are, in order: a mount/install cell; the train cell (`results = model.train(... name='ai_check lot v3' ...)`); the val cell (`best_model = YOLO('/content/runs/detect/ai_check lot v3/weights/best.pt')`); the save cell (`source_model = '/content/runs/detect/ai_check lot v3/weights/best.pt'` → `full_detector.pt` + `eval.json`).

- [ ] **Step 1: Refactor main() to expose build_notebook() (no behavior change yet)**

In `scripts/build_full_training_notebook.py`, split `main()`. Replace the body from `det = detector_cells()` through `OUT.write_text(...)` with a new pure function, and keep upload in `main()`:

```python
def build_notebook() -> dict:
    det = detector_cells()
    cls = classifier_cells()
    assert len(det) >= 2, f"expected >=2 detector cells, got {len(det)}"
    assert len(cls) >= 4, f"expected >=4 classifier cells, got {len(cls)}"
    assert any("iterdir()" in _src(c) for c in cls), "CLASSES auto-discover not applied"
    det_src = "\n".join(_src(c) for c in det)
    assert "full_detector.pt" in det_src and "eval.json" in det_src, \
        "detector cells must save full_detector.pt + write eval.json"

    cells = (
        [_md("# Full Training — Detector + Classifier\n\n"
             "กด `Runtime → Run all`. ถ้า Colab หลุด ให้เปิดแล้ว Run all ใหม่ "
             "(จะ resume ต่อจาก checkpoint ล่าสุดเอง).")]
        + [_md("## ── Detector (YOLO) ──")] + det
        + [_md("## ── Classifier (EfficientNet-V2-S) ──")] + cls
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    load_dotenv()
    nb = build_notebook()
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("wrote %s (%d cells)", OUT, len(nb["cells"]))

    from services.drive_client import DriveClient
    drive = DriveClient()
    file_id = drive.upload_bytes(
        OUT.read_bytes(), name=OUT.name, parent_id="root",
        mime_type="application/vnd.google.colaboratory",
    )
    logger.info("UPLOADED — file_id=%s", file_id)
    logger.info("colab: https://colab.research.google.com/drive/%s", file_id)
```

- [ ] **Step 2: Verify refactor is green**

Run: `python -m pytest tests/test_build_full_training_notebook.py -v`
Expected: PASS (existing test reads the source notebook, still passes)

- [ ] **Step 3: Write the failing detector-resilience test**

Add to `tests/test_build_full_training_notebook.py`:

```python
from scripts.build_full_training_notebook import build_notebook


def _all_code(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_detector_section_has_skip_resume_checkpoint():
    code = _all_code(build_notebook())
    assert "DET_SKIP" in code and "DET_RESUME" in code, "detector skip/resume flags missing"
    assert "CKPT_EVERY" in code, "checkpoint cadence constant missing"
    assert "resume=True" in code, "detector resume call missing"
    assert "_copy_det_ckpt" in code and "on_fit_epoch_end" in code, "checkpoint callback missing"
    assert "if not DET_SKIP" in code, "val/save cells must be guarded by DET_SKIP"
```

Note: `scripts/` needs to be importable — `tests/conftest.py` or the test adds `sys.path.insert(0, repo_root)`. Check whether other tests import from `scripts.`; if `import scripts...` fails, add at the top of the test file: `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))` and ensure `scripts/__init__.py` exists (create an empty one if absent).

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest tests/test_build_full_training_notebook.py::test_detector_section_has_skip_resume_checkpoint -v`
Expected: FAIL — flags not present

- [ ] **Step 5: Implement the detector preflight cell + transforms**

In `scripts/build_full_training_notebook.py`, add the preflight template and rewrite `detector_cells()` to patch + wrap. Add near the top constants:

```python
DET_PREFLIGHT = r'''# ── Resilience config (detector + classifier) ──────────────────────────────
import os, shutil
from pathlib import Path

CKPT_EVERY = 10  # mirror checkpoints to Drive every N epochs (lose <= N on a crash)

DET_ROOT  = Path('/content/drive/MyDrive/data check lot')
DET_EVAL  = DET_ROOT / 'eval.json'                  # detector done-signal
DET_DRIVE = DET_ROOT / '_training' / 'detector'     # durable checkpoint dir on Drive
DET_LOCAL = Path('/content/_det')                   # fast local YOLO project dir
DET_RUN   = 'train'                                 # fixed run name (resume needs a stable path)
DET_DRIVE_LAST = DET_DRIVE / DET_RUN / 'weights' / 'last.pt'
DET_LOCAL_LAST = DET_LOCAL / DET_RUN / 'weights' / 'last.pt'
DET_LOCAL_BEST = DET_LOCAL / DET_RUN / 'weights' / 'best.pt'

DET_SKIP   = DET_EVAL.exists()
DET_RESUME = (not DET_SKIP) and DET_DRIVE_LAST.exists()
print(f'detector: skip={DET_SKIP} resume={DET_RESUME}')

def _copy_det_ckpt(trainer):
    """on_fit_epoch_end → mirror last.pt/best.pt to Drive every CKPT_EVERY epochs."""
    try:
        ep = int(getattr(trainer, 'epoch', 0)) + 1
        if ep % CKPT_EVERY != 0:
            return
        wdir = Path(trainer.save_dir) / 'weights'
        dst = DET_DRIVE / DET_RUN / 'weights'
        dst.mkdir(parents=True, exist_ok=True)
        for f in ('last.pt', 'best.pt'):
            if (wdir / f).exists():
                shutil.copy(wdir / f, dst / f)
        print(f'  [ckpt] mirrored detector weights to Drive @ epoch {ep}')
    except Exception as e:
        print('  [ckpt] detector mirror failed (non-fatal):', e)
'''
```

Add a helper that wraps the lifted train cell. Place these module-level helpers:

```python
import re as _re  # already have `re`; reuse the existing import instead of re-importing

_TRAIN_KWARGS_RE = re.compile(r"model\.train\((.*?)\)\s*$", re.DOTALL | re.MULTILINE)


def _detector_train_cell(train_src: str) -> str:
    """Wrap the lifted YOLO train cell with skip / resume / fresh branches.

    Fresh branch reuses the lifted train kwargs (epochs/imgsz/augmentation =
    source of truth) but patches name=DET_RUN and injects project=DET_LOCAL +
    the checkpoint callback. Resume branch restores Drive's last.pt then
    train(resume=True).
    """
    m = _TRAIN_KWARGS_RE.search(train_src)
    assert m, "could not locate model.train(...) in detector train cell"
    kwargs = m.group(1)
    # patch run name → fixed; drop any existing project=
    kwargs = re.sub(r"name\s*=\s*['\"][^'\"]*['\"]", "name=DET_RUN", kwargs)
    kwargs = re.sub(r"project\s*=\s*[^,]+,", "", kwargs)
    fresh = (
        "    from ultralytics import YOLO\n"
        "    model = YOLO('yolo11s.pt')\n"
        "    model.add_callback('on_fit_epoch_end', _copy_det_ckpt)\n"
        "    results = model.train(\n"
        f"        project=str(DET_LOCAL),\n"
        f"{textwrap.indent(kwargs.strip(), '        ')}\n"
        "    )\n"
    )
    return (
        "if DET_SKIP:\n"
        "    print('detector already trained this run (eval.json exists) — skip')\n"
        "elif DET_RESUME:\n"
        "    from ultralytics import YOLO\n"
        "    DET_LOCAL_LAST.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy(DET_DRIVE_LAST, DET_LOCAL_LAST)\n"
        "    print('resuming detector from Drive checkpoint')\n"
        "    model = YOLO(str(DET_LOCAL_LAST))\n"
        "    model.add_callback('on_fit_epoch_end', _copy_det_ckpt)\n"
        "    results = model.train(resume=True)\n"
        "else:\n"
        f"{fresh}"
        "\n"
        "# final mirror — guarantee Drive has the last/best even if the last epoch\n"
        "# was not a multiple of CKPT_EVERY\n"
        "if not DET_SKIP:\n"
        "    _wd = DET_DRIVE / DET_RUN / 'weights'\n"
        "    _wd.mkdir(parents=True, exist_ok=True)\n"
        "    for _f in ('last.pt', 'best.pt'):\n"
        "        _p = DET_LOCAL / DET_RUN / 'weights' / _f\n"
        "        if _p.exists():\n"
        "            shutil.copy(_p, _wd / _f)\n"
    )
```

Add `import textwrap` at the top of the file. Now rewrite `detector_cells()` to patch the val/save path strings, guard them, and splice in the preflight + wrapped train cell:

```python
def detector_cells() -> list[dict]:
    nb = json.loads(DET_NB.read_text(encoding="utf-8"))
    raw = [_src(c) for c in nb["cells"]
           if c["cell_type"] == "code"
           and not ("yaml_content" in _src(c) and "data.yaml" in _src(c))]
    out = []
    for src in raw:
        if "model.train(" in src:                       # train cell → wrap
            out.append(_code(DET_PREFLIGHT))
            out.append(_code(_detector_train_cell(src)))
            continue
        # patch the hardcoded best.pt path (val + save cells)
        src = src.replace(
            "/content/runs/detect/ai_check lot v3/weights/best.pt",
            "%BEST%",  # placeholder so the replacement is unambiguous
        ).replace("'%BEST%'", "str(DET_LOCAL_BEST)").replace('"%BEST%"', "str(DET_LOCAL_BEST)")
        if "metrics = best_model.val()" in src or "source_model" in src:
            src = "if not DET_SKIP:\n" + textwrap.indent(src, "    ")
        out.append(_code(src))
    return out
```

> Implementation note: confirm by reading `ai_crop_lot.ipynb` that the best.pt path string is byte-identical to the literal above (it currently is). If the source quotes differ, adjust the `.replace` accordingly — the build assertions in Step 3 + Step 7 catch a miss.

- [ ] **Step 6: Run the detector-resilience test**

Run: `python -m pytest tests/test_build_full_training_notebook.py::test_detector_section_has_skip_resume_checkpoint -v`
Expected: PASS

- [ ] **Step 7: Verify the built notebook is valid JSON + Python-parseable**

Add this test (catches indentation/splice bugs the regex could introduce):

```python
import ast

def test_built_notebook_cells_are_valid_python():
    nb = build_notebook()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # strip Colab shell-magics (lines starting with ! or %) before parsing
        clean = "\n".join("" if l.lstrip().startswith(("!", "%")) else l
                          for l in src.splitlines())
        ast.parse(clean)  # raises SyntaxError on a bad splice
```

Run: `python -m pytest tests/test_build_full_training_notebook.py -v`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add scripts/build_full_training_notebook.py tests/test_build_full_training_notebook.py
git commit -m "feat: detector skip/resume/checkpoint in full-training notebook build"
```

---

### Task 4: Build script — classifier resilience

**Files:**
- Modify: `scripts/build_full_training_notebook.py`
- Test: `tests/test_build_full_training_notebook.py`

**Interfaces:**
- Consumes: the classifier config cell defines `MODEL_OUT`, `EPOCHS`, `CLASSES`; cells define `build_model`, `freeze_backbone`, `unfreeze_all`, `run_epoch`, and (in the split cell) `train_loader`, `val_loader`, `criterion`, `LR`, `LR_FINETUNE`, `WEIGHT_DECAY`, `STAGE1_EPOCHS`, `STAGE2_EPOCHS`. The replacement loop reuses all of these names so the recipe (hyperparameters, augmentation, model) stays sourced from the lifted cells.
- Produces: a classifier preamble cell (`CLS_SKIP`/`CLS_DONE`/`CLS_CKPT`/`done.flag` paths) and a replaced orchestration cell with a 2-stage checkpoint/resume loop. The confusion-matrix cell is guarded by `if not CLS_SKIP:`.

Context — the classifier orchestration cell (lifted from `colab_classify_training.ipynb`) is the one containing both `=== Stage 1` and `for epoch in range(STAGE1_EPOCHS)`. It currently: builds dataset/split/sampler/loaders, `model`, `criterion`, sets `best_val_acc=0.0`, `STAGE1_EPOCHS`/`STAGE2_EPOCHS`, runs the Stage-1 head loop then the Stage-2 fine-tune loop, saving best to `MODEL_OUT` each improvement.

- [ ] **Step 1: Write the failing classifier-resilience test**

Add to `tests/test_build_full_training_notebook.py`:

```python
def test_classifier_section_has_skip_resume_checkpoint():
    code = _all_code(build_notebook())
    assert "CLS_SKIP" in code and "CLS_CKPT" in code, "classifier skip/checkpoint flags missing"
    assert "done.flag" in code, "classifier done-flag missing"
    assert "torch.load(CLS_CKPT" in code, "classifier resume load missing"
    assert "'stage'" in code or '"stage"' in code, "checkpoint must record 2-stage marker"
    assert "if not CLS_SKIP" in code, "confusion-matrix cell must be guarded by CLS_SKIP"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_build_full_training_notebook.py::test_classifier_section_has_skip_resume_checkpoint -v`
Expected: FAIL

- [ ] **Step 3: Add the classifier preamble + replacement loop templates**

In `scripts/build_full_training_notebook.py`, add:

```python
CLS_PREAMBLE = r'''# ── Classifier resilience config ───────────────────────────────────────────
CLS_ROOT  = Path('/content/drive/MyDrive/data classify check lot')
CLS_WORK  = CLS_ROOT / '_training' / 'classifier'
CLS_CKPT  = CLS_WORK / 'ckpt.pt'      # {epoch, stage, model, optim, sched, best_val_acc}
CLS_DONE  = CLS_WORK / 'done.flag'    # classifier done-signal (this run)
CLS_WORK.mkdir(parents=True, exist_ok=True)
CLS_SKIP  = CLS_DONE.exists()
print(f'classifier: skip={CLS_SKIP} resume={CLS_CKPT.exists() and not CLS_SKIP}')
'''

# Replacement for the lifted orchestration cell. Reuses names defined by the
# other (lifted) classifier cells, so the recipe stays source-of-truth; only
# the loop orchestration gains checkpoint/resume.
CLS_LOOP = r'''import numpy as np
from sklearn.model_selection import train_test_split
import torch
torch.manual_seed(SEED)

if CLS_SKIP:
    print('classifier already trained this run (done.flag) — skip')
else:
    full_dataset = LotImageDataset(IMAGES_DIR, CLASSES, transform=TRAIN_TRANSFORMS)
    all_indices  = list(range(len(full_dataset)))
    all_labels   = [full_dataset.samples[i][1] for i in all_indices]
    train_indices, val_indices = train_test_split(
        all_indices, test_size=VAL_SPLIT, stratify=all_labels, random_state=SEED)
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_full = LotImageDataset(IMAGES_DIR, CLASSES, transform=VAL_TRANSFORMS)
    val_ds   = torch.utils.data.Subset(val_full, val_indices)

    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    class_counts = torch.zeros(len(CLASSES))
    for lbl in train_labels:
        class_counts[lbl] += 1
    class_weights  = 1.0 / class_counts.clamp(min=1)
    sample_weights = torch.tensor([class_weights[lbl] for lbl in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    model = build_model(num_classes=len(CLASSES)).to(DEVICE)

    STAGE1_EPOCHS = EPOCHS // 2
    STAGE2_EPOCHS = EPOCHS - STAGE1_EPOCHS
    best_val_acc, patience_counter = 0.0, 0

    # ── resume state ──
    start_stage, start_epoch = 'freeze', 0
    if CLS_CKPT.exists():
        st = torch.load(CLS_CKPT, map_location=DEVICE)
        model.load_state_dict(st['model'])
        best_val_acc = st['best_val_acc']
        start_stage  = st['stage']
        start_epoch  = st['epoch'] + 1
        print(f'resuming classifier from stage={start_stage} epoch={start_epoch} best={best_val_acc:.3f}')

    def _save_ckpt(stage, epoch, optim, sched):
        torch.save({'stage': stage, 'epoch': epoch, 'best_val_acc': best_val_acc,
                    'model': model.state_dict(), 'optim': optim.state_dict(),
                    'sched': (sched.state_dict() if sched else None)}, CLS_CKPT)

    def _save_best():
        torch.save({'model_state': model.state_dict(), 'classes': CLASSES}, MODEL_OUT)

    # ── Stage 1: head only ──
    if start_stage == 'freeze':
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        if CLS_CKPT.exists() and start_epoch:
            st = torch.load(CLS_CKPT, map_location=DEVICE)
            if st.get('optim'):
                optimizer.load_state_dict(st['optim'])
        for epoch in range(start_epoch, STAGE1_EPOCHS):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, 'train')
            val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      'val')
            print(f'[S1] {epoch+1}/{STAGE1_EPOCHS} val_acc={val_acc:.3f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc; _save_best()
            if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == STAGE1_EPOCHS:
                _save_ckpt('freeze', epoch, optimizer, None)
        start_epoch = 0  # entering stage 2 from the top

    # ── Stage 2: fine-tune all (OneCycleLR) ──
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_FINETUNE * 10, steps_per_epoch=len(train_loader),
        epochs=STAGE2_EPOCHS, pct_start=0.3)
    if start_stage == 'finetune' and CLS_CKPT.exists():
        st = torch.load(CLS_CKPT, map_location=DEVICE)
        if st.get('optim'):  optimizer.load_state_dict(st['optim'])
        if st.get('sched'):  scheduler.load_state_dict(st['sched'])
    s2_start = start_epoch if start_stage == 'finetune' else 0
    for epoch in range(s2_start, STAGE2_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, 'train')
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      'val')
        scheduler.step()
        print(f'[S2] {epoch+1}/{STAGE2_EPOCHS} val_acc={val_acc:.3f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc; _save_best()
        if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == STAGE2_EPOCHS:
            _save_ckpt('finetune', epoch, optimizer, scheduler)

    # done — write the flag LAST (classifier done-signal for this run)
    CLS_DONE.write_text('ok', encoding='utf-8')
    print(f'classifier done. best val acc={best_val_acc:.3f} → {MODEL_OUT}')
'''
```

> Note on early-stopping: the resilient loop drops the original `patience`/early-stop break to keep the resume bookkeeping simple (free-tier runs are short; OneCycle already regularizes). If early-stop must be preserved, store `patience_counter` in the checkpoint too — out of scope unless requested.

- [ ] **Step 4: Splice the classifier templates into classifier_cells()**

Rewrite `classifier_cells()`:

```python
def classifier_cells() -> list[dict]:
    nb = json.loads(CLS_NB.read_text(encoding="utf-8"))
    out = []
    preamble_done = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _src(c)
        if "drive.mount" in src or src.strip().startswith("!pip install"):
            continue
        if CLASSES_RE.search(src):
            src = CLASSES_RE.sub(CLASSES_REPLACEMENT, src)
            out.append(_code(src))
            out.append(_code(CLS_PREAMBLE))     # right after config cell (MODEL_OUT/EPOCHS defined)
            preamble_done = True
            continue
        if "=== Stage 1" in src and "for epoch in range(STAGE1_EPOCHS)" in src:
            out.append(_code(CLS_LOOP))         # replace orchestration loop
            continue
        if "confusion_matrix" in src and "classification_report" in src:
            src = "if not CLS_SKIP:\n" + textwrap.indent(src, "    ")
        out.append(_code(src))
    assert preamble_done, "classifier config cell (CLASSES=...) not found — preamble not inserted"
    return out
```

- [ ] **Step 5: Run the classifier-resilience test + the validity test**

Run: `python -m pytest tests/test_build_full_training_notebook.py -v`
Expected: PASS (all, including `test_built_notebook_cells_are_valid_python` and `test_classifier_section_has_skip_resume_checkpoint`)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_full_training_notebook.py tests/test_build_full_training_notebook.py
git commit -m "feat: classifier 2-stage skip/resume/checkpoint in notebook build"
```

---

### Task 5: Rebuild, upload, wire constant, manual Colab validation

**Files:**
- Modify: `api/packagings.py:48` (`COMBINED_NOTEBOOK_FILE_ID`)
- Produces: regenerated `lot-checker-full-training.ipynb` uploaded to Drive.

This task has **no automated test** — the notebook can only be validated by running it on Colab. The build assertions (Tasks 3–4) are the automated gate; this task produces and manually verifies the artifact.

- [ ] **Step 1: Run the full test suite first**

Run: `python -m pytest tests/test_build_full_training_notebook.py tests/test_drive_client.py tests/test_dataset_publisher.py tests/test_api_packagings.py -v`
Expected: PASS. (Pre-existing `tests/test_classifier.py` 3 setup errors are unrelated — see CLAUDE.md.)

- [ ] **Step 2: Build + upload the notebook**

Requires Drive OAuth env (`.env` with `DRIVE_OAUTH_*`). Run from repo root:

Run: `python scripts/build_full_training_notebook.py`
Expected: logs `wrote lot-checker-full-training.ipynb (N cells)` then `UPLOADED — file_id=<NEW_ID>` and a `colab: https://...` link. Copy `<NEW_ID>`.

- [ ] **Step 3: Update the constant**

In `api/packagings.py:48`, set `COMBINED_NOTEBOOK_FILE_ID = "<NEW_ID>"`.

- [ ] **Step 4: Commit code + regenerated notebook**

```bash
git add api/packagings.py lot-checker-full-training.ipynb
git commit -m "chore: rebuild full-training notebook (resume-enabled) + point constant at it"
```

- [ ] **Step 5: Manual Colab validation (record results in the PR / bug_fix.md)**

Open the new Colab link and verify, in order:

1. **Fresh run** — with no `_training/` or `eval.json` on Drive: `Run all`. Preflight prints `detector: skip=False resume=False`; detector trains; `full_detector.pt` + `eval.json` appear on Drive; classifier prints `skip=False`; `classifier.pt` + `_training/classifier/done.flag` appear.
2. **Resume detector** — interrupt the runtime mid-detector (Runtime → Disconnect after ≥`CKPT_EVERY` epochs, confirm `_training/detector/train/weights/last.pt` exists on Drive). Reconnect, `Run all`. Preflight prints `resume=True`; training resumes near the last mirrored epoch (not epoch 0).
3. **Skip detector / resume classifier** — with `eval.json` present, interrupt mid-classifier (after a `ckpt.pt` write). Reconnect, `Run all`. Detector prints `skip`; classifier prints `resuming classifier from stage=… epoch=…`.
4. **New run resets** — in the wizard, click **Train** again (this calls `/training/full/start` → publish + `reset_training_state`). Confirm on Drive that `eval.json` + both `_training/` dirs are gone. `Run all` now trains both from scratch (no stale skip).

- [ ] **Step 6: If validation surfaces a notebook bug**

Fix in `scripts/build_full_training_notebook.py` (never hand-edit `lot-checker-full-training.ipynb` — it is generated), re-run Tasks 3–5 build + test, re-upload, update the constant, re-validate.

---

## Self-Review

**Spec coverage:**
- §1 backend reset → Task 2 (`reset_training_state` + endpoint wiring) + Task 1 (`delete_file` it depends on). ✓
- §2 Drive layout (`_training/`, baselines preserved) → encoded in preamble cells (Task 3/4) + reset targets (Task 2). ✓
- §3 detector skip/resume + CKPT_EVERY mirror → Task 3. ✓
- §4 classifier 2-stage skip/resume → Task 4. ✓
- §5 build-script bake-in + assertions → Tasks 3–4 (`build_notebook`, resilience tests, valid-Python test). ✓
- Data flow / reset-on-Train-click → Task 2 endpoint + Task 5 manual step 4. ✓
- Testing (backend mocks, build assertions, manual Colab) → Tasks 2, 3, 4, 5. ✓
- Out-of-scope (`/sync`, recipe, keep-alive) → untouched; `eval.json` contract preserved (Task 3 keeps the save/eval cell). ✓

**Placeholder scan:** No TBD/TODO. The only `%BEST%` token is a deliberate intermediate sentinel inside a documented `.replace` chain, immediately consumed. Manual-validation values (`<NEW_ID>`) are produced by Step 2 and consumed in Step 3.

**Type consistency:** `reset_training_state(drive, det_root, cls_root) -> {"deleted": [...]}` defined in Task 2, called with keyword args matching in the endpoint. `delete_file(file_id)` (Task 1) consumed by `reset_training_state` (Task 2). `build_notebook() -> dict` defined in Task 3 Step 1, consumed by tests in Tasks 3–4. Notebook names `DET_SKIP/DET_RESUME/DET_RUN/DET_LOCAL/_copy_det_ckpt` (Task 3) and `CLS_SKIP/CLS_CKPT/CLS_DONE` (Task 4) are consistent between their preamble definitions and use sites.
