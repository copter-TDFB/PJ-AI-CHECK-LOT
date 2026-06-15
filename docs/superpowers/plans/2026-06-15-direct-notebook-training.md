# Direct-Notebook Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wizard's *generated* training notebook with one combined, proven Drive notebook, and fix the two dataset-upload format mismatches so the published data lands where that notebook reads it.

**Architecture:** The dataset publisher keeps writing annotated images straight into the Drive reference dataset, but (a) classifier copies move under `images/<key>/` and (b) detector destination paths are relativized so absolute `data.yaml` paths no longer break folder resolution. Notebook *generation* is removed; `/training/full/start` returns a single static Colab link to a hand-built combined notebook (detector recipe + classifier recipe, with class auto-discovery and no `data.yaml` overwrite). Model sync after training is fully manual (per design — `docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md`).

**Tech Stack:** Python 3.11, FastAPI, pytest, Google Drive API (`services/drive_client.py`), Colab notebooks (nbformat JSON), vanilla JS wizard (`web/wizard.html`).

---

## Karpathy guardrails (apply to every task)

- **Surgical:** change only lines that trace to this plan. Don't reformat or "improve" adjacent code. Match existing style (module `logger`, no `print`).
- **Simple:** the detector `data.yaml` keeps its absolute paths (they work in Colab); we only relativize *inside the publisher* for folder resolution. Do not add a separate data.yaml rewrite pass.
- **Verify:** every code step has a command + expected output. Loop until green.

## ⚠️ Flagged decision (confirm before Task 7)

The old flow did: Colab uploads `full_detector.pt` to a per-run Drive output folder → `/training/full/done` pulls it → `/deploy` promotes it. The manual design removes the output folder, so **`/training/full/done` auto-pull and the wizard "Deploy" path no longer have a model to fetch.** This plan's Task 7 takes the **minimal** option: neutralize `/training/full/done` (return a clear "manual sync" message instead of crashing) and change the wizard to stop calling it, showing manual instructions instead. The `/deploy` endpoint is left untouched (unused for this flow). If you want deploy kept automated, stop and revisit the spec before Task 7.

## File Structure

- `services/dataset_publisher.py` — **modify**: add `_relativize`, use it in `_resolve_dest_folders` (M2); classifier copies under `images/<key>/` (M1).
- `tests/test_dataset_publisher.py` — **modify**: add tests for `_relativize` and the `images/<key>/` destination.
- `scripts/build_full_training_notebook.py` — **create**: merge the two local notebooks → one combined `.ipynb`, upload to Drive, print the file_id.
- `api/packagings.py` — **modify**: `/training/full/start` returns a static `colab_url`; remove notebook generation + run-folder; `/training/full/done` neutralized (Task 7).
- `tests/test_api_packagings.py` — **modify**: update the two `full/start` tests (drop `notebook_generator` mocks).
- `services/notebook_generator.py` + `tests/test_notebook_generator.py` — **delete**.
- `web/wizard.html` — **modify**: step-5 copy + JS (single link, manual instructions, drop the "Pull model" call).

---

## Task 1: Relativize detector `data.yaml` paths (M2 helper)

**Files:**
- Modify: `services/dataset_publisher.py`
- Test: `tests/test_dataset_publisher.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_dataset_publisher.py` after the `labels_relpath` block:

```python
# ─── _relativize ─────────────────────────────────────────

def test_relativize_strips_absolute_drive_prefix():
    abs_path = "/content/drive/MyDrive/data check lot/train/images"
    assert dp._relativize(abs_path, "train") == "train/images"


def test_relativize_passthrough_when_already_relative():
    assert dp._relativize("val/images", "val") == "val/images"


def test_relativize_falls_back_when_split_absent():
    assert dp._relativize("/a/b/c", "train") == "a/b/c"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset_publisher.py -k relativize -v`
Expected: FAIL with `AttributeError: module 'services.dataset_publisher' has no attribute '_relativize'`

- [ ] **Step 3: Implement `_relativize`**

In `services/dataset_publisher.py`, add directly above `_resolve_dest_folders`:

```python
def _relativize(path: str, split: str) -> str:
    """Reduce a data.yaml split path to one relative to the dataset root.

    YOLO data.yaml may store absolute Colab paths
    (e.g. /content/drive/MyDrive/data check lot/train/images). Splitting that
    raw string would make _resolve_dest_folders nest folders under a bogus
    'content/drive/MyDrive/...' tree, so the published images never land in
    train/val. We keep only the segments from the '<split>' folder onward.
    """
    parts = [p for p in path.strip("/").split("/") if p not in ("", ".")]
    if split in parts:
        return "/".join(parts[parts.index(split):])
    return path.strip("/")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset_publisher.py -k relativize -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/dataset_publisher.py tests/test_dataset_publisher.py
git commit -m "feat: add _relativize for detector data.yaml absolute paths"
```

---

## Task 2: Wire `_relativize` into folder resolution (M2)

**Files:**
- Modify: `services/dataset_publisher.py:171-189` (`_resolve_dest_folders`)
- Test: `tests/test_dataset_publisher.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataset_publisher.py` in the `publish()` block:

```python
def test_publish_handles_absolute_data_yaml_paths(draft, env, fake_drive):
    """Absolute train/val paths in data.yaml must still resolve to train/val
    folders under det-root, not a 'content/drive/...' tree."""
    fake_drive.read_text.return_value = (
        "train: /content/drive/MyDrive/data check lot/train/images\n"
        "val: /content/drive/MyDrive/data check lot/val/images\n"
        "nc: 2\nnames: [old_a, old_b]\n"
    )
    dp.publish(draft, drive=fake_drive)
    # ensure_folder must be called with the split segments, never 'content'
    ensured = [c.args[0] for c in fake_drive.ensure_folder.call_args_list]
    assert "content" not in ensured
    assert "train" in ensured and "images" in ensured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_publisher.py::test_publish_handles_absolute_data_yaml_paths -v`
Expected: FAIL — `assert "content" not in ensured` fails (content IS in ensured).

- [ ] **Step 3: Apply the fix**

In `services/dataset_publisher.py`, in `_resolve_dest_folders`, replace:

```python
    train_rel = str(data_yaml.get("train") or "train/images")
    val_rel = str(data_yaml.get("val") or "val/images")
```

with:

```python
    train_rel = _relativize(str(data_yaml.get("train") or "train/images"), "train")
    val_rel = _relativize(str(data_yaml.get("val") or "val/images"), "val")
```

- [ ] **Step 4: Run the full publisher suite**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: all passed (new test green, no regressions).

- [ ] **Step 5: Commit**

```bash
git add services/dataset_publisher.py tests/test_dataset_publisher.py
git commit -m "fix: resolve detector dest folders from relativized data.yaml paths"
```

---

## Task 3: Classifier copies under `images/<key>/` (M1)

**Files:**
- Modify: `services/dataset_publisher.py:120-121` (inside `publish`)
- Test: `tests/test_dataset_publisher.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataset_publisher.py` in the `publish()` block:

```python
def test_publish_classifier_copies_land_under_images_subfolder(draft, env, fake_drive):
    """Classifier copies must go to images/<key>/ (where the notebook reads),
    not to <key>/ at the classifier-root top level."""
    dp.publish(draft, drive=fake_drive)
    # ensure_folder side_effect: f-{parent}-{name}
    # images under cls-root → 'f-cls-root-images'; key under that →
    # 'f-f-cls-root-images-newpack'
    cls_parent = "f-f-cls-root-images-newpack"
    cls_copies = [e for e in fake_drive.events
                  if e[0] == "upload_file" and e[1] == cls_parent]
    assert len(cls_copies) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_publisher.py::test_publish_classifier_copies_land_under_images_subfolder -v`
Expected: FAIL — copies currently land under `f-cls-root-newpack` (no `images`), so the list is empty.

- [ ] **Step 3: Apply the fix**

In `services/dataset_publisher.py` inside `publish`, replace:

```python
    cls_folder = drive.ensure_folder(key, cls_root)
```

with:

```python
    cls_images = drive.ensure_folder("images", cls_root)
    cls_folder = drive.ensure_folder(key, cls_images)
```

- [ ] **Step 4: Run the full publisher suite**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: all passed (note `test_publish_uploads_images_labels_and_classifier_copies` still expects 6 `upload_file` events — count is unchanged, only the parent moved).

- [ ] **Step 5: Commit**

```bash
git add services/dataset_publisher.py tests/test_dataset_publisher.py
git commit -m "fix: publish classifier copies under images/<key>/ to match notebook layout"
```

---

## Task 4: Build + upload the combined training notebook

**Files:**
- Create: `scripts/build_full_training_notebook.py`

This script merges the two local notebooks into one, applying the two baked-in fixes, then uploads to Drive and prints the file_id (used as a constant in Task 5).

- [ ] **Step 1: Write the builder script**

Create `scripts/build_full_training_notebook.py`:

```python
"""Build the combined Full Training notebook (detector + classifier) from the
two proven local notebooks, apply the wizard fixes, upload to Drive, print the
file_id. Run once (and again whenever the source notebooks change).

Fixes baked in:
- Detector: drop the cell that overwrites data.yaml (use publisher-maintained).
- Classifier: auto-discover CLASSES from images/<class>/ instead of a hardcoded list.

Usage: python scripts/build_full_training_notebook.py
"""

import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DET_NB = Path("ai_crop_lot.ipynb")
CLS_NB = Path("colab_classify_training.ipynb")
OUT = Path("lot-checker-full-training.ipynb")

CLASSES_RE = re.compile(r"CLASSES\s*=\s*sorted\(\[.*?\]\)", re.DOTALL)
CLASSES_REPLACEMENT = (
    "CLASSES      = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])"
)


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def detector_cells() -> list[dict]:
    nb = json.loads(DET_NB.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _src(c)
        if "yaml_content" in src and "data.yaml" in src:
            continue  # drop the data.yaml-overwrite cell
        out.append(_code(src))
    return out


def classifier_cells() -> list[dict]:
    nb = json.loads(CLS_NB.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _src(c)
        if "drive.mount" in src or src.strip().startswith("!pip install"):
            continue  # detector section already mounts + installs
        if CLASSES_RE.search(src):
            src = CLASSES_RE.sub(CLASSES_REPLACEMENT, src)
        out.append(_code(src))
    return out


def main() -> None:
    load_dotenv()
    det = detector_cells()
    cls = classifier_cells()
    # Loud guard: a silent mis-filter would otherwise ship a broken notebook.
    assert len(det) >= 2, f"expected >=2 detector cells, got {len(det)}"
    assert len(cls) >= 4, f"expected >=4 classifier cells, got {len(cls)}"
    assert any("iterdir()" in _src(c) for c in cls), "CLASSES auto-discover not applied"

    cells = (
        [_md("# Full Training — Detector + Classifier\n\n"
             "กด `Runtime → Run all`. Detector (~60 min) แล้วต่อ Classifier (~30 min).\n"
             "Models เซฟลง Drive: `data check lot/<run>.pt` + "
             "`data classify check lot/models/classifier.pt`.")]
        + [_md("## ── Detector (YOLO) ──")] + det
        + [_md("## ── Classifier (EfficientNet-V2-S) ──")] + cls
    )
    nb = {
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
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("wrote %s (%d cells)", OUT, len(cells))

    from services.drive_client import DriveClient
    drive = DriveClient()
    file_id = drive.upload_bytes(
        OUT.read_bytes(), name=OUT.name, parent_id="root",
        mime_type="application/vnd.google.colaboratory",
    )
    logger.info("UPLOADED — file_id=%s", file_id)
    logger.info("colab: https://colab.research.google.com/drive/%s", file_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the builder (local build + Drive upload)**

Run: `python scripts/build_full_training_notebook.py`
Expected: logs `wrote lot-checker-full-training.ipynb (N cells)` then `UPLOADED — file_id=<ID>` and a colab URL. **Copy the `<ID>` — it is the constant for Task 5.**

- [ ] **Step 3: Sanity-check the built notebook locally**

Run: `python -c "import json,sys; nb=json.load(open('lot-checker-full-training.ipynb',encoding='utf-8')); src=''.join(''.join(c['source']) for c in nb['cells']); assert 'iterdir()' in src; assert 'yaml_content' not in src; print('cells:', len(nb['cells']), 'OK')"`
Expected: `cells: <N> OK` (auto-discover present, data.yaml-overwrite gone).

- [ ] **Step 4: Commit the builder + built notebook**

```bash
git add scripts/build_full_training_notebook.py lot-checker-full-training.ipynb
git commit -m "feat: build combined detector+classifier training notebook"
```

---

## Task 5: `/training/full/start` returns the static notebook link

**Files:**
- Modify: `api/packagings.py:466-557`
- Test: `tests/test_api_packagings.py:753-852`

- [ ] **Step 1: Update the two failing tests**

In `tests/test_api_packagings.py`, in `test_training_full_start_publishes_dataset`, **remove** the `notebook_generator` monkeypatch block:

```python
    # build_full_notebook still has the old bundle_file_id signature until
    # Task 5 — mock it so this wiring test stays green in between
    monkeypatch.setattr(
        "services.notebook_generator.build_full_notebook",
        MagicMock(return_value=b"{}"),
    )
```

and replace the assertions block:

```python
    res = client.post("/api/packagings/fullpub/training/full/start")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset"] == fake_summary
    assert "colab.research.google.com" in body["colab_url"]
    publish_mock.assert_called_once_with("fullpub", drive=ANY, progress_cb=ANY)
    # zip bundle must NOT be uploaded anymore
    for call in drive_mock.upload_bytes.call_args_list:
        all_args = list(call.args) + list(call.kwargs.values())
        assert not any(str(a).endswith(".zip") for a in all_args)
```

with:

```python
    res = client.post("/api/packagings/fullpub/training/full/start")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset"] == fake_summary
    assert "colab.research.google.com" in body["colab_url"]
    publish_mock.assert_called_once_with("fullpub", drive=ANY, progress_cb=ANY)
    # no notebook is generated/uploaded anymore — start only publishes the dataset
    drive_mock.upload_bytes.assert_not_called()
    drive_mock.create_folder.assert_not_called()
```

In `test_training_full_start_allows_grouped_multi_field`, **remove** the line:

```python
    monkeypatch.setattr(
        "services.notebook_generator.build_full_notebook", MagicMock(return_value=b"{}"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_packagings.py -k training_full_start -v`
Expected: FAIL — current handler calls `create_folder`/`upload_bytes` and imports `notebook_generator`, so `upload_bytes.assert_not_called()` fails.

- [ ] **Step 3: Rewrite the handler body**

In `api/packagings.py`, add a module-level constant near the top of the file (after imports, with the other module constants):

```python
# Combined Full Training notebook in Drive (built by
# scripts/build_full_training_notebook.py). Replace with the file_id printed
# by that script.
COMBINED_NOTEBOOK_FILE_ID = "PASTE_FILE_ID_FROM_TASK_4"
```

Then replace the body of `training_full_start` from the `from services import ...` line through the `return {...}` with:

```python
    from services import dataset_publisher, progress_store
    from services.drive_client import DriveClient

    progress_store.report(key, "starting")
    try:
        drive = DriveClient()
    except Exception as e:
        logger.exception("Drive client init failed for %s", key)
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, f"Drive client init failed: {e}")

    try:
        dataset_summary = dataset_publisher.publish(
            key,
            drive=drive,
            progress_cb=lambda done, total, name: progress_store.report(
                key, "upload_images", done=done, total=total, detail=name
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("dataset publish failed for %s", key)
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, f"Dataset publish failed: {e}")

    progress_store.report(key, "done")
    packaging_store.update_draft(
        key,
        training_run={
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full",
            "dataset": dataset_summary,
        },
        status="training_full",
    )
    colab_url = (
        f"https://colab.research.google.com/drive/{COMBINED_NOTEBOOK_FILE_ID}"
    )
    return {
        "colab_url": colab_url,
        "dataset": dataset_summary,
        "labeled_count": len(labeled),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_packagings.py -k training_full_start -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: full/start returns static combined-notebook link, drops generation"
```

---

## Task 6: Delete the dead notebook generator

**Files:**
- Delete: `services/notebook_generator.py`, `tests/test_notebook_generator.py`

- [ ] **Step 1: Confirm no remaining references**

Run: `grep -rn "notebook_generator\|build_full_notebook" --include=*.py .`
Expected: no matches (Task 5 removed the last import). If any remain, fix them before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm services/notebook_generator.py tests/test_notebook_generator.py
```

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`
Expected: no import errors; same pass/fail baseline as before (the 3 known `test_classifier.py` setup errors are pre-existing and unrelated).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove generated-notebook builder (replaced by static notebook)"
```

---

## Task 7: Neutralize `/training/full/done` + wizard manual flow

See the **Flagged decision** above. Minimal handling: `/training/full/done` no longer pulls a model; the wizard shows manual instructions and stops calling it.

**Files:**
- Modify: `api/packagings.py:595-638` (`training_full_done`)
- Test: `tests/test_api_packagings.py`
- Modify: `web/wizard.html` (progress label ~3039; step-5 render ~3251-3271; JS ~3371-3402)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_packagings.py` near the other full-training tests:

```python
def test_training_full_done_is_manual_now(client, monkeypatch):
    """Manual model sync: done returns a 400 telling the user to sync via the
    model registry, instead of pulling from a (no-longer-created) output folder."""
    from services import packaging_store
    monkeypatch.setattr(
        packaging_store, "get_draft",
        lambda key: {"key": key, "training_run": {"kind": "full"}},
    )
    res = client.post("/api/packagings/anykey/training/full/done")
    assert res.status_code == 400
    assert "manual" in res.json()["detail"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_packagings.py::test_training_full_done_is_manual_now -v`
Expected: FAIL — current handler raises 404/500 trying to read `run["output_folder_id"]`.

- [ ] **Step 3: Replace the `training_full_done` body**

In `api/packagings.py`, replace the body of `training_full_done` (everything after the docstring) with:

```python
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")
    raise HTTPException(
        400,
        "Full training เป็น manual แล้ว — รัน notebook ใน Colab ให้เสร็จ, "
        "model จะเซฟลง Drive, แล้ว sync เข้าระบบผ่าน model registry/manifest "
        "(ดู docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md)",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_packagings.py::test_training_full_done_is_manual_now -v`
Expected: PASS.

- [ ] **Step 5: Update wizard copy + JS (manual verification)**

In `web/wizard.html`:

(a) Progress label (~line 3039) — replace:
```javascript
  notebook:      'สร้าง Colab notebook…',
```
with:
```javascript
  notebook:      'เตรียม notebook…',
```

(b) Start button label (~line 3254) — replace:
```html
          🚀 เริ่ม Full Training (เปิด Colab)
```
with:
```html
          🚀 Publish dataset + เปิด Training Notebook
```

(c) The "รอ Colab" card (~lines 3263-3271) — replace its `info-box` + done button with manual instructions:
```html
      <div class="info-box">
        <strong>📒 Training notebook เปิดแล้ว</strong> — กด <code>Runtime → Run all</code>
        (detector ~60 นาที + classifier ~30 นาที). Model จะเซฟลง Drive
        (<code>data check lot/</code> + <code>data classify check lot/models/</code>).
        จากนั้น <strong>sync model เข้าระบบเอง</strong> ผ่าน model registry/manifest
        — wizard ไม่ดึงให้อัตโนมัติ
      </div>
```
(Remove the `<div style="text-align:center..."> ... btn-full-done ... </div>` block that called `fullTrainingDone()`.)

(d) Delete the now-unused `fullTrainingDone()` function (~lines 3390-3402).

- [ ] **Step 6: Verify the wizard renders standalone**

Run: `python -m http.server 8090 --directory web` then, via Playwright/agent-browser, load `http://localhost:8090/wizard.html` and `page.evaluate('goStep(5)')`.
Expected: step-5 shows the new start button and (after a mocked start) the manual-instructions card; no JS console error about `fullTrainingDone`.

- [ ] **Step 7: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py web/wizard.html
git commit -m "feat: manual model sync — neutralize full/done, wizard shows manual steps"
```

---

## Task 8: Heal the orphaned `new_tea_bag_box` class (one-off ops)

The detector `data.yaml` already lists `new_tea_bag_box_*` but its images never landed (see spec). After Tasks 1-3, re-publishing uploads the missing images to the correct folders.

- [ ] **Step 1: Re-run publish for the affected draft**

Run (dev server must have the new env vars from `.env`):
`python -c "from dotenv import load_dotenv; load_dotenv(); from services.dataset_publisher import publish; print(publish('new_tea_bag_box'))"`
Expected: a summary dict with `images_uploaded > 0` and `new_classes == []` (names already present; only images upload).

- [ ] **Step 2: Verify images landed**

Run: `python -c "from dotenv import load_dotenv; load_dotenv(); from services.drive_client import DriveClient; d=DriveClient(); det='1xmhCGoUrhPpDOHGdsewusPn57twkQXEr'; t=[x for x in d.list_folder(d.find_in_folder(det,'train') or det)]; print('see train/images via wizard or re-list')"`
Better: re-run the inspection from the design session — confirm `new_tea_bag_box`-prefixed files now appear in `train/images`/`val/images`, and that the classifier copies appear under `data classify check lot/images/new_tea_bag_box/`.
Expected: non-zero `new_tea_bag_box` files in both detector splits and the classifier `images/` subfolder.

- [ ] **Step 3: No commit** (data-only operation on Drive).

---

## Self-Review (completed during authoring)

- **Spec coverage:** M1 → Task 3; M2 → Tasks 1-2; remove generation + single link → Task 5; remove `notebook_generator` → Task 6; combined notebook (auto-discover classes, no data.yaml overwrite) → Task 4; wizard step-5 copy → Task 7; data heal → Task 8; lost `full/done` automation → Task 7 (flagged). Split 80/20 train/val: unchanged, no task (correct).
- **Placeholders:** only `PASTE_FILE_ID_FROM_TASK_4` — intentional, filled from Task 4's printed output.
- **Type/name consistency:** `_relativize(path, split)` defined in Task 1, used identically in Task 2; `COMBINED_NOTEBOOK_FILE_ID` defined and used in Task 5; classifier copy parent id math matches the `fake_drive.ensure_folder` side_effect.
