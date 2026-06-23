# Train-Done Sync + Deploy Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a wizard button that pulls the freshly-trained detector + classifier + eval metrics from Drive into a draft, then routes the user into the existing Eval → Deploy flow so a manually-Colab-trained model can ship to production.

**Architecture:** Colab writes `full_detector.pt` + `classifier.pt` + `eval.json` to fixed Drive paths (`eval.json` last = "done" signal). A new `POST /{key}/training/full/sync` endpoint downloads the three files into `data/drafts/{key}/models/` and flips the draft to `trained`. The existing `/deploy` then promotes both models — extended to also promote the classifier and to run promotion/backup on the fresh-deploy path (previously detector-only, overwrite-path-only).

**Tech Stack:** FastAPI, pytest + `unittest.mock`, Google Drive API (`services/drive_client.DriveClient`), Ultralytics YOLO (notebook), vanilla JS (`web/wizard.html`).

## Global Constraints

- Python 3.11. Run tests with `python -m pytest` (pytest is NOT on PATH).
- `print()` is forbidden — use the module `logger`.
- Read/write source files containing Thai with `encoding="utf-8"`.
- `web/wizard.html` is the SINGLE source of truth; never edit `test wizzard/wizard.html` or `dist/portable-bundle/static/wizard.html` (generated copies).
- Registry reloads go through `main.reload_registry()`, never `PackagingRegistry()` directly.
- Drive dataset folder ids come from env: `DRIVE_DETECTOR_DATASET_FOLDER_ID` (folder `data check lot`) and `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` (folder `data classify check lot`).
- `eval.json` schema (hard floor checks first three): `detector_mAP_50` (≥0.65), `precision` (≥0.70), `recall` (≥0.60), plus `epochs`, `imgsz`, `train_count`, `val_count`.
- Stage files deliberately with `git add <file>` (working tree holds other in-progress work); `git add -p` is unavailable.

---

### Task 1: Promote detector AND classifier from a draft

**Files:**
- Modify: `services/cloudrun_deployer.py:173-186` (`promote_draft_model`)
- Test: `tests/test_cloudrun_deployer.py` (create)

**Interfaces:**
- Produces: `promote_draft_model(draft_key: str) -> dict[str, Path]` — returns `{"detector": Path, "classifier": Path}` for whichever of `full_detector.pt` / `full_classifier.pt` exist under `DRAFT_DIR/{draft_key}/models/`, copied to `MODELS_DIR/detector.pt` / `MODELS_DIR/classifier.pt`. Empty dict if neither present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cloudrun_deployer.py`:

```python
"""Unit tests for services.cloudrun_deployer model promotion."""

import importlib


def _reload_with_dirs(monkeypatch, draft_dir, models_dir):
    monkeypatch.setenv("DRAFT_DIR", str(draft_dir))
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    from services import cloudrun_deployer
    importlib.reload(cloudrun_deployer)
    return cloudrun_deployer


def test_promote_draft_model_promotes_detector_and_classifier(tmp_path, monkeypatch):
    draft_models = tmp_path / "drafts" / "k1" / "models"
    draft_models.mkdir(parents=True)
    (draft_models / "full_detector.pt").write_bytes(b"DET")
    (draft_models / "full_classifier.pt").write_bytes(b"CLS")
    models_dir = tmp_path / "models"

    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", models_dir)
    promoted = dep.promote_draft_model("k1")

    assert (models_dir / "detector.pt").read_bytes() == b"DET"
    assert (models_dir / "classifier.pt").read_bytes() == b"CLS"
    assert set(promoted) == {"detector", "classifier"}


def test_promote_draft_model_detector_only(tmp_path, monkeypatch):
    draft_models = tmp_path / "drafts" / "k1" / "models"
    draft_models.mkdir(parents=True)
    (draft_models / "full_detector.pt").write_bytes(b"DET")
    models_dir = tmp_path / "models"

    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", models_dir)
    promoted = dep.promote_draft_model("k1")

    assert (models_dir / "detector.pt").read_bytes() == b"DET"
    assert not (models_dir / "classifier.pt").exists()
    assert set(promoted) == {"detector"}


def test_promote_draft_model_none_present(tmp_path, monkeypatch):
    (tmp_path / "drafts" / "k1" / "models").mkdir(parents=True)
    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", tmp_path / "models")
    assert dep.promote_draft_model("k1") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cloudrun_deployer.py -v`
Expected: FAIL — `test_..._detector_and_classifier` errors because `promote_draft_model` returns a `Path`/`None`, not a dict (no `full_classifier.pt` handling).

- [ ] **Step 3: Write minimal implementation**

Replace `promote_draft_model` (`services/cloudrun_deployer.py:173-186`):

```python
def promote_draft_model(draft_key: str) -> dict[str, Path]:
    """Copy a draft's trained detector + classifier to production models/.

    Returns {"detector": Path, "classifier": Path} for whichever artifacts
    exist under DRAFT_DIR/{draft_key}/models/. Caller backs up the previous
    models first.
    """
    draft_models = Path(os.getenv("DRAFT_DIR", "data/drafts")) / draft_key / "models"
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    promoted: dict[str, Path] = {}
    for src_name, dst_name, label in (
        ("full_detector.pt", "detector.pt", "detector"),
        ("full_classifier.pt", "classifier.pt", "classifier"),
    ):
        src = draft_models / src_name
        if src.exists():
            dst = _MODELS_DIR / dst_name
            shutil.copy2(src, dst)
            logger.info("Promoted draft %s: %s → %s", label, src, dst)
            promoted[label] = dst
    return promoted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cloudrun_deployer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add services/cloudrun_deployer.py tests/test_cloudrun_deployer.py
git commit -m "feat: promote_draft_model promotes detector + classifier"
```

---

### Task 2: Deploy promotes synced models on the fresh path too

**Files:**
- Modify: `api/packagings.py:640-724` (`deploy_packaging`)
- Test: `tests/test_api_packagings.py` (add)

**Interfaces:**
- Consumes: `promote_draft_model(key) -> dict[str, Path]` (Task 1); `cloudrun_deployer.backup_artifacts(key) -> dict`; `cloudrun_deployer.write_packaging_yaml(key, draft) -> Path`; `main.reload_registry()`.
- Produces: `deploy_packaging` runs backup → promote → reload whenever the draft has synced models (`full_detector.pt` or `full_classifier.pt`), on both fresh and overwrite paths. Response `model_promoted` becomes `dict[str,str] | None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_packagings.py`:

```python
def test_fresh_deploy_promotes_synced_models(client, tmp_path, monkeypatch):
    import json as _json
    from pathlib import Path as _Path
    import main
    from services import packaging_store, cloudrun_deployer, eval_thresholds

    key = "newpack"
    draft_models = _Path(packaging_store._DRAFT_DIR) / key / "models"
    draft_models.mkdir(parents=True, exist_ok=True)
    (draft_models / "full_detector.pt").write_bytes(b"DET")
    (draft_models / "full_classifier.pt").write_bytes(b"CLS")
    (draft_models / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.9, "precision": 0.9, "recall": 0.9,
    }), encoding="utf-8")

    monkeypatch.setattr(packaging_store, "get_draft",
                        lambda k: {"key": key, "config": {}, "pipeline": "detector_ocr"})
    monkeypatch.setattr(packaging_store, "update_draft", lambda *a, **k: None)
    monkeypatch.setattr(main.registry, "get", lambda k: None)
    monkeypatch.setattr(main, "reload_registry", lambda: None)
    monkeypatch.setattr(cloudrun_deployer, "write_packaging_yaml",
                        lambda k, d: _Path("config/packagings") / f"{k}.yaml")
    monkeypatch.setattr(cloudrun_deployer, "backup_artifacts", lambda k: {"timestamp": "x", "files": []})
    promoted_calls = []
    monkeypatch.setattr(cloudrun_deployer, "promote_draft_model",
                        lambda k: promoted_calls.append(k) or {"detector": _Path("models/detector.pt"),
                                                               "classifier": _Path("models/classifier.pt")})
    monkeypatch.setattr(cloudrun_deployer, "trigger_cloud_run_revision",
                        lambda: {"triggered": False, "reason": "test"})
    monkeypatch.setattr(eval_thresholds, "check_hard_floor",
                        lambda e: {"passed": True, "failures": [], "hard_floor": {}})

    r = client.post(f"/api/packagings/{key}/deploy")
    assert r.status_code == 200, r.text
    assert promoted_calls == [key]            # promotion ran on the fresh path
    assert r.json()["model_promoted"]["detector"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py::test_fresh_deploy_promotes_synced_models -v`
Expected: FAIL — `promoted_calls == []` (current code only promotes when `parent_key is not None`).

- [ ] **Step 3: Write minimal implementation**

In `deploy_packaging`, replace the backup + promote block (`api/packagings.py:673-685`):

```python
    # 2. Detect freshly-synced models (present on both fresh + edit-draft paths)
    draft_models = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models"
    has_synced = (draft_models / "full_detector.pt").exists() or \
                 (draft_models / "full_classifier.pt").exists()

    # 3. Backup active artifacts before overwrite. For a fresh key the YAML
    #    backup is a no-op; the global model backup still protects rollback.
    backup_manifest = (
        cloudrun_deployer.backup_artifacts(target_key)
        if (parent_key is not None or has_synced) else None
    )

    try:
        # 4. Write packaging YAML under target_key (parent_key on overwrite)
        yaml_path = cloudrun_deployer.write_packaging_yaml(target_key, draft)

        # 5. Promote freshly-synced detector + classifier (either path)
        promoted = cloudrun_deployer.promote_draft_model(key) if has_synced else {}
```

Then update the registry-reload block to keep using `promoted`, and the response (`api/packagings.py:717-724`) `model_promoted` line:

```python
        "model_promoted": {k: str(v) for k, v in promoted.items()} or None,
```

(Remove the old `promoted_model = None` / `if parent_key is not None: promoted_model = ...` lines and the `str(promoted_model) if parent_key and promoted_model else None` expression.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_packagings.py -v -k deploy`
Expected: PASS (new test + any existing deploy tests green)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: deploy promotes synced models on fresh path"
```

---

### Task 3: `/training/full/sync` endpoint downloads model + eval from Drive

**Files:**
- Modify: `api/packagings.py` (add endpoint near `training_full_done`, ~line 638)
- Test: `tests/test_api_packagings.py` (add)

**Interfaces:**
- Consumes: `DriveClient().find_in_folder(parent_id, name) -> str | None`; `DriveClient().download_file(file_id, dest: Path) -> None`; `packaging_store.get_draft`, `packaging_store.update_draft`; env `DRIVE_DETECTOR_DATASET_FOLDER_ID`, `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`.
- Produces: `POST /api/packagings/{key}/training/full/sync` → downloads `eval.json`, `full_detector.pt` (detector folder) and `classifier.pt`→`full_classifier.pt` (classifier folder `models/` subfolder) into `DRAFT_DIR/{key}/models/`, sets `status="trained"`, returns `{"synced": true, "eval": <dict>}`. 409 if Drive has no `eval.json`; 404 if draft missing.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_packagings.py`:

```python
def test_sync_downloads_and_marks_trained(client, monkeypatch):
    import json as _json
    from pathlib import Path as _Path
    from services import packaging_store
    import api.packagings as apk

    key = "syncpack"
    monkeypatch.setenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", "DETFOLDER")
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSFOLDER")
    monkeypatch.setattr(packaging_store, "get_draft", lambda k: {"key": key})
    updates = {}
    monkeypatch.setattr(packaging_store, "update_draft",
                        lambda k, **kw: updates.update(kw))

    eval_obj = {"detector_mAP_50": 0.9, "precision": 0.9, "recall": 0.9,
                "epochs": 60, "imgsz": 640, "train_count": 40, "val_count": 10}

    def fake_find(parent, name):
        return {"eval.json": "E", "full_detector.pt": "D",
                "models": "M", "classifier.pt": "C"}.get(name)

    def fake_download(file_id, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "E":
            dest.write_text(_json.dumps(eval_obj), encoding="utf-8")
        else:
            dest.write_bytes(b"PT")

    fake = type("D", (), {"find_in_folder": staticmethod(fake_find),
                          "download_file": staticmethod(fake_download)})()
    monkeypatch.setattr(apk, "DriveClient", lambda: fake)

    r = client.post(f"/api/packagings/{key}/training/full/sync")
    assert r.status_code == 200, r.text
    models = _Path(packaging_store._DRAFT_DIR) / key / "models"
    assert (models / "eval.json").exists()
    assert (models / "full_detector.pt").read_bytes() == b"PT"
    assert (models / "full_classifier.pt").read_bytes() == b"PT"
    assert updates["status"] == "trained"
    assert r.json()["eval"]["detector_mAP_50"] == 0.9


def test_sync_409_when_eval_missing(client, monkeypatch):
    from services import packaging_store
    import api.packagings as apk
    monkeypatch.setenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", "DETFOLDER")
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSFOLDER")
    monkeypatch.setattr(packaging_store, "get_draft", lambda k: {"key": "x"})
    fake = type("D", (), {"find_in_folder": staticmethod(lambda p, n: None),
                          "download_file": staticmethod(lambda *a: None)})()
    monkeypatch.setattr(apk, "DriveClient", lambda: fake)
    r = client.post("/api/packagings/x/training/full/sync")
    assert r.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py -v -k sync`
Expected: FAIL — 404/405 (endpoint does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Confirm `DriveClient` is importable at module top of `api/packagings.py` (add `from services.drive_client import DriveClient` if not already imported; otherwise import inside the function to match existing lazy-import style). Add the endpoint after `training_full_done`:

```python
@router.post("/{key}/training/full/sync")
def training_full_sync(key: str):
    """Pull the freshly Colab-trained detector + classifier + eval from Drive
    into the draft, then mark it `trained` so the Eval/Deploy screen appears.

    `eval.json` is written last by the notebook, so its absence means training
    is not finished yet (→ 409).
    """
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    det_folder = os.getenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", "")
    cls_folder = os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "")
    if not det_folder or not cls_folder:
        raise HTTPException(500, "DRIVE_*_DATASET_FOLDER_ID not configured")

    client = DriveClient()
    eval_id = client.find_in_folder(det_folder, "eval.json")
    if eval_id is None:
        raise HTTPException(409, "ยังเทรนไม่เสร็จ — รัน Colab notebook ให้จบก่อน")

    models_dir = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # eval.json + detector live directly in the detector dataset folder
    client.download_file(eval_id, models_dir / "eval.json")
    det_id = client.find_in_folder(det_folder, "full_detector.pt")
    if det_id is None:
        raise HTTPException(409, "Drive มี eval แต่ไม่พบ full_detector.pt — เทรนยังไม่ครบ")
    client.download_file(det_id, models_dir / "full_detector.pt")

    # classifier lives under <classifier folder>/models/classifier.pt
    cls_models = client.find_in_folder(cls_folder, "models")
    cls_id = client.find_in_folder(cls_models, "classifier.pt") if cls_models else None
    if cls_id is None:
        raise HTTPException(409, "ไม่พบ classifier.pt ใน Drive — เทรนยังไม่ครบ")
    client.download_file(cls_id, models_dir / "full_classifier.pt")

    eval_data = json.loads((models_dir / "eval.json").read_text(encoding="utf-8"))
    packaging_store.update_draft(key, status="trained")
    logger.info("Synced trained model for '%s' from Drive", key)
    return {"synced": True, "eval": eval_data}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_packagings.py -v -k sync`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: /training/full/sync pulls trained model + eval from Drive"
```

---

### Task 4: Notebook writes eval.json + fixed model name

**Files:**
- Modify: `ai_crop_lot.ipynb` (the detector save cell, `ai_crop_lot.ipynb:1263-1294`)
- Modify: `scripts/build_full_training_notebook.py` (add an assertion the eval cell survives the build)
- Test: `tests/test_build_full_training_notebook.py` (create)

**Interfaces:**
- Produces: built notebook `lot-checker-full-training.ipynb` contains a cell that (a) copies `best.pt` → `data check lot/full_detector.pt`, (b) writes `data check lot/eval.json` with the schema in Global Constraints. The build asserts this cell is present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_full_training_notebook.py`:

```python
"""Guards that the Full Training notebook source emits eval.json + a fixed
detector filename so the /sync endpoint can find them."""

import json
from pathlib import Path


def test_detector_notebook_writes_eval_json_and_fixed_name():
    nb = json.loads(Path("ai_crop_lot.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"]
                    if c["cell_type"] == "code")
    assert "full_detector.pt" in src, "detector must save to fixed full_detector.pt"
    assert "eval.json" in src, "notebook must write eval.json"
    assert "detector_mAP_50" in src, "eval.json must carry detector_mAP_50"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_full_training_notebook.py -v`
Expected: FAIL — current notebook saves to `ai_check lot v3.pt` and never writes `eval.json`.

- [ ] **Step 3: Edit the notebook save cell**

In `ai_crop_lot.ipynb`, replace the body of the final detector save cell (currently copying to `ai_check lot v3.pt`, `ai_crop_lot.ipynb:1263-1294`) so its `source` array contains this code (the cell runs after `metrics = best_model.val()`):

```python
import shutil, json, os

DRIVE_DET = '/content/drive/MyDrive/data check lot'
source_model = '/content/runs/detect/ai_check lot v3/weights/best.pt'
destination_model = f'{DRIVE_DET}/full_detector.pt'
shutil.copy(source_model, destination_model)
print('saved detector to', destination_model)

train_imgs = len(list(Path(f'{DRIVE_DET}/train/images').glob('*')))
val_imgs   = len(list(Path(f'{DRIVE_DET}/val/images').glob('*')))
eval_data = {
    'detector_mAP_50': float(metrics.box.map50),
    'precision':       float(metrics.box.mp),
    'recall':          float(metrics.box.mr),
    'epochs':          int(getattr(results, 'epoch', 0) or 0) or 300,
    'imgsz':           640,
    'train_count':     train_imgs,
    'val_count':       val_imgs,
}
# eval.json written LAST — its presence is the "training done" signal
with open(f'{DRIVE_DET}/eval.json', 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)
print('wrote eval.json:', eval_data)
```

Ensure `from pathlib import Path` is available in the notebook (Ultralytics import already pulls it in most cells; if the linter test still needs it, the string `full_detector.pt` / `eval.json` / `detector_mAP_50` presence is what the test checks).

- [ ] **Step 4: Add the build-time assertion**

In `scripts/build_full_training_notebook.py`, inside `main()` after the existing `assert any("iterdir()" ...)` (line ~84), add:

```python
    det_src = "\n".join(_src(c) for c in det)
    assert "full_detector.pt" in det_src and "eval.json" in det_src, \
        "detector cells must save full_detector.pt + write eval.json"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_full_training_notebook.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add ai_crop_lot.ipynb scripts/build_full_training_notebook.py tests/test_build_full_training_notebook.py
git commit -m "feat: notebook writes eval.json + fixed full_detector.pt"
```

> **Post-merge manual step (not a code task):** rerun `python scripts/build_full_training_notebook.py` to rebuild and re-upload `lot-checker-full-training.ipynb` to Drive, and update `COMBINED_NOTEBOOK_FILE_ID` in `api/packagings.py` if the file_id changes. Document this in the PR description.

---

### Task 5: Wizard "Train เสร็จแล้ว — ดึง model" button

**Files:**
- Modify: `web/wizard.html` (`renderStep5_training`, `web/wizard.html:3364-3376`; add `syncTrainedModel()` near `startFullTraining`, `web/wizard.html:3473`)

**Interfaces:**
- Consumes: `api('POST', '/api/packagings/{key}/training/full/sync')` (Task 3); existing `loadStep5()`, `curDraftKey`, `esc()`.
- Produces: `syncTrainedModel()` — POSTs `/sync`; on success calls `loadStep5()` (status now `trained` → `renderStep5_eval` shows metrics + existing Deploy button); on error shows an inline message and re-enables the button.

- [ ] **Step 1: Add the button to `renderStep5_training`**

Replace `renderStep5_training` (`web/wizard.html:3364-3376`) with:

```javascript
function renderStep5_training(pkg) {
  document.getElementById('sp5-body').innerHTML = `
    <div class="wiz-card">
      <div class="wiz-card-title">Full Training — รอ Colab</div>
      <div class="info-box">
        <strong>📒 Training notebook เปิดแล้ว</strong> — กด <code>Runtime → Run all</code>
        (detector ~60 นาที + classifier ~30 นาที). Model จะเซฟลง Drive
        (<code>data check lot/</code> + <code>data classify check lot/models/</code>).
      </div>
      <div style="text-align:center;margin-top:14px">
        <button class="btn btn-primary" id="btn-sync-model" onclick="syncTrainedModel()">
          ✅ Train เสร็จแล้ว — ดึง model จาก Drive
        </button>
      </div>
      <div id="sync-msg" style="margin-top:10px;text-align:center;color:var(--t3);font-size:13px"></div>
    </div>`;
}
```

- [ ] **Step 2: Add the `syncTrainedModel()` handler**

Insert before `startFullTraining()` (`web/wizard.html:3473`):

```javascript
async function syncTrainedModel() {
  if (!curDraftKey) return;
  const btn = document.getElementById('btn-sync-model');
  const msg = document.getElementById('sync-msg');
  btn.disabled = true;
  btn.textContent = '⏳ กำลังดึง model จาก Drive...';
  msg.textContent = '';
  try {
    await api('POST', `/api/packagings/${encodeURIComponent(curDraftKey)}/training/full/sync`);
    await loadStep5();   // status → trained → renders Eval + Deploy
  } catch (err) {
    btn.disabled = false;
    btn.textContent = '✅ Train เสร็จแล้ว — ดึง model จาก Drive';
    msg.style.color = 'var(--err)';
    msg.textContent = err.message.includes('ยังเทรนไม่เสร็จ')
      ? 'ยังเทรนไม่เสร็จ — รอ Colab รันให้จบแล้วลองใหม่'
      : `ดึง model ไม่สำเร็จ: ${err.message}`;
  }
}
```

- [ ] **Step 3: Verify the step renders without a backend**

Run: `python -m http.server 8090 --directory web` then, via Playwright/agent-browser, `page.evaluate("goStep(5)")` after seeding `curDraftKey`, or load the page and call `renderStep5_training({})` directly:

```javascript
// in browser console / page.evaluate
renderStep5_training({});
document.getElementById('btn-sync-model') !== null   // expect true
```

Expected: the button `#btn-sync-model` exists and shows the Thai label.

- [ ] **Step 4: Commit**

```bash
git add web/wizard.html
git commit -m "feat: wizard button to sync trained model and reach Deploy"
```

---

## Self-Review

**Spec coverage:**
- Notebook eval.json + fixed name → Task 4 ✓
- `/training/full/sync` endpoint (download 3 files, 409 on missing eval, status=trained) → Task 3 ✓
- Promote detector + classifier → Task 1 ✓
- Fresh-deploy promotion/backup gap → Task 2 ✓
- Wizard button + route into Eval/Deploy → Task 5 ✓
- Error handling (409, status not flipped on failure, hard-floor/rollback reuse existing) → Tasks 2/3 ✓
- Testing (api sync, deployer promote, fresh-path deploy, notebook build) → Tasks 1–4 ✓

**Type consistency:** `promote_draft_model` returns `dict[str, Path]` in Task 1 and is consumed as a dict in Task 2 (`promoted.items()`, `model_promoted` dict). `find_in_folder`/`download_file` signatures match `services/drive_client.py`. `eval.json` keys identical across Tasks 3, 4, and the hard-floor config.

**Placeholder scan:** No TBD/TODO; every code step carries full code. The notebook-rebuild + `COMBINED_NOTEBOOK_FILE_ID` update is explicitly flagged as a manual post-merge step, not a silent gap.

**Notes / residual risk:**
- Fresh-deploy rollback restores backed-up global models but does not delete a newly-written YAML if a later step fails — pre-existing rollback limitation, low impact (next deploy overwrites). Out of scope to fix here.
- `results.epoch` access in the notebook is defensive (`getattr(... 0) or 300`); exact epoch count is cosmetic (not hard-floor gated).
