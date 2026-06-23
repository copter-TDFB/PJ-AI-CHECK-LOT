# Drive Dataset Direct Write — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Full-Training zip bundle with direct writes of images + YOLO labels into the Drive reference dataset, so the dataset on Drive is always complete and the Colab notebook just mounts and trains.

**Architecture:** A new `services/dataset_publisher.py` reads `data.yaml` from the detector dataset folder on Drive, appends new class names (never reordering), uploads images/labels with global class ids into train/val (deterministic 80/20 split), uploads classifier copies into a per-class folder, and writes `data.yaml` back **last** (commit point). `DriveClient` gains `read_text` / `update_file_content` / `list_folder` / `ensure_folder` and switches scope from `drive.file` to `drive` (service account — no OAuth verification issue, see spec). The full-training notebook loses all bundle/merge logic. Seed training is untouched.

**Tech Stack:** Python 3.11, FastAPI, google-api-python-client (Drive v3), PyYAML, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-drive-dataset-direct-write-design.md`

**Manual prerequisite (not a code task):** share `data check lot` + `data classify check lot` with the SA email as Editor; set `DRIVE_DETECTOR_DATASET_FOLDER_ID` + `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` in `.env` / Cloud Run.

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/drive_client.py` | Modify | Drive API wrapper — scope change + 4 new methods |
| `tests/test_drive_client.py` | Create | Unit tests for new DriveClient methods (API mocked) |
| `services/dataset_publisher.py` | Create | Publish draft → Drive reference dataset |
| `tests/test_dataset_publisher.py` | Create | Pure-logic + orchestration tests (DriveClient mocked) |
| `api/packagings.py` | Modify (`training_full_start`, ~line 491) | Call publisher instead of zip bundle |
| `tests/test_api_packagings.py` | Modify | Endpoint wiring test |
| `services/notebook_generator.py` | Modify (`build_full_notebook`) | Drop bundle download + merge cells |
| `tests/test_notebook_generator.py` | Create | Full notebook has no bundle/merge logic |
| `docs/adr/0003-direct-dataset-write-via-service-account.md` | Create | Decision record |
| `docs/adr/0001-...md`, `.env.example`, `CLAUDE.md` | Modify | Cross-references + env vars |
| `services/training_bundle.py` | **Unchanged** | Still used by seed training |

---

### Task 1: DriveClient — scope change + new methods

**Files:**
- Modify: `services/drive_client.py`
- Test: `tests/test_drive_client.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_drive_client.py`:

```python
"""Unit tests for DriveClient extensions — Drive API fully mocked."""

import io
from unittest.mock import MagicMock, patch

import pytest


class FakeDownloader:
    """Stands in for MediaIoBaseDownload — writes fixed bytes into the buffer."""

    payload = b"names: [a, b]\n"

    def __init__(self, buf, request, chunksize=None):
        self._buf = buf

    def next_chunk(self):
        self._buf.write(self.payload)
        return None, True


@pytest.fixture
def client_and_svc():
    with patch("services.drive_client.google_auth_default",
               return_value=(MagicMock(), None)), \
         patch("services.drive_client.build") as mock_build:
        svc = MagicMock()
        mock_build.return_value = svc
        from services.drive_client import DriveClient
        yield DriveClient(), svc


def test_scope_is_full_drive():
    from services import drive_client
    assert drive_client._SCOPES == ["https://www.googleapis.com/auth/drive"]


def test_read_text_decodes_utf8(client_and_svc):
    client, _ = client_and_svc
    with patch("services.drive_client.MediaIoBaseDownload", FakeDownloader):
        assert client.read_text("fid") == "names: [a, b]\n"


def test_update_file_content_calls_files_update(client_and_svc):
    client, svc = client_and_svc
    client.update_file_content("fid123", b"new content", mime_type="text/yaml")
    _, kwargs = svc.files.return_value.update.call_args
    assert kwargs["fileId"] == "fid123"
    assert "media_body" in kwargs


def test_list_folder_paginates(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.side_effect = [
        {"files": [{"id": "1", "name": "a.jpg"}], "nextPageToken": "tok"},
        {"files": [{"id": "2", "name": "b.jpg"}]},
    ]
    files = client.list_folder("parent")
    assert [f["name"] for f in files] == ["a.jpg", "b.jpg"]
    assert svc.files.return_value.list.call_count == 2


def test_ensure_folder_returns_existing(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "existing-id"}]
    }
    assert client.ensure_folder("train", "root") == "existing-id"
    svc.files.return_value.create.assert_not_called()


def test_ensure_folder_creates_when_missing(client_and_svc):
    client, svc = client_and_svc
    svc.files.return_value.list.return_value.execute.return_value = {"files": []}
    svc.files.return_value.create.return_value.execute.return_value = {"id": "new-id"}
    assert client.ensure_folder("train", "root") == "new-id"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drive_client.py -v`
Expected: FAIL — `test_scope_is_full_drive` asserts wrong scope; `read_text`, `update_file_content`, `list_folder`, `ensure_folder` raise `AttributeError`.

- [ ] **Step 3: Implement**

In `services/drive_client.py`:

3a. Replace the scope block (the comment is now wrong too):

```python
# Full drive scope — required to write into the reference dataset folders
# that the user shares with this service account. SAs do not go through
# OAuth consent verification, so the "restricted scope" problem in ADR 0001
# does not apply (see ADR 0003).
_SCOPES = ["https://www.googleapis.com/auth/drive"]
```

3b. Refactor the download path and add `read_text` — replace the body of `read_json` with a shared helper:

```python
    def _download_bytes(self, file_id: str) -> bytes:
        request = self._svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue()

    def read_json(self, file_id: str) -> dict:
        """Download file from Drive และ parse เป็น dict."""
        return json.loads(self._download_bytes(file_id))

    def read_text(self, file_id: str) -> str:
        """Download file from Drive เป็น UTF-8 text."""
        return self._download_bytes(file_id).decode("utf-8")
```

3c. Add to the upload/folders section:

```python
    def update_file_content(
        self, file_id: str, content: bytes, mime_type: str = "text/plain"
    ) -> None:
        """เขียนทับเนื้อหาไฟล์เดิม (file_id คงเดิม)."""
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=False)
        self._svc.files().update(fileId=file_id, media_body=media).execute()
        logger.info("Updated file content → %s (%.1f KB)", file_id, len(content) / 1024)

    def ensure_folder(self, name: str, parent_id: str) -> str:
        """Find folder by name under parent — create ถ้ายังไม่มี. Return folder_id."""
        safe = name.replace("'", "\\'")
        q = (
            f"name = '{safe}' and '{parent_id}' in parents and trashed = false "
            "and mimeType = 'application/vnd.google-apps.folder'"
        )
        resp = self._svc.files().list(q=q, fields="files(id)", pageSize=1).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        return self.create_folder(name, parent_id)
```

3d. Add to the search section:

```python
    def list_folder(self, parent_id: str) -> list[dict]:
        """List ทุกไฟล์/โฟลเดอร์ใน parent — return [{id, name, mimeType}]."""
        files: list[dict] = []
        token = None
        while True:
            resp = self._svc.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageSize=1000,
                pageToken=token,
            ).execute()
            files.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_drive_client.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `pytest`
Expected: everything that passed before still passes (`read_json` refactor is behaviour-preserving).

- [ ] **Step 6: Commit**

```bash
git add services/drive_client.py tests/test_drive_client.py
git commit -m "feat: DriveClient full-drive scope + read_text/update/list/ensure_folder"
```

---

### Task 2: dataset_publisher — pure helpers

**Files:**
- Create: `services/dataset_publisher.py`
- Test: `tests/test_dataset_publisher.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_publisher.py`:

```python
"""Tests for dataset_publisher — pure helpers + publish orchestration."""

import pytest

from services import dataset_publisher as dp


# ─── merge_class_names ───────────────────────────────────

def test_merge_appends_new_names_at_end():
    assert dp.merge_class_names(["a", "b"], ["c", "d"]) == ["a", "b", "c", "d"]


def test_merge_never_reorders_or_duplicates():
    assert dp.merge_class_names(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_merge_noop_when_all_exist():
    assert dp.merge_class_names(["a", "b"], ["a", "b"]) == ["a", "b"]


# ─── split_for ───────────────────────────────────────────

def test_split_is_deterministic():
    assert dp.split_for("box_001.jpg") == dp.split_for("box_001.jpg")


def test_split_ratio_roughly_80_20():
    names = [f"pkg_{i:04d}.jpg" for i in range(500)]
    train = sum(1 for n in names if dp.split_for(n) == "train")
    assert 0.70 <= train / len(names) <= 0.90


def test_split_returns_only_train_or_val():
    assert {dp.split_for(f"x{i}.jpg") for i in range(50)} <= {"train", "val"}


# ─── labels_relpath ──────────────────────────────────────

def test_labels_relpath_replaces_last_images_segment():
    assert dp.labels_relpath("train/images") == "train/labels"
    assert dp.labels_relpath("images/train") == "labels/train"


def test_labels_relpath_raises_without_images_segment():
    with pytest.raises(ValueError):
        dp.labels_relpath("train/imgs")


# ─── label_lines ─────────────────────────────────────────

def test_label_lines_uses_global_ids_and_normalises():
    bboxes = [{"x1": 0, "y1": 0, "x2": 50, "y2": 100, "label": "box"}]
    lines = dp.label_lines(bboxes, {"box": 9}, w=100, h=200, default_label="box")
    assert lines == ["9 0.250000 0.250000 0.500000 0.500000"]


def test_label_lines_skips_unknown_label():
    bboxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "mystery"}]
    assert dp.label_lines(bboxes, {"box": 0}, w=100, h=100, default_label="box") == []


def test_label_lines_defaults_missing_label():
    bboxes = [{"x1": 10, "y1": 10, "x2": 20, "y2": 20}]
    lines = dp.label_lines(bboxes, {"lot": 3}, w=100, h=100, default_label="lot")
    assert lines[0].startswith("3 ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: FAIL — `ModuleNotFoundError: services.dataset_publisher`.

- [ ] **Step 3: Implement the helpers**

Create `services/dataset_publisher.py`:

```python
"""Publish a draft's labeled images + YOLO labels directly into the Drive
reference dataset (detector + classifier folders).

Replaces the zip bundle for the FULL training path only (seed training still
uses services/training_bundle.py — see ADR 0003).

Key invariants:
- data.yaml `names` are only ever APPENDED to — label files reference numeric
  class ids, so reordering existing names corrupts the dataset.
- data.yaml is written LAST. It is the commit point: if an upload run dies
  mid-way, the new class names were never declared and the dataset stays
  valid. Retries skip files that already exist (idempotent).
- Filenames are prefixed with the packaging key to avoid cross-packaging
  collisions.
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DRAFT_DIR = Path("data/drafts")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_DET_ENV = "DRIVE_DETECTOR_DATASET_FOLDER_ID"
_CLS_ENV = "DRIVE_CLASSIFIER_DATASET_FOLDER_ID"
_TRAIN_BUCKETS = 8  # out of 10 → 80/20 split


def merge_class_names(existing: list[str], new: list[str]) -> list[str]:
    """Append names not already present. NEVER reorders existing entries."""
    return list(existing) + [n for n in new if n not in existing]


def split_for(filename: str) -> str:
    """Deterministic train/val assignment — same name always lands the same."""
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return "train" if int(digest, 16) % 10 < _TRAIN_BUCKETS else "val"


def labels_relpath(images_relpath: str) -> str:
    """YOLO convention: swap the last 'images' path segment for 'labels'."""
    parts = images_relpath.strip("/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return "/".join(parts)
    raise ValueError(f"no 'images' segment in dataset path: {images_relpath!r}")


def label_lines(
    bboxes: list[dict],
    label_to_id: dict[str, int],
    w: int,
    h: int,
    default_label: str,
) -> list[str]:
    """Convert pixel bboxes → YOLO lines with GLOBAL class ids."""
    lines = []
    for b in bboxes:
        label = b.get("label") or default_label
        if label not in label_to_id:
            logger.warning("unknown label '%s' — skipping bbox", label)
            continue
        cid = label_to_id[label]
        cx = ((b["x1"] + b["x2"]) / 2) / w
        cy = ((b["y1"] + b["y2"]) / 2) / h
        bw = (b["x2"] - b["x1"]) / w
        bh = (b["y2"] - b["y1"]) / h
        cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
        bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add services/dataset_publisher.py tests/test_dataset_publisher.py
git commit -m "feat: dataset publisher pure helpers (merge/split/labels)"
```

---

### Task 3: dataset_publisher — `publish()` orchestration

**Files:**
- Modify: `services/dataset_publisher.py`
- Modify: `tests/test_dataset_publisher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset_publisher.py`:

```python
# ─── publish() orchestration ─────────────────────────────

import json
from unittest.mock import MagicMock

from PIL import Image


@pytest.fixture
def draft(tmp_path, monkeypatch):
    """Minimal draft on disk: meta + 3 labeled images, sub_regions=[lot]."""
    from services import packaging_store

    draft_dir = tmp_path / "drafts"
    monkeypatch.setattr(dp, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(packaging_store, "_DRAFT_DIR", draft_dir)

    key = "newpack"
    (draft_dir / key / "images").mkdir(parents=True)
    (draft_dir / key / "annotations").mkdir(parents=True)
    (draft_dir / key / "meta.json").write_text(
        json.dumps({"key": key, "sub_regions": ["lot"]}), encoding="utf-8"
    )
    for i in range(3):
        name = f"img_{i}.jpg"
        Image.new("RGB", (100, 200)).save(draft_dir / key / "images" / name)
        ann = {"bboxes": [{"x1": 10, "y1": 10, "x2": 50, "y2": 60, "label": "lot"}]}
        (draft_dir / key / "annotations" / f"{name}.json").write_text(
            json.dumps(ann), encoding="utf-8"
        )
    return key


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", "det-root")
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "cls-root")


@pytest.fixture
def fake_drive():
    """MagicMock DriveClient recording call order in .events."""
    drive = MagicMock()
    drive.events = []

    drive.find_in_folder.return_value = "yaml-id"
    drive.read_text.return_value = (
        "train: train/images\nval: val/images\nnc: 2\nnames: [old_a, old_b]\n"
    )
    drive.ensure_folder.side_effect = (
        lambda name, parent_id: f"f-{parent_id}-{name}"
    )
    drive.list_folder.return_value = []

    drive.upload_file.side_effect = (
        lambda src, parent_id=None, name=None, public=False:
        drive.events.append(("upload_file", parent_id, name)) or f"id-{name}"
    )
    drive.upload_bytes.side_effect = (
        lambda content, name, parent_id=None, mime_type=None, public=False:
        drive.events.append(("upload_bytes", parent_id, name)) or f"id-{name}"
    )
    drive.update_file_content.side_effect = (
        lambda file_id, content, mime_type=None:
        drive.events.append(("update_yaml", file_id, content))
    )
    return drive


def test_publish_fails_fast_without_env(draft, fake_drive, monkeypatch):
    monkeypatch.delenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", raising=False)
    monkeypatch.delenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", raising=False)
    with pytest.raises(RuntimeError, match="DATASET_FOLDER_ID"):
        dp.publish(draft, drive=fake_drive)


def test_publish_uploads_images_labels_and_classifier_copies(draft, env, fake_drive):
    summary = dp.publish(draft, drive=fake_drive)
    assert summary["images_uploaded"] == 3
    assert summary["images_skipped"] == 0
    assert summary["new_classes"] == ["newpack_lot"]
    # every image: detector image + label + classifier copy
    kinds = [e[0] for e in fake_drive.events]
    assert kinds.count("upload_file") == 6   # 3 detector imgs + 3 classifier copies
    assert kinds.count("upload_bytes") == 3  # 3 label files


def test_publish_uses_global_class_ids(draft, env, fake_drive):
    dp.publish(draft, drive=fake_drive)
    label_events = [e for e in fake_drive.events if e[0] == "upload_bytes"]
    for _, _, name in label_events:
        assert name.startswith("newpack_") and name.endswith(".txt")
    # global id = 2 (after old_a, old_b) — check via summary
    assert dp.publish(draft, drive=fake_drive)["class_ids"] == {"lot": 2}


def test_publish_writes_data_yaml_last(draft, env, fake_drive):
    dp.publish(draft, drive=fake_drive)
    assert fake_drive.events[-1][0] == "update_yaml"
    content = fake_drive.events[-1][2].decode("utf-8")
    assert "newpack_lot" in content and "nc: 3" in content
    # existing names stay first
    assert content.index("old_a") < content.index("newpack_lot")


def test_publish_skips_existing_files(draft, env, fake_drive):
    uploaded = {"newpack_img_0.jpg", "newpack_img_0.txt"}
    fake_drive.list_folder.side_effect = lambda fid: [
        {"id": "x", "name": n} for n in uploaded
    ]
    summary = dp.publish(draft, drive=fake_drive)
    assert summary["images_skipped"] == 1
    assert summary["images_uploaded"] == 2


def test_publish_skips_yaml_update_when_class_exists(draft, env, fake_drive):
    fake_drive.read_text.return_value = (
        "train: train/images\nval: val/images\nnc: 3\n"
        "names: [old_a, old_b, newpack_lot]\n"
    )
    dp.publish(draft, drive=fake_drive)
    fake_drive.update_file_content.assert_not_called()


def test_publish_raises_when_no_labeled_images(tmp_path, env, fake_drive, monkeypatch):
    from services import packaging_store

    draft_dir = tmp_path / "drafts"
    monkeypatch.setattr(dp, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(packaging_store, "_DRAFT_DIR", draft_dir)
    (draft_dir / "empty" / "images").mkdir(parents=True)
    (draft_dir / "empty" / "meta.json").write_text(
        json.dumps({"key": "empty", "sub_regions": ["lot"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no labeled images"):
        dp.publish("empty", drive=fake_drive)
```

> Note: check how `packaging_store.get_annotation` names annotation files — if it
> uses `{stem}.json` instead of `{filename}.json`, adjust the fixture to match
> (read `packaging_store._safe_filename` / `get_annotation` first).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'publish'`; the 11 helper tests still pass.

- [ ] **Step 3: Implement `publish()`**

Append to `services/dataset_publisher.py`:

```python
def publish(key: str, drive=None) -> dict:
    """Publish draft's labeled data into the Drive reference dataset.

    Returns {images_uploaded, images_skipped, train_count, val_count,
             new_classes, class_ids, total_classes}.
    Raises RuntimeError (setup problems), FileNotFoundError (no draft),
    ValueError (no labeled images).
    """
    from services import packaging_store

    det_root = os.getenv(_DET_ENV, "")
    cls_root = os.getenv(_CLS_ENV, "")
    if not det_root or not cls_root:
        raise RuntimeError(
            f"{_DET_ENV} / {_CLS_ENV} not set — share the dataset folders with "
            "the service account email (Editor) and set both env vars"
        )

    draft = packaging_store.get_draft(key)
    if draft is None:
        raise FileNotFoundError(f"draft '{key}' not found")
    sub_regions: list[str] = draft.get("sub_regions") or ["lot"]
    new_names = [f"{key}_{sr}" for sr in sub_regions]

    if drive is None:
        from services.drive_client import DriveClient
        drive = DriveClient()

    yaml_id, data_yaml, existing_names = _load_data_yaml(drive, det_root)
    merged = merge_class_names(existing_names, new_names)
    label_to_id = {sr: merged.index(f"{key}_{sr}") for sr in sub_regions}

    dest = _resolve_dest_folders(drive, det_root, data_yaml)
    cls_folder = drive.ensure_folder(key, cls_root)
    existing_files = {
        fid: {f["name"] for f in drive.list_folder(fid)}
        for fid in set(dest.values()) | {cls_folder}
    }

    stats = _upload_items(
        drive, key, label_to_id, sub_regions[0], dest, cls_folder, existing_files
    )
    if stats["images_uploaded"] == 0 and stats["images_skipped"] == 0:
        raise ValueError(f"draft '{key}' has no labeled images yet")

    # data.yaml LAST — the commit point (see module docstring)
    if merged != existing_names:
        import yaml as _yaml
        data_yaml["names"] = merged
        data_yaml["nc"] = len(merged)
        drive.update_file_content(
            yaml_id,
            _yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True).encode("utf-8"),
            mime_type="text/yaml",
        )

    summary = {
        **stats,
        "new_classes": [n for n in merged if n not in existing_names],
        "class_ids": label_to_id,
        "total_classes": len(merged),
    }
    logger.info("Dataset published for '%s': %s", key, summary)
    return summary


def _load_data_yaml(drive, det_root: str) -> tuple[str, dict, list[str]]:
    """Fetch data.yaml from the detector dataset folder. Fail fast if absent."""
    import yaml as _yaml

    yaml_id = drive.find_in_folder(det_root, "data.yaml")
    if yaml_id is None:
        raise RuntimeError(
            "data.yaml not found in the detector dataset folder — check "
            f"{_DET_ENV} and that the folder is shared with the service account"
        )
    data_yaml = _yaml.safe_load(drive.read_text(yaml_id)) or {}
    return yaml_id, data_yaml, list(data_yaml.get("names") or [])


def _resolve_dest_folders(drive, det_root: str, data_yaml: dict) -> dict:
    """Map (split, kind) → Drive folder id, derived from data.yaml paths."""
    train_rel = str(data_yaml.get("train") or "train/images")
    val_rel = str(data_yaml.get("val") or "val/images")
    rels = {
        ("train", "images"): train_rel,
        ("train", "labels"): labels_relpath(train_rel),
        ("val", "images"): val_rel,
        ("val", "labels"): labels_relpath(val_rel),
    }
    dest = {}
    for k, rel in rels.items():
        fid = det_root
        for seg in rel.strip("/").split("/"):
            if seg in ("", "."):
                continue
            fid = drive.ensure_folder(seg, fid)
        dest[k] = fid
    return dest


def _upload_items(
    drive,
    key: str,
    label_to_id: dict[str, int],
    default_label: str,
    dest: dict,
    cls_folder: str,
    existing_files: dict[str, set[str]],
) -> dict:
    """Upload each labeled image + label + classifier copy. Skip existing."""
    from PIL import Image

    from services import packaging_store

    img_dir = _DRAFT_DIR / key / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"no images for draft '{key}'")

    uploaded = skipped = 0
    counts = {"train": 0, "val": 0}
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        ann = packaging_store.get_annotation(key, img_path.name)
        if not ann or not ann.get("bboxes"):
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            logger.warning("skip unreadable: %s", img_path.name)
            continue
        if w == 0 or h == 0:
            continue
        lines = label_lines(ann["bboxes"], label_to_id, w, h, default_label)
        if not lines:
            continue

        new_name = f"{key}_{img_path.name}"
        lbl_name = f"{Path(new_name).stem}.txt"
        split = split_for(new_name)
        counts[split] += 1
        img_fid = dest[(split, "images")]
        lbl_fid = dest[(split, "labels")]

        if new_name in existing_files[img_fid]:
            skipped += 1
        else:
            drive.upload_file(img_path, parent_id=img_fid, name=new_name)
            uploaded += 1
        if lbl_name not in existing_files[lbl_fid]:
            drive.upload_bytes(
                ("\n".join(lines) + "\n").encode("utf-8"),
                name=lbl_name, parent_id=lbl_fid, mime_type="text/plain",
            )
        if new_name not in existing_files[cls_folder]:
            drive.upload_file(img_path, parent_id=cls_folder, name=new_name)

    return {
        "images_uploaded": uploaded,
        "images_skipped": skipped,
        "train_count": counts["train"],
        "val_count": counts["val"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset_publisher.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add services/dataset_publisher.py tests/test_dataset_publisher.py
git commit -m "feat: dataset_publisher.publish — direct Drive dataset write with data.yaml commit point"
```

---

### Task 4: Wire `training_full_start` to the publisher

**Files:**
- Modify: `api/packagings.py:491-548` (`training_full_start`)
- Test: `tests/test_api_packagings.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_packagings.py` (reuse the existing `client` fixture; mock everything heavy):

```python
def test_training_full_start_publishes_dataset(client, monkeypatch):
    """full/start must publish dataset (no zip) then gen + upload notebook."""
    from services import packaging_store

    client.post("/api/packagings/drafts", json={"key": "fullpub", "display_name": "x"})

    monkeypatch.setattr(
        packaging_store, "list_annotation_status",
        lambda key: [{"name": f"i{n}.jpg", "labeled": True, "bbox_count": 1}
                     for n in range(10)],
    )

    fake_summary = {
        "images_uploaded": 10, "images_skipped": 0,
        "train_count": 8, "val_count": 2,
        "new_classes": ["fullpub_lot"], "class_ids": {"lot": 9},
        "total_classes": 10,
    }
    publish_mock = MagicMock(return_value=fake_summary)
    monkeypatch.setattr("services.dataset_publisher.publish", publish_mock)

    drive_mock = MagicMock()
    drive_mock.create_folder.return_value = "run-folder"
    drive_mock.upload_bytes.return_value = "nb-id"
    monkeypatch.setattr(
        "services.drive_client.DriveClient", MagicMock(return_value=drive_mock)
    )
    # build_full_notebook still has the old bundle_file_id signature until
    # Task 5 — mock it so this wiring test stays green in between
    monkeypatch.setattr(
        "services.notebook_generator.build_full_notebook",
        MagicMock(return_value=b"{}"),
    )

    res = client.post("/api/packagings/fullpub/training/full/start")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset"] == fake_summary
    assert "colab.research.google.com" in body["colab_url"]
    publish_mock.assert_called_once()
    # zip bundle must NOT be uploaded anymore
    for call in drive_mock.upload_bytes.call_args_list:
        assert not str(call.kwargs.get("name", "")).endswith(".zip")
```

> Adjust the draft-creation call to match the existing draft-creation route used
> elsewhere in this test file (search for `drafts` POST in the file and copy its
> shape) — the route/payload above is a best guess and the existing tests are
> the source of truth.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_packagings.py::test_training_full_start_publishes_dataset -v`
Expected: FAIL — response still contains `bundle_size_kb`, no `dataset` key, and `publish_mock` not called.

- [ ] **Step 3: Rewrite `training_full_start`**

Replace the body after the `labeled` check (keep the 404/min-10 guards unchanged):

```python
    from services import dataset_publisher, notebook_generator
    from services.drive_client import DriveClient

    try:
        drive = DriveClient()
        dataset_summary = dataset_publisher.publish(key, drive=drive)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("dataset publish failed for %s", key)
        raise HTTPException(500, f"Dataset publish failed: {e}")

    try:
        run_folder_id = drive.create_folder(f"lot-checker-training-{key}-full")
        nb_bytes = notebook_generator.build_full_notebook(
            packaging_key=key,
            output_folder_id=run_folder_id,
        )
        nb_file_id = drive.upload_bytes(
            nb_bytes, name=f"{key}-full-training.ipynb", parent_id=run_folder_id,
            mime_type="application/vnd.google.colaboratory",
        )
    except Exception as e:
        logger.exception("training/full/start failed for %s", key)
        raise HTTPException(500, f"Drive upload failed: {e}")

    packaging_store.update_draft(
        key,
        training_run={
            "notebook_file_id": nb_file_id,
            "output_folder_id": run_folder_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full",
            "dataset": dataset_summary,
        },
        status="training_full",
    )
    return {
        "colab_url": notebook_generator.colab_url(nb_file_id),
        "notebook_file_id": nb_file_id,
        "output_folder_id": run_folder_id,
        "dataset": dataset_summary,
        "labeled_count": len(labeled),
    }
```

Also delete `training_bundle` from this function's lazy import (seed still imports it in `training_seed_start` — leave that one alone).

> Note: `build_full_notebook` loses its `bundle_file_id` parameter in Task 5.
> Tasks 4+5 must land together before the suite is green — run the full suite
> at the end of Task 5, not Task 4. (Alternatively implement Task 5 first;
> order here matches the data flow for readability.)

- [ ] **Step 4: Run the new test**

Run: `pytest tests/test_api_packagings.py::test_training_full_start_publishes_dataset -v`
Expected: PASS (`build_full_notebook` is mocked in this test, so the old
signature doesn't matter; the real call only works after Task 5 — do not run
the server between Tasks 4 and 5).

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: training/full/start publishes dataset to Drive instead of zip bundle"
```

---

### Task 5: Simplify the full-training notebook

**Files:**
- Modify: `services/notebook_generator.py:108-381` (`build_full_notebook`)
- Test: `tests/test_notebook_generator.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_notebook_generator.py`:

```python
"""Full-training notebook must train straight from the reference dataset."""

import json

from services import notebook_generator


def _full_nb_source() -> str:
    nb_bytes = notebook_generator.build_full_notebook(
        packaging_key="testpack", output_folder_id="out-folder",
    )
    nb = json.loads(nb_bytes)
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_full_notebook_has_no_bundle_or_merge_logic():
    src = _full_nb_source()
    assert "gdown" not in src
    assert "bundle" not in src
    assert "addition" not in src
    assert "offset" not in src


def test_full_notebook_trains_from_reference_dataset():
    src = _full_nb_source()
    assert "data check lot" in src
    assert "data classify check lot" in src
    assert "full_detector.pt" in src
    assert "full_classifier.pt" in src
    assert "OUTPUT_FOLDER_ID = 'out-folder'" in src or '"out-folder"' in src


def test_seed_notebook_unchanged_still_uses_bundle():
    nb_bytes = notebook_generator.build_seed_notebook(
        packaging_key="x", bundle_file_id="bid", output_folder_id="oid",
    )
    src = "".join("".join(c["source"]) for c in json.loads(nb_bytes)["cells"])
    assert "gdown" in src and "bundle" in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notebook_generator.py -v`
Expected: FAIL — `build_full_notebook()` requires `bundle_file_id`, and source contains `gdown`/`bundle`/`addition`/`offset`.

- [ ] **Step 3: Rewrite `build_full_notebook`**

New signature (drop `bundle_file_id`):

```python
def build_full_notebook(
    packaging_key: str,
    output_folder_id: str,
    detector_ref_path: str = DEFAULT_DETECTOR_REF,
    classifier_ref_path: str = DEFAULT_CLASSIFIER_REF,
    epochs: int = 250,
    imgsz: int = 1024,
) -> bytes:
    """Full = retrain Classifier + Detector straight from the reference dataset.

    The backend has already published the new packaging's images + labels into
    the reference dataset on Drive (services/dataset_publisher.py), so there is
    nothing to merge here — mount, copy to local disk for I/O speed, train.
    """
```

Cells (replace setup + merge cells; the two **training cells and the upload
cell keep their existing code** with only the variable renames noted below):

```python
    cells = [
        _cell("markdown", (
            f"# Full Training — {packaging_key}\n\n"
            "ขั้นตอน:\n"
            "1. กด `Runtime → Run all` (Ctrl+F9)\n"
            "2. Authorize Drive access ตอน popup\n"
            "3. รอประมาณ **1-2 ชั่วโมง** (detector ~60 min + classifier ~30 min)\n"
            "4. เมื่อขึ้น **TRAINING DONE** กลับไปที่ wizard\n\n"
            f"**Dataset (in My Drive):** `{detector_ref_path}/`, `{classifier_ref_path}/`"
        )),
        _cell("code", "!pip install -q ultralytics==8.3.0 timm==1.0.11\n"),
        _cell("code", (
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
        )),
        _cell("code", (
            "import json, os, shutil, yaml\n"
            f"OUTPUT_FOLDER_ID = {output_folder_id!r}\n"
            f"PACKAGING_KEY = {packaging_key!r}\n"
            f"DETECTOR_REF = '/content/drive/MyDrive/{detector_ref_path}'\n"
            f"CLASSIFIER_REF = '/content/drive/MyDrive/{classifier_ref_path}'\n"
            "assert os.path.exists(DETECTOR_REF), f'missing {DETECTOR_REF}'\n"
            "assert os.path.exists(CLASSIFIER_REF), f'missing {CLASSIFIER_REF}'\n"
            "ref_yaml = yaml.safe_load(open(f'{DETECTOR_REF}/data.yaml'))\n"
            "class_names = ref_yaml['names']\n"
            "print('classes:', class_names)\n"
        )),
        _cell("markdown", "## 1. Copy detector dataset to local disk (Drive I/O is slow)"),
        _cell("code", (
            "MD = '/content/detector_data'\n"
            "for split in ['train', 'val']:\n"
            "    os.makedirs(f'{MD}/images/{split}', exist_ok=True)\n"
            "    os.makedirs(f'{MD}/labels/{split}', exist_ok=True)\n"
            "    for kind in ['images', 'labels']:\n"
            "        src = f'{DETECTOR_REF}/{split}/{kind}'\n"
            "        if os.path.isdir(src):\n"
            "            for fn in os.listdir(src):\n"
            "                shutil.copy(f'{src}/{fn}', f'{MD}/{kind}/{split}/{fn}')\n"
            "print(f\"train={len(os.listdir(f'{MD}/images/train'))}, \"\n"
            "      f\"val={len(os.listdir(f'{MD}/images/val'))}\")\n"
            "with open(f'{MD}/data.yaml', 'w') as f:\n"
            "    f.write(f'path: {MD}\\n')\n"
            "    f.write('train: images/train\\n')\n"
            "    f.write('val: images/val\\n')\n"
            "    f.write(f'nc: {len(class_names)}\\n')\n"
            "    f.write(f'names: {class_names}\\n')\n"
        )),
        # ── 2. Train Detector: keep the existing cell verbatim, but change
        #      data='/content/merged_detector/data.yaml' → data=f'{MD}/data.yaml'
        #      (both in model.train and model.val)
        # ── 3. Classifier: keep the existing merge+train cells, but the merge
        #      cell shrinks to a plain copy (no addition branch):
        _cell("markdown", "## 3. Copy classifier dataset + Train EfficientNet"),
        _cell("code", (
            "MC = '/content/classifier_data'\n"
            "os.makedirs(MC, exist_ok=True)\n"
            "for cls_dir in os.listdir(CLASSIFIER_REF):\n"
            "    src = os.path.join(CLASSIFIER_REF, cls_dir)\n"
            "    if os.path.isdir(src):\n"
            "        dst = os.path.join(MC, cls_dir)\n"
            "        if not os.path.exists(dst):\n"
            "            shutil.copytree(src, dst)\n"
            "classifier_classes = sorted(os.listdir(MC))\n"
            "print('classifier classes:', classifier_classes)\n"
            "for c in classifier_classes:\n"
            "    print(f'  {c}: {len(os.listdir(os.path.join(MC, c)))} imgs')\n"
        )),
        # ── EfficientNet training cell: keep verbatim (it already reads MC)
        # ── 4. Upload cell: keep verbatim, but replace
        #      'merged_class_count': len(merged_classes)
        #      → 'merged_class_count': len(class_names)   # key kept for eval compat
    ]
```

The instructions in comments above are for the implementer: the detector
training cell, EfficientNet cell, and upload cell are copied from the current
file unchanged except the three renames called out (`/content/merged_detector`
→ `{MD}`, `merged_classes` → `class_names`, classifier folder `MC` path).
The module docstring (lines 3-12) must also be rewritten — it describes the
bundle-merge architecture; replace with: backend publishes into the reference
dataset, notebook mounts + copies locally + trains.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_notebook_generator.py tests/test_api_packagings.py -v`
Expected: PASS — then `pytest` (full suite) to confirm nothing else referenced `build_full_notebook(bundle_file_id=...)`:

Run: `grep -rn "build_full_notebook" --include="*.py" .`
Expected: only `services/notebook_generator.py` (def) and `api/packagings.py` (call without bundle_file_id).

- [ ] **Step 5: Commit**

```bash
git add services/notebook_generator.py tests/test_notebook_generator.py
git commit -m "feat: full-training notebook trains straight from reference dataset (no bundle merge)"
```

---

### Task 6: Docs — ADR 0003, env examples, CLAUDE.md

**Files:**
- Create: `docs/adr/0003-direct-dataset-write-via-service-account.md`
- Modify: `docs/adr/0001-wizard-trains-both-models-via-reference-dataset.md` (status line only)
- Modify: `.env.example`
- Modify: `CLAUDE.md` (Environment section)

- [ ] **Step 1: Write ADR 0003**

```markdown
# 0003 — Backend writes the reference dataset directly via service account

Date: 2026-06-10
Status: Accepted (supersedes the addition-bundle data path of ADR 0001;
the "notebook trains both models" decision in ADR 0001 stays in force)

## Context

ADR 0001 routed new-packaging data through a backend-owned "addition" folder
because the full `drive` OAuth scope is restricted for unverified OAuth
clients. But `services/drive_client.py` authenticates with
`google_auth_default()` — a service account, not a user OAuth client. Service
accounts never go through OAuth consent verification, so the restriction that
motivated the bundle/merge design does not apply: sharing the dataset folders
with the SA's email is enough.

The bundle design also left real costs: the Drive dataset was never complete
on its own, addition folders piled up and needed manual cleanup, and the
notebook duplicated dataset-layout knowledge (class-id offsets, splits).

## Decision

On Full Training start, the backend writes the draft's images + YOLO labels
directly into the reference dataset folders shared with the service account:

- `data check lot/` — detector train/val images + labels + `data.yaml`
- `data classify check lot/{class}/` — classifier images

Rules (implemented in `services/dataset_publisher.py`):

- `data.yaml` `names` are append-only — existing entries are never reordered.
- `data.yaml` is written LAST (commit point): a half-finished upload leaves
  the dataset valid; retries skip already-uploaded files (idempotent).
- Filenames are prefixed with the packaging key to prevent collisions.
- Deterministic 80/20 train/val split by filename hash.

Seed training keeps the small zip bundle — it runs before publication and its
model is throwaway.

## Consequences

**Positive** — the Drive dataset is the complete, single source of truth;
the notebook shrinks to mount-copy-train; "add images to an existing class"
works for free through the same path; no addition folders to clean up.

**Negative** — requires one-time sharing of both folders with the SA + two
env vars (`DRIVE_DETECTOR_DATASET_FOLDER_ID`, `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`);
files created by the SA count against the SA's 15 GB quota (moving the dataset
to a Workspace Shared Drive removes this if it ever matters); the backend now
holds full-Drive scope on the SA, so dataset-write bugs can corrupt the
dataset — mitigated by append-only names + commit-point ordering.
```

- [ ] **Step 2: Update ADR 0001 status line**

In `docs/adr/0001-...md` change `Status: Accepted` →
`Status: Accepted — data path superseded by ADR 0003 (training recipe still in force)`

- [ ] **Step 3: Update `.env.example`**

Append:

```bash
# Drive reference dataset folders (share both with the SA email as Editor)
DRIVE_DETECTOR_DATASET_FOLDER_ID=
DRIVE_CLASSIFIER_DATASET_FOLDER_ID=
```

- [ ] **Step 4: Update CLAUDE.md**

In the **Environment** key-vars list add:

```markdown
- `DRIVE_DETECTOR_DATASET_FOLDER_ID` / `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` —
  Drive folder ids of the reference dataset (`data check lot` /
  `data classify check lot`), shared with the SA. Required for the wizard's
  Full Training (dataset publish). See ADR 0003.
```

In the **Wizard API** paragraph, after the ADR 0002 sentence add:
`ADR 0003 explains "backend writes the reference dataset directly" (Full Training publishes images/labels to Drive; data.yaml is the commit point).`

- [ ] **Step 5: Run full suite + commit**

Run: `pytest`
Expected: all green.

```bash
git add docs/adr/0003-direct-dataset-write-via-service-account.md docs/adr/0001-wizard-trains-both-models-via-reference-dataset.md .env.example CLAUDE.md
git commit -m "docs: ADR 0003 direct dataset write via service account + env vars"
```

---

## Manual verification (after all tasks)

1. Share both dataset folders with the SA email (Editor); put their folder ids
   in `.env`.
2. **Smoke test against scratch folders first:** create two scratch Drive
   folders (copy a handful of dataset files + `data.yaml` into the detector
   one), point the env vars at them, run the wizard Full Training start on an
   existing draft, and verify on Drive: images/labels landed in train+val,
   `data.yaml` got the new class appended at the end, classifier subfolder
   created. Re-run start → response shows `images_skipped` = previous count.
3. Point env vars at the real folders.
4. Run one real Full Training end-to-end (wizard → Colab → done) and confirm
   the notebook trains without the merge cells.
