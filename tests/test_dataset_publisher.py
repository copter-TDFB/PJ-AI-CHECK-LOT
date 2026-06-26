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


# ─── _relativize ─────────────────────────────────────────

def test_relativize_strips_absolute_drive_prefix():
    abs_path = "/content/drive/MyDrive/data check lot/train/images"
    assert dp._relativize(abs_path, "train") == "train/images"


def test_relativize_passthrough_when_already_relative():
    assert dp._relativize("val/images", "val") == "val/images"


def test_relativize_falls_back_when_split_absent():
    assert dp._relativize("/a/b/c", "train") == "a/b/c"


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
        # _safe_filename("img_0.jpg") = "img_0.jpg", so annotation is img_0.jpg.json
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
    drive.copy_file.side_effect = (
        lambda file_id, parent_id, name=None:
        drive.events.append(("copy_file", parent_id, name)) or f"copy-{name}"
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
    kinds = [e[0] for e in fake_drive.events]
    assert kinds.count("upload_file") == 3   # 3 detector image uploads
    assert kinds.count("copy_file") == 3     # 3 classifier copies (server-side, no re-upload)
    assert kinds.count("upload_bytes") == 3  # 3 label files


def test_publish_handles_absolute_data_yaml_paths(draft, env, fake_drive):
    """Absolute train/val paths in data.yaml must still resolve to train/val
    folders under det-root, not a 'content/drive/...' tree."""
    fake_drive.read_text.return_value = (
        "train: /content/drive/MyDrive/data check lot/train/images\n"
        "val: /content/drive/MyDrive/data check lot/val/images\n"
        "nc: 2\nnames: [old_a, old_b]\n"
    )
    dp.publish(draft, drive=fake_drive)
    ensured = [c.args[0] for c in fake_drive.ensure_folder.call_args_list]
    assert "content" not in ensured
    assert "train" in ensured and "images" in ensured


def test_publish_classifier_copies_land_under_images_subfolder(draft, env, fake_drive):
    """Classifier copies must go to images/<key>/ (where the notebook reads),
    not to <key>/ at the classifier-root top level."""
    dp.publish(draft, drive=fake_drive)
    # ensure_folder side_effect: f-{parent}-{name}
    cls_parent = "f-f-cls-root-images-newpack"
    cls_copies = [e for e in fake_drive.events
                  if e[0] == "copy_file" and e[1] == cls_parent]
    assert len(cls_copies) == 3


def test_publish_uses_global_class_ids(draft, env, fake_drive):
    dp.publish(draft, drive=fake_drive)
    label_events = [e for e in fake_drive.events if e[0] == "upload_bytes"]
    for _, _, name in label_events:
        assert name.startswith("newpack_") and name.endswith(".txt")
    assert dp.publish(draft, drive=fake_drive)["class_ids"] == {"lot": 2}


def test_publish_writes_data_yaml_last(draft, env, fake_drive):
    dp.publish(draft, drive=fake_drive)
    assert fake_drive.events[-1][0] == "update_yaml"
    content = fake_drive.events[-1][2].decode("utf-8")
    assert "newpack_lot" in content and "nc: 3" in content
    assert content.index("old_a") < content.index("newpack_lot")


def test_publish_skips_existing_files(draft, env, fake_drive):
    uploaded = {"newpack_img_0.jpg", "newpack_img_0.txt"}
    fake_drive.list_folder.side_effect = lambda fid: [
        {"id": "x", "name": n} for n in uploaded
    ]
    summary = dp.publish(draft, drive=fake_drive)
    assert summary["images_skipped"] == 1
    assert summary["images_uploaded"] == 2
    # img_0's label was also pre-existing — only 2 label uploads should occur
    kinds = [e[0] for e in fake_drive.events]
    assert kinds.count("upload_bytes") == 2


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


def test_publish_multi_sub_region_global_ids(tmp_path, env, fake_drive, monkeypatch):
    """sub_regions [box, sachet] → global ids continue after existing names."""
    import json as _json

    from PIL import Image as _Image

    from services import packaging_store

    draft_dir = tmp_path / "drafts"
    monkeypatch.setattr(dp, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(packaging_store, "_DRAFT_DIR", draft_dir)
    key = "multi"
    (draft_dir / key / "images").mkdir(parents=True)
    (draft_dir / key / "annotations").mkdir(parents=True)
    (draft_dir / key / "meta.json").write_text(
        _json.dumps({"key": key, "sub_regions": ["box", "sachet"]}), encoding="utf-8"
    )
    _Image.new("RGB", (100, 100)).save(draft_dir / key / "images" / "a.jpg")
    ann = {
        "bboxes": [
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "box"},
            {"x1": 20, "y1": 20, "x2": 40, "y2": 40, "label": "sachet"},
        ]
    }
    (draft_dir / key / "annotations" / "a.jpg.json").write_text(
        _json.dumps(ann), encoding="utf-8"
    )

    summary = dp.publish(key, drive=fake_drive)
    assert summary["class_ids"] == {"box": 2, "sachet": 3}
    assert summary["new_classes"] == ["multi_box", "multi_sachet"]

    # Verify label file content: both bboxes with global ids 2 and 3
    args, kwargs = fake_drive.upload_bytes.call_args
    content = (args[0] if args else kwargs["content"]).decode("utf-8")
    leading_ids = {line.split()[0] for line in content.splitlines() if line.strip()}
    assert leading_ids == {"2", "3"}
