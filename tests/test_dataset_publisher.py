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
