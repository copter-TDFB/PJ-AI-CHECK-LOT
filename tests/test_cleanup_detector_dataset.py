import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from scripts.cleanup_detector_dataset import (
    TARGET_NAMES, REMAP, remap_label_text, extract_split_paths, build_data_yaml,
)


def test_target_names_has_11_entries():
    assert len(TARGET_NAMES) == 11
    assert TARGET_NAMES[9] == "print_sticker_back_lot_exp"
    assert TARGET_NAMES[10] == "print_sticker_back_product_size"


def test_remap_rewrites_only_leading_id():
    text = "12 0.5 0.5 0.2 0.2\n13 0.1 0.1 0.3 0.3\n"
    out = remap_label_text(text, REMAP)
    assert out == "9 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.3 0.3\n"


def test_remap_leaves_unmapped_ids_untouched():
    text = "0 0.5 0.5 0.2 0.2\n8 0.1 0.1 0.3 0.3\n"
    assert remap_label_text(text, REMAP) == text


def test_remap_skips_blank_lines():
    text = "12 0.5 0.5 0.2 0.2\n\n"
    assert remap_label_text(text, REMAP) == "9 0.5 0.5 0.2 0.2\n\n"


def test_extract_split_paths():
    raw = (
        "train: /content/drive/MyDrive/data check lot/train/images\n"
        "val: /content/drive/MyDrive/data check lot/val/images\n"
        "nc: 16\nnames:\n- back_label_lot\n-test_u\n"
    )
    train, val = extract_split_paths(raw)
    assert train == "/content/drive/MyDrive/data check lot/train/images"
    assert val == "/content/drive/MyDrive/data check lot/val/images"


def test_build_data_yaml_roundtrips():
    out = build_data_yaml("t/images", "v/images", TARGET_NAMES)
    parsed = yaml.safe_load(out)
    assert parsed["nc"] == 11
    assert parsed["names"] == TARGET_NAMES
    assert parsed["train"] == "t/images"
    assert parsed["val"] == "v/images"
