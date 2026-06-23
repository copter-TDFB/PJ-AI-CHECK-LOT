"""Multi-field detection mode — config load, routing, schema, label mapping."""

import pytest
from pydantic import ValidationError

from api.schemas import PackagingCreate
from pipeline.detector import DetectionResult
from pipeline.packaging_registry import PackagingRegistry


def test_existing_packagings_have_expected_detection_mode():
    reg = PackagingRegistry()
    assert reg.get("container_label").detection_mode == "cross_check"
    for k in ("back_label", "grade_bag", "capsule_box", "retail_sachet"):
        assert reg.get(k).detection_mode == "single"


def test_detection_mode_defaults_to_single_when_missing():
    reg = PackagingRegistry()
    assert reg.get("import_sticker").detection_mode == "single"


def test_detection_result_has_class_name_default_none():
    d = DetectionResult(cropped_bytes=b"x", bbox=[0, 0, 1, 1])
    assert d.class_name is None
    d2 = DetectionResult(cropped_bytes=b"x", bbox=[0, 0, 1, 1], class_name="k_lot")
    assert d2.class_name == "k_lot"


import pipeline.pipeline_runner as pr
from pipeline.pipeline_runner import PipelineRunner
from pipeline.packaging_registry import PackagingConfig


def _cfg_multi(key="k"):
    return PackagingConfig(
        key=key, display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot", "exp", "product", "size"],
        sheet_checks=["lot"], post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label=key, detector_yolo_prefixes=[f"{key}_lot"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=["lot", "exp", "product", "size"], detection_mode="multi_field",
    )


class _FakeDetector:
    def __init__(self, dets): self._dets = dets
    def crop_all(self, image_bytes, key): return self._dets


class _FakePre:
    def run(self, b, key): return b


class _FakeOcr:
    def run(self, b, config=None):
        return {"raw_text": b.decode(), "lot_number": None}


def test_multi_field_routes_each_crop_to_its_field(monkeypatch):
    dets = [
        DetectionResult(cropped_bytes=b"LOTTEXT", bbox=[0, 0, 1, 1], class_name="k_lot"),
        DetectionResult(cropped_bytes=b"EXPTEXT", bbox=[0, 1, 1, 2], class_name="k_exp"),
        DetectionResult(cropped_bytes=b"NAMETEXT", bbox=[0, 2, 1, 3], class_name="k_product"),
        DetectionResult(cropped_bytes=b"SIZETEXT", bbox=[0, 3, 1, 4], class_name="k_size"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: f"LOT::{t}")
    monkeypatch.setattr(pr, "find_expiry", lambda t: f"EXP::{t}")
    monkeypatch.setattr(pr, "find_product_name", lambda t, aliases=None: f"PROD::{t}")
    monkeypatch.setattr(pr, "find_size", lambda t: f"SIZE::{t}")

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, bbox = runner._run_multi_field(b"img", _cfg_multi())

    assert result["lot_number"] == "LOT::LOTTEXT"
    assert result["exp_date"] == "EXP::EXPTEXT"
    assert result["product_name"] == "PROD::NAMETEXT"
    assert result["size"] == "SIZE::SIZETEXT"
    assert result["lot_box"] is None
    assert result["status"] == "ok"
    assert bbox == [0, 0, 1, 1]


def test_multi_field_requires_lot_sub_region():
    with pytest.raises(ValidationError):
        PackagingCreate(key="k", display_name="K",
                        detection_mode="multi_field", sub_regions=["exp", "size"])


def test_multi_field_accepts_lot_plus_fields():
    m = PackagingCreate(key="k", display_name="K",
                        detection_mode="multi_field",
                        sub_regions=["lot", "exp", "product", "size"])
    assert m.detection_mode == "multi_field"
    assert "lot" in m.sub_regions


def test_cross_check_requires_two_sub_regions():
    with pytest.raises(ValidationError):
        PackagingCreate(key="k", display_name="K",
                        detection_mode="cross_check", sub_regions=["box"])


def test_default_detection_mode_is_single():
    m = PackagingCreate(key="k", display_name="K")
    assert m.detection_mode == "single"


def test_deployer_writes_multi_field_mode_and_prefixes(tmp_path, monkeypatch):
    import services.cloudrun_deployer as dep
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path))
    draft_meta = {
        "display_name": "New Pkg",
        "pipeline": "detector_ocr",
        "sub_regions": ["lot", "exp", "product", "size"],
        "detection_mode": "multi_field",
        "config": {
            "lot_patterns": ["LOT(\\w+)"], "fields_extracted": ["lot", "exp", "product", "size"],
            "sheet_checks": ["lot"], "message_template_key": "default_full",
        },
    }
    out = dep.write_packaging_yaml("newpkg", draft_meta)
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["detection_mode"] == "multi_field"
    assert data["detector_yolo_prefixes"] == [
        "newpkg_lot", "newpkg_exp", "newpkg_product", "newpkg_size"]


def test_deployer_writes_grouped_class_prefixes(tmp_path, monkeypatch):
    import services.cloudrun_deployer as dep
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path))
    draft_meta = {
        "display_name": "Grouped Pkg",
        "pipeline": "detector_ocr",
        "sub_regions": ["lot_exp", "product_size"],
        "detection_mode": "multi_field",
        "config": {
            "lot_patterns": ["LOT(\\w+)"],
            "fields_extracted": ["lot", "exp", "product", "size"],
            "sheet_checks": ["lot"], "message_template_key": "default_full",
        },
    }
    out = dep.write_packaging_yaml("grouppkg", draft_meta)
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["detection_mode"] == "multi_field"
    assert data["sub_regions"] == ["lot_exp", "product_size"]
    assert data["detector_yolo_prefixes"] == ["grouppkg_lot_exp", "grouppkg_product_size"]


def test_deployer_persists_product_aliases(tmp_path, monkeypatch):
    import services.cloudrun_deployer as dep
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path))
    aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
    draft_meta = {
        "display_name": "Alias Pkg",
        "pipeline": "detector_ocr",
        "config": {
            "lot_patterns": ["LOT(\\w+)"],
            "fields_extracted": ["lot", "product"],
            "sheet_checks": ["lot", "product"],
            "message_template_key": "default_full",
            "product_aliases": aliases,
        },
    }
    out = dep.write_packaging_yaml("aliaspkg", draft_meta)
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["product_aliases"] == aliases


def test_multi_field_label_lines_map_each_field_to_its_class():
    from services.dataset_publisher import label_lines, merge_class_names
    sub_regions = ["lot", "exp", "product", "size"]
    merged = merge_class_names([], [f"k_{s}" for s in sub_regions])
    label_to_id = {s: merged.index(f"k_{s}") for s in sub_regions}

    bboxes = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "size"},
        {"x1": 0, "y1": 20, "x2": 10, "y2": 30, "label": "lot"},
    ]
    lines = label_lines(bboxes, label_to_id, 100, 100, default_label="lot")
    class_ids = [int(line.split()[0]) for line in lines]
    assert class_ids == [label_to_id["size"], label_to_id["lot"]]


def _cfg_grouped(sub_regions):
    cfg = _cfg_multi()
    return cfg.__class__(**{**cfg.__dict__, "sub_regions": sub_regions})


def test_multi_field_shared_box_feeds_both_extractors(monkeypatch):
    # one crop holds lot AND exp; one crop holds product AND size
    dets = [
        DetectionResult(cropped_bytes=b"LOTEXP", bbox=[0, 0, 1, 1], class_name="k_lot_exp"),
        DetectionResult(cropped_bytes=b"PRODSIZE", bbox=[0, 1, 1, 2], class_name="k_product_size"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: f"LOT::{t}")
    monkeypatch.setattr(pr, "find_expiry", lambda t: f"EXP::{t}")
    monkeypatch.setattr(pr, "find_product_name", lambda t, aliases=None: f"PROD::{t}")
    monkeypatch.setattr(pr, "find_size", lambda t: f"SIZE::{t}")

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, _ = runner._run_multi_field(b"img", _cfg_grouped(["lot_exp", "product_size"]))

    assert result["lot_number"] == "LOT::LOTEXP"
    assert result["exp_date"] == "EXP::LOTEXP"        # exp read from the SAME crop as lot
    assert result["product_name"] == "PROD::PRODSIZE"
    assert result["size"] == "SIZE::PRODSIZE"
    assert result["status"] == "ok"


def test_multi_field_raw_text_not_duplicated_for_shared_crop(monkeypatch):
    dets = [
        DetectionResult(cropped_bytes=b"LOTEXP", bbox=[0, 0, 1, 1], class_name="k_lot_exp"),
    ]
    monkeypatch.setattr(pr, "find_lot", lambda t, image_class=None, patterns=None: "L")
    monkeypatch.setattr(pr, "find_expiry", lambda t: "E")
    monkeypatch.setattr(pr, "find_product_name", lambda t, aliases=None: None)
    monkeypatch.setattr(pr, "find_size", lambda t: None)

    runner = PipelineRunner(_FakeDetector(dets), _FakePre(), _FakeOcr(), object())
    result, _ = runner._run_multi_field(b"img", _cfg_grouped(["lot_exp"]))

    assert result["raw_text"].count("LOTEXP") == 1
