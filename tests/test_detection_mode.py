"""Multi-field detection mode — config load, routing, schema, label mapping."""

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
