import re

import pytest

from pipeline.message_builder import build_verify_message
from pipeline.packaging_registry import MessageTemplate, PackagingConfig


def _config(**overrides) -> PackagingConfig:
    base = dict(
        key="back_label",
        display_name="Back Label",
        pipeline="detector_ocr",
        lot_patterns=[re.compile(r".*")],
        fields_extracted=["lot", "exp"],
        sheet_checks=["lot", "exp"],
        post_ocr_fixes=[],
        message_template_key="lot_exp",
        model_classifier_label="back_label",
        detector_yolo_prefixes=["back_label"],
        conf_threshold=0.6,
        accuracy=None,
        gate_on_lot=True,
        lot_short_fallback=False,
        sub_regions=[],
        detection_mode="single",
    )
    base.update(overrides)
    return PackagingConfig(**base)


def _template() -> MessageTemplate:
    return MessageTemplate(
        key="lot_exp",
        ok_message="ข้อมูลถูกต้อง",
        fail_template="ข้อมูลไม่ถูกต้อง ตรวจสอบ{fields}",
        field_labels={"lot": "เลข Lot", "exp": "วันหมดอายุ"},
    )


def _sheet(lot_match=True, exp_match=True, **extra) -> dict:
    base = {
        "lot_match": lot_match,
        "exp_match": exp_match,
        "product_match": None,
        "sachet_match": None,
        "sheet_product_name": None,
    }
    base.update(extra)
    return base


def test_ok_message_appends_lot_and_exp_as_ddmmyyyy():
    msg = build_verify_message(
        _config(), _template(), _sheet(), lot="A12345", exp="2026-12-01"
    )
    assert msg == "ข้อมูลถูกต้อง | LOT: A12345 | EXP: 01/12/2026"


def test_fail_message_appends_lot_and_exp():
    msg = build_verify_message(
        _config(), _template(), _sheet(exp_match=False), lot="A12345", exp="2026-12-01"
    )
    assert "LOT: A12345 | EXP: 01/12/2026" in msg
    assert "วันหมดอายุ" in msg  # field label of the failing exp check


def test_exp_omitted_when_not_read():
    msg = build_verify_message(
        _config(), _template(), _sheet(), lot="A12345", exp=None
    )
    assert msg == "ข้อมูลถูกต้อง | LOT: A12345"
    assert "EXP:" not in msg


def test_exp_omitted_when_unparseable():
    msg = build_verify_message(
        _config(), _template(), _sheet(), lot="A12345", exp="garbage"
    )
    assert "EXP:" not in msg
    assert "LOT: A12345" in msg


def test_exp_shown_even_without_lot():
    msg = build_verify_message(
        _config(gate_on_lot=False, sheet_checks=["exp"]),
        _template(),
        _sheet(lot_match=None),
        lot=None,
        exp="2026-12-01",
    )
    assert "EXP: 01/12/2026" in msg
    assert "LOT:" not in msg


def test_exp_value_driven_shown_even_if_not_in_sheet_checks():
    # class reads exp (fields_extracted) but does not cross-check it (sheet_checks)
    msg = build_verify_message(
        _config(sheet_checks=["lot"], message_template_key="lot_only"),
        _template(),
        _sheet(exp_match=None),
        lot="A12345",
        exp="2026-12-01",
    )
    assert "EXP: 01/12/2026" in msg


def test_default_exp_arg_is_optional():
    # backward compatible: callers that pass no exp still work
    msg = build_verify_message(_config(), _template(), _sheet(), lot="A12345")
    assert msg == "ข้อมูลถูกต้อง | LOT: A12345"
