"""Routing keys off detection_mode (single | cross_check | multi_field), the
explicit router that replaced the old len(sub_regions) inference. See
CONTEXT.md "Sub-region" and spec 2026-06-14-multi-field-detection."""

import pytest

from pipeline.packaging_registry import PackagingConfig
from pipeline.pipeline_runner import PipelineRunner
from utils.sheet_checker import SheetChecker


def _cfg(detection_mode="single", sub_regions=(), sheet_checks=("lot",)):
    return PackagingConfig(
        key="k", display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot"], sheet_checks=list(sheet_checks),
        post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label="k", detector_yolo_prefixes=["k_lot"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=list(sub_regions), detection_mode=detection_mode,
    )


@pytest.fixture
def runner():
    return PipelineRunner(
        detector=object(), preprocessor=object(),
        ocr_engine=object(), qr_scanner=object(),
    )


@pytest.mark.parametrize("mode, expected", [
    ("single", "single"),
    ("cross_check", "multi"),
    ("multi_field", "field"),
])
def test_run_dispatch(runner, monkeypatch, mode, expected):
    calls = []
    monkeypatch.setattr(runner, "_run_single_region",
                        lambda b, c: (calls.append("single"), ({}, None))[1])
    monkeypatch.setattr(runner, "_run_multi_region",
                        lambda b, c: (calls.append("multi"), ({}, None))[1])
    monkeypatch.setattr(runner, "_run_multi_field",
                        lambda b, c: (calls.append("field"), ({}, None))[1])
    runner.run(b"img", _cfg(detection_mode=mode))
    assert calls == [expected]


@pytest.mark.parametrize("mode, expected_lot_key", [
    ("single", "lot_number"),
    ("cross_check", "lot_box"),
    ("multi_field", "lot_number"),
])
def test_sheet_checker_lot_source(monkeypatch, mode, expected_lot_key):
    checker = SheetChecker()
    seen = {}

    def fake_find(lot, sheet_id, gid):
        seen["lot"] = lot
        return None

    monkeypatch.setattr(checker, "_find_row_by_lot", fake_find)
    checker.check(
        _cfg(detection_mode=mode),
        sheet_id="sid", gid=0,
        lot_number="LOTNUM", lot_box="LOTBOX", lot_sachet="LOTSACHET",
        exp_date="2026-01-01", exp_box="2026-01-01", exp_sachet="2026-01-01",
    )
    expected = {"lot_number": "LOTNUM", "lot_box": "LOTBOX"}[expected_lot_key]
    assert seen["lot"] == expected
