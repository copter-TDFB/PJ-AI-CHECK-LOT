"""_run_multi_region must select box/sachet by YOLO class_name, not by
position in the detections list — see
docs/superpowers/specs/2026-08-07-container-label-box-sachet-selection-design.md"""

from pipeline.detector import DetectionResult
from pipeline.packaging_registry import PackagingConfig
from pipeline.pipeline_runner import PipelineRunner


def _cfg():
    return PackagingConfig(
        key="k", display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot", "exp"], sheet_checks=["lot", "exp"],
        post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label="k", detector_yolo_prefixes=["k_box", "k_sachet"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=["box", "sachet"], detection_mode="cross_check",
    )


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def crop_all(self, image_bytes, key):
        return self._detections


class _StubPreprocessor:
    def run(self, cropped_bytes, key):
        return cropped_bytes


class _StubOcrEngine:
    def __init__(self, results_by_bytes):
        self._results_by_bytes = results_by_bytes

    def run(self, image_bytes, config=None):
        return self._results_by_bytes[image_bytes]


def _det(tag, class_name, conf):
    return DetectionResult(cropped_bytes=tag, bbox=[0, 0, 1, 1], class_name=class_name, conf=conf)


def _ocr(lot_number, exp_date):
    return {
        "raw_text": "enough text to pass the degraded-crop check",
        "lot_number": lot_number,
        "exp_date": exp_date,
    }


def _run(detections, ocr_by_bytes):
    runner = PipelineRunner(
        detector=_StubDetector(detections),
        preprocessor=_StubPreprocessor(),
        ocr_engine=_StubOcrEngine(ocr_by_bytes),
        qr_scanner=object(),
    )
    return runner._run_multi_region(b"unused_image_bytes", _cfg())


def test_one_box_zero_sachet():
    """SKU with no sachet at all — legitimate case, must stay correct."""
    box = _det(b"box", "k_box", 0.9)
    result, bbox = _run([box], {b"box": _ocr("LOTBOX", "2026-01-01")})
    assert result["lot_box"] == "LOTBOX"
    assert result["exp_box"] == "2026-01-01"
    assert result["lot_sachet"] is None
    assert result["exp_sachet"] is None
    assert bbox == [0, 0, 1, 1]


def test_one_box_one_sachet():
    """Normal case — both regions detected, one of each class."""
    box = _det(b"box", "k_box", 0.9)
    sachet = _det(b"sachet", "k_sachet", 0.8)
    result, _ = _run(
        [box, sachet],
        {b"box": _ocr("LOTBOX", "2026-01-01"), b"sachet": _ocr("LOTSACHET", "2026-01-01")},
    )
    assert result["lot_box"] == "LOTBOX"
    assert result["lot_sachet"] == "LOTSACHET"


def test_duplicate_box_does_not_become_sachet():
    """Regression guard for the confirmed bug: 2 detections of the SAME
    'k_box' class (no real sachet at all) must never populate lot_sachet/
    exp_sachet — the lower-confidence duplicate must be discarded, not
    misread as the sachet crop."""
    box_high = _det(b"box_high", "k_box", 0.81)
    box_low = _det(b"box_low", "k_box", 0.20)
    result, bbox = _run(
        [box_high, box_low],
        {
            b"box_high": _ocr("REALLOT", "2026-01-01"),
            b"box_low": _ocr("GHOSTLOT", "2026-01-01"),
        },
    )
    assert result["lot_box"] == "REALLOT"
    assert result["lot_sachet"] is None
    assert result["exp_sachet"] is None
    assert bbox == [0, 0, 1, 1]


def test_zero_box_one_sachet():
    """Box occluded/undetected but sachet visible — must not mislabel the
    sachet detection as the box."""
    sachet = _det(b"sachet", "k_sachet", 0.8)
    result, _ = _run([sachet], {b"sachet": _ocr("LOTSACHET", "2026-01-01")})
    assert result["lot_box"] is None
    assert result["exp_box"] is None
    assert result["lot_sachet"] == "LOTSACHET"


def test_heuristic_fallback_single_detection_treated_as_box():
    """YOLO didn't run / found nothing — the single heuristic detection
    (class_name=None) must still be treated as the box, as before."""
    heuristic = _det(b"full_image", None, None)
    result, _ = _run([heuristic], {b"full_image": _ocr("LOTBOX", "2026-01-01")})
    assert result["lot_box"] == "LOTBOX"
    assert result["lot_sachet"] is None


def test_zero_detections():
    result, bbox = _run([], {})
    assert result["lot_box"] is None
    assert result["lot_sachet"] is None
    assert result["status"] == "not_found"
    assert bbox is None


def test_custom_sub_region_names_are_not_hardcoded_to_box_sachet():
    """A cross_check packaging need not name its regions 'box'/'sachet' —
    config.sub_regions=['front', 'back'] must drive the YOLO class names
    used for selection (k_front / k_back), not a hardcoded '_box'/'_sachet'
    suffix. The result dict's own key names (lot_box/lot_sachet) stay fixed
    regardless of the packaging's region-name choice — that's the existing
    external contract."""
    front = _det(b"front", "k_front", 0.9)
    back = _det(b"back", "k_back", 0.8)
    runner = PipelineRunner(
        detector=_StubDetector([front, back]),
        preprocessor=_StubPreprocessor(),
        ocr_engine=_StubOcrEngine(
            {b"front": _ocr("LOTFRONT", "2026-01-01"), b"back": _ocr("LOTBACK", "2026-01-01")}
        ),
        qr_scanner=object(),
    )
    config = PackagingConfig(
        key="k", display_name="K", pipeline="detector_ocr",
        lot_patterns=[], fields_extracted=["lot", "exp"], sheet_checks=["lot", "exp"],
        post_ocr_fixes=[], message_template_key="default_full",
        model_classifier_label="k", detector_yolo_prefixes=["k_front", "k_back"],
        conf_threshold=0.6, accuracy=None, gate_on_lot=True, lot_short_fallback=False,
        sub_regions=["front", "back"], detection_mode="cross_check",
    )
    result, _ = runner._run_multi_region(b"unused_image_bytes", config)
    assert result["lot_box"] == "LOTFRONT"
    assert result["lot_sachet"] == "LOTBACK"
