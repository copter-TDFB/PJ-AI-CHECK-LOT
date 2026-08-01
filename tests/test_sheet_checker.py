from pipeline.packaging_registry import PackagingConfig
from utils.sheet_checker import SheetChecker


def _cfg(sheet_checks=("lot", "exp"), lot_short_fallback=False):
    return PackagingConfig(
        key="retail_sachet",
        display_name="Retail Sachet",
        pipeline="detector_ocr",
        lot_patterns=[],
        fields_extracted=["lot", "exp"],
        sheet_checks=list(sheet_checks),
        post_ocr_fixes=[],
        message_template_key="lot_exp",
        model_classifier_label="retail_sachet",
        detector_yolo_prefixes=["retail_sachet_lot"],
        conf_threshold=0.6,
        accuracy=None,
        gate_on_lot=True,
        lot_short_fallback=lot_short_fallback,
        sub_regions=[],
        detection_mode="single",
    )


def _row(lot, mfg="", exp="", product="Rootroute Wheatgrass Powder 5 g"):
    return {"Lot.": lot, "MFG": mfg, "EXP": exp, "Product Name": product}


def test_single_matching_row_behaves_like_before(monkeypatch):
    checker = SheetChecker()
    monkeypatch.setattr(
        checker, "_find_rows_by_lot",
        lambda lot, sheet_id, gid: [_row("109W", exp="01022027")],
    )
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0,
        lot_number="109W", exp_date="2027-02-01",
    )
    assert result["lot_match"] is True
    assert result["exp_match"] is True
    assert result["sheet_product_name"] == "Rootroute Wheatgrass Powder 5 g"


def test_duplicate_rows_blank_placeholder_then_filled_resolves_to_filled(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp=""),  # placeholder row added before the label was printed
        _row("109W", mfg="01/08/2026", exp="01022027"),  # real data added later
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0,
        lot_number="109W", exp_date="2027-02-01",
    )
    assert result["lot_match"] is True
    assert result["exp_match"] is True


def test_duplicate_rows_conflicting_exp_picks_whichever_matches_ocr(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp="01/02/2027"),
        _row("109W", exp="05/03/2027"),
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)

    result_first = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2027-02-01"
    )
    assert result_first["exp_match"] is True

    result_second = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2027-03-05"
    )
    assert result_second["exp_match"] is True


def test_duplicate_rows_none_match_ocr_falls_back_to_first_row(monkeypatch):
    checker = SheetChecker()
    rows = [
        _row("109W", exp="01/02/2027"),
        _row("109W", exp="05/03/2027"),
    ]
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: rows)
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="109W", exp_date="2099-01-01"
    )
    assert result["lot_match"] is True   # lot itself still matched
    assert result["exp_match"] is False  # but no row's EXP agrees with OCR


def test_no_rows_match_lot(monkeypatch):
    checker = SheetChecker()
    monkeypatch.setattr(checker, "_find_rows_by_lot", lambda lot, sheet_id, gid: [])
    result = checker.check(
        _cfg(), sheet_id="sid", gid=0, lot_number="ZZZZ", exp_date="2027-02-01"
    )
    assert result["lot_match"] is False
    assert result["exp_match"] is False


def test_lot_short_fallback_retries_via_new_method(monkeypatch):
    checker = SheetChecker()
    calls = []

    def fake_find(lot, sheet_id, gid):
        calls.append(lot)
        if lot == "AAAAAAAAA1234XX":
            return []
        return [_row("1234", exp="01022027")]

    monkeypatch.setattr(checker, "_find_rows_by_lot", fake_find)
    result = checker.check(
        _cfg(lot_short_fallback=True), sheet_id="sid", gid=0,
        lot_number="AAAAAAAAA1234XX", exp_date="2027-02-01",
    )
    assert calls == ["AAAAAAAAA1234XX", "1234"]
    assert result["lot_match"] is True
    assert result["exp_match"] is True


def test_find_rows_by_lot_returns_every_match_in_sheet_order(monkeypatch):
    checker = SheetChecker()
    monkeypatch.setattr(checker, "_get_rows", lambda sheet_id, gid: [
        _row(" 109w ", exp=""), _row("OTHER"), _row("109W", exp="01022027"),
    ])
    found = checker._find_rows_by_lot("109W", "sid", 0)
    assert [r["EXP"] for r in found] == ["", "01022027"]
