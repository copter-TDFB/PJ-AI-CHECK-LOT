"""
tests สำหรับ utils/validators.py — ทดสอบ regex extraction ต่อ class
รัน: python -m pytest tests/test_ocr.py -v
"""

import pytest

from utils.validators import find_expiry, find_lot, find_mfg, normalize_date


# ─── normalize_date ────────────────────────────────────────────────────────────

class TestNormalizeDate:
    def test_slash_4digit_year(self):
        assert normalize_date("23/03/2026") == "2026-03-23"

    def test_slash_2digit_year_2000s(self):
        assert normalize_date("23/03/26") == "2026-03-23"

    def test_slash_2digit_year_1900s(self):
        assert normalize_date("23/03/75") == "1975-03-23"

    def test_compact_8digit(self):
        assert normalize_date("23032026") == "2026-03-23"

    def test_invalid_date(self):
        assert normalize_date("32/13/2026") is None

    def test_empty(self):
        assert normalize_date("") is None


# ─── find_lot — generic ────────────────────────────────────────────────────────

class TestFindLotGeneric:
    def test_lot_colon(self):
        assert find_lot("LOT: AB20250101") == "AB20250101"

    def test_lot_no(self):
        assert find_lot("LOT NO: TH202501") == "TH202501"

    def test_batch_no(self):
        assert find_lot("BATCH NO: XYZ1234") == "XYZ1234"

    def test_lot_newline(self):
        assert find_lot("LOT\nTH250601") == "TH250601"

    def test_no_lot(self):
        assert find_lot("EXP 23/03/2026 MFG 23/03/2025") is None

    def test_ignores_date_as_lot(self):
        assert find_lot("LOT 23/03/2026") is None


# ─── find_lot — per class ──────────────────────────────────────────────────────

class TestFindLotByClass:
    def test_back_label_lot(self):
        text = "สินค้าไทย\nน้ำหนัก 500g\nLOT: TH250601\nEXP 12/2027"
        assert find_lot(text, "back_label") == "TH250601"

    def test_container_label_lot(self):
        text = "LOT TDF-2025-001\nMFG 01/06/2025"
        assert find_lot(text, "container_label") == "TDF-2025-001"

    def test_grade_bag_lot(self):
        text = "MEDIUM GRADE\nLOT: GR250601\nEXP 06/2026"
        assert find_lot(text, "grade_bag") == "GR250601"

    def test_retail_sachet_lot(self):
        text = "EXCELLENT\nLOT RS25060101\nBBD 01/06/2026"
        assert find_lot(text, "retail_sachet") == "RS25060101"

    def test_class_specific_before_generic(self):
        # back_label pattern ควรจับ L-prefix ได้
        text = "L 250601A"
        result = find_lot(text, "back_label")
        assert result is not None

    def test_unknown_class_falls_back_to_generic(self):
        text = "LOT: AB1234"
        assert find_lot(text, "unknown_class") == "AB1234"


# ─── find_expiry ───────────────────────────────────────────────────────────────

class TestFindExpiry:
    def test_exp_keyword(self):
        assert find_expiry("EXP 23/03/2026") == "2026-03-23"

    def test_bbd_keyword(self):
        assert find_expiry("BBD: 01/12/2027") == "2027-12-01"

    def test_best_before(self):
        assert find_expiry("BEST BEFORE 15/06/2025") == "2025-06-15"

    def test_thai_keyword(self):
        assert find_expiry("วันหมดอายุ 23/03/2026") == "2026-03-23"

    def test_2digit_year(self):
        assert find_expiry("EXP 23/03/26") == "2026-03-23"

    def test_fallback_no_keyword(self):
        # ไม่มี keyword → fallback เอาวันแรกที่เจอ
        assert find_expiry("23/03/2026") == "2026-03-23"

    def test_no_date(self):
        assert find_expiry("LOT AB1234") is None


# ─── find_mfg ─────────────────────────────────────────────────────────────────

class TestFindMfg:
    def test_mfg_keyword(self):
        assert find_mfg("MFG 23/03/2025") == "2025-03-23"

    def test_manufactured_on(self):
        assert find_mfg("MANUFACTURED ON 01/01/2025") == "2025-01-01"

    def test_thai_keyword(self):
        assert find_mfg("วันผลิต 23/03/2025") == "2025-03-23"

    def test_mfg_date_label(self):
        assert find_mfg("MFG DATE: 01/06/2025") == "2025-06-01"

    def test_no_mfg(self):
        assert find_mfg("EXP 23/03/2026") is None

    def test_mfg_and_exp_together(self):
        text = "MFG 23/03/2025 EXP 23/03/2027"
        assert find_mfg(text) == "2025-03-23"
        assert find_expiry(text) == "2027-03-23"
