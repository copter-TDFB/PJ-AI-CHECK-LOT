"""
tests สำหรับ utils/validators.py — ทดสอบ regex extraction ต่อ class
รัน: python -m pytest tests/test_ocr.py -v
"""

import re

import pytest

from utils.validators import (
    find_expiry,
    find_lot,
    find_mfg,
    find_product_name,
    find_size,
    normalize_date,
    resolve_product_template,
)


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

    def test_back_label_lot_new_format(self):
        assert (
            find_lot(
                "LOT: ER0010TDF019616726P\nEXP 12/2027",
                "back_label",
            )
            == "ER0010TDF019616726P"
        )

    def test_back_label_lot_old_format(self):
        assert (
            find_lot(
                "LOT: ER0010000019616726P\nEXP 12/2027",
                "back_label",
            )
            == "ER0010000019616726P"
        )

    def test_container_label_lot(self):
        text = "LOT TDF-2025-001\nMFG 01/06/2025"
        assert find_lot(text, "container_label") == "TDF-2025-001"

    def test_container_label_lot_new_format(self):
        assert (
            find_lot(
                "ER0010TDF019616726P\nMFG 01/06/2025",
                "container_label",
            )
            == "ER0010TDF019616726P"
        )

    def test_container_label_lot_old_format(self):
        assert (
            find_lot(
                "ER0010000019616726P1\nMFG 01/06/2025",
                "container_label",
            )
            == "ER0010000019616726P1"
        )

    def test_container_label_lot_suffix_p2(self):
        assert (
            find_lot(
                "ER0010TDF019616726P2\nMFG 01/06/2025",
                "container_label",
            )
            == "ER0010TDF019616726P2"
        )

    def test_grade_bag_lot(self):
        text = "MEDIUM GRADE\nLOT: GR250601\nEXP 06/2026"
        assert find_lot(text, "grade_bag") == "GR250601"

    def test_grade_bag_lot_new_format(self):
        assert (
            find_lot(
                "MEDIUM GRADE\nLOT: ER0010TDF019616726P1\nEXP 06/2026",
                "grade_bag",
            )
            == "ER0010TDF019616726P1"
        )

    def test_grade_bag_lot_old_format(self):
        assert (
            find_lot(
                "MEDIUM GRADE\nER0010000019616726P\nEXP 06/2026",
                "grade_bag",
            )
            == "ER0010000019616726P"
        )

    def test_grade_bag_lot_new_format_bare(self):
        # bare line (no LOT: prefix) → exercises the word-bound pattern
        assert (
            find_lot(
                "MEDIUM GRADE\nER0010TDF019616726P\nEXP 06/2026",
                "grade_bag",
            )
            == "ER0010TDF019616726P"
        )

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


# ─── find_product_name — config-driven aliases ──────────────────────────────────

class TestFindProductNameAliases:
    """aliases รูป [{"canonical": str, "keywords": [str]}] — match ตามลำดับ (priority)."""

    def test_keyword_found_in_noisy_text(self):
        # detector ตีกรอบกว้าง → OCR รก ๆ แต่ยังเจอ keyword
        aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
        text = "NET WT 100g\nHOUJICHA\nLOT TH123456"
        assert find_product_name(text, aliases) == "Houjicha Powder"

    def test_priority_specific_keyword_first(self):
        aliases = [
            {"canonical": "Excellent Rich 95%", "keywords": ["excellent rich"]},
            {"canonical": "Excellent", "keywords": ["excellent"]},
        ]
        assert find_product_name("brand excellent rich blend", aliases) == "Excellent Rich 95%"

    def test_no_keyword_match_returns_none(self):
        aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
        assert find_product_name("genmaicha green tea", aliases) is None

    def test_word_boundary_no_substring_match(self):
        # "rich" ต้องไม่ match ใน "enriched"
        aliases = [{"canonical": "Rich", "keywords": ["rich"]}]
        assert find_product_name("enriched flavor blend", aliases) is None

    def test_case_insensitive(self):
        aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
        assert find_product_name("HOUJICHA POWDER", aliases) == "Houjicha Powder"

    def test_keyword_with_punctuation_is_literal(self):
        # keyword มีอักขระ regex (%) — ต้อง treat เป็น literal ไม่ crash และ match ท้าย %
        aliases = [{"canonical": "Matcha 95%", "keywords": ["matcha 95%"]}]
        assert find_product_name("contains matcha 95% green", aliases) == "Matcha 95%"

    def test_no_aliases_falls_back_to_hardcoded(self):
        assert find_product_name("a houjicha b") == "Houjicha Powder"

    def test_empty_aliases_falls_back_to_hardcoded(self):
        assert find_product_name("classic blend", aliases=[]) == "Classic"

    def test_nonempty_aliases_with_blank_keywords_returns_none(self):
        # อธิบายพฤติกรรม: aliases ที่ "ไม่ว่าง" (opt-in) แต่ไม่มี keyword ใช้งานได้
        # → ไม่ fallback hardcode, คืน None (wizard กรองทิ้งก่อนส่งแล้ว — ดู collectConfig)
        aliases = [{"canonical": "Houjicha Powder", "keywords": []}]
        assert find_product_name("houjicha classic", aliases) is None


# ─── resolve_product_template ───────────────────────────────

class TestResolveProductTemplate:
    def test_substitutes_size_token(self):
        assert resolve_product_template("Medium {size}", "40 g") == "Medium 40 g"

    def test_no_token_returns_literal(self):
        assert resolve_product_template("Houjicha Powder", "40 g") == "Houjicha Powder"

    def test_no_token_ignores_missing_size(self):
        assert resolve_product_template("Excellent", None) == "Excellent"

    def test_token_present_but_size_missing_returns_none(self):
        assert resolve_product_template("Medium {size}", None) is None

    def test_collapses_double_space(self):
        # awkward template should not leave a double space when size has its own spacing
        assert resolve_product_template("Medium  {size}", "40 g") == "Medium 40 g"

    def test_token_anywhere_and_suffix(self):
        assert resolve_product_template("Excellent {size} Powder", "200 g") == "Excellent 200 g Powder"

    def test_empty_template_returns_none(self):
        assert resolve_product_template("", "40 g") is None
        assert resolve_product_template(None, "40 g") is None


# ─── product composition paths (mirrors ocr_engine branch) ──────────

class TestProductCompositionPaths:
    def test_aliases_path_resolves_size_token(self):
        aliases = [{"canonical": "Medium {size}", "keywords": ["medium"]}]
        text = "medium 40 g"
        name = find_product_name(text, aliases)        # -> "Medium {size}"
        size = find_size(text)                          # -> "40 g"
        assert resolve_product_template(name, size) == "Medium 40 g"

    def test_aliases_path_literal_when_no_token(self):
        aliases = [{"canonical": "Houjicha Powder", "keywords": ["houjicha"]}]
        text = "houjicha tea"
        name = find_product_name(text, aliases)         # -> "Houjicha Powder"
        size = find_size(text)                          # -> None (no size printed)
        assert resolve_product_template(name, size) == "Houjicha Powder"

    def test_legacy_fallback_appends_size(self):
        # no aliases -> hardcoded fallback; ocr_engine appends size as before
        name = find_product_name("houjicha tea", None)  # -> "Houjicha Powder"
        size = "40 g"
        composed = f"{name} {size}" if (name and size) else name
        assert composed == "Houjicha Powder 40 g"


def test_find_lot_config_pattern_without_capture_group():
    # Wizard-generated lot_patterns may have no capture group (e.g. M3/M4 print
    # sticker). find_lot must fall back to the full match, not crash on group(1).
    pat = [re.compile(r"(?i)[A-Z]{1}\d{17,}[A-Z0-9]*")]
    assert find_lot("A12345678901234567", patterns=pat) == "A12345678901234567"


def test_print_sticker_back_new_format():
    pat = [
        re.compile(
            r"(?i)([A-Z]{1}(?:\d{17,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)"
        )
    ]
    assert find_lot("M40001YST001613226R", patterns=pat) == "M40001YST001613226R"
