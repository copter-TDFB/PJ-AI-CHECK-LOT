"""Unit tests for multi_field group encoding (utils/field_groups)."""

import pytest

from utils.field_groups import (
    FIELD_ORDER,
    parse_group,
    canonicalize_group,
    validate_groups,
)


def test_field_order_is_the_four_known_fields():
    assert FIELD_ORDER == ["lot", "exp", "product", "size"]


def test_parse_group_splits_on_underscore():
    assert parse_group("lot_exp") == ["lot", "exp"]
    assert parse_group("lot") == ["lot"]
    assert parse_group("") == []


def test_canonicalize_sorts_by_field_order_and_dedupes():
    assert canonicalize_group("exp_lot") == "lot_exp"
    assert canonicalize_group("size_product") == "product_size"
    assert canonicalize_group("lot_lot") == "lot"


def test_validate_groups_canonicalizes_each_entry():
    assert validate_groups(["exp_lot", "size_product"]) == ["lot_exp", "product_size"]


def test_validate_groups_accepts_single_token_groups_backward_compat():
    assert validate_groups(["lot", "exp", "product", "size"]) == [
        "lot", "exp", "product", "size"]


def test_validate_groups_rejects_unknown_token():
    with pytest.raises(ValueError, match="ไม่รู้จัก"):
        validate_groups(["lot_weight"])


def test_validate_groups_rejects_field_in_two_groups():
    with pytest.raises(ValueError, match="กรอบเดียว"):
        validate_groups(["lot_exp", "exp_size"])


def test_validate_groups_rejects_union_without_lot():
    with pytest.raises(ValueError, match="lot"):
        validate_groups(["exp_product", "size"])


def test_validate_groups_rejects_empty_group():
    with pytest.raises(ValueError, match="ว่าง"):
        validate_groups(["lot", ""])


def test_validate_groups_rejects_duplicate_group_after_canonicalize():
    with pytest.raises(ValueError, match="ซ้ำ"):
        validate_groups(["lot_exp", "exp_lot"])
