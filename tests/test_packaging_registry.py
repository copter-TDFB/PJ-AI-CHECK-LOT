"""Tests สำหรับ PackagingRegistry — overrides merge (ADR 0004)."""

from pipeline.packaging_registry import PackagingRegistry


def test_no_overrides_uses_yaml_value():
    reg = PackagingRegistry()
    assert reg.get("back_label").conf_threshold == 0.8


def test_override_replaces_conf_threshold():
    reg = PackagingRegistry(overrides={"back_label": {"conf_threshold": 0.55}})
    assert reg.get("back_label").conf_threshold == 0.55


def test_override_leaves_other_keys_untouched():
    reg = PackagingRegistry(overrides={"back_label": {"conf_threshold": 0.55}})
    assert reg.get("grade_bag").conf_threshold == 0.6


def test_override_unknown_key_ignored():
    reg = PackagingRegistry(overrides={"no_such_pkg": {"conf_threshold": 0.9}})
    assert reg.get("no_such_pkg") is None
    assert reg.get("back_label").conf_threshold == 0.8


def test_override_invalid_value_keeps_yaml_value():
    reg = PackagingRegistry(overrides={"back_label": {"conf_threshold": "abc"}})
    assert reg.get("back_label").conf_threshold == 0.8


def test_override_non_dict_entry_ignored():
    reg = PackagingRegistry(overrides={"back_label": 0.55})
    assert reg.get("back_label").conf_threshold == 0.8
