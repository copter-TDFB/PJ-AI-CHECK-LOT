"""Tests สำหรับ services/config_overrides.py — runtime tuning overrides (ADR 0004)."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def local_mode(tmp_path, monkeypatch):
    """Local fallback mode — env var ว่าง, เขียนลงไฟล์ใน tmp."""
    path = tmp_path / "config_overrides.json"
    monkeypatch.delenv("DRIVE_CONFIG_OVERRIDES_FILE_ID", raising=False)
    monkeypatch.setenv("CONFIG_OVERRIDES_PATH", str(path))
    return path


@pytest.fixture
def drive_mode(monkeypatch):
    """Drive mode — env var ชี้ file id, DriveClient ถูก mock."""
    monkeypatch.setenv("DRIVE_CONFIG_OVERRIDES_FILE_ID", "fake-file-id")
    with patch("services.config_overrides.DriveClient") as mock_cls:
        yield mock_cls.return_value


# ─── load() — local mode ─────────────────────────────────

def test_load_returns_empty_when_file_missing(local_mode):
    from services import config_overrides

    assert config_overrides.load() == {}


def test_load_returns_saved_overrides(local_mode):
    from services import config_overrides

    local_mode.write_text(json.dumps({"back_label": {"conf_threshold": 0.75}}))
    assert config_overrides.load() == {"back_label": {"conf_threshold": 0.75}}


def test_load_corrupt_json_returns_empty(local_mode):
    from services import config_overrides

    local_mode.write_text("{not json")
    assert config_overrides.load() == {}


def test_load_non_dict_payload_returns_empty(local_mode):
    from services import config_overrides

    local_mode.write_text(json.dumps(["not", "a", "dict"]))
    assert config_overrides.load() == {}


# ─── save_conf_threshold() — local mode ──────────────────

def test_save_writes_file_and_returns_merged(local_mode):
    from services import config_overrides

    merged = config_overrides.save_conf_threshold("back_label", 0.75)
    assert merged == {"back_label": {"conf_threshold": 0.75}}
    assert json.loads(local_mode.read_text(encoding="utf-8")) == merged


def test_save_preserves_other_keys(local_mode):
    from services import config_overrides

    local_mode.write_text(json.dumps({"grade_bag": {"conf_threshold": 0.6}}))
    merged = config_overrides.save_conf_threshold("back_label", 0.9)
    assert merged["grade_bag"] == {"conf_threshold": 0.6}
    assert merged["back_label"] == {"conf_threshold": 0.9}


# ─── save_product_aliases() — local mode ─────────────────

def test_save_product_aliases_writes_and_returns_merged(local_mode):
    from services import config_overrides

    aliases = [{"canonical": "Medium {size}", "keywords": ["medium", "med"]}]
    merged = config_overrides.save_product_aliases("matcha_sachet", aliases)
    assert merged == {"matcha_sachet": {"product_aliases": aliases}}
    assert json.loads(local_mode.read_text(encoding="utf-8")) == merged


def test_save_product_aliases_coexists_with_conf(local_mode):
    from services import config_overrides

    config_overrides.save_conf_threshold("matcha_sachet", 0.7)
    aliases = [{"canonical": "Excellent", "keywords": ["excellent"]}]
    merged = config_overrides.save_product_aliases("matcha_sachet", aliases)
    assert merged["matcha_sachet"]["conf_threshold"] == 0.7
    assert merged["matcha_sachet"]["product_aliases"] == aliases
    # and the reverse: saving conf must not clobber aliases
    merged2 = config_overrides.save_conf_threshold("matcha_sachet", 0.8)
    assert merged2["matcha_sachet"]["product_aliases"] == aliases
    assert merged2["matcha_sachet"]["conf_threshold"] == 0.8


def test_save_product_aliases_normalizes_shape(local_mode):
    from services import config_overrides

    # extra keys dropped, keywords coerced to list
    merged = config_overrides.save_product_aliases(
        "x", [{"canonical": "A", "keywords": ("a", "b"), "junk": 1}])
    assert merged["x"]["product_aliases"] == [{"canonical": "A", "keywords": ["a", "b"]}]


# ─── Drive mode ──────────────────────────────────────────

def test_load_reads_from_drive(drive_mode):
    from services import config_overrides

    drive_mode.read_json.return_value = {"capsule_box": {"conf_threshold": 0.8}}
    assert config_overrides.load() == {"capsule_box": {"conf_threshold": 0.8}}
    drive_mode.read_json.assert_called_once_with("fake-file-id")


def test_load_drive_failure_returns_empty(drive_mode):
    from services import config_overrides

    drive_mode.read_json.side_effect = RuntimeError("drive down")
    assert config_overrides.load() == {}


def test_save_writes_drive_first(drive_mode):
    from services import config_overrides

    drive_mode.read_json.return_value = {}
    merged = config_overrides.save_conf_threshold("back_label", 0.7)
    assert merged == {"back_label": {"conf_threshold": 0.7}}
    drive_mode.update_file_content.assert_called_once()
    file_id, content = drive_mode.update_file_content.call_args[0][:2]
    assert file_id == "fake-file-id"
    assert json.loads(content.decode("utf-8")) == merged


def test_save_drive_failure_raises(drive_mode):
    from services import config_overrides

    drive_mode.read_json.return_value = {}
    drive_mode.update_file_content.side_effect = RuntimeError("quota")
    with pytest.raises(RuntimeError):
        config_overrides.save_conf_threshold("back_label", 0.7)
