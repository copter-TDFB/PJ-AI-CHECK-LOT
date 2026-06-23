"""Phase 2 read-path tests: GCS overlay for registry, model_registry, config_overrides.

Uses a fake-backed GcsStore injected via monkeypatching gcs_store.get_store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from services import config_overrides, gcs_store, model_registry
from services.gcs_store import GcsStore
from tests.test_gcs_store import FakeClient
from pipeline.packaging_registry import PackagingRegistry


def _seeded_store(seed: dict[str, bytes]) -> GcsStore:
    client = FakeClient()
    store = GcsStore("test-bucket", client=client)
    for path, data in seed.items():
        store.put_bytes(path, data)
    return store


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(gcs_store, "get_store", lambda: store)


# ── PackagingRegistry overlay ────────────────────────────────────────────────
def test_overlay_adds_new_class(monkeypatch):
    store = _seeded_store({})
    store.write_json(gcs_store.MANIFEST_PATH, {"packagings": {"newclass": {"sha": "x"}}})
    store.put_text(
        "packagings/newclass.yaml",
        "key: newclass\ndisplay_name: New Class\nlot_patterns: ['LOT (\\d+)']\n",
    )
    _patch_store(monkeypatch, store)

    reg = PackagingRegistry()
    cfg = reg.get("newclass")
    assert cfg is not None
    assert cfg.display_name == "New Class"
    # existing image classes still present
    assert reg.get("retail_sachet") is not None


def test_overlay_overrides_existing_class(monkeypatch):
    store = _seeded_store({})
    store.write_json(gcs_store.MANIFEST_PATH, {"packagings": {"retail_sachet": {"sha": "x"}}})
    store.put_text(
        "packagings/retail_sachet.yaml",
        "key: retail_sachet\ndisplay_name: OVERRIDDEN\nlot_patterns: []\n",
    )
    _patch_store(monkeypatch, store)

    reg = PackagingRegistry()
    assert reg.get("retail_sachet").display_name == "OVERRIDDEN"


def test_overlay_tombstone_removes_class(monkeypatch):
    store = _seeded_store({})
    store.write_json(gcs_store.MANIFEST_PATH, {"archived": ["retail_sachet"]})
    _patch_store(monkeypatch, store)

    reg = PackagingRegistry()
    assert reg.get("retail_sachet") is None
    # other classes survive
    assert reg.get("back_label") is not None


def test_no_store_uses_image_only(monkeypatch):
    monkeypatch.setattr(gcs_store, "get_store", lambda: None)
    reg = PackagingRegistry()
    assert reg.get("retail_sachet") is not None  # baked-in image config


def test_overlay_missing_yaml_skips(monkeypatch):
    store = _seeded_store({})
    store.write_json(gcs_store.MANIFEST_PATH, {"packagings": {"ghost": {"sha": "x"}}})
    _patch_store(monkeypatch, store)
    reg = PackagingRegistry()
    assert reg.get("ghost") is None  # manifest lists it but no YAML → skipped, no crash


# ── model_registry GCS-first ─────────────────────────────────────────────────
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_model_sync_from_gcs(monkeypatch, tmp_path):
    clf, det = b"classifier-weights", b"detector-weights"
    store = _seeded_store({"models/classifier.pt": clf, "models/detector.pt": det})
    store.write_json(gcs_store.MANIFEST_PATH, {
        "models": {
            "classifier": {"object": "models/classifier.pt", "sha256": _sha(clf)},
            "detector": {"object": "models/detector.pt", "sha256": _sha(det)},
        }
    })
    _patch_store(monkeypatch, store)

    clf_path, det_path = model_registry.sync(manifest_file_id="", cache_dir=tmp_path)
    assert clf_path.read_bytes() == clf
    assert det_path.read_bytes() == det


def test_model_sync_gcs_sha_mismatch_falls_back(monkeypatch, tmp_path):
    store = _seeded_store({"models/classifier.pt": b"x", "models/detector.pt": b"y"})
    store.write_json(gcs_store.MANIFEST_PATH, {
        "models": {
            "classifier": {"object": "models/classifier.pt", "sha256": "wrong"},
            "detector": {"object": "models/detector.pt", "sha256": "wrong"},
        }
    })
    _patch_store(monkeypatch, store)
    # mismatch → _sync_from_gcs returns None → with empty drive id → local fallback paths
    clf_path, det_path = model_registry.sync(manifest_file_id="", cache_dir=tmp_path)
    assert clf_path == model_registry._LOCAL_CLF
    assert det_path == model_registry._LOCAL_DET


def test_model_sync_no_models_in_manifest_falls_back(monkeypatch, tmp_path):
    store = _seeded_store({})
    store.write_json(gcs_store.MANIFEST_PATH, {"packagings": {}})
    _patch_store(monkeypatch, store)
    clf_path, det_path = model_registry.sync(manifest_file_id="", cache_dir=tmp_path)
    assert clf_path == model_registry._LOCAL_CLF


# ── config_overrides on GCS ──────────────────────────────────────────────────
def test_conf_overrides_load_from_gcs(monkeypatch):
    store = _seeded_store({})
    store.write_json("config_overrides.json", {"back_label": {"conf_threshold": 0.8}})
    _patch_store(monkeypatch, store)
    assert config_overrides.load() == {"back_label": {"conf_threshold": 0.8}}


def test_conf_overrides_absent_returns_empty(monkeypatch):
    _patch_store(monkeypatch, _seeded_store({}))
    assert config_overrides.load() == {}


def test_conf_overrides_save_to_gcs(monkeypatch):
    store = _seeded_store({})
    _patch_store(monkeypatch, store)
    merged = config_overrides.save_conf_threshold("capsule_box", 0.7)
    assert merged["capsule_box"]["conf_threshold"] == 0.7
    assert store.read_json("config_overrides.json")["capsule_box"]["conf_threshold"] == 0.7
