"""Phase 3 write-path tests: cloudrun_deployer GCS publish/archive helpers."""

from __future__ import annotations

import pytest

from services import cloudrun_deployer, gcs_store
from services.gcs_store import GcsStore
from tests.test_gcs_store import FakeClient


def _setup(monkeypatch, tmp_path, with_models=True):
    cfg_dir = tmp_path / "config" / "packagings"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "newclass.yaml").write_text("key: newclass\ndisplay_name: New\n", encoding="utf-8")
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    models = tmp_path / "models"
    models.mkdir()
    if with_models:
        (models / "detector.pt").write_bytes(b"det-weights")
        (models / "classifier.pt").write_bytes(b"clf-weights")
    monkeypatch.setattr(cloudrun_deployer, "_MODELS_DIR", models)

    store = GcsStore("test-bucket", client=FakeClient())
    monkeypatch.setattr(gcs_store, "get_store", lambda: store)
    return store


def test_publish_uploads_yaml_models_and_manifest_last(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    res = cloudrun_deployer.publish_packaging_to_gcs("newclass")

    assert res["published"] is True
    assert store.get_text("packagings/newclass.yaml").startswith("key: newclass")
    m = store.read_json("manifest.json")
    assert "newclass" in m["packagings"]
    assert m["models"]["detector"]["object"] == "models/detector.pt"
    assert m["models"]["classifier"]["sha256"]  # sha computed
    assert m["version"] == 1
    assert "updated_at" in m


def test_publish_removes_key_from_archived(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    store.write_json("manifest.json", {"version": 4, "packagings": {}, "archived": ["newclass"], "models": {}})
    cloudrun_deployer.publish_packaging_to_gcs("newclass")
    m = store.read_json("manifest.json")
    assert "newclass" not in m["archived"]
    assert "newclass" in m["packagings"]
    assert m["version"] == 5


def test_publish_manifest_untouched_on_upload_failure(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    store.write_json("manifest.json", {"version": 9, "packagings": {}, "archived": [], "models": {}})

    def boom(*a, **k):
        raise RuntimeError("gcs upload down")

    monkeypatch.setattr(store, "upload_file", boom)
    with pytest.raises(RuntimeError):
        cloudrun_deployer.publish_packaging_to_gcs("newclass")
    # commit point (manifest) never reached → unchanged
    assert store.read_json("manifest.json")["version"] == 9


def test_publish_missing_yaml_raises(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        cloudrun_deployer.publish_packaging_to_gcs("ghost")


def test_publish_no_bucket_is_noop(monkeypatch):
    monkeypatch.setattr(gcs_store, "get_store", lambda: None)
    res = cloudrun_deployer.publish_packaging_to_gcs("anything")
    assert res["published"] is False


def test_publish_rejects_unsafe_key(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        cloudrun_deployer.publish_packaging_to_gcs("../manifest")
    with pytest.raises(ValueError):
        cloudrun_deployer.publish_packaging_to_gcs("a/b")


def test_publish_manifest_untouched_on_yaml_upload_failure(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    store.write_json("manifest.json", {"version": 7, "packagings": {}, "archived": [], "models": {}})

    def boom(*a, **k):
        raise RuntimeError("gcs put_text down")

    monkeypatch.setattr(store, "put_text", boom)
    with pytest.raises(RuntimeError):
        cloudrun_deployer.publish_packaging_to_gcs("newclass")
    assert store.read_json("manifest.json")["version"] == 7


def test_publish_skips_unchanged_model_upload(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    det_sha = cloudrun_deployer.sha256_file(tmp_path / "models" / "detector.pt")
    clf_sha = cloudrun_deployer.sha256_file(tmp_path / "models" / "classifier.pt")
    store.write_json("manifest.json", {
        "version": 2, "packagings": {}, "archived": [],
        "models": {
            "detector": {"object": "models/detector.pt", "sha256": det_sha},
            "classifier": {"object": "models/classifier.pt", "sha256": clf_sha},
        },
    })
    calls = []
    monkeypatch.setattr(store, "upload_file", lambda obj, src: calls.append(obj))

    res = cloudrun_deployer.publish_packaging_to_gcs("newclass")
    assert calls == []  # unchanged models not re-uploaded
    m = store.read_json("manifest.json")
    assert m["models"]["detector"]["sha256"] == det_sha  # still recorded
    assert res["published"] is True


def test_set_archived_toggles_manifest(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path)
    store.write_json("manifest.json", {"version": 1, "packagings": {"foo": {"sha": "x"}}, "archived": [], "models": {}})

    cloudrun_deployer.set_archived_in_gcs("foo", True)
    m = store.read_json("manifest.json")
    assert "foo" in m["archived"] and "foo" not in m["packagings"]
    assert m["version"] == 2

    store.put_text("packagings/foo.yaml", "key: foo\n")
    cloudrun_deployer.set_archived_in_gcs("foo", False)
    m = store.read_json("manifest.json")
    assert "foo" not in m["archived"] and "foo" in m["packagings"]
    assert m["version"] == 3
