"""Unit tests for services.gcs_store using an in-memory fake GCS client.

No real bucket / google-cloud-storage package needed — the fake mimics the
minimal google.cloud.storage surface GcsStore uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import gcs_store
from services.gcs_store import GcsStore


# ── in-memory fake GCS client ────────────────────────────────────────────────
class _FakeBlob:
    def __init__(self, store: dict, name: str):
        self._store = store
        self.name = name

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, data, content_type="application/octet-stream"):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._store[self.name] = bytes(data)

    def download_as_bytes(self) -> bytes:
        return self._store[self.name]

    def delete(self):
        self._store.pop(self.name, None)

    def upload_from_filename(self, src):
        self._store[self.name] = Path(src).read_bytes()

    def download_to_filename(self, dest):
        Path(dest).write_bytes(self._store[self.name])


class _FakeBucket:
    def __init__(self, store: dict):
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)


class _FakeListed:
    def __init__(self, name: str):
        self.name = name


class FakeClient:
    """Mimics google.cloud.storage.Client for one bucket."""

    def __init__(self):
        self._buckets: dict[str, dict] = {}

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._buckets.setdefault(name, {}))

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        store = self._buckets.setdefault(bucket_name, {})
        return [_FakeListed(n) for n in store if n.startswith(prefix)]


@pytest.fixture
def store() -> GcsStore:
    return GcsStore("test-bucket", client=FakeClient())


# ── tests ─────────────────────────────────────────────────────────────────────
def test_put_get_text_round_trip(store):
    store.put_text("packagings/foo.yaml", "key: foo\n")
    assert store.get_text("packagings/foo.yaml") == "key: foo\n"


def test_get_missing_returns_none(store):
    assert store.get_text("nope.yaml") is None
    assert store.get_bytes("nope.bin") is None
    assert store.read_json("nope.json") is None


def test_exists(store):
    assert not store.exists("packagings/foo.yaml")
    store.put_text("packagings/foo.yaml", "x")
    assert store.exists("packagings/foo.yaml")


def test_json_round_trip(store):
    payload = {"version": 3, "packagings": {"foo": {"sha": "abc"}}}
    store.write_json("manifest.json", payload)
    assert store.read_json("manifest.json") == payload


def test_delete(store):
    store.put_text("packagings/foo.archived", "1")
    assert store.exists("packagings/foo.archived")
    store.delete("packagings/foo.archived")
    assert not store.exists("packagings/foo.archived")
    # deleting a missing object is a no-op, not an error
    store.delete("packagings/foo.archived")


def test_list_prefix(store):
    store.put_text("packagings/a.yaml", "a")
    store.put_text("packagings/b.yaml", "b")
    store.put_text("models/detector.pt", "x")
    assert store.list_prefix("packagings/") == ["packagings/a.yaml", "packagings/b.yaml"]


def test_upload_download_file(tmp_path, store):
    src = tmp_path / "detector.pt"
    src.write_bytes(b"weights")
    store.upload_file("models/detector.pt", src)

    dest = tmp_path / "out" / "detector.pt"
    assert store.download_to("models/detector.pt", dest) is True
    assert dest.read_bytes() == b"weights"


def test_download_missing_returns_false(tmp_path, store):
    assert store.download_to("models/missing.pt", tmp_path / "x.pt") is False


def test_get_store_none_without_env(monkeypatch):
    monkeypatch.delenv(gcs_store.ENV_BUCKET, raising=False)
    assert gcs_store.get_store() is None


def test_get_store_returns_store_with_env(monkeypatch):
    monkeypatch.setenv(gcs_store.ENV_BUCKET, "my-bucket")
    s = gcs_store.get_store()
    assert isinstance(s, GcsStore)
    assert s.bucket_name == "my-bucket"
