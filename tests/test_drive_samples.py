import pytest


class _FakeDrive:
    def __init__(self, list_calls):
        self._list_calls = list_calls

    def find_in_folder(self, parent, name):
        return {
            ("CLSROOT", "images"): "IMG",
            ("IMG", "back_label"): "CLASS",
        }.get((parent, name))

    def list_folder(self, parent):
        self._list_calls.append(parent)
        return [
            {"id": "1", "name": "a.jpg", "mimeType": "image/jpeg"},
            {"id": "2", "name": "b.png", "mimeType": "image/png"},
            {"id": "3", "name": "notes.txt", "mimeType": "text/plain"},
        ]


def test_class_images_resolves_chain_and_filters(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr("services.drive_client.DriveClient", lambda: _FakeDrive([]))

    out = drive_samples.class_images("back_label")
    assert [f["name"] for f in out] == ["a.jpg", "b.png"]
    assert out[0]["id"] == "1"


def test_class_images_cached_within_ttl(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    calls = []
    monkeypatch.setattr("services.drive_client.DriveClient", lambda: _FakeDrive(calls))

    drive_samples.class_images("back_label")
    drive_samples.class_images("back_label")
    assert len(calls) == 1  # second call served from cache

    drive_samples.clear_cache()
    drive_samples.class_images("back_label")
    assert len(calls) == 2


def test_class_images_env_unset_returns_empty(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.delenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", raising=False)
    assert drive_samples.class_images("back_label") == []


def test_class_images_exception_safe(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")

    def boom():
        raise RuntimeError("drive down")

    monkeypatch.setattr("services.drive_client.DriveClient", boom)
    assert drive_samples.class_images("back_label") == []
