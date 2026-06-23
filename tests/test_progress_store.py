"""Tests for services/progress_store.py — in-memory training upload progress."""

import pytest

from services import progress_store


@pytest.fixture(autouse=True)
def _clean():
    progress_store.clear("k1")
    yield
    progress_store.clear("k1")


def test_get_unknown_key_returns_idle():
    snap = progress_store.get("k1")
    assert snap["phase"] == "idle"
    assert snap["percent"] is None


def test_report_computes_percent():
    progress_store.report("k1", "upload_images", done=3, total=12, detail="a.jpg")
    snap = progress_store.get("k1")
    assert snap == {
        "phase": "upload_images",
        "done": 3,
        "total": 12,
        "percent": 25,
        "detail": "a.jpg",
    }


def test_report_without_total_has_no_percent():
    progress_store.report("k1", "notebook")
    snap = progress_store.get("k1")
    assert snap["phase"] == "notebook"
    assert snap["percent"] is None


def test_report_replaces_previous_snapshot():
    progress_store.report("k1", "upload_images", done=1, total=4)
    progress_store.report("k1", "done")
    assert progress_store.get("k1")["phase"] == "done"


def test_clear_resets_to_idle():
    progress_store.report("k1", "upload_images", done=1, total=4)
    progress_store.clear("k1")
    assert progress_store.get("k1")["phase"] == "idle"
