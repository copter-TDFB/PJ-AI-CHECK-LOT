"""TEST_MODE lets the backend boot without Vision/Sheets creds.

The wizard test harness never calls Vision/Sheets, but lifespan constructs them
eagerly. `_init_optional` swallows construction errors only in TEST_MODE.
"""

import pytest

import main


def _boom():
    raise RuntimeError("no GCP credentials")


def test_init_optional_returns_none_in_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    assert main._init_optional(_boom, "OcrEngine") is None


def test_init_optional_reraises_when_not_test_mode(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    with pytest.raises(RuntimeError):
        main._init_optional(_boom, "OcrEngine")


def test_init_optional_passes_through_success(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    sentinel = object()
    assert main._init_optional(lambda: sentinel, "X") is sentinel
