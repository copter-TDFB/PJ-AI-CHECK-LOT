"""Unit tests for services.cloudrun_deployer model promotion."""

import importlib


def _reload_with_dirs(monkeypatch, draft_dir, models_dir):
    monkeypatch.setenv("DRAFT_DIR", str(draft_dir))
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    from services import cloudrun_deployer
    importlib.reload(cloudrun_deployer)
    return cloudrun_deployer


def test_promote_draft_model_promotes_detector_and_classifier(tmp_path, monkeypatch):
    draft_models = tmp_path / "drafts" / "k1" / "models"
    draft_models.mkdir(parents=True)
    (draft_models / "full_detector.pt").write_bytes(b"DET")
    (draft_models / "full_classifier.pt").write_bytes(b"CLS")
    models_dir = tmp_path / "models"

    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", models_dir)
    promoted = dep.promote_draft_model("k1")

    assert (models_dir / "detector.pt").read_bytes() == b"DET"
    assert (models_dir / "classifier.pt").read_bytes() == b"CLS"
    assert set(promoted) == {"detector", "classifier"}


def test_promote_draft_model_detector_only(tmp_path, monkeypatch):
    draft_models = tmp_path / "drafts" / "k1" / "models"
    draft_models.mkdir(parents=True)
    (draft_models / "full_detector.pt").write_bytes(b"DET")
    models_dir = tmp_path / "models"

    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", models_dir)
    promoted = dep.promote_draft_model("k1")

    assert (models_dir / "detector.pt").read_bytes() == b"DET"
    assert not (models_dir / "classifier.pt").exists()
    assert set(promoted) == {"detector"}


def test_promote_draft_model_none_present(tmp_path, monkeypatch):
    (tmp_path / "drafts" / "k1" / "models").mkdir(parents=True)
    dep = _reload_with_dirs(monkeypatch, tmp_path / "drafts", tmp_path / "models")
    assert dep.promote_draft_model("k1") == {}
