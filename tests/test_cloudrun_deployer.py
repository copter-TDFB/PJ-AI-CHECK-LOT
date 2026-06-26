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


def test_trigger_cloud_run_routes_traffic_to_latest(monkeypatch):
    # The triggered revision must RECEIVE traffic, else a wizard deploy publishes
    # new models to GCS but the new revision sits at 0% (traffic stays pinned).
    from services import cloudrun_deployer as dep
    captured = {}

    class _Exec:
        def __init__(self, ret): self.ret = ret
        def execute(self): return self.ret

    class _Services:
        def get(self, name): return _Exec({"template": {"annotations": {}}})
        def patch(self, name, body, updateMask):
            captured["body"] = body
            captured["updateMask"] = updateMask
            return _Exec({"name": "operations/op1"})

    class _Run:
        def projects(self):
            class _P:
                def locations(self_inner):
                    class _L:
                        def services(self_i2): return _Services()
                    return _L()
            return _P()

    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: _Run())
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (object(), "proj"))

    res = dep.trigger_cloud_run_revision()
    assert res["triggered"] is True
    assert "traffic" in captured["updateMask"]
    assert {"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "percent": 100} in captured["body"]["traffic"]
