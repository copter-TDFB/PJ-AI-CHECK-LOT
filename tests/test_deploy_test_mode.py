import json

import api.packagings as pk


def test_test_mode_skips_cloud_run_trigger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    # eval.json the endpoint reads at the hardcoded relative path data/drafts/<key>/models/
    eval_path = tmp_path / "data" / "drafts" / "demo" / "models" / "eval.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps({"detector": {"map50": 0.99}}), encoding="utf-8")

    draft = {
        "display_name": "Demo",
        "parent_key": None,
        "config": {
            "lot_patterns": ["LOT[0-9]+"],
            "fields_extracted": ["lot"],
            "sheet_checks": ["lot"],
            "message_template_key": "lot_only",
        },
    }
    monkeypatch.setattr(pk.packaging_store, "get_draft", lambda key: draft)
    monkeypatch.setattr(pk.packaging_store, "update_draft", lambda *a, **k: None)

    from services import cloudrun_deployer, eval_thresholds
    monkeypatch.setattr(
        eval_thresholds, "check_hard_floor",
        lambda data: {"passed": True, "failures": []},
    )

    def _boom():
        raise AssertionError("trigger_cloud_run_revision must NOT run in TEST_MODE")
    monkeypatch.setattr(cloudrun_deployer, "trigger_cloud_run_revision", _boom)

    import main
    monkeypatch.setattr(main, "registry", None)
    monkeypatch.setattr(main, "reload_registry", lambda: None)

    result = pk.deploy_packaging("demo")

    assert result["deployed"] is True
    assert result["cloud_run"] == {"triggered": False, "reason": "test mode (simulated)"}


def test_deploy_eval_gate_honors_draft_dir(tmp_path, monkeypatch):
    # eval.json under a CUSTOM DRAFT_DIR must be found — proves the gate does not
    # read the hardcoded data/drafts.
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("DRAFT_DIR", str(tmp_path / "mydrafts"))
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    eval_path = tmp_path / "mydrafts" / "demo" / "models" / "eval.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps({"detector": {"map50": 0.99}}), encoding="utf-8")

    draft = {
        "display_name": "Demo",
        "parent_key": None,
        "config": {
            "lot_patterns": ["LOT[0-9]+"],
            "fields_extracted": ["lot"],
            "sheet_checks": ["lot"],
            "message_template_key": "lot_only",
        },
    }
    monkeypatch.setattr(pk.packaging_store, "get_draft", lambda key: draft)
    monkeypatch.setattr(pk.packaging_store, "update_draft", lambda *a, **k: None)

    from services import cloudrun_deployer, eval_thresholds
    monkeypatch.setattr(
        eval_thresholds, "check_hard_floor",
        lambda data: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(cloudrun_deployer, "trigger_cloud_run_revision", lambda: {"triggered": False})

    import main
    monkeypatch.setattr(main, "registry", None)
    monkeypatch.setattr(main, "reload_registry", lambda: None)

    result = pk.deploy_packaging("demo")
    assert result["deployed"] is True
