import pytest

from pipeline.packaging_registry import PackagingRegistry


def _write_minimal_config(root):
    pkg = root / "config" / "packagings"
    pkg.mkdir(parents=True)
    (root / "config" / "message_templates").mkdir(parents=True)
    (pkg / "demo.yaml").write_text("key: demo\nlot_patterns:\n  - 'LOT[0-9]+'\n", encoding="utf-8")


def test_registry_reads_config_dir_env(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    reg = PackagingRegistry()

    assert reg.get("demo") is not None
    assert reg.get("demo").key == "demo"


def test_registry_defaults_to_repo_config(monkeypatch):
    monkeypatch.delenv("OCR_CONFIG_DIR", raising=False)

    reg = PackagingRegistry()

    # The repo ships these production packagings under config/packagings/
    assert "back_label" in reg.all_keys()


def test_deployer_packaging_dir_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    from services import cloudrun_deployer

    draft = {
        "display_name": "Demo",
        "config": {
            "lot_patterns": ["LOT[0-9]+"],
            "fields_extracted": ["lot"],
            "sheet_checks": ["lot"],
            "message_template_key": "lot_only",
        },
    }
    out = cloudrun_deployer.write_packaging_yaml("demo", draft)

    assert out == tmp_path / "config" / "packagings" / "demo.yaml"
    assert out.exists()
