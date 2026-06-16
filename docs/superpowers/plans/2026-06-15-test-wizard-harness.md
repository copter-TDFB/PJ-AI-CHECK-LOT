# Test Wizard Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an isolated test copy of the packaging wizard that runs the full flow (draft → upload → publish dataset → notebook link → simulated Deploy) against a dedicated test Drive folder, never touching production Cloud Run or the real Drive dataset.

**Architecture:** One codebase, two run profiles separated by env file + port. A `TEST_MODE=1` flag gates the single Cloud Run trigger call. Two hardcoded config-dir constants become call-time helpers reading `OCR_CONFIG_DIR`, so the test backend writes YAML/models into `data/test/*` instead of the real `config/`/`models/`. A launch script exports `.env.test` into the process env, bootstraps `data/test/`, and serves a copy of `wizard.html` pointed at the test backend on port 8081.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, python-dotenv, Google Drive API (via existing `DriveClient`), PowerShell launch script, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-test-wizard-harness-design.md`

---

## File Structure

**Modified (code, backward-compatible — defaults preserve current behavior):**
- `pipeline/packaging_registry.py` — `_CONFIG_DIR` constant → `_config_dir()` helper reading `OCR_CONFIG_DIR`.
- `services/cloudrun_deployer.py` — `_PACKAGING_DIR` constant → `_packaging_dir()` helper reading `OCR_CONFIG_DIR`.
- `api/packagings.py` — gate `trigger_cloud_run_revision()` behind `TEST_MODE` in `deploy_packaging`.
- `.gitignore` — ignore `data/test/` and `.env.test`.

**Created:**
- `.env.test.example` — committed template for the test profile env.
- `scripts/setup_test_drive.py` — creates the `OCR-LOT-TEST/` Drive tree, prints file_ids.
- `scripts/run_test_wizard.ps1` — bootstraps `data/test/`, exports `.env.test`, serves the test wizard + backend.
- `test wizzard/wizard.html` — copy of `web/wizard.html` with `API_BASE_OVERRIDE` → `:8081`.
- `test wizzard/README.md` — how to run the harness.

**Tests:**
- `tests/test_config_dir_env.py` — `OCR_CONFIG_DIR` redirects registry + deployer.
- `tests/test_deploy_test_mode.py` — `TEST_MODE` skips the Cloud Run trigger.

---

## Task 1: `OCR_CONFIG_DIR` env support in PackagingRegistry

**Files:**
- Modify: `pipeline/packaging_registry.py:1-10`, `:68`, `:97`, `:127`, `:131`
- Test: `tests/test_config_dir_env.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_dir_env.py`:

```python
import importlib

import pytest


def _write_minimal_config(root):
    pkg = root / "config" / "packagings"
    pkg.mkdir(parents=True)
    (pkg / "demo.yaml").write_text("key: demo\nlot_patterns:\n  - 'LOT[0-9]+'\n", encoding="utf-8")


def test_registry_reads_config_dir_env(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    import pipeline.packaging_registry as reg_mod
    importlib.reload(reg_mod)
    reg = reg_mod.PackagingRegistry()

    assert reg.get("demo") is not None
    assert reg.get("demo").key == "demo"


def test_registry_defaults_to_repo_config(monkeypatch):
    monkeypatch.delenv("OCR_CONFIG_DIR", raising=False)

    import pipeline.packaging_registry as reg_mod
    importlib.reload(reg_mod)
    reg = reg_mod.PackagingRegistry()

    # The repo ships these production packagings under config/packagings/
    assert "back_label" in reg.all_keys()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_dir_env.py::test_registry_reads_config_dir_env -v`
Expected: FAIL — registry still loads from the hardcoded repo `config/`, so `demo` is not found (`assert reg.get("demo") is not None` fails).

- [ ] **Step 3: Add `import os` and the helper, replace the constant**

In `pipeline/packaging_registry.py`, change the imports block (top of file) from:

```python
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"
```

to:

```python
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _config_dir() -> Path:
    """Config root — `OCR_CONFIG_DIR` env (test harness) or the repo `config/`."""
    override = os.getenv("OCR_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR
```

- [ ] **Step 4: Replace the four `_CONFIG_DIR` usages with `_config_dir()`**

Line 68 (in `_load`):
```python
        pkg_dir = _config_dir() / "packagings"
```
Line 97 (in `_load`):
```python
        tmpl_dir = _config_dir() / "message_templates"
```
Line 127 (in `is_archived`):
```python
        return (_config_dir() / "packagings" / f"{key}.yaml.archived").exists()
```
Line 131 (in `archived_keys`):
```python
        pkg_dir = _config_dir() / "packagings"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_dir_env.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add pipeline/packaging_registry.py tests/test_config_dir_env.py
git commit -m "feat: PackagingRegistry honors OCR_CONFIG_DIR env"
```

---

## Task 2: `OCR_CONFIG_DIR` env support in cloudrun_deployer

**Files:**
- Modify: `services/cloudrun_deployer.py:25`, `:46`, `:98`, `:120`, `:122`, `:151`, `:190`, `:193`, `:203`, `:206`
- Test: `tests/test_config_dir_env.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_dir_env.py`:

```python
def test_deployer_packaging_dir_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))

    import services.cloudrun_deployer as dep_mod
    importlib.reload(dep_mod)

    draft = {
        "display_name": "Demo",
        "config": {
            "lot_patterns": ["LOT[0-9]+"],
            "fields_extracted": ["lot"],
            "sheet_checks": ["lot"],
            "message_template_key": "lot_only",
        },
    }
    out = dep_mod.write_packaging_yaml("demo", draft)

    assert out == tmp_path / "config" / "packagings" / "demo.yaml"
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_dir_env.py::test_deployer_packaging_dir_reads_env -v`
Expected: FAIL — `write_packaging_yaml` writes to the hardcoded `config/packagings/demo.yaml`, so the path assertion fails (and it would pollute the real config dir).

- [ ] **Step 3: Replace the constant with a call-time helper**

In `services/cloudrun_deployer.py`, change line 25 from:

```python
_PACKAGING_DIR = Path("config/packagings")
```

to:

```python
def _packaging_dir() -> Path:
    """Packagings dir — `OCR_CONFIG_DIR` env (test harness) or repo `config/packagings`."""
    return Path(os.getenv("OCR_CONFIG_DIR", "config")) / "packagings"
```

(`import os` already exists at line 15.)

- [ ] **Step 4: Replace every `_PACKAGING_DIR` usage with `_packaging_dir()`**

Update each site (the surrounding code is unchanged — only the constant reference):

Line 46 (`write_packaging_yaml`):
```python
    out = _packaging_dir() / f"{key}.yaml"
```
Line 98 (`write_packaging_yaml`):
```python
    _packaging_dir().mkdir(parents=True, exist_ok=True)
```
Line 120 (`backup_artifacts`):
```python
    yaml_src = _packaging_dir() / f"{parent_key}.yaml"
```
Line 122 (`backup_artifacts`):
```python
        dst = _packaging_dir() / f"{parent_key}.yaml.bak-{ts}"
```
Line 151 (`_rotate_backups`):
```python
    yaml_baks = sorted(_packaging_dir().glob(f"{parent_key}.yaml.bak-*"))
```
Line 190 (`archive_packaging`):
```python
    src = _packaging_dir() / f"{key}.yaml"
```
Line 193 (`archive_packaging`):
```python
    dst = _packaging_dir() / f"{key}.yaml.archived"
```
Line 203 (`unarchive_packaging`):
```python
    src = _packaging_dir() / f"{key}.yaml.archived"
```
Line 206 (`unarchive_packaging`):
```python
    dst = _packaging_dir() / f"{key}.yaml"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_dir_env.py -v`
Expected: PASS (all three tests).

- [ ] **Step 6: Verify no other reference to the old constant remains**

Run: `python -m pytest tests/test_api_packagings.py -q`
Expected: PASS (no regression — existing deploy/archive tests still green). If any test referenced `_PACKAGING_DIR` directly, update it to `_packaging_dir()`.

- [ ] **Step 7: Commit**

```bash
git add services/cloudrun_deployer.py tests/test_config_dir_env.py
git commit -m "feat: cloudrun_deployer honors OCR_CONFIG_DIR env"
```

---

## Task 3: Gate the Cloud Run trigger behind `TEST_MODE`

**Files:**
- Modify: `api/packagings.py` (`deploy_packaging`, around line 659-660)
- Test: `tests/test_deploy_test_mode.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deploy_test_mode.py`:

```python
import json

import pytest

import api.packagings as pk


@pytest.fixture
def deployable_draft(tmp_path, monkeypatch):
    """A draft that passes the hard-floor gate, wired to temp dirs."""
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))

    draft_root = tmp_path / "drafts" / "demo"
    (draft_root / "models").mkdir(parents=True)
    eval_path = draft_root / "models" / "eval.json"
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
    monkeypatch.setattr(pk.packaging_store, "delete_draft", lambda *a, **k: None)

    import main
    monkeypatch.setattr(main, "registry", None)        # fresh-deploy collision check passes
    monkeypatch.setattr(main, "reload_registry", lambda: None)

    # Point the endpoint's eval.json lookup at our temp draft dir.
    monkeypatch.setattr(pk, "Path", pk.Path)  # no-op, keeps Path importable
    return tmp_path, eval_path


def test_test_mode_skips_cloud_run_trigger(deployable_draft, monkeypatch):
    tmp_path, _ = deployable_draft
    monkeypatch.setenv("TEST_MODE", "1")

    # Hard fail if the real trigger is ever called.
    from services import cloudrun_deployer
    def _boom():
        raise AssertionError("trigger_cloud_run_revision must NOT run in TEST_MODE")
    monkeypatch.setattr(cloudrun_deployer, "trigger_cloud_run_revision", _boom)

    # Redirect the eval.json the endpoint reads (data/drafts/<key>/models/eval.json).
    monkeypatch.setenv("DRAFT_DIR", str(tmp_path / "drafts"))
    monkeypatch.setattr(pk, "_draft_eval_path", None, raising=False)

    result = pk.deploy_packaging("demo")

    assert result["deployed"] is True
    assert result["cloud_run"] == {"triggered": False, "reason": "test mode (simulated)"}
```

> NOTE for the implementer: `deploy_packaging` reads the eval file via a hardcoded
> `Path("data/drafts") / key / "models" / "eval.json"`. Confirm the exact construction
> in `api/packagings.py` (~line 622) and, if it does not honor `DRAFT_DIR`, set the test
> draft up under the real relative path `data/drafts/demo/models/eval.json` inside a
> `monkeypatch.chdir(tmp_path)` instead of the env redirect. Adjust the fixture so the
> gate passes; the assertion under test is purely the `cloud_run` field.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deploy_test_mode.py -v`
Expected: FAIL — current code always calls `cloudrun_deployer.trigger_cloud_run_revision()`, so `_boom` raises `AssertionError`.

- [ ] **Step 3: Add the `TEST_MODE` gate in `deploy_packaging`**

In `api/packagings.py`, find (around line 659-660):

```python
    # 6. Trigger Cloud Run revision (non-fatal if IAM lacks role)
    cr_result = cloudrun_deployer.trigger_cloud_run_revision()
```

Replace with:

```python
    # 6. Trigger Cloud Run revision (non-fatal if IAM lacks role).
    #    In TEST_MODE the trigger is the one call that can touch production
    #    Cloud Run, so it is skipped entirely and the deploy is simulated.
    if os.getenv("TEST_MODE") == "1":
        cr_result = {"triggered": False, "reason": "test mode (simulated)"}
        logger.info("TEST_MODE — skipping Cloud Run trigger for '%s'", target_key)
    else:
        cr_result = cloudrun_deployer.trigger_cloud_run_revision()
```

(`import os` already exists at `api/packagings.py:5`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deploy_test_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_deploy_test_mode.py
git commit -m "feat: TEST_MODE skips Cloud Run trigger on deploy"
```

---

## Task 4: `.env.test.example` template + gitignore

**Files:**
- Create: `.env.test.example`
- Modify: `.gitignore`

- [ ] **Step 1: Create `.env.test.example`**

Create `.env.test.example` (committed template — real `.env.test` is gitignored and filled with ids from Task 5):

```bash
# Test wizard harness profile. Copy to .env.test and fill the DRIVE_* ids
# printed by `python scripts/setup_test_drive.py`. NEVER point these at prod.

# --- Test isolation ---
TEST_MODE=1
OCR_CONFIG_DIR=data/test/config
MODELS_DIR=data/test/models
DRAFT_DIR=data/test/drafts
CROP_CACHE_DIR=data/test/crops

# Local model fallback (DRIVE_MANIFEST_FILE_ID stays empty → load these)
DRIVE_MANIFEST_FILE_ID=
MODEL_CLASSIFIER_PATH=data/test/models/classifier.pt
MODEL_DETECTOR_PATH=data/test/models/detector.pt

# --- Test Drive (filled by scripts/setup_test_drive.py) ---
DRIVE_DETECTOR_DATASET_FOLDER_ID=
DRIVE_CLASSIFIER_DATASET_FOLDER_ID=
DRIVE_CONFIG_OVERRIDES_FILE_ID=

# --- Same OAuth user as prod (Drive writes) ---
DRIVE_OAUTH_CLIENT_ID=
DRIVE_OAUTH_CLIENT_SECRET=
DRIVE_OAUTH_REFRESH_TOKEN=

# --- Reference dataset folder names inside the test Drive root ---
REFERENCE_DETECTOR_PATH=data check lot
REFERENCE_CLASSIFIER_PATH=data classify check lot

# --- GCP Vision (local SA key, same as dev) ---
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
LOG_LEVEL=INFO
```

- [ ] **Step 2: Add ignores**

Append to `.gitignore`:

```
# Test wizard harness
.env.test
data/test/
```

- [ ] **Step 3: Verify the ignore works**

Run: `git check-ignore .env.test data/test/`
Expected output:
```
.env.test
data/test/
```

- [ ] **Step 4: Commit**

```bash
git add .env.test.example .gitignore
git commit -m "chore: test wizard env template + gitignore data/test"
```

---

## Task 5: `scripts/setup_test_drive.py`

**Files:**
- Create: `scripts/setup_test_drive.py`

Creates the test Drive tree as the OAuth user and prints ids to paste into `.env.test`.

- [ ] **Step 1: Write the script**

Create `scripts/setup_test_drive.py`:

```python
"""Create (or reset) the OCR-LOT-TEST Drive tree for the test wizard harness.

Run from the repo root with the prod .env loaded so DRIVE_OAUTH_* are present
(the OAuth user has full `drive` scope, avoiding the drive.file blind spot on
manually-created folders):

    python scripts/setup_test_drive.py            # create, skip existing
    python scripts/setup_test_drive.py --reset     # delete the root, recreate

Prints the file_ids to paste into .env.test. Creates:

    OCR-LOT-TEST/
    ├── data check lot/            -> DRIVE_DETECTOR_DATASET_FOLDER_ID
    ├── data classify check lot/   -> DRIVE_CLASSIFIER_DATASET_FOLDER_ID
    └── config_overrides.json {}   -> DRIVE_CONFIG_OVERRIDES_FILE_ID
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

# Standalone script — bootstrap repo root so `services` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from googleapiclient.http import MediaIoBaseUpload  # noqa: E402

from services.drive_client import DriveClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("setup_test_drive")

ROOT_NAME = "OCR-LOT-TEST"
FOLDER_MIME = "application/vnd.google-apps.folder"


def _find_root(svc) -> str | None:
    q = (
        f"name = '{ROOT_NAME}' and mimeType = '{FOLDER_MIME}' "
        "and trashed = false and 'me' in owners"
    )
    res = svc.files().list(q=q, fields="files(id,name)", spaces="drive").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _make_folder(svc, name: str, parent: str | None = None) -> str:
    body = {"name": name, "mimeType": FOLDER_MIME}
    if parent:
        body["parents"] = [parent]
    return svc.files().create(body=body, fields="id").execute()["id"]


def _make_json_file(svc, name: str, parent: str, payload: dict) -> str:
    body = {"name": name, "parents": [parent]}
    media = MediaIoBaseUpload(
        io.BytesIO(json.dumps(payload).encode("utf-8")),
        mimetype="application/json",
    )
    return svc.files().create(body=body, media_body=media, fields="id").execute()["id"]


def main() -> None:
    load_dotenv()
    reset = "--reset" in sys.argv

    client = DriveClient()
    svc = client._svc  # same authenticated service the app uses

    who = svc.about().get(fields="user").execute().get("user", {})
    logger.info("Authenticated as: %s", who.get("emailAddress", "<unknown>"))

    existing = _find_root(svc)
    if existing and reset:
        svc.files().delete(fileId=existing).execute()
        logger.info("Deleted existing %s (%s)", ROOT_NAME, existing)
        existing = None
    if existing and not reset:
        logger.warning(
            "%s already exists (%s). Re-run with --reset to recreate.",
            ROOT_NAME, existing,
        )
        sys.exit(1)

    root = _make_folder(svc, ROOT_NAME)
    det = _make_folder(svc, "data check lot", root)
    cls = _make_folder(svc, "data classify check lot", root)
    overrides = _make_json_file(svc, "config_overrides.json", root, {})

    logger.info("\n# --- paste into .env.test ---")
    logger.info("DRIVE_DETECTOR_DATASET_FOLDER_ID=%s", det)
    logger.info("DRIVE_CLASSIFIER_DATASET_FOLDER_ID=%s", cls)
    logger.info("DRIVE_CONFIG_OVERRIDES_FILE_ID=%s", overrides)
    logger.info("# OCR-LOT-TEST root: %s", root)


if __name__ == "__main__":
    main()
```

> NOTE for the implementer: confirm `DriveClient` exposes the authenticated service as
> `._svc` (the CLAUDE.md debug one-liner uses `DriveClient()._svc.about()...`, so it
> does). If the attribute name differs, adjust `svc = client._svc` accordingly.

- [ ] **Step 2: Smoke-check it imports (no Drive call)**

Run: `python -c "import ast; ast.parse(open('scripts/setup_test_drive.py', encoding='utf-8').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run for real (requires DRIVE_OAUTH_* in .env)**

Run: `python scripts/setup_test_drive.py`
Expected: prints `Authenticated as: <workspace user email>` then three `DRIVE_*` lines + the root id. Copy these three ids into `.env.test`.

> If it prints a service-account identity (no email) or 403s, the OAuth user creds are
> missing — see CLAUDE.md "Drive identity" notes. Do NOT fall back to the SA (zero quota).

- [ ] **Step 4: Commit**

```bash
git add scripts/setup_test_drive.py
git commit -m "feat: setup_test_drive script creates OCR-LOT-TEST tree"
```

---

## Task 6: Launch script `scripts/run_test_wizard.ps1`

**Files:**
- Create: `scripts/run_test_wizard.ps1`

Bootstraps `data/test/`, exports `.env.test` into the process env (so module-level env reads at import see the test values), starts the backend on :8081 and serves the test wizard on :8091.

- [ ] **Step 1: Write the launch script**

Create `scripts/run_test_wizard.ps1`:

```powershell
# Test wizard harness launcher. Run from repo root:  .\scripts\run_test_wizard.ps1
# Starts the backend on :8081 (TEST_MODE) and serves the test wizard on :8091.
# NEVER touches production Cloud Run.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$envFile = Join-Path $repo ".env.test"
if (-not (Test-Path $envFile)) {
    Write-Error ".env.test not found. Copy .env.test.example to .env.test and fill DRIVE_* ids (run scripts/setup_test_drive.py first)."
}

# Export .env.test into the process env BEFORE python starts, so module-level
# os.getenv reads (config dir, models, drafts) pick up the test values.
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $idx = $line.IndexOf("=")
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        Set-Item -Path "Env:$key" -Value $val
    }
}

# Bootstrap test dirs — mirror prod config + models so the test wizard starts
# with the same 6 packagings and loadable model weights.
$testRoot = Join-Path $repo "data\test"
$testConfig = Join-Path $testRoot "config"
$testModels = Join-Path $testRoot "models"
New-Item -ItemType Directory -Force -Path $testRoot, $testModels, (Join-Path $testRoot "drafts"), (Join-Path $testRoot "crops") | Out-Null

if (-not (Test-Path $testConfig)) {
    Copy-Item -Recurse -Path (Join-Path $repo "config") -Destination $testConfig
    Write-Host "Seeded data/test/config from config/"
}
foreach ($m in @("classifier.pt", "detector.pt")) {
    $dst = Join-Path $testModels $m
    $src = Join-Path $repo "models\$m"
    if ((-not (Test-Path $dst)) -and (Test-Path $src)) {
        Copy-Item -Path $src -Destination $dst
        Write-Host "Seeded data/test/models/$m"
    }
}

Write-Host "TEST_MODE backend -> http://localhost:8081"
Write-Host "Test wizard       -> http://localhost:8091/wizard.html"

# Serve the test wizard (static) in the background, then run the backend.
$wizardDir = Join-Path $repo "test wizzard"
Start-Process -NoNewWindow python -ArgumentList @("-m", "http.server", "8091", "--directory", "`"$wizardDir`"")

python -m uvicorn main:app --port 8081
```

- [ ] **Step 2: Lint the script parses (PowerShell syntax check)**

Run: `powershell -NoProfile -Command "[void][System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw scripts/run_test_wizard.ps1), [ref]$null); 'ok'"`
Expected: `ok` (no parse errors).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_test_wizard.ps1
git commit -m "feat: run_test_wizard launcher (port 8081/8091, env export, dir bootstrap)"
```

---

## Task 7: Test wizard frontend copy

**Files:**
- Create: `test wizzard/wizard.html`
- Create: `test wizzard/README.md`

- [ ] **Step 1: Copy the wizard**

Run: `python -c "import shutil; shutil.copyfile('web/wizard.html', 'test wizzard/wizard.html'); print('copied')"`
Expected: `copied` (creates the file; `test wizzard/` may need creating first — `New-Item -ItemType Directory -Force "test wizzard"`).

- [ ] **Step 2: Inject the API_BASE override**

In `test wizzard/wizard.html`, the `API_BASE` IIFE (around line 1711) already honors `window.API_BASE_OVERRIDE` first. Add an inline script that sets it, placed in `<head>` BEFORE the main script block. Find the opening `<head>` tag and insert immediately after it:

```html
<head>
<script>
  // Test wizard harness — pin the backend to the TEST_MODE instance on :8081.
  window.API_BASE_OVERRIDE = 'http://localhost:8081';
</script>
```

> NOTE for the implementer: if `window.API_BASE_OVERRIDE` is read by the IIFE at parse
> time, the inline script must execute before that `<script>` runs. Placing it as the
> first child of `<head>` guarantees that. Verify by opening the served page and
> checking the API base chip (rendered at `web/wizard.html:1971`) shows
> `http://localhost:8081`.

- [ ] **Step 3: Write the README**

Create `test wizzard/README.md`:

```markdown
# Test Wizard Harness

Isolated copy of the packaging wizard for safe testing. **Never touches production
Cloud Run or the real Drive dataset.**

## One-time setup

1. Ensure the prod `.env` has `DRIVE_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN` set.
2. Create the test Drive tree and copy the printed ids:
   ```
   python scripts/setup_test_drive.py
   ```
3. Copy `.env.test.example` → `.env.test`, paste the three `DRIVE_*` ids and the
   same `DRIVE_OAUTH_*` values as prod.

## Run

```
.\scripts\run_test_wizard.ps1
```

- Backend (TEST_MODE): http://localhost:8081
- Test wizard UI:      http://localhost:8091/wizard.html

## What is simulated

- **Deploy** writes YAML/models into `data/test/` and reloads the test registry, but
  the Cloud Run revision trigger is skipped (`{"triggered": false, "reason": "test
  mode (simulated)"}`).
- **Dataset publish** uploads real images/labels into the `OCR-LOT-TEST/` Drive
  folders only.
- **Notebook** returns the existing combined-notebook link (training is not run here).

## Safety

Production is protected by: `TEST_MODE=1` gating the trigger, port 8081, and
`data/test/*` + `OCR-LOT-TEST/` isolation. The only shared resource is the Google
account / OAuth user; the harness writes nowhere outside `OCR-LOT-TEST/`.
```

- [ ] **Step 4: Commit**

```bash
git add "test wizzard/wizard.html" "test wizzard/README.md"
git commit -m "feat: test wizard frontend copy pinned to :8081 + README"
```

---

## Task 8: End-to-end verification

**Files:** none (manual + automated checks)

- [ ] **Step 1: Full test suite — no regression**

Run: `python -m pytest -q`
Expected: PASS except the known pre-existing `tests/test_classifier.py` 3 setup errors (documented in CLAUDE.md). No NEW failures.

- [ ] **Step 2: Prod-default guard (env unset → original behavior)**

Run: `python -c "import os; assert 'OCR_CONFIG_DIR' not in os.environ; from pipeline.packaging_registry import _config_dir; from services.cloudrun_deployer import _packaging_dir; print(_config_dir()); print(_packaging_dir())"`
Expected: prints the repo `config` path and `config\packagings` — confirming defaults are unchanged when no test env is set.

- [ ] **Step 3: Launch the harness**

Run: `.\scripts\run_test_wizard.ps1`
Expected: seeds `data/test/`, backend on :8081, wizard on :8091. Open http://localhost:8091/wizard.html — the API base chip shows `http://localhost:8081`, dashboard lists the mirrored packagings.

- [ ] **Step 4: Confirm isolation after a simulated deploy**

In the test wizard, create a draft → add a packaging → click Deploy (use a draft that passes the hard-floor gate, or an edit-draft). Then verify on disk:

Run: `git status --porcelain config/packagings models`
Expected: EMPTY — no changes to the real `config/packagings/` or `models/`. All writes landed under `data/test/` (gitignored).

Run (check the deploy response in the browser network tab or backend log): the deploy response `cloud_run` field equals `{"triggered": false, "reason": "test mode (simulated)"}`.

- [ ] **Step 5: Confirm Drive isolation**

After a dataset publish from the test wizard, confirm in Drive that new images appear under `OCR-LOT-TEST/data check lot/` (or `data classify check lot/`) and NOT in the production dataset folders.

- [ ] **Step 6: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "test: verify test wizard harness isolation"
```

---

## Self-Review Notes

- **Spec coverage:** Arch/profiles (Tasks 4,6,7), env isolation table (Task 4), 3 code changes (Tasks 1-3), Drive setup script (Task 5), bootstrap dirs (Task 6), simulated deploy flow (Tasks 3,8), safety guarantees (Task 8 verification), CORS open item (resolved — `allow_origins=["*"]`, noted in plan header; no task needed).
- **Notebook (spec Q4=B):** no code change — `/training/full/start` already returns the existing `COMBINED_NOTEBOOK_FILE_ID`; in test it simply publishes to the test Drive folders via the env-driven `dataset_publisher`. No task required beyond env config.
- **Type/name consistency:** helpers named `_config_dir()` (registry) and `_packaging_dir()` (deployer) used consistently; `cloud_run` response shape `{"triggered": false, "reason": "test mode (simulated)"}` matches between Task 3 implementation and Task 8 assertion.
