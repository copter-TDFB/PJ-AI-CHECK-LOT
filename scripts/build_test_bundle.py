"""Build a self-contained Docker test bundle under dist/test-bundle/.

The bundle bundles the backend + models + a wizard frontend + the caller's Drive
creds so a recipient can run the WHOLE test stack with one command:

    docker compose up        # backend :8081, wizard :8091
    # then open http://localhost:8091/wizard.html

It is a TEST harness: TEST_MODE=1 makes deploy skip the Cloud Run trigger, and the
DRIVE_* ids point at the OCR-LOT-TEST Drive folder — production is never touched.

SECURITY: the generated .env + gcp-key.json contain real Drive OAuth + GCP creds.
Anyone with the bundle can act as you on Drive and spend your Vision quota. Only
send to trusted people; revoke the OAuth token afterwards if needed.

Run from the repo root (reads .env.test for ids/creds):

    python scripts/build_test_bundle.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_test_bundle")

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "dist" / "test-bundle"

# Backend source the FastAPI app needs at runtime.
SRC_FILES = ["main.py", "requirements.txt", "Dockerfile"]
SRC_DIRS = ["api", "pipeline", "services", "utils", "config"]
MODELS = ["classifier.pt", "detector.pt"]

# Keys copied verbatim from the local .env.test into the bundle .env.
# NOTE: we deliberately DO NOT carry OCR_CONFIG_DIR / MODELS_DIR / DRAFT_DIR /
# CROP_CACHE_DIR — inside the container the defaults (config/, models/, data/)
# are already an isolated sandbox; TEST_MODE + the test Drive ids do the
# production-protection.
ENV_KEYS = [
    "DRIVE_DETECTOR_DATASET_FOLDER_ID",
    "DRIVE_CLASSIFIER_DATASET_FOLDER_ID",
    "DRIVE_CONFIG_OVERRIDES_FILE_ID",
    "DRIVE_OAUTH_CLIENT_ID",
    "DRIVE_OAUTH_CLIENT_SECRET",
    "DRIVE_OAUTH_REFRESH_TOKEN",
    "REFERENCE_DETECTOR_PATH",
    "REFERENCE_CLASSIFIER_PATH",
]

COMPOSE = """\
# Test bundle — run:  docker compose up
# backend (TEST_MODE) -> http://localhost:8081
# wizard UI           -> http://localhost:8091/wizard.html
services:
  backend:
    build: .
    image: ocr-lot-test:latest
    command: uvicorn main:app --host 0.0.0.0 --port 8081
    ports:
      - "8081:8081"
    env_file: .env
  wizard:
    image: ocr-lot-test:latest
    command: python -m http.server 8091 --directory /app/static
    ports:
      - "8091:8091"
    depends_on:
      - backend
"""

START_BAT = """\
@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   OCR Lot Checker - Test Wizard
echo ============================================
echo.

docker info >nul 2>&1
if errorlevel 1 (
  echo [!] Docker is not running. Install/open Docker Desktop, then run START.bat again.
  pause
  exit /b 1
)

echo Starting containers ^(first run builds the image, ~2-5 min^)...
docker compose up -d --build
if errorlevel 1 (
  echo [!] Failed to start - see the error above.
  pause
  exit /b 1
)

echo.
echo Waiting for the backend to be ready...
set /a tries=0
:wait
set /a tries+=1
curl -sf -o nul http://localhost:8081/openapi.json
if not errorlevel 1 goto ready
if %tries% GEQ 60 (
  echo [!] Backend did not become ready. Check logs:  docker compose logs
  pause
  exit /b 1
)
timeout /t 3 >nul
goto wait

:ready
echo Backend is up. Opening the wizard in your browser...
start "" http://localhost:8091/wizard.html
echo.
echo   Wizard:  http://localhost:8091/wizard.html
echo   To stop: double-click STOP.bat
echo.
pause
"""

STOP_BAT = """\
@echo off
cd /d "%~dp0"
echo Stopping containers...
docker compose down
echo Done.
pause
"""

START_SH = """\
#!/usr/bin/env bash
# macOS / Linux launcher
cd "$(dirname "$0")"
if ! docker info >/dev/null 2>&1; then
  echo "[!] Docker is not running. Install/open Docker Desktop, then run again."
  exit 1
fi
echo "Starting containers (first run builds the image, ~2-5 min)..."
docker compose up -d --build || exit 1
echo "Waiting for the backend to be ready..."
for i in $(seq 1 60); do
  curl -sf -o /dev/null http://localhost:8081/openapi.json && break
  sleep 3
done
URL="http://localhost:8091/wizard.html"
( command -v open >/dev/null && open "$URL" ) || ( command -v xdg-open >/dev/null && xdg-open "$URL" ) || echo "Open: $URL"
echo "Wizard: $URL   |   stop: docker compose down"
"""

README = """\
# OCR Lot Checker — Test Wizard Bundle

Self-contained TEST harness. **Never touches production** (TEST_MODE: deploy is
simulated; Drive writes go only to the shared `OCR-LOT-TEST` test folder).

## Requirements
- Docker Desktop installed and **running** (Python/deps are inside the image)

## Run — one click
- **Windows:** double-click **`START.bat`**
- **macOS/Linux:** run `./start.sh`

It builds + starts everything, waits for the backend, and opens the wizard in your
browser automatically: **http://localhost:8091/wizard.html**

To stop: double-click **`STOP.bat`** (or `docker compose down`).

### Manual alternative
```
docker compose up          # then open http://localhost:8091/wizard.html
```

## What you can test
Create a packaging draft -> upload images -> annotate -> Full Training
(uploads to the shared test Drive) -> get a Colab notebook link -> Deploy
(simulated — returns `{"triggered": false, "reason": "test mode (simulated)"}`,
no real Cloud Run revision).

## Notes
- Drafts/images you create live INSIDE the container and vanish on
  `docker compose down` — that's expected for a sandbox.
- This bundle contains the sender's Google Drive/Vision credentials. Treat it as
  sensitive; don't re-share.
"""


def _read_env_test() -> dict:
    path = REPO / ".env.test"
    if not path.exists():
        logger.error(".env.test not found — run scripts/setup_test_drive.py and create .env.test first")
        sys.exit(1)
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _ignore_pycache(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def main() -> None:
    env = _read_env_test()
    missing = [k for k in ("DRIVE_OAUTH_REFRESH_TOKEN", "DRIVE_DETECTOR_DATASET_FOLDER_ID") if not env.get(k)]
    if missing:
        logger.error("Missing required keys in .env.test: %s", missing)
        sys.exit(1)

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    # 1. backend source
    for f in SRC_FILES:
        shutil.copy2(REPO / f, BUNDLE / f)
    for d in SRC_DIRS:
        shutil.copytree(REPO / d, BUNDLE / d, ignore=_ignore_pycache)

    # 2. models
    (BUNDLE / "models").mkdir()
    for m in MODELS:
        shutil.copy2(REPO / "models" / m, BUNDLE / "models" / m)

    # 3. Vision SA key (referenced by GOOGLE_APPLICATION_CREDENTIALS)
    gcp = REPO / "gcp-key.json"
    if gcp.exists():
        shutil.copy2(gcp, BUNDLE / "gcp-key.json")
    else:
        logger.warning("gcp-key.json not found — Vision OCR will be unavailable in the bundle")

    # 4. wizard frontend, pinned to the bundle's backend on :8081
    wizard = (REPO / "web" / "wizard.html").read_text(encoding="utf-8")
    inject = (
        "<head>\n<script>\n"
        "  // Test bundle — backend runs in the same compose stack on :8081.\n"
        "  window.API_BASE_OVERRIDE = 'http://localhost:8081';\n"
        "</script>"
    )
    if "window.API_BASE_OVERRIDE = 'http://localhost:8081'" not in wizard:
        wizard = wizard.replace("<head>", inject, 1)
    (BUNDLE / "static").mkdir()
    (BUNDLE / "static" / "wizard.html").write_text(wizard, encoding="utf-8")

    # 5. .env (TEST_MODE + test Drive ids + creds; container defaults for paths)
    lines = [
        "# Generated by scripts/build_test_bundle.py — TEST harness, do not point at prod.",
        "TEST_MODE=1",
        "DRIVE_MANIFEST_FILE_ID=",
        "GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json",
        "LOG_LEVEL=INFO",
    ]
    for k in ENV_KEYS:
        lines.append(f"{k}={env.get(k, '')}")
    (BUNDLE / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 6. compose + launchers + readme
    (BUNDLE / "docker-compose.yml").write_text(COMPOSE, encoding="utf-8")
    # .bat files use CRLF so Windows double-click runs them cleanly
    (BUNDLE / "START.bat").write_text(START_BAT.replace("\n", "\r\n"), encoding="utf-8")
    (BUNDLE / "STOP.bat").write_text(STOP_BAT.replace("\n", "\r\n"), encoding="utf-8")
    (BUNDLE / "start.sh").write_text(START_SH, encoding="utf-8", newline="\n")
    (BUNDLE / "README.md").write_text(README, encoding="utf-8")

    # summary
    total = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    logger.info("Built bundle at: %s", BUNDLE)
    logger.info("Size: %.1f MB", total / 1e6)
    logger.info("Zip it and send:  Compress-Archive -Path '%s\\*' -DestinationPath test-bundle.zip", BUNDLE)
    logger.info("SECURITY: .env + gcp-key.json hold real creds — send only to trusted recipients.")


if __name__ == "__main__":
    main()
