"""One-command bootstrap for the test wizard harness (tdfb.co Workspace users).

Runs the Google Drive OAuth consent, creates the runner's OWN `OCR-LOT-TEST`
Drive tree, and writes `.env.test`. `setup_and_run.ps1` calls this on first run
so the harness comes up ready. Idempotent: if `.env.test` already carries the
Drive ids, it does nothing.

Requires an OAuth client config in the repo root (`oauth_client.json` or
`client_secret_*.json`) — the sender ships this with the project. The signed-in
account must be a tdfb.co Workspace member (the consent screen is Internal).
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import sys
from pathlib import Path

# Standalone script — bootstrap repo root so `services`/sibling scripts import.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402
from googleapiclient.discovery import build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bootstrap_test_env")

SCOPES = ["https://www.googleapis.com/auth/drive"]
OAUTH_PORT = 8765  # must match the OAuth client's Authorized redirect URI
ENV_TEST = _REPO / ".env.test"
ENV_EXAMPLE = _REPO / ".env.test.example"


def _load_setup_test_drive():
    """Load scripts/setup_test_drive.py as a module (single source of helpers)."""
    spec = importlib.util.spec_from_file_location(
        "setup_test_drive", Path(__file__).resolve().parent / "setup_test_drive.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _already_configured() -> bool:
    if not ENV_TEST.is_file():
        return False
    for line in ENV_TEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("DRIVE_DETECTOR_DATASET_FOLDER_ID=") and line.split("=", 1)[1].strip():
            return True
    return False


def _resolve_client_file() -> Path:
    matches = sorted(glob.glob(str(_REPO / "oauth_client.json"))) + sorted(
        glob.glob(str(_REPO / "client_secret_*.json"))
    )
    if not matches:
        raise SystemExit(
            "No OAuth client JSON in the project root. Ask the sender for "
            "'oauth_client.json', put it in this folder, then re-run."
        )
    return Path(matches[0])


def _write_env_test(ids: dict, creds) -> None:
    fill = {
        **ids,
        "DRIVE_OAUTH_CLIENT_ID": creds.client_id,
        "DRIVE_OAUTH_CLIENT_SECRET": creds.client_secret,
        "DRIVE_OAUTH_REFRESH_TOKEN": creds.refresh_token,
    }
    out = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            key = s.split("=", 1)[0].strip()
            if key in fill:
                out.append(f"{key}={fill[key]}")
                continue
        out.append(line)
    ENV_TEST.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    if _already_configured():
        logger.info(".env.test already configured — skipping Drive setup.")
        return
    if not ENV_EXAMPLE.is_file():
        raise SystemExit(".env.test.example not found — run from the project root.")

    std = _load_setup_test_drive()
    client_file = _resolve_client_file()

    logger.info("Opening a browser for Google consent — sign in with your tdfb.co account.")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), scopes=SCOPES)
    creds = flow.run_local_server(port=OAUTH_PORT, prompt="consent", access_type="offline")
    if not creds.refresh_token:
        raise SystemExit(
            "No refresh_token returned. Revoke prior access at "
            "https://myaccount.google.com/permissions and re-run."
        )

    svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    who = svc.about().get(fields="user").execute().get("user", {})
    logger.info("Authenticated as: %s", who.get("emailAddress", "<unknown>"))

    if std._find_root(svc):
        raise SystemExit(
            f"{std.ROOT_NAME} already exists in your Drive but .env.test is missing. "
            "Delete that folder (or run `python scripts/setup_test_drive.py --reset` "
            "with DRIVE_OAUTH_* set) and re-run."
        )

    root = std._make_folder(svc, std.ROOT_NAME)
    det = std._make_folder(svc, "data check lot", root)
    cls = std._make_folder(svc, "data classify check lot", root)
    overrides = std._make_json_file(svc, "config_overrides.json", root, {})
    std._make_text_file(svc, "data.yaml", det, std._STARTER_DATA_YAML, "text/yaml")

    _write_env_test(
        {
            "DRIVE_DETECTOR_DATASET_FOLDER_ID": det,
            "DRIVE_CLASSIFIER_DATASET_FOLDER_ID": cls,
            "DRIVE_CONFIG_OVERRIDES_FILE_ID": overrides,
        },
        creds,
    )
    logger.info("Created %s (root %s) and wrote .env.test — ready to launch.",
                std.ROOT_NAME, root)


if __name__ == "__main__":
    main()
