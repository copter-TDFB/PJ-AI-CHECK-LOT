"""One-time helper: exchange an OAuth client for a long-lived Drive refresh token.

Run once locally, sign in as the Workspace user whose Drive quota should own
the published dataset (e.g. sarutipong@tdfb.co), then copy the printed values
into `.env` (local) and Cloud Run env vars (prod). See ADR 0005.

Usage:
    python scripts/generate_drive_token.py [path/to/oauth_client.json] [port]

Falls back to the single `client_secret_*.json` in the project root if no path
is given. The OAuth client must be type "Web application" whose Authorized
redirect URIs include http://localhost:{port}/ (default port 8765 — 8080 is
usually taken by the dev server). "Desktop app" clients accept any loopback port.
"""

import glob
import logging
import os
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Full drive scope — lets the backend write into the existing (hand-created)
# reference dataset folders. Allowed without Google verification because the
# OAuth consent screen is configured as Internal within the tdfb.co Workspace.
SCOPES = ["https://www.googleapis.com/auth/drive"]
# 8080 is usually taken by the dev server (uvicorn --reload --port 8080), so
# default to a free port. Override with argv[2] or DRIVE_OAUTH_PORT. Whatever
# port is used must appear in the OAuth client's Authorized redirect URIs.
DEFAULT_PORT = 8765


def _resolve_client_file(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    matches = sorted(glob.glob("client_secret_*.json")) + sorted(glob.glob("oauth_client.json"))
    if not matches:
        raise SystemExit(
            "No OAuth client JSON found. Pass the path explicitly:\n"
            "    python scripts/generate_drive_token.py path/to/client.json"
        )
    return Path(matches[0])


def _resolve_port(argv: list[str]) -> int:
    if len(argv) > 2:
        return int(argv[2])
    return int(os.environ.get("DRIVE_OAUTH_PORT", DEFAULT_PORT))


def main(argv: list[str]) -> None:
    client_file = _resolve_client_file(argv)
    if not client_file.is_file():
        raise SystemExit(f"File not found: {client_file}")
    port = _resolve_port(argv)

    logger.info("Using OAuth client: %s", client_file)
    logger.info("Listening on http://localhost:%d/ — make sure that exact URI", port)
    logger.info("is in the client's Authorized redirect URIs.")
    logger.info("A browser window will open — sign in as the Workspace user")
    logger.info("whose Drive should own the dataset (e.g. sarutipong@tdfb.co).\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), scopes=SCOPES)
    creds = flow.run_local_server(port=port, prompt="consent", access_type="offline")

    if not creds.refresh_token:
        raise SystemExit(
            "No refresh_token returned. Revoke prior access at "
            "https://myaccount.google.com/permissions and re-run "
            "(prompt=consent forces a fresh one)."
        )

    logger.info("\n%s", "=" * 70)
    logger.info("SUCCESS — paste these into .env (local) and Cloud Run (prod):")
    logger.info("%s", "=" * 70)
    logger.info("DRIVE_OAUTH_CLIENT_ID=%s", creds.client_id)
    logger.info("DRIVE_OAUTH_CLIENT_SECRET=%s", creds.client_secret)
    logger.info("DRIVE_OAUTH_REFRESH_TOKEN=%s", creds.refresh_token)
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main(sys.argv)
