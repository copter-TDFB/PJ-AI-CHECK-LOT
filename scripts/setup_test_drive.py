"""Create (or reset) the OCR-LOT-TEST Drive tree for the test wizard harness.

Run from the repo root with the prod .env loaded so DRIVE_OAUTH_* are present
(the OAuth user has full `drive` scope, avoiding the drive.file blind spot on
manually-created folders):

    python scripts/setup_test_drive.py            # create, skip if exists
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


def _make_text_file(svc, name: str, parent: str, text: str, mime: str) -> str:
    body = {"name": name, "parents": [parent]}
    media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype=mime)
    return svc.files().create(body=body, media_body=media, fields="id").execute()["id"]


# Starter YOLO data.yaml the detector dataset folder must contain — dataset_publisher
# reads it to learn existing class names (it only ever APPENDS). Empty to start;
# relative split paths are normalized by dataset_publisher._relativize.
_STARTER_DATA_YAML = "train: train/images\nval: val/images\nnc: 0\nnames: []\n"


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
    # The detector folder must ship a data.yaml or dataset_publisher.publish()
    # fails fast with "data.yaml not found".
    _make_text_file(svc, "data.yaml", det, _STARTER_DATA_YAML, "text/yaml")

    logger.info("\n# --- paste into .env.test ---")
    logger.info("DRIVE_DETECTOR_DATASET_FOLDER_ID=%s", det)
    logger.info("DRIVE_CLASSIFIER_DATASET_FOLDER_ID=%s", cls)
    logger.info("DRIVE_CONFIG_OVERRIDES_FILE_ID=%s", overrides)
    logger.info("# OCR-LOT-TEST root: %s", root)


if __name__ == "__main__":
    main()
