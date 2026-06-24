"""Resolve a packaging class's classifier-dataset images on Drive (ADR 0003 layout).

Production does NOT ship the local `images/` dataset, so the dashboard's image
count and sample thumbnails come from Drive: `<CLS_FOLDER>/images/<key>/`, where
CLS_FOLDER = env DRIVE_CLASSIFIER_DATASET_FOLDER_ID and the class folder is named
by `key` (see services/dataset_publisher.py).

`class_images()` never raises -- a Drive outage must not break the dashboard. Results
(including empty ones) are cached per-instance for _TTL_SECONDS.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CLS_ENV = "DRIVE_CLASSIFIER_DATASET_FOLDER_ID"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_TTL_SECONDS = 600

# key -> (monotonic_timestamp, list[{"id","name"}])
_CACHE: dict[str, tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    """Drop all cached entries (tests + future cache-busting)."""
    _CACHE.clear()


def class_images(key: str) -> list[dict]:
    """Return [{"id","name"}] of the Drive classifier-dataset images for `key`.

    Resolves <CLS_FOLDER>/images/<key>. Returns [] when the env var is unset, the
    folder chain is missing, or Drive errors. Cached per-instance for _TTL_SECONDS.
    """
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    result: list[dict] = []
    cls_root = os.getenv(_CLS_ENV, "").strip()
    if cls_root:
        try:
            from services.drive_client import DriveClient

            drive = DriveClient()
            images_id = drive.find_in_folder(cls_root, "images")
            class_id = drive.find_in_folder(images_id, key) if images_id else None
            if class_id:
                result = [
                    {"id": f["id"], "name": f["name"]}
                    for f in drive.list_folder(class_id)
                    if Path(f["name"]).suffix.lower() in _IMG_EXTS
                ]
        except Exception as e:  # noqa: BLE001 -- dashboard must survive Drive outages
            logger.warning("drive_samples.class_images(%s) failed: %s", key, e)
            result = []

    _CACHE[key] = (now, result)
    return result
