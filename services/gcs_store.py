"""Durable config + model store backed by Google Cloud Storage.

The service account (`ocr-lot-checker-sa`) reads/writes GCS natively via ADC
(`roles/storage.objectAdmin` on the bucket) — no OAuth, no Drive quota. This is
the persistence layer that makes wizard "Deploy" survive Cloud Run revisions:
startup overlays the GCS copy on top of the baked-in image config + models.

Bucket layout (`gs://$GCS_CONFIG_BUCKET/`):
    manifest.json            — commit pointer (packagings + models shas, overrides)
    packagings/<key>.yaml     — full packaging YAML per class
    packagings/<key>.archived — tombstone marker for archived/deleted classes
    models/detector.pt, models/classifier.pt

`get_store()` returns None when `GCS_CONFIG_BUCKET` is unset, so local dev,
tests, and zero-env-var prod all fall back to the baked-in image cleanly. The
`google.cloud.storage` import is lazy — it only happens when a real client is
created, so the package is not required unless GCS is actually configured.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_BUCKET = "GCS_CONFIG_BUCKET"

MANIFEST_PATH = "manifest.json"
PACKAGINGS_PREFIX = "packagings/"
MODELS_PREFIX = "models/"


class GcsStore:
    """Thin wrapper over a single GCS bucket.

    `client` is injectable for tests (an in-memory fake); when None, a real
    `google.cloud.storage.Client` is created lazily on first use.
    """

    def __init__(self, bucket_name: str, client: object | None = None) -> None:
        self.bucket_name = bucket_name
        self._client = client

    @property
    def client(self):  # lazy real client
        if self._client is None:
            from google.cloud import storage  # lazy: not needed unless GCS configured

            self._client = storage.Client()
        return self._client

    def _blob(self, path: str):
        return self.client.bucket(self.bucket_name).blob(path)

    # ── bytes / text ────────────────────────────────────────────────────────
    def exists(self, path: str) -> bool:
        return self._blob(path).exists()

    def put_bytes(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._blob(path).upload_from_string(data, content_type=content_type)

    def get_bytes(self, path: str) -> bytes | None:
        blob = self._blob(path)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def put_text(self, path: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(path, text.encode("utf-8"), content_type=content_type)

    def get_text(self, path: str) -> str | None:
        data = self.get_bytes(path)
        return data.decode("utf-8") if data is not None else None

    def delete(self, path: str) -> None:
        blob = self._blob(path)
        if blob.exists():
            blob.delete()

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self.client.list_blobs(self.bucket_name, prefix=prefix))

    # ── json ─────────────────────────────────────────────────────────────────
    def read_json(self, path: str) -> dict | None:
        text = self.get_text(path)
        if text is None:
            return None
        return json.loads(text)

    def write_json(self, path: str, obj: dict) -> None:
        self.put_text(path, json.dumps(obj, ensure_ascii=False, indent=2), content_type="application/json")

    # ── files (models) ─────────────────────────────────────────────────────
    def upload_file(self, path: str, src: Path) -> None:
        self._blob(path).upload_from_filename(str(src))

    def download_to(self, path: str, dest: Path) -> bool:
        blob = self._blob(path)
        if not blob.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        return True


def bucket_name() -> str:
    return os.getenv(ENV_BUCKET, "").strip()


_store_singleton: GcsStore | None = None


def get_store() -> GcsStore | None:
    """Return a GcsStore if `GCS_CONFIG_BUCKET` is set, else None (fall back to image).

    The instance is cached so reloads reuse one GCS client (one ADC init) instead
    of constructing a fresh client on every registry reload.
    """
    global _store_singleton
    name = bucket_name()
    if not name:
        return None
    if _store_singleton is None or _store_singleton.bucket_name != name:
        _store_singleton = GcsStore(name)
    return _store_singleton
