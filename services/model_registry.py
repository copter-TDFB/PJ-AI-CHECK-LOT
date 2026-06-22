"""Model registry — sync classifier + detector weights from Google Drive.

manifest.json format (อยู่ใน Drive):
{
  "version": "v1",
  "deployed_at": "2026-06-06T00:00:00Z",
  "classifier": {"file_id": "DRIVE_FILE_ID", "sha256": "abc..."},
  "detector":   {"file_id": "DRIVE_FILE_ID", "sha256": "def..."}
}

Env vars:
  DRIVE_MANIFEST_FILE_ID  — file_id ของ manifest.json ใน Drive (required บน Cloud Run)
  MODEL_CACHE_DIR         — local cache directory (default /tmp/models)

ถ้า DRIVE_MANIFEST_FILE_ID ไม่ถูกตั้งค่า → fallback ใช้ local models/*.pt (local dev)
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MANIFEST_FILE_ID = os.getenv("DRIVE_MANIFEST_FILE_ID", "")
_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/models"))
_LOCAL_CLF = Path(os.getenv("MODEL_CLASSIFIER_PATH", "models/classifier.pt"))
_LOCAL_DET = Path(os.getenv("MODEL_DETECTOR_PATH", "models/detector.pt"))


def sync(
    manifest_file_id: str = _MANIFEST_FILE_ID,
    cache_dir: Path = _CACHE_DIR,
) -> tuple[Path, Path]:
    """คืน (classifier_path, detector_path).

    ลำดับแหล่ง: GCS (GCS_CONFIG_BUCKET) → Drive manifest → local models/*.pt.
    ทุกชั้น fall back ลงชั้นถัดไปเมื่อ error เพื่อให้ /predict ไม่ล่ม.
    """
    gcs_result = _sync_from_gcs(cache_dir)
    if gcs_result is not None:
        return gcs_result

    if not manifest_file_id:
        logger.info(
            "DRIVE_MANIFEST_FILE_ID not set — using local models: clf=%s det=%s",
            _LOCAL_CLF, _LOCAL_DET,
        )
        return _LOCAL_CLF, _LOCAL_DET

    from services.drive_client import DriveClient  # import ใน function เพื่อ lazy init

    client = DriveClient()
    logger.info("Fetching manifest from Drive (file_id=%s)", manifest_file_id)
    manifest = client.read_json(manifest_file_id)
    logger.info(
        "Manifest: version=%s deployed_at=%s",
        manifest.get("version"), manifest.get("deployed_at"),
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    clf_path = _ensure_model(client, manifest["classifier"], cache_dir / "classifier.pt")
    det_path = _ensure_model(client, manifest["detector"], cache_dir / "detector.pt")
    return clf_path, det_path


def _sync_from_gcs(cache_dir: Path) -> tuple[Path, Path] | None:
    """Try GCS first. Return (clf, det) or None to fall back to Drive/local."""
    from services import gcs_store

    store = gcs_store.get_store()
    if store is None:
        return None
    try:
        manifest = store.read_json(gcs_store.MANIFEST_PATH) or {}
        models = manifest.get("models") or {}
        clf_entry, det_entry = models.get("classifier"), models.get("detector")
        if not clf_entry or not det_entry:
            logger.info("GCS manifest has no models — falling back to Drive/local")
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        clf = _ensure_gcs_model(store, clf_entry, cache_dir / "classifier.pt")
        det = _ensure_gcs_model(store, det_entry, cache_dir / "detector.pt")
        logger.info("Models synced from GCS bucket=%s", store.bucket_name)
        return clf, det
    except Exception as e:
        logger.warning("GCS model sync failed — falling back: %s", e)
        return None


def _ensure_gcs_model(store, entry: dict, dest: Path) -> Path:
    """Download model from GCS if cache miss / sha256 mismatch. Verify sha256."""
    expected = entry["sha256"]
    obj = entry["object"]
    if dest.exists() and _sha256(dest) == expected:
        logger.info("Cache hit — %s (sha256 match)", dest.name)
        return dest
    if not store.download_to(obj, dest):
        raise RuntimeError(f"GCS object missing: {obj}")
    actual = _sha256(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 verification failed for {dest.name}: expected={expected} got={actual}"
        )
    logger.info("%s verified from GCS (sha256 OK)", dest.name)
    return dest


def _ensure_model(client, entry: dict, dest: Path) -> Path:
    """Download model ถ้า cache ไม่มีหรือ sha256 ไม่ตรง."""
    expected = entry["sha256"]
    file_id = entry["file_id"]

    if dest.exists():
        actual = _sha256(dest)
        if actual == expected:
            logger.info("Cache hit — %s (sha256 match)", dest.name)
            return dest
        logger.warning("sha256 mismatch for %s — re-downloading", dest.name)
    else:
        logger.info("Cache miss — downloading %s", dest.name)

    client.download_file(file_id, dest)

    actual = _sha256(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 verification failed for {dest.name}: "
            f"expected={expected} got={actual}"
        )

    logger.info("✓ %s verified (sha256 OK)", dest.name)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
