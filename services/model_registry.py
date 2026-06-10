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
    """คืน (classifier_path, detector_path) — download จาก Drive ถ้าจำเป็น.

    ถ้า manifest_file_id ว่าง → ใช้ local model files (local dev / ไม่มี Drive)
    """
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
