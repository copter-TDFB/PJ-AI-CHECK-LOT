"""Deploy a newly-trained packaging:

  1. Write config/packagings/{key}.yaml from draft config + lot_patterns
  2. Update Drive manifest.json with the new detector file_id + sha256
  3. (Optional) Trigger Cloud Run revision spawn via admin API

Step 3 requires roles/run.developer on the service account. If credentials
lack that scope, we skip silently and surface a hint to the operator.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PACKAGING_DIR = Path("config/packagings")
_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
_BACKUP_KEEP = 3
_CLOUD_RUN_PROJECT = os.getenv("CLOUD_RUN_PROJECT", "pj-ai-detect-lot-no")
_CLOUD_RUN_REGION = os.getenv("CLOUD_RUN_REGION", "asia-southeast1")
_CLOUD_RUN_SERVICE = os.getenv("CLOUD_RUN_SERVICE", "ocr-lot-checker")


def write_packaging_yaml(key: str, draft_meta: dict[str, Any]) -> Path:
    """Convert draft meta → config/packagings/{key}.yaml.

    Fresh deploys produce a YAML in the same shape as the existing 6 classes.
    Overwrite deploys (when an active YAML for `key` already exists, e.g. via
    an edit-draft) preserve every field on disk that the draft did not change
    — so things like `conf_threshold`, `accuracy`, `post_ocr_fixes`,
    `sub_regions` stay intact unless explicitly edited.
    """
    cfg = draft_meta.get("config") or {}
    pipeline = draft_meta.get("pipeline", "detector_ocr")

    existing: dict[str, Any] = {}
    out = _PACKAGING_DIR / f"{key}.yaml"
    if out.exists():
        try:
            existing = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            logger.warning("Existing YAML at %s unreadable — falling back to defaults", out)
            existing = {}

    # Fields the wizard owns — always pull from draft.config
    edited = {
        "lot_patterns": cfg.get("lot_patterns"),
        "fields_extracted": cfg.get("fields_extracted"),
        "sheet_checks": cfg.get("sheet_checks"),
        "message_template_key": cfg.get("message_template_key"),
        "product_aliases": cfg.get("product_aliases"),
    }

    sub_regions_final = draft_meta.get("sub_regions") or existing.get("sub_regions", [])
    detection_mode = draft_meta.get("detection_mode") or existing.get("detection_mode", "single")
    default_prefixes = (
        [f"{key}_{sr}" for sr in sub_regions_final]
        if detection_mode == "multi_field" and sub_regions_final
        else [f"{key}_lot"]
    )

    data = {
        "key": key,
        "display_name": draft_meta.get("display_name") or existing.get("display_name", key),
        "pipeline": pipeline,
        "conf_threshold": existing.get("conf_threshold", 0.6),
        "accuracy": existing.get("accuracy"),  # cleared by /eval if retrained
        "gate_on_lot": existing.get("gate_on_lot", True),
        "lot_short_fallback": existing.get("lot_short_fallback", False),
        "sub_regions": sub_regions_final,
        "detection_mode": detection_mode,
        "lot_patterns": edited["lot_patterns"] if edited["lot_patterns"] is not None
            else existing.get("lot_patterns", []),
        "fields_extracted": edited["fields_extracted"] if edited["fields_extracted"] is not None
            else existing.get("fields_extracted", ["lot"]),
        "sheet_checks": edited["sheet_checks"] if edited["sheet_checks"] is not None
            else existing.get("sheet_checks", []),
        "post_ocr_fixes": existing.get("post_ocr_fixes", []),
        "message_template_key": edited["message_template_key"]
            if edited["message_template_key"] is not None
            else existing.get("message_template_key", "lot_only"),
        "product_aliases": edited["product_aliases"]
            if edited["product_aliases"] is not None
            else existing.get("product_aliases", []),
        "model_classifier_label": existing.get("model_classifier_label", key),
        "detector_yolo_prefixes": existing.get("detector_yolo_prefixes", default_prefixes),
    }

    _PACKAGING_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Wrote packaging YAML: %s", out)
    return out


def _bak_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def backup_artifacts(parent_key: str) -> dict[str, Any]:
    """Snapshot active YAML + model weights before an overwrite deploy.

    Returns a manifest with the paths of every file backed up so callers can
    restore them if the deploy fails downstream.
    """
    ts = _bak_suffix()
    manifest: dict[str, Any] = {"timestamp": ts, "files": []}

    yaml_src = _PACKAGING_DIR / f"{parent_key}.yaml"
    if yaml_src.exists():
        dst = _PACKAGING_DIR / f"{parent_key}.yaml.bak-{ts}"
        shutil.copy2(yaml_src, dst)
        manifest["files"].append({"src": str(yaml_src), "bak": str(dst)})
        logger.info("Backed up YAML: %s", dst)

    for model_name in ("detector.pt", "classifier.pt"):
        src = _MODELS_DIR / model_name
        if src.exists():
            dst = _MODELS_DIR / f"{model_name}.bak-{ts}"
            shutil.copy2(src, dst)
            manifest["files"].append({"src": str(src), "bak": str(dst)})
            logger.info("Backed up model: %s", dst)

    _rotate_backups(parent_key)
    return manifest


def restore_backup(manifest: dict[str, Any]) -> None:
    """Restore files from a backup_artifacts() manifest after a failed deploy."""
    for entry in manifest.get("files", []):
        src = Path(entry["bak"])
        dst = Path(entry["src"])
        if src.exists():
            shutil.copy2(src, dst)
            logger.warning("Restored from backup: %s ← %s", dst, src)


def _rotate_backups(parent_key: str, keep: int = _BACKUP_KEEP) -> None:
    """Keep only the most recent N backups per artifact type."""
    yaml_baks = sorted(_PACKAGING_DIR.glob(f"{parent_key}.yaml.bak-*"))
    for old in yaml_baks[:-keep]:
        try:
            old.unlink()
            logger.info("Rotated old backup: %s", old)
        except OSError:
            pass

    for model_name in ("detector.pt", "classifier.pt"):
        model_baks = sorted(_MODELS_DIR.glob(f"{model_name}.bak-*"))
        for old in model_baks[:-keep]:
            try:
                old.unlink()
                logger.info("Rotated old backup: %s", old)
            except OSError:
                pass


def promote_draft_model(draft_key: str) -> Path | None:
    """Copy a draft's trained detector to production `models/detector.pt`.

    Returns the destination path, or None if the draft has no trained model.
    Caller is responsible for backing up the previous detector first.
    """
    src = Path("data/drafts") / draft_key / "models" / "full_detector.pt"
    if not src.exists():
        return None
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = _MODELS_DIR / "detector.pt"
    shutil.copy2(src, dst)
    logger.info("Promoted draft detector: %s → %s", src, dst)
    return dst


def archive_packaging(key: str) -> Path:
    """Soft-delete: rename {key}.yaml → {key}.yaml.archived.

    Returns the new path. Raises FileNotFoundError if no active YAML exists.
    """
    src = _PACKAGING_DIR / f"{key}.yaml"
    if not src.exists():
        raise FileNotFoundError(f"no active YAML for '{key}'")
    dst = _PACKAGING_DIR / f"{key}.yaml.archived"
    if dst.exists():
        dst.unlink()  # collapse stale archive if any
    src.rename(dst)
    logger.info("Archived packaging: %s → %s", src, dst)
    return dst


def unarchive_packaging(key: str) -> Path:
    """Restore: rename {key}.yaml.archived → {key}.yaml."""
    src = _PACKAGING_DIR / f"{key}.yaml.archived"
    if not src.exists():
        raise FileNotFoundError(f"no archived YAML for '{key}'")
    dst = _PACKAGING_DIR / f"{key}.yaml"
    if dst.exists():
        raise FileExistsError(f"active '{key}' already exists — refusing to overwrite")
    src.rename(dst)
    logger.info("Unarchived packaging: %s → %s", src, dst)
    return dst


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def trigger_cloud_run_revision() -> dict[str, Any]:
    """Trigger a new Cloud Run revision so the running app re-syncs config + Drive manifest.

    Uses Cloud Run admin API. Requires roles/run.developer on the SA.
    If permission denied → return {'triggered': False, 'reason': ...} (non-fatal).
    """
    try:
        from googleapiclient.discovery import build
        from google.auth import default as google_auth_default

        creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        run = build("run", "v2", credentials=creds, cache_discovery=False)
        svc_name = (
            f"projects/{_CLOUD_RUN_PROJECT}/locations/{_CLOUD_RUN_REGION}"
            f"/services/{_CLOUD_RUN_SERVICE}"
        )
        # GET current service, then PATCH with a no-op annotation bump to force new revision
        svc = run.projects().locations().services().get(name=svc_name).execute()
        template = svc.get("template", {})
        annotations = template.setdefault("annotations", {})
        from datetime import datetime, timezone
        annotations["lot-checker/restarted-at"] = datetime.now(timezone.utc).isoformat()
        patched = run.projects().locations().services().patch(
            name=svc_name,
            body={"template": template},
            updateMask="template",
        ).execute()
        op_name = patched.get("name")
        logger.info("Cloud Run revision triggered: op=%s", op_name)
        return {"triggered": True, "operation": op_name}
    except Exception as e:
        logger.warning("Cloud Run trigger skipped: %s", e)
        return {"triggered": False, "reason": str(e)}
