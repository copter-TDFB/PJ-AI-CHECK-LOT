"""Runtime tuning overrides — Drive-persisted, merged over packaging YAML (ADR 0004).

Stores ONLY runtime tuning fields keyed by packaging key:

    {"back_label": {"conf_threshold": 0.75}}

Storage backend:
  - `DRIVE_CONFIG_OVERRIDES_FILE_ID` set → that file on Drive (SA-shared,
    same pattern as DRIVE_MANIFEST_FILE_ID).
  - empty → local `data/config_overrides.json` (path overridable via
    CONFIG_OVERRIDES_PATH for tests).

`load()` never raises — a broken tuning file must not take the service down.
`save_conf_threshold()` raises on persist failure so the API can refuse the
change without diverging from Drive.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from services.drive_client import DriveClient

logger = logging.getLogger(__name__)

_ENV_FILE_ID = "DRIVE_CONFIG_OVERRIDES_FILE_ID"
_ENV_LOCAL_PATH = "CONFIG_OVERRIDES_PATH"
_DEFAULT_LOCAL_PATH = "data/config_overrides.json"
_GCS_OBJECT = "config_overrides.json"


def _drive_file_id() -> str:
    return os.getenv(_ENV_FILE_ID, "").strip()


def _local_path() -> Path:
    return Path(os.getenv(_ENV_LOCAL_PATH, _DEFAULT_LOCAL_PATH))


def _validated(raw: object) -> dict[str, dict]:
    """Accept only {key: {field: value}} shapes — anything else → {}."""
    if not isinstance(raw, dict):
        logger.warning("config overrides payload is not a dict — ignoring")
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def load() -> dict[str, dict]:
    """Load overrides from GCS (or Drive / local fallback). Never raises."""
    from services import gcs_store

    store = gcs_store.get_store()
    if store is not None:
        try:
            data = store.read_json(_GCS_OBJECT)
            return _validated(data) if data is not None else {}
        except Exception as e:
            # transient GCS error → fall through to Drive/local rather than
            # silently dropping operator-tuned conf_threshold overrides
            logger.warning("config overrides (GCS) unreadable — falling back to Drive/local: %s", e)

    file_id = _drive_file_id()
    try:
        if file_id:
            return _validated(DriveClient().read_json(file_id))
        path = _local_path()
        if not path.exists():
            return {}
        return _validated(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.warning("config overrides unreadable — using YAML values only: %s", e)
        return {}


def _persist(merged: dict[str, dict]) -> None:
    """Write the merged overrides to GCS (or Drive / local fallback). Raises on failure."""
    from services import gcs_store

    store = gcs_store.get_store()
    if store is not None:
        store.write_json(_GCS_OBJECT, merged)
        return
    content = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")
    file_id = _drive_file_id()
    if file_id:
        DriveClient().update_file_content(file_id, content, mime_type="application/json")
    else:
        path = _local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.decode("utf-8"), encoding="utf-8")


def save_conf_threshold(key: str, value: float) -> dict[str, dict]:
    """Persist a conf_threshold override and return the merged overrides.

    Raises on persist failure — caller must NOT apply the change locally
    in that case, so instances never diverge from the stored overrides.
    """
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    merged.setdefault(key, {})["conf_threshold"] = float(value)
    _persist(merged)
    logger.info("Saved conf_threshold override: %s=%.2f", key, value)
    return merged


def save_product_aliases(key: str, aliases: list[dict]) -> dict[str, dict]:
    """Persist a product_aliases override and return the merged overrides.

    Raises on persist failure — caller must NOT apply the change locally
    in that case, so instances never diverge from the stored overrides.
    """
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    merged.setdefault(key, {})["product_aliases"] = [
        {"canonical": a["canonical"], "keywords": list(a["keywords"])}
        for a in aliases
    ]
    _persist(merged)
    logger.info("Saved product_aliases override: %s (%d aliases)", key, len(aliases))
    return merged


def delete_product_aliases(key: str) -> dict[str, dict]:
    """Remove a product_aliases override → revert to the YAML/hardcoded default.

    Idempotent (no-op if none stored). Raises on persist failure — caller must
    NOT apply the change locally in that case.
    """
    current = load()
    merged = {k: dict(v) for k, v in current.items()}
    entry = merged.get(key)
    if entry is not None:
        entry.pop("product_aliases", None)
        if not entry:
            merged.pop(key, None)
    _persist(merged)
    logger.info("Deleted product_aliases override: %s", key)
    return merged
