"""Filesystem-based store สำหรับ draft packagings.

โครงสร้าง:
    data/drafts/{key}/
        meta.json           — metadata + status + config
        images/             — รูปที่ ops อัพโหลด
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DRAFT_DIR = Path(os.getenv("DRAFT_DIR", "data/drafts"))
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_drafts() -> list[dict]:
    """List all drafts."""
    if not _DRAFT_DIR.exists():
        return []
    drafts = []
    for d in sorted(_DRAFT_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["image_count"] = _count_images(d / "images")
            drafts.append(meta)
        except Exception:
            logger.exception("Failed to read draft meta: %s", d)
    return drafts


def get_draft(key: str) -> dict | None:
    meta_file = _DRAFT_DIR / key / "meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["image_count"] = _count_images(_DRAFT_DIR / key / "images")
    return meta


def create_draft(
    key: str,
    display_name: str,
    description: str | None,
    pipeline: str,
    sub_regions: list[str] | None = None,
    detection_mode: str = "single",
) -> dict:
    """สร้าง draft ใหม่ — raise ValueError ถ้ามี key ซ้ำ."""
    draft_path = _DRAFT_DIR / key
    if draft_path.exists():
        raise ValueError(f"draft '{key}' already exists")

    draft_path.mkdir(parents=True, exist_ok=False)
    (draft_path / "images").mkdir()

    meta = {
        "key": key,
        "display_name": display_name,
        "description": description,
        "pipeline": pipeline,
        "sub_regions": sub_regions or ["lot"],
        "detection_mode": detection_mode,
        "status": "draft",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "config": None,
    }
    _write_meta(key, meta)
    return meta


def update_draft(key: str, **updates) -> dict | None:
    meta = _read_meta(key)
    if meta is None:
        return None
    for k, v in updates.items():
        if v is not None:
            meta[k] = v
    meta["updated_at"] = _now_iso()
    _write_meta(key, meta)
    return meta


def save_config(key: str, config: dict) -> dict | None:
    meta = _read_meta(key)
    if meta is None:
        return None
    meta["config"] = config
    meta["status"] = "configured"
    meta["updated_at"] = _now_iso()
    _write_meta(key, meta)
    return meta


def delete_draft(key: str) -> bool:
    draft_path = _DRAFT_DIR / key
    if not draft_path.exists():
        return False
    shutil.rmtree(draft_path)
    return True


def clone_from_active(
    parent_key: str,
    active_yaml: dict,
    description: str | None = None,
) -> dict:
    """Create an edit-draft `{parent_key}__edit` cloned from an active packaging.

    Copies metadata + config from the active YAML. Does NOT copy images —
    UI links to parent's `images/{parent_key}/` for read-only display.

    Raises ValueError if the edit-draft already exists.
    """
    edit_key = f"{parent_key}__edit"
    draft_path = _DRAFT_DIR / edit_key
    if draft_path.exists():
        raise ValueError(f"edit-draft '{edit_key}' already exists — continue or discard first")

    draft_path.mkdir(parents=True, exist_ok=False)
    (draft_path / "images").mkdir()

    # Re-shape active YAML into draft.config format (matches save_config payload)
    config = {
        "lot_patterns": active_yaml.get("lot_patterns", []),
        "fields_extracted": active_yaml.get("fields_extracted", ["lot"]),
        "sheet_checks": active_yaml.get("sheet_checks", []),
        "message_template_key": active_yaml.get("message_template_key", "default_full"),
        "product_aliases": active_yaml.get("product_aliases", []),
    }

    meta = {
        "key": edit_key,
        "parent_key": parent_key,
        "display_name": active_yaml.get("display_name", parent_key),
        "description": description or f"Edit draft cloned from active {parent_key}",
        "pipeline": active_yaml.get("pipeline", "detector_ocr"),
        "sub_regions": active_yaml.get("sub_regions", []) or ["lot"],
        "detection_mode": active_yaml.get("detection_mode", "single"),
        "status": "draft",  # no images of its own yet — first upload bumps to "uploading"
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "config": config,
    }
    _write_meta(edit_key, meta)
    return meta


def parent_images_dir(parent_key: str) -> Path:
    """Path to parent active packaging's reference images (read-only from edit-draft POV)."""
    return Path("images") / parent_key


def save_image(key: str, filename: str, content: bytes) -> Path:
    img_dir = _DRAFT_DIR / key / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    dest = img_dir / safe_name
    # Avoid overwriting — add suffix ถ้าซ้ำ
    i = 1
    while dest.exists():
        stem, suf = safe_name.rsplit(".", 1) if "." in safe_name else (safe_name, "")
        dest = img_dir / f"{stem}_{i}.{suf}" if suf else img_dir / f"{stem}_{i}"
        i += 1
    dest.write_bytes(content)
    # Bump status → uploading (so wizard "Continue setup" resumes at step 3)
    meta_file = _DRAFT_DIR / key / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("status") == "draft":
                meta["status"] = "uploading"
                meta["updated_at"] = _now_iso()
                meta_file.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except (json.JSONDecodeError, OSError):
            pass
    return dest


def list_images(key: str) -> list[dict]:
    img_dir = _DRAFT_DIR / key / "images"
    if not img_dir.exists():
        return []
    return [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(img_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    ]


def delete_image(key: str, filename: str) -> bool:
    img = _DRAFT_DIR / key / "images" / _safe_filename(filename)
    if not img.exists() or not img.is_file():
        return False
    img.unlink()
    return True


def image_path(key: str, filename: str) -> Path | None:
    """Return absolute path for serving an image. None ถ้าไม่มี."""
    img = _DRAFT_DIR / key / "images" / _safe_filename(filename)
    if img.exists() and img.is_file():
        return img
    return None


# ─── Annotations ─────────────────────────────────────────

def save_annotation(key: str, filename: str, bboxes: list[dict]) -> dict | None:
    """Save bbox annotations สำหรับรูป — overwrite. Return saved data หรือ None."""
    if _DRAFT_DIR / key not in [_DRAFT_DIR / key] or not (_DRAFT_DIR / key).exists():
        return None
    ann_dir = _DRAFT_DIR / key / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(filename)
    data = {"bboxes": bboxes, "updated_at": _now_iso()}
    (ann_dir / f"{safe}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return data


def get_annotation(key: str, filename: str) -> dict | None:
    """Return annotation JSON หรือ None ถ้าไม่มี."""
    safe = _safe_filename(filename)
    p = _DRAFT_DIR / key / "annotations" / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_annotation_status(key: str) -> list[dict]:
    """Return [{name, labeled, bbox_count}] สำหรับทุกรูปใน draft."""
    img_dir = _DRAFT_DIR / key / "images"
    ann_dir = _DRAFT_DIR / key / "annotations"
    if not img_dir.exists():
        return []
    out = []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in _IMG_EXTS:
            continue
        ann_file = ann_dir / f"{img.name}.json"
        if ann_file.exists():
            try:
                data = json.loads(ann_file.read_text(encoding="utf-8"))
                count = len(data.get("bboxes", []))
                out.append({"name": img.name, "labeled": count > 0, "bbox_count": count})
                continue
            except (json.JSONDecodeError, OSError):
                pass
        out.append({"name": img.name, "labeled": False, "bbox_count": 0})
    return out


def delete_annotation(key: str, filename: str) -> bool:
    p = _DRAFT_DIR / key / "annotations" / f"{_safe_filename(filename)}.json"
    if not p.exists():
        return False
    p.unlink()
    return True


def _count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    return sum(
        1 for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )


def _read_meta(key: str) -> dict | None:
    meta_file = _DRAFT_DIR / key / "meta.json"
    if not meta_file.exists():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _write_meta(key: str, meta: dict) -> None:
    meta_file = _DRAFT_DIR / key / "meta.json"
    meta_file.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_filename(name: str) -> str:
    """Strip path components — กัน path traversal."""
    return Path(name).name
