"""Publish a draft's labeled images + YOLO labels directly into the Drive
reference dataset (detector + classifier folders).

Replaces the zip bundle for the FULL training path only (seed training still
uses services/training_bundle.py — see ADR 0003).

Key invariants:
- data.yaml `names` are only ever APPENDED to — label files reference numeric
  class ids, so reordering existing names corrupts the dataset.
- data.yaml is written LAST. It is the commit point: if an upload run dies
  mid-way, the new class names were never declared and the dataset stays
  valid. Retries skip files that already exist (idempotent).
- Filenames are prefixed with the packaging key to avoid cross-packaging
  collisions.
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DRAFT_DIR = Path("data/drafts")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_DET_ENV = "DRIVE_DETECTOR_DATASET_FOLDER_ID"
_CLS_ENV = "DRIVE_CLASSIFIER_DATASET_FOLDER_ID"
_TRAIN_BUCKETS = 8  # out of 10 → 80/20 split


def merge_class_names(existing: list[str], new: list[str]) -> list[str]:
    """Append names not already present. NEVER reorders existing entries."""
    return list(existing) + [n for n in new if n not in existing]


def split_for(filename: str) -> str:
    """Deterministic train/val assignment — same name always lands the same."""
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return "train" if int(digest, 16) % 10 < _TRAIN_BUCKETS else "val"


def labels_relpath(images_relpath: str) -> str:
    """YOLO convention: swap the last 'images' path segment for 'labels'."""
    parts = images_relpath.strip("/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return "/".join(parts)
    raise ValueError(f"no 'images' segment in dataset path: {images_relpath!r}")


def label_lines(
    bboxes: list[dict],
    label_to_id: dict[str, int],
    w: int,
    h: int,
    default_label: str,
) -> list[str]:
    """Convert pixel bboxes → YOLO lines with GLOBAL class ids."""
    lines = []
    for b in bboxes:
        label = b.get("label") or default_label
        if label not in label_to_id:
            logger.warning("unknown label '%s' — skipping bbox", label)
            continue
        cid = label_to_id[label]
        cx = ((b["x1"] + b["x2"]) / 2) / w
        cy = ((b["y1"] + b["y2"]) / 2) / h
        bw = (b["x2"] - b["x1"]) / w
        bh = (b["y2"] - b["y1"]) / h
        if bw <= 0 or bh <= 0:
            logger.warning("zero-area bbox skipped (label '%s')", label)
            continue
        cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
        bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def publish(key: str, drive=None) -> dict:
    """Publish draft's labeled data into the Drive reference dataset.

    Returns {images_uploaded, images_skipped, train_count, val_count,
             new_classes, class_ids, total_classes}.
    Raises RuntimeError (setup problems), FileNotFoundError (no draft),
    ValueError (no labeled images).
    """
    from services import packaging_store

    det_root = os.getenv(_DET_ENV, "")
    cls_root = os.getenv(_CLS_ENV, "")
    if not det_root or not cls_root:
        raise RuntimeError(
            f"{_DET_ENV} / {_CLS_ENV} not set — share the dataset folders with "
            "the service account email (Editor) and set both env vars"
        )

    draft = packaging_store.get_draft(key)
    if draft is None:
        raise FileNotFoundError(f"draft '{key}' not found")
    sub_regions: list[str] = draft.get("sub_regions") or ["lot"]
    new_names = [f"{key}_{sr}" for sr in sub_regions]

    if drive is None:
        from services.drive_client import DriveClient
        drive = DriveClient()

    yaml_id, data_yaml, existing_names = _load_data_yaml(drive, det_root)
    merged = merge_class_names(existing_names, new_names)
    label_to_id = {sr: merged.index(f"{key}_{sr}") for sr in sub_regions}

    dest = _resolve_dest_folders(drive, det_root, data_yaml)
    cls_folder = drive.ensure_folder(key, cls_root)
    # Snapshot read once per publish() run. Files created mid-run by a
    # concurrent/partial uploader aren't visible here; they'll be seen on the
    # next call (acceptable: retry rebuilds the snapshot cleanly).
    existing_files = {
        fid: {f["name"] for f in drive.list_folder(fid)}
        for fid in set(dest.values()) | {cls_folder}
    }

    stats = _upload_items(
        drive, key, label_to_id, sub_regions[0], dest, cls_folder, existing_files
    )
    if stats["images_uploaded"] == 0 and stats["images_skipped"] == 0:
        raise ValueError(f"draft '{key}' has no labeled images yet")

    # data.yaml LAST — the commit point (see module docstring)
    if merged != existing_names:
        import yaml as _yaml
        updated_yaml = {**data_yaml, "names": merged, "nc": len(merged)}
        drive.update_file_content(
            yaml_id,
            _yaml.safe_dump(updated_yaml, sort_keys=False, allow_unicode=True).encode("utf-8"),
            mime_type="text/yaml",
        )

    summary = {
        **stats,
        "new_classes": [n for n in merged if n not in existing_names],
        "class_ids": label_to_id,
        "total_classes": len(merged),
    }
    logger.info("Dataset published for '%s': %s", key, summary)
    return summary


def _load_data_yaml(drive, det_root: str) -> tuple[str, dict, list[str]]:
    """Fetch data.yaml from the detector dataset folder. Fail fast if absent."""
    import yaml as _yaml

    yaml_id = drive.find_in_folder(det_root, "data.yaml")
    if yaml_id is None:
        raise RuntimeError(
            "data.yaml not found in the detector dataset folder — check "
            f"{_DET_ENV} and that the folder is shared with the service account"
        )
    data_yaml = _yaml.safe_load(drive.read_text(yaml_id)) or {}
    return yaml_id, data_yaml, list(data_yaml.get("names") or [])


def _resolve_dest_folders(drive, det_root: str, data_yaml: dict) -> dict:
    """Map (split, kind) → Drive folder id, derived from data.yaml paths."""
    train_rel = str(data_yaml.get("train") or "train/images")
    val_rel = str(data_yaml.get("val") or "val/images")
    rels = {
        ("train", "images"): train_rel,
        ("train", "labels"): labels_relpath(train_rel),
        ("val", "images"): val_rel,
        ("val", "labels"): labels_relpath(val_rel),
    }
    dest = {}
    for k, rel in rels.items():
        fid = det_root
        for seg in rel.strip("/").split("/"):
            if seg in ("", "."):
                continue
            fid = drive.ensure_folder(seg, fid)
        dest[k] = fid
    return dest


def _upload_items(
    drive,
    key: str,
    label_to_id: dict[str, int],
    default_label: str,
    dest: dict,
    cls_folder: str,
    existing_files: dict[str, set[str]],
) -> dict:
    """Upload each labeled image + label + classifier copy. Skip existing."""
    from PIL import Image

    from services import packaging_store

    img_dir = _DRAFT_DIR / key / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"no images for draft '{key}'")

    uploaded = skipped = 0
    counts = {"train": 0, "val": 0}
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        ann = packaging_store.get_annotation(key, img_path.name)
        if not ann or not ann.get("bboxes"):
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            logger.warning("skip unreadable: %s", img_path.name)
            continue
        if w == 0 or h == 0:
            continue
        lines = label_lines(ann["bboxes"], label_to_id, w, h, default_label)
        if not lines:
            continue

        new_name = f"{key}_{img_path.name}"
        lbl_name = f"{Path(new_name).stem}.txt"
        split = split_for(new_name)
        counts[split] += 1
        img_fid = dest[(split, "images")]
        lbl_fid = dest[(split, "labels")]

        if new_name in existing_files[img_fid]:
            skipped += 1
        else:
            drive.upload_file(img_path, parent_id=img_fid, name=new_name)
            uploaded += 1
        if lbl_name not in existing_files[lbl_fid]:
            drive.upload_bytes(
                ("\n".join(lines) + "\n").encode("utf-8"),
                name=lbl_name, parent_id=lbl_fid, mime_type="text/plain",
            )
        if new_name not in existing_files[cls_folder]:
            drive.upload_file(img_path, parent_id=cls_folder, name=new_name)

    return {
        "images_uploaded": uploaded,
        "images_skipped": skipped,
        "train_count": counts["train"],
        "val_count": counts["val"],
    }
