"""Build a zip bundle of images + YOLO-format labels for training in Colab.

Bundle ONLY the new packaging's data. The notebook merges with the existing
reference dataset (which lives on the user's personal Drive) at training time.

Layout inside zip:
    data_addition.yaml          # describes new class names + paths
    images/{filename}.jpg
    labels/{stem}.txt           # YOLO format
    classifier_images/{key}/{filename}.jpg
"""

import io
import json
import logging
import zipfile
from pathlib import Path

from PIL import Image

from services import packaging_store

logger = logging.getLogger(__name__)

_DRAFT_DIR = Path("data/drafts")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_zip(key: str) -> bytes:
    """Build training bundle ZIP. Returns bytes ready to upload to Drive.

    YOLO class ordering: sub_regions list order from draft meta.
    YOLO class names: f"{key}_{sub_region}" for each entry.
    """
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise FileNotFoundError(f"draft '{key}' not found")

    sub_regions: list[str] = draft.get("sub_regions") or ["lot"]
    label_to_id = {sr: i for i, sr in enumerate(sub_regions)}
    yolo_class_names = [f"{key}_{sr}" for sr in sub_regions]

    img_dir = _DRAFT_DIR / key / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"no images for draft '{key}'")

    buf = io.BytesIO()
    img_count = 0
    bbox_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Addition manifest — notebook reads this to know how to merge
        manifest = {
            "key": key,
            "sub_regions": sub_regions,
            "yolo_class_names": yolo_class_names,
        }
        zf.writestr("addition_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
                continue
            ann = packaging_store.get_annotation(key, img_path.name)
            if not ann or not ann.get("bboxes"):
                continue

            # Image size for normalisation
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                logger.warning("skip unreadable: %s", img_path.name)
                continue
            if w == 0 or h == 0:
                continue

            label_lines = []
            for b in ann["bboxes"]:
                label = b.get("label") or sub_regions[0]
                if label not in label_to_id:
                    logger.warning("unknown label '%s' for %s — skipping bbox", label, img_path.name)
                    continue
                cls_id = label_to_id[label]
                cx = ((b["x1"] + b["x2"]) / 2) / w
                cy = ((b["y1"] + b["y2"]) / 2) / h
                bw = (b["x2"] - b["x1"]) / w
                bh = (b["y2"] - b["y1"]) / h
                cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                bbox_count += 1

            if not label_lines:
                continue

            # Detector: images + labels (YOLO convention)
            zf.write(img_path, f"images/{img_path.name}")
            zf.writestr(f"labels/{img_path.stem}.txt", "\n".join(label_lines) + "\n")

            # Classifier: separate folder named after the packaging key
            zf.write(img_path, f"classifier_images/{key}/{img_path.name}")

            img_count += 1

    if img_count == 0:
        raise ValueError(f"draft '{key}' has no labeled images yet")

    logger.info(
        "Bundle built: key=%s images=%d bboxes=%d size=%.1fKB classes=%s",
        key, img_count, bbox_count, len(buf.getvalue()) / 1024, yolo_class_names,
    )
    return buf.getvalue()
