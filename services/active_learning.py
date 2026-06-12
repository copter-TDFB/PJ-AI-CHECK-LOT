"""Active-learning loop — use seed model to pre-annotate remaining images.

After ops verifies + manual-labels 20 รูป, seed model is trained on those.
This service runs the seed model on ALL OTHER images of the draft and saves
the predictions as pre-annotations (so ops verifies/edits instead of labeling
from scratch).
"""

import logging
from pathlib import Path

from services import packaging_store

logger = logging.getLogger(__name__)

_DRAFT_DIR = Path("data/drafts")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_PRELABEL_CONF = 0.25  # min YOLO confidence to keep a prediction


def filter_prelabel_bboxes(boxes, class_prefixes=None):
    """boxes: iterable of (x1, y1, x2, y2, class_name).

    Keep boxes whose class_name starts with one of class_prefixes (when given)
    and that have positive area. Returns a list of annotation bbox dicts.
    """
    kept = []
    for x1, y1, x2, y2, name in boxes:
        if class_prefixes and not any(name.startswith(p) for p in class_prefixes):
            continue
        if x2 > x1 and y2 > y1:
            kept.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2), "label": "prelabel",
            })
    return kept


def prelabel_remaining(key: str, model_path: Path, conf: float = _PRELABEL_CONF,
                       class_prefixes: list[str] | None = None) -> dict:
    """Run seed model on all unlabeled images → save bbox predictions.

    Returns {prelabeled: int, skipped_already_labeled: int, errors: int}.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"seed model not found: {model_path}")

    # Lazy import — ultralytics is heavy
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    class_names = model.names if hasattr(model, "names") else {}
    img_dir = _DRAFT_DIR / key / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"no images for draft '{key}'")

    prelabeled = 0
    skipped = 0
    errors = 0

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        existing = packaging_store.get_annotation(key, img_path.name)
        if existing and existing.get("bboxes"):
            skipped += 1
            continue
        try:
            results = model.predict(str(img_path), conf=conf, verbose=False)
            r = results[0]
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.int().tolist()
                for box, cid in zip(xyxy, cls_ids):
                    name = class_names.get(int(cid), "") if isinstance(class_names, dict) else ""
                    boxes.append((float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3]), name))
            bboxes = filter_prelabel_bboxes(boxes, class_prefixes)
            if bboxes:
                packaging_store.save_annotation(key, img_path.name, bboxes)
                prelabeled += 1
        except Exception:
            logger.exception("prelabel failed for %s", img_path.name)
            errors += 1

    logger.info(
        "Active learning done: key=%s prelabeled=%d skipped=%d errors=%d",
        key, prelabeled, skipped, errors,
    )
    return {"prelabeled": prelabeled, "skipped_already_labeled": skipped, "errors": errors}
