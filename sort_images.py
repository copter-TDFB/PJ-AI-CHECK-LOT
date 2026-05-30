"""
แยกรูปใน 'data collect pj check lot' ด้วย classifier.pt
- confidence >= 0.60 → subfolder ตาม class
- confidence <  0.60 → subfolder 'others'

วิธีรัน:
    python sort_images.py
"""

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DIR   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data collect pj check lot")
CONF_CUTOFF  = 0.60
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    from pipeline.classifier import ImageClassifier

    model = ImageClassifier("models/classifier.pt")

    images = [p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    logger.info("พบรูป %d ใบ — เริ่ม classify", len(images))

    counts: dict[str, int] = {}
    errors = 0

    for img_path in images:
        try:
            class_name, confidence = model.predict(img_path.read_bytes())
        except Exception as exc:
            logger.warning("อ่านรูปไม่ได้ %s: %s", img_path.name, exc)
            errors += 1
            dest_dir = SOURCE_DIR / "others"
            dest_dir.mkdir(exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)
            continue

        dest_folder = class_name if confidence >= CONF_CUTOFF else "others"
        dest_dir = SOURCE_DIR / dest_folder
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

        counts[dest_folder] = counts.get(dest_folder, 0) + 1
        logger.debug("%-40s → %-20s (%.2f%%)", img_path.name, dest_folder, confidence * 100)

    # สรุปผล
    logger.info("─" * 50)
    logger.info("สรุปผลการแยกรูป:")
    for folder, n in sorted(counts.items()):
        logger.info("  %-20s : %d รูป", folder, n)
    if errors:
        logger.info("  %-20s : %d รูป (อ่านไม่ได้)", "error→others", errors)
    logger.info("รวม: %d รูป", sum(counts.values()))


if __name__ == "__main__":
    main()
