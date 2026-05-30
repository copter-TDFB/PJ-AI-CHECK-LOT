"""
รัน pipeline บนรูปจริง class ละ 5 รูป แล้ว save ผลไว้ใน preview/
โครงสร้าง:
    preview/
    └── <class>/
        └── <filename>/
            ├── 1_original.jpg
            ├── 2_cropped.jpg
            ├── 3_preprocessed.jpg
            └── result.txt

วิธีรัน:
    python preview_pipeline.py
    python preview_pipeline.py --n 10          # class ละ 10 รูป
    python preview_pipeline.py --class grade_bag  # เฉพาะ class นั้น
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLASSES    = ["back_label", "container_label", "grade_bag", "import_sticker", "retail_sachet"]
IMAGES_DIR = Path("images")
PREVIEW_DIR = Path("preview")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
SEED       = 42


def save_image(img_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(img_bytes)


def bytes_to_jpeg(img_np: np.ndarray) -> bytes:
    import io
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run_preview(cls: str, img_path: Path, clf, det, pre, qr, ocr) -> dict:
    img_bytes = img_path.read_bytes()

    # ── 1. Classify ─────────────────────────────────────────────
    pred_class, class_conf = clf.predict(img_bytes)

    # ── 2. Detect / QR ──────────────────────────────────────────
    if cls == "import_sticker":
        qr_result = qr.scan(img_bytes)
        cropped_bytes = img_bytes          # QR scan ใช้รูปเต็ม
        preprocessed_bytes = img_bytes
        bbox = None
        lot    = qr_result["lot_number"]
        conf   = qr_result["confidence"]
        raw    = qr_result["raw_text"]
        mfg    = qr_result["mfg_date"]
        exp    = qr_result["exp_date"]
        status = qr_result["status"]
    else:
        detection = det.crop(img_bytes, cls)
        cropped_bytes = detection.cropped_bytes
        bbox = detection.bbox

        # ── 3. Preprocess ────────────────────────────────────────
        preprocessed_bytes = pre.run(cropped_bytes, cls)

        # ── 4. OCR ───────────────────────────────────────────────
        if ocr:
            ocr_result = ocr.run(preprocessed_bytes, image_class=cls)
            lot    = ocr_result["lot_number"]
            conf   = ocr_result["confidence"]
            raw    = ocr_result["raw_text"]
            mfg    = ocr_result["mfg_date"]
            exp    = ocr_result["exp_date"]
            status = ocr_result["status"]
        else:
            lot = conf = raw = mfg = exp = None
            status = "ocr_skipped"

    return {
        "file":            img_path.name,
        "true_class":      cls,
        "pred_class":      pred_class,
        "class_conf":      round(class_conf, 4),
        "bbox":            bbox,
        "lot_number":      lot,
        "ocr_confidence":  round(conf, 4) if conf else None,
        "raw_text":        raw,
        "mfg_date":        mfg,
        "exp_date":        exp,
        "status":          status,
        "cropped_bytes":   cropped_bytes,
        "preprocessed_bytes": preprocessed_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=5, help="จำนวนรูปต่อ class")
    parser.add_argument("--class", dest="cls",          help="เฉพาะ class นี้")
    parser.add_argument("--no-ocr", action="store_true", help="ข้าม OCR step")
    args = parser.parse_args()

    # โหลด pipeline
    print("Loading pipeline...")
    from pipeline.classifier import ImageClassifier
    from pipeline.detector   import RegionDetector
    from pipeline.preprocessor import Preprocessor
    from pipeline.qr_scanner  import QrScanner

    clf = ImageClassifier("models/classifier.pt")
    det = RegionDetector("models/detector.pt")
    pre = Preprocessor()
    qr  = QrScanner()
    ocr = None

    if not args.no_ocr:
        try:
            from pipeline.ocr_engine import OcrEngine
            ocr = OcrEngine()
            print("OCR: Google Cloud Vision API ✓")
        except Exception as e:
            print(f"OCR: ข้าม (credentials ไม่พบ) — {e}")

    # ล้าง preview/ เก่า
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir()

    target_classes = [args.cls] if args.cls else CLASSES
    random.seed(SEED)

    summary = []

    for cls in target_classes:
        cls_dir = IMAGES_DIR / cls
        if not cls_dir.exists():
            print(f"  ⚠️  ไม่พบ images/{cls}/ — ข้าม")
            continue

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        sample = random.sample(images, min(args.n, len(images)))
        print(f"\n[{cls}] {len(sample)} รูป")

        for img_path in sample:
            try:
                result = run_preview(cls, img_path, clf, det, pre, qr, ocr)
            except Exception as exc:
                print(f"  ❌ {img_path.name}: {exc}")
                continue

            # บันทึกไฟล์
            out_dir = PREVIEW_DIR / cls / img_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            save_image(img_path.read_bytes(),         out_dir / "1_original.jpg")
            save_image(result["cropped_bytes"],        out_dir / "2_cropped.jpg")
            save_image(result["preprocessed_bytes"],   out_dir / "3_preprocessed.jpg")

            # result.txt
            txt = f"""=== {img_path.name} ===
Classify:   {result['pred_class']} (conf={result['class_conf']:.2%})  [true={result['true_class']}]
Bbox:       {result['bbox']}
Lot Number: {result['lot_number']}
MFG Date:   {result['mfg_date']}
EXP Date:   {result['exp_date']}
Status:     {result['status']}
OCR Conf:   {result['ocr_confidence']}

--- Raw OCR Text ---
{result['raw_text'] or '(ไม่มี)'}
"""
            (out_dir / "result.txt").write_text(txt, encoding="utf-8")

            clf_ok = "✅" if result["pred_class"] == cls else "❌"
            lot_ok = "✅" if result["lot_number"] else "—"
            print(f"  {clf_ok} {img_path.name:<35} lot={lot_ok} {result['lot_number'] or ''}")

            summary.append({k: v for k, v in result.items()
                            if k not in ("cropped_bytes", "preprocessed_bytes")})

    # สรุปภาพรวม
    (PREVIEW_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ บันทึกผลไว้ที่ preview/  (summary: preview/summary.json)")


if __name__ == "__main__":
    main()
