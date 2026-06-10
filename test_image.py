"""
test_image.py — รัน pipeline ตรงๆ โดยไม่ต้องเปิด server

Usage:
  python test_image.py <image_path>                          # ไม่เช็ค sheet
  python test_image.py <image_path> <sheet_id>               # เช็ค sheet tab 0
  python test_image.py <image_path> <sheet_id> <sheet_gid>   # เช็ค sheet tab ระบุ
"""
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_image")


def run(image_path: str, sheet_id: str | None = None, sheet_gid: int = 0) -> dict:
    import os
    from pipeline.classifier import ImageClassifier
    from pipeline.detector import RegionDetector
    from pipeline.ocr_engine import OcrEngine
    from pipeline.preprocessor import Preprocessor
    from pipeline.qr_scanner import QrScanner
    from utils.image_utils import stack_images_vertically

    image_bytes = Path(image_path).read_bytes()
    logger.info("Image: %s (%d bytes)", image_path, len(image_bytes))

    # 1. Classify
    classifier_path = os.getenv("MODEL_CLASSIFIER_PATH", "models/classifier.pt")
    classifier = ImageClassifier(classifier_path)
    image_class, class_confidence = classifier.predict(image_bytes)
    logger.info("Class: %s  Confidence: %.4f", image_class, class_confidence)

    if class_confidence < 0.6:
        return {
            "status": "low_confidence",
            "class": image_class,
            "class_confidence": class_confidence,
            "verify_message": "⚠️ ไม่แน่ใจประเภทรูป กรุณาตรวจสอบด้วยตนเอง",
        }

    detector    = RegionDetector()
    preprocessor = Preprocessor()
    ocr_engine  = OcrEngine()

    # 2. Route by class
    if image_class == "import_sticker":
        qr_scanner = QrScanner()
        result = qr_scanner.scan(image_bytes)
        result.setdefault("product_name", None)
        result.setdefault("size", None)
        bbox = None

    elif image_class == "container_label":
        detections = detector.crop_all(image_bytes, image_class)
        crops: list[dict] = []
        for i, det in enumerate(detections):
            processed = preprocessor.run(det.cropped_bytes, image_class)
            ocr_res = ocr_engine.run(processed, image_class=image_class)
            text_ok = len(ocr_res["raw_text"].strip()) >= 8
            data_ok = bool(ocr_res["lot_number"] or ocr_res["exp_date"])
            if not text_ok and not data_ok:
                logger.warning("Crop %d degraded — retrying with original bytes", i)
                ocr_res = ocr_engine.run(det.cropped_bytes, image_class=image_class)
            crops.append({"lot_number": ocr_res["lot_number"], "exp_date": ocr_res["exp_date"]})
            logger.info("Crop %d → lot=%s  exp=%s  raw=%r", i, ocr_res["lot_number"], ocr_res["exp_date"], ocr_res["raw_text"])

        box    = crops[0] if len(crops) > 0 else {}
        sachet = crops[1] if len(crops) > 1 else {}
        result = {
            "lot_number":   None,
            "exp_date":     None,
            "mfg_date":     None,
            "raw_text":     "",
            "confidence":   None,
            "product_name": None,
            "size":         None,
            "lot_box":      box.get("lot_number"),
            "lot_sachet":   sachet.get("lot_number"),
            "exp_box":      box.get("exp_date"),
            "exp_sachet":   sachet.get("exp_date"),
            "status":       "ok" if any(c.get("lot_number") for c in crops) else "not_found",
        }
        bbox = detections[0].bbox if detections else None

    else:
        detections = detector.crop_all(image_bytes, image_class)
        processed  = [preprocessor.run(det.cropped_bytes, image_class) for det in detections]
        combined   = stack_images_vertically(processed)
        result     = ocr_engine.run(combined, image_class=image_class)
        bbox       = detections[0].bbox if detections else None

    # 3. Sheet check (optional)
    sheet: dict = {
        "lot_match": None, "exp_match": None,
        "product_match": None, "sachet_match": None,
        "sheet_product_name": None,
    }
    if sheet_id:
        from utils.sheet_checker import SheetChecker
        checker = SheetChecker()
        sheet = checker.check(
            image_class,
            sheet_id=sheet_id,
            gid=sheet_gid,
            lot_number=result.get("lot_number"),
            exp_date=result.get("exp_date"),
            product_name=result.get("product_name"),
            lot_box=result.get("lot_box"),
            lot_sachet=result.get("lot_sachet"),
            exp_box=result.get("exp_box"),
            exp_sachet=result.get("exp_sachet"),
        )

    return {
        "class":            image_class,
        "class_confidence": round(class_confidence, 4),
        "status":           result.get("status", "ok"),
        "lot_number":       result.get("lot_number"),
        "exp_date":         result.get("exp_date"),
        "mfg_date":         result.get("mfg_date"),
        "product_name":     result.get("product_name"),
        "size":             result.get("size"),
        "lot_box":          result.get("lot_box"),
        "lot_sachet":       result.get("lot_sachet"),
        "exp_box":          result.get("exp_box"),
        "exp_sachet":       result.get("exp_sachet"),
        "raw_text":         result.get("raw_text", ""),
        "bbox":             bbox,
        # sheet results (None ถ้าไม่ได้ส่ง sheet_id)
        "lot_match":        sheet["lot_match"],
        "exp_match":        sheet["exp_match"],
        "product_match":    sheet["product_match"],
        "sachet_match":     sheet["sachet_match"],
        "sheet_product_name": sheet.get("sheet_product_name"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    sheet_id   = sys.argv[2] if len(sys.argv) > 2 else None
    sheet_gid  = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    output = run(image_path, sheet_id, sheet_gid)
    print("\n" + "=" * 60)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("=" * 60)
