import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from pipeline.classifier import ImageClassifier
from pipeline.detector import RegionDetector
from pipeline.ocr_engine import OcrEngine
from pipeline.preprocessor import Preprocessor
from pipeline.qr_scanner import QrScanner
from utils.image_utils import stack_images_vertically
from utils.sheet_checker import SheetChecker


def _build_verify_message(image_class: str, sheet: dict) -> str:
    """สร้าง summary message สำหรับส่ง Slack"""
    lot_match     = sheet["lot_match"]
    exp_match     = sheet["exp_match"]
    product_match = sheet["product_match"]
    sachet_match  = sheet["sachet_match"]

    if image_class == "import_sticker":
        if lot_match:
            return "✅ ตรวจสอบผ่าน"
        return "❌ ไม่พบ lot ใน sheet"

    if image_class in ("back_label", "grade_bag"):
        if not lot_match:
            return "❌ ไม่พบ lot ใน sheet"
        errors = []
        if not exp_match:
            errors.append("exp ไม่ตรง")
        if not product_match:
            errors.append("product ไม่ตรง")
        if errors:
            return f"❌ {', '.join(errors)}"
        return "✅ ตรวจสอบผ่าน"

    if image_class in ("retail_sachet", "capsule_box"):
        if not lot_match:
            return "❌ ไม่พบ lot ใน sheet"
        if not exp_match:
            return "❌ exp ไม่ตรง"
        return "✅ ตรวจสอบผ่าน"

    if image_class == "container_label":
        errors = []
        if not lot_match:
            errors.append("ไม่พบ lot ใน sheet")
        if not exp_match:
            errors.append("exp ไม่ตรง")
        if not sachet_match:
            errors.append("กล่องกับซองไม่ตรงกัน")
        if errors:
            return f"❌ {', '.join(errors)}"
        return "✅ ตรวจสอบผ่าน"

    return "✅ ตรวจสอบผ่าน"

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_CLASSIFIER_PATH = os.getenv("MODEL_CLASSIFIER_PATH", "models/classifier.pt")

classifier: ImageClassifier | None = None
detector: RegionDetector | None = None
preprocessor: Preprocessor | None = None
ocr_engine: OcrEngine | None = None
qr_scanner: QrScanner | None = None
sheet_checker: SheetChecker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, detector, preprocessor, ocr_engine, qr_scanner, sheet_checker
    logger.info("Startup — loading models")
    classifier = ImageClassifier(_CLASSIFIER_PATH)
    detector = RegionDetector()
    preprocessor = Preprocessor()
    ocr_engine = OcrEngine()
    qr_scanner = QrScanner()
    sheet_checker = SheetChecker()
    yield
    logger.info("Shutdown")


app = FastAPI(title="OCR Lot Checker", lifespan=lifespan)


@app.get("/health")
def health():
    """Health check สำหรับ Cloud Run"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sheet_id: str = Query(..., description="Google Sheet ID"),
    sheet_gid: int = Query(0, description="Google Sheet GID (tab)"),
):
    """
    รับรูปภาพและคืนผล OCR เลข lot

    n8n ส่ง multipart/form-data field ชื่อ 'file'
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="file is empty")

    logger.info("Received: %s (%d bytes)", file.filename, len(image_bytes))

    try:
        image_class, class_confidence = classifier.predict(image_bytes)

        if class_confidence < 0.6:
            logger.warning("Low confidence %.4f for class=%s — skipping pipeline", class_confidence, image_class)
            return JSONResponse({
                "lot_number":       None,
                "confidence":       None,
                "class":            image_class,
                "class_confidence": class_confidence,
                "raw_text":         "",
                "mfg_date":         None,
                "exp_date":         None,
                "product_name":     None,
                "size":             None,
                "lot_box":          None,
                "lot_sachet":       None,
                "exp_box":          None,
                "exp_sachet":       None,
                "lot_match":        False,
                "exp_match":        False,
                "product_match":    False,
                "sachet_match":     False,
                "verify_message":   "⚠️ ไม่แน่ใจประเภทรูป กรุณาตรวจสอบด้วยตนเอง",
                "bbox":             None,
                "status":           "low_confidence",
            })

        if image_class == "import_sticker":
            # import_sticker มี QR code — สแกนโดยตรง ข้าม crop/preprocess/OCR
            result = qr_scanner.scan(image_bytes)
            result.setdefault("product_name", None)
            result.setdefault("size", None)
            bbox = None
        elif image_class == "container_label":
            # container_label มี 2 crop (lot ยาว + lot สั้น) แต่ละ crop มี lot + date ของตัวเอง
            # ต้อง OCR แยกเพื่อจับคู่ lot↔date ให้ถูก
            detections = detector.crop_all(image_bytes, image_class)
            crops: list[dict] = []
            for det in detections:
                processed = preprocessor.run(det.cropped_bytes, image_class)
                ocr_res = ocr_engine.run(processed, image_class=image_class)
                text_ok = len(ocr_res["raw_text"].strip()) >= 8
                data_ok = bool(ocr_res["lot_number"] or ocr_res["exp_date"])
                if not text_ok and not data_ok:
                    logger.warning("Preprocessed crop appears degraded — retrying with original bytes")
                    ocr_res = ocr_engine.run(det.cropped_bytes, image_class=image_class)
                crops.append({
                    "lot_number": ocr_res["lot_number"],
                    "exp_date":   ocr_res["exp_date"],
                })
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
                "status":       "ok" if any(c["lot_number"] for c in crops) else "not_found",
            }
            bbox = detections[0].bbox
        else:
            # OCR path: crop → preprocess → stack รวมเป็นรูปเดียว → OCR 1 ครั้ง
            detections = detector.crop_all(image_bytes, image_class)
            processed = [preprocessor.run(det.cropped_bytes, image_class) for det in detections]
            combined = stack_images_vertically(processed)
            result = ocr_engine.run(combined, image_class=image_class)
            bbox = detections[0].bbox   # bbox ของ region แรก (บนสุด)

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    sheet = sheet_checker.check(
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

    response = {
        "lot_number":       result["lot_number"],
        "confidence":       result["confidence"],
        "class":            image_class,
        "class_confidence": class_confidence,
        "raw_text":         result["raw_text"],
        "mfg_date":         result["mfg_date"],
        "exp_date":         result["exp_date"],
        "product_name":     result.get("product_name"),
        "size":             result.get("size"),
        "lot_box":          result.get("lot_box"),
        "lot_sachet":       result.get("lot_sachet"),
        "exp_box":          result.get("exp_box"),
        "exp_sachet":       result.get("exp_sachet"),
        "lot_match":        sheet["lot_match"],
        "exp_match":        sheet["exp_match"],
        "product_match":    sheet["product_match"],
        "sachet_match":     sheet["sachet_match"],
        "verify_message":   _build_verify_message(image_class, sheet),
        "bbox":             bbox,
        "status":           result["status"],
    }
    return JSONResponse(response)
