import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.packagings import router as packagings_router
from pipeline.classifier import ImageClassifier
from pipeline.detector import RegionDetector
from pipeline.message_builder import build_verify_message
from pipeline.ocr_engine import OcrEngine
from pipeline.packaging_registry import PackagingRegistry
from pipeline.pipeline_runner import PipelineRunner
from pipeline.preprocessor import Preprocessor
from pipeline.qr_scanner import QrScanner
from services import config_overrides, model_registry
from utils.sheet_checker import SheetChecker

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONF_THRESHOLD = 0.6

classifier: ImageClassifier | None = None
detector: RegionDetector | None = None
pipeline_runner: PipelineRunner | None = None
sheet_checker: SheetChecker | None = None
registry: PackagingRegistry | None = None


def reload_registry() -> None:
    """Rebuild the registry with runtime tuning overrides merged in (ADR 0004)."""
    global registry
    registry = PackagingRegistry(overrides=config_overrides.load())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, detector, pipeline_runner, sheet_checker, registry
    logger.info("Startup — loading models and config")
    reload_registry()
    clf_path, det_path = model_registry.sync()
    classifier = ImageClassifier(clf_path)
    detector = RegionDetector(det_path)
    preprocessor = Preprocessor()
    ocr_engine = OcrEngine()
    qr_scanner = QrScanner()
    pipeline_runner = PipelineRunner(detector, preprocessor, ocr_engine, qr_scanner)
    sheet_checker = SheetChecker()
    yield
    logger.info("Shutdown")


app = FastAPI(title="OCR Lot Checker", lifespan=lifespan)

# CORS — wizard frontend อาจ deploy บน Netlify หรือเปิด file:// local
# internal tool — allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(packagings_router)


@app.get("/health")
def health():
    """Health check สำหรับ Cloud Run"""
    return {"status": "ok", "test_mode": os.getenv("TEST_MODE") == "1"}


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

        config = registry.get(image_class)
        conf_threshold = config.conf_threshold if config else _DEFAULT_CONF_THRESHOLD

        if config is None and registry.is_archived(image_class):
            logger.warning("Archived class predicted: %s (conf=%.4f)", image_class, class_confidence)
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
                "verify_message":   "⚠️ ประเภทรูปนี้ถูกปิดใช้งานชั่วคราว กรุณาตรวจสอบด้วยตนเอง",
                "bbox":             None,
                "status":           "archived_class",
            })

        if class_confidence < conf_threshold:
            logger.warning(
                "Low confidence %.4f for class=%s — skipping pipeline",
                class_confidence,
                image_class,
            )
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

        if config is None:
            logger.warning("Classified class has no config: %s (conf=%.4f)", image_class, class_confidence)
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
                "verify_message":   "⚠️ ตรวจพบประเภทรูปที่ยังไม่ได้ตั้งค่าในระบบ กรุณาตรวจสอบด้วยตนเอง",
                "bbox":             None,
                "status":           "unconfigured_class",
            })

        result, bbox = pipeline_runner.run(image_bytes, config)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    sheet = sheet_checker.check(
        config,
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

    template = registry.get_template(config.message_template_key)
    lot_for_message = (
        result.get("lot_box")
        if config.detection_mode == "cross_check"
        else result.get("lot_number")
    )
    exp_for_message = (
        result.get("exp_box")
        if config.detection_mode == "cross_check"
        else result.get("exp_date")
    )
    verify_message = build_verify_message(
        config, template, sheet, lot_for_message, exp_for_message
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
        "verify_message":   verify_message,
        "bbox":             bbox,
        "status":           result["status"],
    }
    return JSONResponse(response)
