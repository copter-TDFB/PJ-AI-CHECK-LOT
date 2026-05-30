import logging

from google.cloud import vision

from utils.validators import find_expiry, find_lot, find_mfg, find_product_name, find_size

logger = logging.getLogger(__name__)


class OcrEngine:
    """Wrapper ของ Google Cloud Vision Text Detection"""

    def __init__(self) -> None:
        self._client = vision.ImageAnnotatorClient()
        logger.info("OcrEngine initialised (Google Cloud Vision)")

    def run(self, image_bytes: bytes, image_class: str | None = None) -> dict:
        """
        รัน OCR และ extract lot number, dates จาก image_bytes

        Args:
            image_bytes: raw image bytes
            image_class: class ของรูปจาก classifier (Phase 2) ใช้ช่วย filter ในอนาคต

        Returns:
            dict ที่มี lot_number, confidence, raw_text, mfg_date, exp_date, bbox, status
        """
        logger.info("OCR start — class=%s size=%d bytes", image_class, len(image_bytes))

        vision_image = vision.Image(content=image_bytes)
        response = self._client.text_detection(image=vision_image)

        if response.error.message:
            logger.error("Vision API error: %s", response.error.message)
            raise RuntimeError(f"Vision API error: {response.error.message}")

        annotations = response.text_annotations
        if not annotations:
            logger.warning("No text detected in image")
            return self._empty_result("not_found")

        full_text = annotations[0].description
        logger.info("Text detected (%d chars)", len(full_text))

        lot = find_lot(full_text, image_class=image_class)
        exp = find_expiry(full_text)
        mfg = find_mfg(full_text)
        size = find_size(full_text) if image_class in ("back_label", "grade_bag") else None
        product_name = find_product_name(full_text) if image_class in ("back_label", "grade_bag") else None
        if product_name and size:
            product_name = f"{product_name} {size}"

        # Vision API TEXT_DETECTION มักไม่มี page confidence — ใช้ 1.0 เป็น default
        confidence: float | None = None
        try:
            pages = response.full_text_annotation.pages
            if pages and pages[0].confidence > 0:
                confidence = round(float(pages[0].confidence), 4)
        except Exception:
            pass

        status = "ok" if lot else "not_found"
        logger.info("OCR done — lot=%s mfg=%s exp=%s status=%s", lot, mfg, exp, status)

        return {
            "lot_number": lot,
            "confidence": confidence,
            "raw_text": full_text,
            "mfg_date": mfg,
            "exp_date": exp,
            "product_name": product_name,
            "size": size,
            "bbox": None,
            "status": status,
        }

    @staticmethod
    def _empty_result(status: str) -> dict:
        return {
            "lot_number": None,
            "confidence": None,
            "raw_text": "",
            "mfg_date": None,
            "exp_date": None,
            "product_name": None,
            "size": None,
            "bbox": None,
            "status": status,
        }
