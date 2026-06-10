import logging

from pipeline.packaging_registry import PackagingConfig
from utils.image_utils import stack_images_vertically

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Dispatches image through the correct pipeline based on PackagingConfig."""

    def __init__(self, detector, preprocessor, ocr_engine, qr_scanner) -> None:
        self._detector = detector
        self._preprocessor = preprocessor
        self._ocr_engine = ocr_engine
        self._qr_scanner = qr_scanner

    def run(self, image_bytes: bytes, config: PackagingConfig) -> tuple[dict, object]:
        """
        Run the pipeline for the given packaging config.

        Returns:
            (result_dict, bbox)  — bbox is None for QR-only pipelines
        """
        if config.pipeline == "qr_scanner":
            result = self._qr_scanner.scan(image_bytes)
            result.setdefault("product_name", None)
            result.setdefault("size", None)
            return result, None

        if config.sub_regions:
            return self._run_multi_region(image_bytes, config)

        return self._run_single_region(image_bytes, config)

    def _run_single_region(
        self, image_bytes: bytes, config: PackagingConfig
    ) -> tuple[dict, object]:
        detections = self._detector.crop_all(image_bytes, config.key)
        processed = [
            self._preprocessor.run(det.cropped_bytes, config.key)
            for det in detections
        ]
        combined = stack_images_vertically(processed)
        result = self._ocr_engine.run(combined, config=config)
        bbox = detections[0].bbox if detections else None
        return result, bbox

    def _run_multi_region(
        self, image_bytes: bytes, config: PackagingConfig
    ) -> tuple[dict, object]:
        """Multi-crop OCR ใช้กับ container_label (กล่อง + ซอง แยก lot/date)"""
        detections = self._detector.crop_all(image_bytes, config.key)
        crops: list[dict] = []
        for det in detections:
            processed = self._preprocessor.run(det.cropped_bytes, config.key)
            ocr_res = self._ocr_engine.run(processed, config=config)
            text_ok = len(ocr_res["raw_text"].strip()) >= 8
            data_ok = bool(ocr_res["lot_number"] or ocr_res["exp_date"])
            if not text_ok and not data_ok:
                logger.warning(
                    "Preprocessed crop appears degraded — retrying with original bytes"
                )
                ocr_res = self._ocr_engine.run(det.cropped_bytes, config=config)
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
            "status":       "ok" if any(c.get("lot_number") for c in crops) else "not_found",
        }
        bbox = detections[0].bbox if detections else None
        return result, bbox
