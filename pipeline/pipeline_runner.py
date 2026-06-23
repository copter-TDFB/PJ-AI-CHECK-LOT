import logging

from pipeline.packaging_registry import PackagingConfig
from utils.field_groups import parse_group
from utils.image_utils import stack_images_vertically
from utils.validators import find_lot, find_expiry, find_product_name, find_size, resolve_product_template

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

        # Detection mode is the explicit router (see CONTEXT.md / spec
        # 2026-06-14-multi-field-detection). cross_check = compare same value
        # across crops; multi_field = each crop is a different field.
        if config.detection_mode == "cross_check":
            return self._run_multi_region(image_bytes, config)
        if config.detection_mode == "multi_field":
            return self._run_multi_field(image_bytes, config)
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

    def _run_multi_field(
        self, image_bytes: bytes, config: PackagingConfig
    ) -> tuple[dict, object]:
        """multi_field — each field has its own crop class {key}_{field}.
        OCR each crop separately and assign its text to that field's extractor.
        Returns the SAME dict shape as _run_single_region (no lot_box/sachet)."""
        detections = self._detector.crop_all(image_bytes, config.key)
        prefix = f"{config.key}_"
        texts: dict[str, list[str]] = {}
        raw_parts: list[str] = []
        for det in detections:
            cls = det.class_name or ""
            if not cls.startswith(prefix):
                continue
            group = cls[len(prefix):]
            processed = self._preprocessor.run(det.cropped_bytes, config.key)
            text = self._ocr_engine.run(processed, config=config)["raw_text"]
            raw_parts.append(text)
            for field in parse_group(group):
                texts.setdefault(field, []).append(text)

        def joined(field: str) -> str:
            return "\n".join(texts.get(field, [])).strip()

        lot_text = joined("lot")
        lot = (
            find_lot(lot_text, image_class=config.key, patterns=config.lot_patterns)
            if lot_text else None
        )
        size = find_size(joined("size")) if texts.get("size") else None
        product_name = (
            find_product_name(joined("product"), config.product_aliases)
            if texts.get("product") else None
        )
        if product_name and config.product_aliases:
            # alias canonical may carry a {size} token — resolve it with the OCR'd size
            product_name = resolve_product_template(product_name, size)

        result = {
            "lot_number":   lot,
            "exp_date":     find_expiry(joined("exp")) if texts.get("exp") else None,
            "mfg_date":     None,
            "product_name": product_name,
            "size":         size,
            "raw_text":     "\n".join(raw_parts),
            "confidence":   None,
            "lot_box":      None,
            "lot_sachet":   None,
            "exp_box":      None,
            "exp_sachet":   None,
            "status":       "ok" if lot else "not_found",
        }
        bbox = detections[0].bbox if detections else None
        return result, bbox
