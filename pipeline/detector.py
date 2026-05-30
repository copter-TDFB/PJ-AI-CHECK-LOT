import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import bytes_to_numpy

logger = logging.getLogger(__name__)

Bbox = list[int]   # [x1, y1, x2, y2] pixel coordinates

_DETECTOR_PATH  = os.getenv("MODEL_DETECTOR_PATH", "models/detector.pt")
_CONF_THRESHOLD = float(os.getenv("DETECTOR_CONF", "0.25"))


@dataclass
class DetectionResult:
    cropped_bytes: bytes   # JPEG bytes ของ region ที่ crop แล้ว
    bbox: Bbox | None      # [x1, y1, x2, y2] relative ต่อรูปต้นฉบับ



class RegionDetector:
    """
    YOLOv8-based lot region detector
    - crop_all() คืน DetectionResult ทุกจุดที่ detect ได้ (class ละ 1-2 จุด)
    - ถ้า YOLO ไม่พบ หรือยังไม่มี models/detector.pt → heuristic fallback
    """

    def __init__(self, model_path: str | Path = _DETECTOR_PATH) -> None:
        self._model = None
        model_path = Path(model_path)
        if model_path.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(model_path))
                logger.info("YOLO detector loaded: %s", model_path)
            except Exception as exc:
                logger.warning("YOLO load failed (%s) — using heuristic fallback", exc)
        else:
            logger.info("models/detector.pt not found — using heuristic fallback")

    def crop_all(self, image_bytes: bytes, image_class: str) -> list[DetectionResult]:
        """
        Crop ทุก region ที่ detect ได้สำหรับ image_class นั้น
        เรียงจากบนลงล่าง (top y ก่อน) ตามลำดับการอ่าน

        Returns:
            list[DetectionResult] อย่างน้อย 1 รายการ (fallback ถ้าไม่เจอ)
        """
        img_np = bytes_to_numpy(image_bytes)
        logger.info("Detecting lot regions — class=%s size=%dx%d",
                    image_class, img_np.shape[1], img_np.shape[0])

        if self._model is not None:
            results = self._yolo_crop_all(img_np, image_class)
            if results:
                return results
            logger.warning("YOLO: no detection for class '%s' — heuristic fallback", image_class)

        return [self._heuristic_crop(img_np, image_class)]

    def crop(self, image_bytes: bytes, image_class: str) -> DetectionResult:
        """Backward compat — คืน detection แรก (บนสุด) เท่านั้น"""
        return self.crop_all(image_bytes, image_class)[0]

    # ─── YOLO inference ───────────────────────────────────────────────────────

    def _yolo_crop_all(self, img: np.ndarray, image_class: str) -> list[DetectionResult]:
        """
        คืน DetectionResult ทุก box ที่ตรงกับ image_class
        เรียงตาม y1 (บนลงล่าง)
        """
        results = self._model.predict(img, conf=_CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        class_names: dict[int, str] = self._model.names
        cls_ids = boxes.cls.int().tolist()

        # จับทุก YOLO class ที่ขึ้นต้นด้วย "{image_class}_"
        prefix = f"{image_class}_"
        target_cls_ids = {cid for cid, name in class_names.items() if name.startswith(prefix)}
        if target_cls_ids:
            matched_indices = [i for i, c in enumerate(cls_ids) if c in target_cls_ids]
            logger.info("YOLO: target classes for '%s' → %s", image_class, sorted(target_cls_ids))
        else:
            logger.warning("YOLO: ไม่มี class ที่ขึ้นต้นด้วย '%s' — ใช้ทุก box", prefix)
            matched_indices = list(range(len(boxes)))

        if not matched_indices:
            return []

        # เรียงจาก y1 น้อย → มาก (บนลงล่าง)
        matched_indices.sort(key=lambda i: float(boxes.xyxy[i][1]))

        h, w = img.shape[:2]
        detections: list[DetectionResult] = []
        for idx in matched_indices:
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx].tolist())
            conf = float(boxes.conf[idx])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            logger.info("YOLO box: class=%s bbox=[%d,%d,%d,%d] conf=%.3f",
                        image_class, x1, y1, x2, y2, conf)
            cropped = img[y1:y2, x1:x2]
            detections.append(DetectionResult(cropped_bytes=_to_jpeg(cropped), bbox=[x1, y1, x2, y2]))

        return detections

    # ─── Heuristic fallback ───────────────────────────────────────────────────

    _CROP_RULES: dict[str, dict] = {
        "back_label":    {"top": 0.65, "bottom": 1.0,  "left": 0.0, "right": 1.0},
        "grade_bag":     {"top": 0.70, "bottom": 1.0,  "left": 0.0, "right": 1.0},
        "retail_sachet": {"top": 0.60, "bottom": 1.0,  "left": 0.0, "right": 1.0},
    }

    def _heuristic_crop(self, img: np.ndarray, image_class: str) -> DetectionResult:
        if image_class == "container_label":
            return self._crop_container_label(img)
        rule = self._CROP_RULES.get(image_class)
        if rule is None:
            logger.warning("Unknown class '%s' — returning full image", image_class)
            return DetectionResult(cropped_bytes=_to_jpeg(img), bbox=None)
        return self._crop_by_rule(img, rule)

    @staticmethod
    def _crop_by_rule(img: np.ndarray, rule: dict) -> DetectionResult:
        h, w = img.shape[:2]
        y1 = int(h * rule["top"])
        y2 = int(h * rule["bottom"])
        x1 = int(w * rule["left"])
        x2 = int(w * rule["right"])
        return DetectionResult(cropped_bytes=_to_jpeg(img[y1:y2, x1:x2]), bbox=[x1, y1, x2, y2])

    @staticmethod
    def _crop_container_label(img: np.ndarray) -> DetectionResult:
        """หากล่องขาวที่มีเลข lot ด้วย contour detection"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: tuple | None = None
        best_score = 0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < (w * h) * 0.01 or area > (w * h) * 0.35:
                continue
            if (bw / max(bh, 1)) < 1.5:
                continue
            if area > best_score:
                best_score = area
                best = (x, y, x + bw, y + bh)

        if best:
            pad_x, pad_y = int(w * 0.02), int(h * 0.02)
            x1 = max(0, best[0] - pad_x)
            y1 = max(0, best[1] - pad_y)
            x2 = min(w, best[2] + pad_x)
            y2 = min(h, best[3] + pad_y)
            logger.info("container_label: white box found bbox=[%d,%d,%d,%d]", x1, y1, x2, y2)
            return DetectionResult(cropped_bytes=_to_jpeg(img[y1:y2, x1:x2]), bbox=[x1, y1, x2, y2])

        logger.warning("container_label: white box not found — center crop fallback")
        return RegionDetector._crop_by_rule(img, {"top": 0.15, "bottom": 0.65, "left": 0.1, "right": 0.9})


def _to_jpeg(img_rgb: np.ndarray, quality: int = 90) -> bytes:
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
