import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessing stub — Google Cloud Vision จัดการ image enhancement เองได้ดีกว่า
    ส่ง crop bytes ดิบตรงไป OCR โดยไม่แตะรูป
    """

    def run(self, image_bytes: bytes, image_class: str | None = None) -> bytes:
        """คืน image_bytes ดิบโดยไม่แก้ไข"""
        logger.info("Preprocessing — pass-through (class=%s)", image_class)
        return image_bytes
