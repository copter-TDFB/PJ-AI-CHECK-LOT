import logging
import re

import cv2
import numpy as np

from utils.image_utils import bytes_to_numpy

try:
    import zxingcpp as _zx
    logger_init = logging.getLogger(__name__)
    logger_init.info("zxing-cpp available — ใช้เป็น final fallback decoder")
    _ZXING_OK = True
except ImportError:
    _ZXING_OK = False

logger = logging.getLogger(__name__)

_detector = cv2.QRCodeDetector()

# ลอง WeChatQRCode (มาใน opencv-contrib หรือ opencv บางรุ่น — handle perspective ได้ดีกว่ามาก)
_wechat: object | None = None
try:
    _wechat = cv2.wechat_qrcode_WeChatQRCode()
    logger.info("WeChatQRCode available — ใช้เป็น primary decoder")
except Exception:
    logger.info("WeChatQRCode ไม่มีใน OpenCV รุ่นนี้ — ใช้ QRCodeDetector แทน")


class QrScanner:
    """สแกน QR code จากรูปและดึงเลข lot ออกมาโดยตรง"""

    def scan(self, image_bytes: bytes) -> dict:
        """
        Decode QR code จาก image bytes

        Returns:
            dict ที่มี lot_number, raw_text, confidence, mfg_date, exp_date, status
        """
        img = bytes_to_numpy(image_bytes)
        logger.info("Scanning QR code — size=%dx%d", img.shape[1], img.shape[0])

        raw_text = _decode_all_attempts(img)
        if not raw_text:
            logger.warning("QR code not found after all attempts")
            return {
                "lot_number": None,
                "raw_text": "",
                "confidence": 0.0,
                "mfg_date": None,
                "exp_date": None,
                "status": "qr_not_found",
            }

        logger.info("QR decoded: %s", raw_text)
        lot = _extract_lot(raw_text)

        return {
            "lot_number": lot,
            "raw_text": raw_text,
            "confidence": 1.0 if lot else 0.5,
            "mfg_date": None,
            "exp_date": None,
            "status": "ok" if lot else "lot_not_found",
        }


# ─── Decode pipeline ──────────────────────────────────────────────────────────

def _try_decode_basic(img: np.ndarray) -> str:
    """ลอง decode ด้วย cv2.QRCodeDetector คืน data หรือ ''"""
    data, points, straight = _detector.detectAndDecode(img)
    if data:
        return data
    # detector หา corner ได้ แต่ decode ไม่ได้ → ลอง decode straight_qr ที่ rectify แล้ว
    if points is not None and straight is not None and straight.size > 0:
        if len(straight.shape) == 3:
            straight = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY)
        data2, _, _ = _detector.detectAndDecode(straight)
        if data2:
            return data2
        _, bin_s = cv2.threshold(straight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data3, _, _ = _detector.detectAndDecode(bin_s)
        if data3:
            return data3
    return ""


def _try_wechat(img: np.ndarray) -> str:
    """ลอง decode ด้วย WeChatQRCode (ถ้ามี) — handle perspective ได้ดีกว่า"""
    if _wechat is None:
        return ""
    try:
        texts, _ = _wechat.detectAndDecode(img)
        return texts[0] if texts else ""
    except Exception:
        return ""


def _candidates(gray: np.ndarray) -> list[np.ndarray]:
    """สร้าง preprocessing variants จาก grayscale image"""
    variants: list[np.ndarray] = [gray]

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    _, e_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(e_otsu)

    sharpened = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    variants.append(sharpened)

    return variants


def _rotate_padded(img: np.ndarray, angle: float) -> np.ndarray:
    """หมุนภาพโดยขยาย canvas เพื่อไม่ให้ขอบโดน crop"""
    h, w = img.shape[:2]
    diag = int(np.ceil(np.sqrt(h * h + w * w)))
    pad_h, pad_w = (diag - h) // 2, (diag - w) // 2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=255)
    ph, pw = padded.shape[:2]
    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)
    return cv2.warpAffine(padded, M, (pw, ph),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=255)


_SWEEP_MAX_DIM = 900   # ย่อก่อน rotation sweep — 900px รักษา QR module resolution ได้ดีกว่า 640
_SCAN_MAX_DIM = 2048   # cap ก่อนเริ่ม scan — QR ไม่ต้องการ resolution เกินนี้, ป้องกันค้างบนรูปใหญ่


def _shrink_for_sweep(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    longest = max(h, w)
    if longest <= _SWEEP_MAX_DIM:
        return gray
    scale = _SWEEP_MAX_DIM / longest
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _try_perspective_warp(gray: np.ndarray) -> str:
    """
    ใช้ QRCodeDetector.detect() หา 4 corners แล้ว perspective warp → decode
    แก้ปัญหา QR ที่ถ่ายเฉียง (phone-style) โดยไม่ต้อง rotation sweep
    ลองบน original + resized versions เผื่อ detect() ทำงานได้ดีกว่าที่ขนาดต่างกัน
    """
    h, w = gray.shape
    test_imgs = [gray]
    # ลอง scale variants — ข้าม scale up ถ้ารูปใหญ่อยู่แล้ว (> 1000px) เพื่อป้องกันช้า
    _large = max(h, w) > 1000
    for scale in (0.5, 2.0, 0.75):
        if scale > 1.0 and _large:
            continue
        test_imgs.append(cv2.resize(gray, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC))

    for base in test_imgs:
        for cand in _candidates(base):
            ret, points = _detector.detect(cand)
            if not ret or points is None:
                continue
            try:
                pts = points.reshape(4, 2).astype(np.float32)
                side = 400
                dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
                               dtype=np.float32)
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(cand, M, (side, side))
                for w_cand in _candidates(warped):
                    if d := _try_decode_basic(w_cand):
                        logger.info("QR decoded via perspective warp")
                        return d
            except Exception:
                continue
    return ""


def _try_zxing(img: np.ndarray) -> str:
    """ลอง decode ด้วย zxing-cpp — handle perspective สูงมาก (ZXing algorithm)"""
    if not _ZXING_OK:
        return ""
    try:
        # zxingcpp ต้องการ uint8 grayscale หรือ BGR/RGB
        results = _zx.read_barcodes(img)
        for r in results:
            if r.valid and r.text:
                return r.text
        # ลอง variants ถ้า decode ไม่ได้ตรงๆ
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for cand in _candidates(gray):
            results = _zx.read_barcodes(cand)
            for r in results:
                if r.valid and r.text:
                    return r.text
    except Exception:
        pass
    return ""


def _find_sticker_crop(img: np.ndarray) -> np.ndarray | None:
    """
    หา QR code region ใน frame
    1. ใช้ QRCodeDetector.detect() หา corners → crop + padding
    2. fallback: หา QR finder patterns (3 nested square contours) ด้วย contour hierarchy
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    src = img if img.ndim == 3 else gray

    # Phase 1: QRCodeDetector.detect() หา 4 corners ของ QR
    for scale in (0.5, 0.25, 1.0):
        sw, sh = int(w * scale), int(h * scale)
        test = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        ret, points = _detector.detect(test)
        if ret and points is not None:
            pts = (points.reshape(4, 2) / scale).astype(int)
            pad = int(max(w, h) * 0.05)
            x1 = max(0, int(pts[:, 0].min()) - pad)
            y1 = max(0, int(pts[:, 1].min()) - pad)
            x2 = min(w, int(pts[:, 0].max()) + pad)
            y2 = min(h, int(pts[:, 1].max()) + pad)
            logger.info("QR detector crop: [%d,%d,%d,%d] scale=%.2f", x1, y1, x2, y2, scale)
            return src[y1:y2, x1:x2]

    # Phase 2: หา QR finder patterns ด้วย contour hierarchy (nested dark squares)
    # ย่อรูปก่อนเพื่อความเร็ว
    max_dim = 1200
    scale2 = min(1.0, max_dim / max(h, w))
    small = cv2.resize(gray, (int(w * scale2), int(h * scale2)), interpolation=cv2.INTER_AREA)
    sh2, sw2 = small.shape

    adaptive = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return None

    # หา contour ที่เป็นสี่เหลี่ยมมีลูก (finder pattern = dark square มี white square ข้างใน)
    finder_boxes: list[tuple[int, int, int, int]] = []
    for i, cnt in enumerate(contours):
        # ตรวจว่ามีลูก (nested) และมีพ่อ
        child = hierarchy[0][i][2]
        parent = hierarchy[0][i][3]
        if child < 0 or parent < 0:
            continue
        area = cv2.contourArea(cnt)
        if area < 100 or area > (sw2 * sh2) * 0.1:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # finder pattern เกือบจะเป็นสี่เหลี่ยมจัตุรัส
        if not (0.6 < bw / max(bh, 1) < 1.6):
            continue
        finder_boxes.append((x, y, x + bw, y + bh))

    if len(finder_boxes) < 3:
        return None

    # ใช้ bounding box รวมของ finder_boxes ทั้งหมด × เพิ่ม padding 50%
    all_x1 = min(b[0] for b in finder_boxes)
    all_y1 = min(b[1] for b in finder_boxes)
    all_x2 = max(b[2] for b in finder_boxes)
    all_y2 = max(b[3] for b in finder_boxes)
    qr_w = all_x2 - all_x1
    qr_h = all_y2 - all_y1
    pad2 = max(qr_w, qr_h) // 2
    x1 = max(0, int((all_x1 - pad2) / scale2))
    y1 = max(0, int((all_y1 - pad2) / scale2))
    x2 = min(w, int((all_x2 + pad2) / scale2))
    y2 = min(h, int((all_y2 + pad2) / scale2))
    logger.info("QR finder pattern crop: [%d,%d,%d,%d] finders=%d", x1, y1, x2, y2, len(finder_boxes))
    return src[y1:y2, x1:x2]


def _decode_all_attempts(img: np.ndarray) -> str:
    """
    ลอง decode QR แบบ exhaustive (เร็วก่อน → ช้าหลัง):
      Phase 0 — WeChatQRCode (ถ้ามี) จัดการ perspective ได้เอง
      Phase 1 — preprocessing variants ไม่หมุน + scale variations
      Phase 2 — perspective warp จาก detected corners (แก้ QR เอียงเฉียง)
      Phase 3 — rotation sweep 15° บนรูปย่อ 640px (fallback สุดท้าย)
    """
    # ย่อรูปก่อนถ้าใหญ่เกิน _SCAN_MAX_DIM — QR ไม่ต้อง full resolution ป้องกัน scale-up phase ค้าง
    h0, w0 = img.shape[:2]
    if max(h0, w0) > _SCAN_MAX_DIM:
        scale_pre = _SCAN_MAX_DIM / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale_pre), int(h0 * scale_pre)), interpolation=cv2.INTER_AREA)
        logger.debug("Pre-resize %dx%d → %dx%d before scan", w0, h0, img.shape[1], img.shape[0])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Phase 0: zxing-cpp บน full image ก่อนเสมอ — แม่นที่สุด ไม่มี false positive
    if d := _try_zxing(gray):
        logger.info("QR decoded via zxing-cpp (full image)")
        return d

    # Phase 0b: WeChatQRCode full image (handle perspective ได้ดีถ้ามี)
    if d := _try_wechat(img):
        logger.info("QR decoded via WeChatQRCode")
        return d

    # Phase 1: ลอง detect sticker region แล้ว decode จาก crop
    # หมายเหตุ: ทำหลัง zxing full เพราะ crop ที่ใหญ่เกินทำให้ cv2 false positive
    sticker = _find_sticker_crop(img)
    if sticker is not None:
        if d := _try_wechat(sticker):
            logger.info("QR decoded from sticker crop via WeChatQRCode")
            return d
        s_gray = sticker if sticker.ndim == 2 else cv2.cvtColor(sticker, cv2.COLOR_RGB2GRAY)
        if d := _try_zxing(s_gray):
            logger.info("QR decoded from sticker crop via zxing-cpp")
            return d
        for cand in _candidates(s_gray):
            if d := _try_decode_basic(cand):
                logger.info("QR decoded from sticker crop")
                return d

    # Phase 2: preprocessing variants ไม่หมุน บน full image (fallback)
    for cand in _candidates(gray):
        if d := _try_decode_basic(cand):
            logger.info("QR decoded (no rotation)")
            return d

    # Phase 3: ย่อรูปก่อน แล้วลอง zxing + basic อีกรอบ (รูปใหญ่เกินไปบางครั้ง detector miss)
    h, w = gray.shape
    longest = max(h, w)
    if longest > 1200:
        scale_down = 1200 / longest
        small_gray = cv2.resize(gray, (int(w * scale_down), int(h * scale_down)),
                                interpolation=cv2.INTER_AREA)
        if d := _try_zxing(small_gray):
            logger.info("QR decoded via zxing-cpp (resized)")
            return d
        for cand in _candidates(small_gray):
            if d := _try_decode_basic(cand):
                logger.info("QR decoded (resized)")
                return d

    # Phase 4: perspective warp จาก corners
    if d := _try_perspective_warp(gray):
        return d

    # Phase 5: scale up (QR เล็กเกินไปใน frame)
    for scale in (2.0, 1.5, 3.0):
        resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        for cand in _candidates(resized):
            if d := _try_decode_basic(cand):
                logger.info("QR decoded at %.1f× scale", scale)
                return d
        if d := _try_zxing(resized):
            logger.info("QR decoded via zxing-cpp at %.1f× scale", scale)
            return d

    # Phase 4: rotation sweep บนรูปย่อ 900px (last resort — ช้า)
    small = _shrink_for_sweep(gray)
    _, otsu_s = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    _, e_otsu_s = cv2.threshold(clahe.apply(small), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for angle in range(10, 360, 10):
        for base in (small, otsu_s, e_otsu_s):
            rotated = _rotate_padded(base, angle)
            if d := _try_decode_basic(rotated):
                logger.info("QR decoded at %d° rotation (sweep)", angle)
                return d

    # Phase 4b: fine sweep ทุก 5° รอบมุมเฉียง (QR เอียงไม่ตรง 10° interval พอดี)
    for band_center in (45, 135, 225, 315):
        for offset in range(-4, 5):
            angle = band_center + offset
            if angle % 10 == 0:
                continue
            rotated = _rotate_padded(otsu_s, angle)
            if d := _try_decode_basic(rotated):
                logger.info("QR decoded at %d° (fine sweep)", angle)
                return d

    return ""


# ─── Lot extraction ───────────────────────────────────────────────────────────

def _extract_lot(text: str) -> str | None:
    """
    ดึงเลข lot จาก QR text
    QR มักเป็น lot โดยตรง แต่บางครั้งมี prefix เช่น 'LOT:TH20250601'
    """
    for pat in (r'LOT[:\s]+([A-Z0-9\-]+)', r'Lot[:\s]+([A-Z0-9\-]+)', r'lot[:\s]+([A-Z0-9\-]+)'):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return text.strip() or None
