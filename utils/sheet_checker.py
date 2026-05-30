import logging
import re
import time
from datetime import datetime

import gspread
from google.auth import default as google_auth_default

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_CACHE_TTL = 300  # refresh sheet ทุก 5 นาที


def _normalize_sheet_date(value: str) -> str | None:
    """แปลง dd/mm/yyyy หรือ ddmmyyyy จาก sheet เป็น ISO yyyy-mm-dd"""
    s = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d%m%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None



class SheetChecker:
    """เช็คผล OCR กับข้อมูลใน Google Sheet — รับ sheet_id per request"""

    def __init__(self) -> None:
        creds, _ = google_auth_default(scopes=_SCOPES)
        self._client = gspread.Client(auth=creds)
        # cache: key=(sheet_id, gid) → (rows, last_fetch)
        self._cache: dict[tuple, tuple[list[dict], float]] = {}
        logger.info("SheetChecker initialised")

    def _get_rows(self, sheet_id: str, gid: int) -> list[dict]:
        """โหลด rows จาก sheet พร้อม cache 5 นาที ต่อ sheet_id"""
        key = (sheet_id, gid)
        rows, last_fetch = self._cache.get(key, ([], 0.0))
        now = time.time()
        if now - last_fetch > _CACHE_TTL or not rows:
            ws = self._client.open_by_key(sheet_id).get_worksheet_by_id(gid)
            all_values = ws.get_all_values()

            header_idx = None
            for i, row in enumerate(all_values):
                if "Lot." in row or "Lot" in row:
                    header_idx = i
                    break

            if header_idx is None:
                rows = []
                logger.warning("Sheet: header row not found (sheet=%s gid=%s)", sheet_id, gid)
            else:
                headers = all_values[header_idx]
                rows = [
                    dict(zip(headers, row))
                    for row in all_values[header_idx + 1:]
                    if any(cell.strip() for cell in row)
                ]

            self._cache[key] = (rows, now)
            logger.info("Sheet refreshed — sheet=%s gid=%s rows=%d", sheet_id, gid, len(rows))
        return rows

    def _find_row_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> dict | None:
        """หา row แรกที่ column 'Lot.' ตรงกับ lot (case-insensitive)"""
        if not lot:
            return None
        lot_key = lot.upper().strip()
        for row in self._get_rows(sheet_id, gid):
            if str(row.get("Lot.", "")).upper().strip() == lot_key:
                return row
        return None

    def check(self, image_class: str, sheet_id: str, gid: int, **kwargs) -> dict:
        """
        เช็ค lot/exp/product กับ sheet ตาม class

        Returns:
            dict มี lot_match, exp_match, product_match, sachet_match
            ค่า None = ไม่เกี่ยวกับ class นั้น, False = ไม่ตรง, True = ตรง
        """
        result = {
            "lot_match":     None,
            "exp_match":     None,
            "product_match": None,
            "sachet_match":  None,
        }

        try:
            if image_class == "import_sticker":
                lot = kwargs.get("lot_number")
                row = self._find_row_by_lot(lot, sheet_id, gid)
                if row is None and lot and len(lot) >= 13:
                    # fallback: ตัด 4 หลัก [9:13] จาก barcode ยาว เช่น HR0008001014613926R → 0146
                    short_lot = lot[9:13]
                    logger.info("import_sticker lot not found — retrying with short lot %s", short_lot)
                    row = self._find_row_by_lot(short_lot, sheet_id, gid)
                result["lot_match"] = row is not None

            elif image_class in ("back_label", "grade_bag"):
                lot     = kwargs.get("lot_number")
                exp     = kwargs.get("exp_date")
                product = kwargs.get("product_name")
                row = self._find_row_by_lot(lot, sheet_id, gid)
                result["lot_match"] = row is not None
                if row:
                    sheet_exp = _normalize_sheet_date(row.get("EXP", ""))
                    result["exp_match"] = sheet_exp == exp if exp else False
                    sheet_product = str(row.get("Product Name", "")).strip().lower()
                    result["product_match"] = bool(product and product.strip().lower() == sheet_product)
                else:
                    result["exp_match"]     = False
                    result["product_match"] = False

            elif image_class in ("retail_sachet", "capsule_box"):
                lot = kwargs.get("lot_number")
                exp = kwargs.get("exp_date")
                row = self._find_row_by_lot(lot, sheet_id, gid)
                result["lot_match"] = row is not None
                if row:
                    sheet_exp = _normalize_sheet_date(row.get("EXP", ""))
                    result["exp_match"] = sheet_exp == exp if exp else False
                else:
                    result["exp_match"] = False

            elif image_class == "container_label":
                lot_box    = kwargs.get("lot_box")
                exp_box    = kwargs.get("exp_box")
                exp_sachet = kwargs.get("exp_sachet")
                row = self._find_row_by_lot(lot_box, sheet_id, gid)
                result["lot_match"] = row is not None
                if row:
                    sheet_exp = _normalize_sheet_date(row.get("EXP", ""))
                    result["exp_match"] = sheet_exp == exp_box if exp_box else False
                else:
                    result["exp_match"] = False
                lot_sachet = kwargs.get("lot_sachet")
                if not exp_sachet and not lot_sachet:
                    # ไม่มีซองเลย — ถือว่าผ่าน (สแกนแค่กล่อง)
                    result["sachet_match"] = True
                elif exp_box and exp_sachet:
                    result["sachet_match"] = exp_box == exp_sachet
                else:
                    result["sachet_match"] = False

        except Exception:
            logger.exception("Sheet check failed — returning None for all match fields")

        return result
