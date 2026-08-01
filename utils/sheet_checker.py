import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import gspread
from google.auth import default as google_auth_default

if TYPE_CHECKING:
    from pipeline.packaging_registry import PackagingConfig

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

    def _find_rows_by_lot(self, lot: str | None, sheet_id: str, gid: int) -> list[dict]:
        """หาทุก row ที่ column 'Lot.' ตรงกับ lot (case-insensitive) — อาจเจอมากกว่า 1 แถว
        ถ้าชีทมีข้อมูลซ้ำ (เช่น แถว placeholder ที่ยังไม่กรอก MFG/EXP + แถวจริงที่กรอกภายหลัง)"""
        if not lot:
            return []
        lot_key = lot.upper().strip()
        return [
            row for row in self._get_rows(sheet_id, gid)
            if str(row.get("Lot.", "")).upper().strip() == lot_key
        ]

    def check(
        self,
        config: "PackagingConfig",
        sheet_id: str,
        gid: int,
        **kwargs,
    ) -> dict:
        """
        เช็ค lot/exp/product กับ sheet ตาม PackagingConfig

        Returns:
            dict มี lot_match, exp_match, product_match, sachet_match
            ค่า None = ไม่เกี่ยวกับ class นั้น, False = ไม่ตรง, True = ตรง
        """
        result: dict = {
            "lot_match":          None,
            "exp_match":          None,
            "product_match":      None,
            "sachet_match":       None,
            "sheet_product_name": None,
        }

        try:
            checks = config.sheet_checks
            # Container = cross_check mode (box+sachet compare same value);
            # single and multi_field both read lot_number at top level.
            is_container = config.detection_mode == "cross_check"

            lot = kwargs.get("lot_box") if is_container else kwargs.get("lot_number")
            rows = self._find_rows_by_lot(lot, sheet_id, gid)

            if not rows and config.lot_short_fallback and lot and len(lot) >= 13:
                short_lot = lot[9:13]
                logger.info(
                    "%s lot not found — retrying with short lot %s",
                    config.key,
                    short_lot,
                )
                rows = self._find_rows_by_lot(short_lot, sheet_id, gid)

            if "lot" in checks:
                result["lot_match"] = bool(rows)

            if rows:
                if len(rows) > 1:
                    logger.warning(
                        "%s: %d rows matched Lot %s — resolving by EXP",
                        config.key, len(rows), lot,
                    )

                exp_key = "exp_box" if is_container else "exp_date"
                exp = kwargs.get(exp_key)

                row = None
                if "exp" in checks and exp:
                    row = next(
                        (r for r in rows if _normalize_sheet_date(r.get("EXP", "")) == exp),
                        None,
                    )
                row = row or rows[0]

                result["sheet_product_name"] = row.get("Product Name", "").strip() or None

                if "exp" in checks:
                    sheet_exp = _normalize_sheet_date(row.get("EXP", ""))
                    result["exp_match"] = sheet_exp == exp if exp else False

                if "product" in checks:
                    product = kwargs.get("product_name")
                    sheet_product = str(row.get("Product Name", "")).strip().lower()
                    result["product_match"] = bool(
                        product and product.strip().lower() == sheet_product
                    )
            else:
                if "exp" in checks:
                    result["exp_match"] = False
                if "product" in checks:
                    result["product_match"] = False

            if "sachet" in checks:
                exp_box    = kwargs.get("exp_box")
                exp_sachet = kwargs.get("exp_sachet")
                lot_sachet = kwargs.get("lot_sachet")
                if not exp_sachet and not lot_sachet:
                    # ไม่มีซองเลย — ถือว่าผ่าน
                    result["sachet_match"] = True
                elif exp_box and exp_sachet:
                    result["sachet_match"] = exp_box == exp_sachet
                elif not exp_sachet:
                    # เจอซอง (lot_sachet) แต่ OCR อ่าน exp_sachet ไม่ได้ — ผ่าน
                    result["sachet_match"] = True
                else:
                    result["sachet_match"] = False

        except Exception:
            logger.exception("Sheet check failed — returning None for all match fields")

        return result
