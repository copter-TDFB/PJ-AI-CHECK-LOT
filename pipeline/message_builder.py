from datetime import datetime

from pipeline.packaging_registry import MessageTemplate, PackagingConfig


def _format_exp_display(exp: str | None) -> str | None:
    """แปลง exp (ISO yyyy-mm-dd ที่ OCR อ่านได้) เป็น dd/mm/yyyy สำหรับแสดงในข้อความ

    คืน None ถ้าไม่มีค่า หรือ parse ไม่ได้ — ผู้เรียกจะข้าม EXP ไป (เหมือน lot)
    """
    if not exp:
        return None
    try:
        return datetime.strptime(exp.strip(), "%Y-%m-%d").strftime("%d/%m/%Y")
    except (ValueError, AttributeError):
        return None


def build_verify_message(
    config: PackagingConfig,
    template: MessageTemplate,
    sheet: dict,
    lot: str | None,
    exp: str | None = None,
) -> str:
    """
    สร้าง verify_message สำหรับส่ง Slack ตาม packaging config

    Logic:
    - gate_on_lot=True: ถ้า lot fail → return fail(lot) ทันที, ไม่เช็ค field อื่น
    - จากนั้น (หรือถ้า gate_on_lot=False): collect failures ของทุก field ที่เหลือ

    `exp` = วันหมดอายุที่ OCR อ่านได้ (ISO) — value-driven: มีค่าก็ต่อท้าย EXP,
    ไม่มี/parse ไม่ได้ก็ข้าม (mirror พฤติกรรมของ lot)
    """
    exp_display = _format_exp_display(exp)

    def _trailer() -> str:
        parts: list[str] = []
        if lot:
            parts.append(f"LOT: {lot}")
        if exp_display:
            parts.append(f"EXP: {exp_display}")
        return " | ".join(parts)
    match_map: dict[str, bool | None] = {
        "lot":     sheet.get("lot_match"),
        "exp":     sheet.get("exp_match"),
        "product": sheet.get("product_match"),
        "sachet":  sheet.get("sachet_match"),
    }
    sheet_product = sheet.get("sheet_product_name")

    def _ok() -> str:
        parts = [template.ok_message]
        if sheet_product:
            parts.append(sheet_product)
        trailer = _trailer()
        if trailer:
            parts.append(trailer)
        return " | ".join(parts)

    def _fail(labels: list[str]) -> str:
        joined = " และ".join(labels)
        base = template.fail_template.format(fields=joined)
        trailer = _trailer()
        return f"{base} | {trailer}" if trailer else base

    checks = config.sheet_checks  # e.g. ["lot", "exp", "product"]

    if config.gate_on_lot and "lot" in checks:
        if not match_map.get("lot"):
            return _fail([template.field_labels.get("lot", "เลข Lot")])
        remaining = [c for c in checks if c != "lot"]
    else:
        remaining = checks

    failures = [
        template.field_labels.get(c, c)
        for c in remaining
        if not match_map.get(c)
    ]
    if failures:
        return _fail(failures)
    return _ok()
