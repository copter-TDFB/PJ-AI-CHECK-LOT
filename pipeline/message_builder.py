from pipeline.packaging_registry import MessageTemplate, PackagingConfig


def build_verify_message(
    config: PackagingConfig,
    template: MessageTemplate,
    sheet: dict,
    lot: str | None,
) -> str:
    """
    สร้าง verify_message สำหรับส่ง Slack ตาม packaging config

    Logic:
    - gate_on_lot=True: ถ้า lot fail → return fail(lot) ทันที, ไม่เช็ค field อื่น
    - จากนั้น (หรือถ้า gate_on_lot=False): collect failures ของทุก field ที่เหลือ
    """
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
        if lot:
            parts.append(f"LOT: {lot}")
        return " | ".join(parts)

    def _fail(labels: list[str]) -> str:
        joined = " และ".join(labels)
        base = template.fail_template.format(fields=joined)
        return f"{base} | LOT: {lot}" if lot else base

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
