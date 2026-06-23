"""Field grouping for multi_field detection mode.

A *group* is one or more field tokens that share a single YOLO crop. It is
encoded as a `sub_regions` entry — tokens joined by '_' in canonical
`FIELD_ORDER` (e.g. 'lot_exp' = lot and expiry printed in one box). A
single-token group ('lot') is the 1:1 case and behaves exactly as before.

Field tokens never contain '_', so splitting a group on '_' is unambiguous
once the '{key}_' class prefix has been stripped by the caller.
"""

FIELD_ORDER = ["lot", "exp", "product", "size"]


def parse_group(group: str) -> list[str]:
    """Split a group entry into its field tokens (drops empties)."""
    return [t for t in group.split("_") if t]


def canonicalize_group(group: str) -> str:
    """Return the group's tokens sorted by FIELD_ORDER, deduped."""
    tokens = parse_group(group)
    return "_".join(f for f in FIELD_ORDER if f in tokens)


def validate_groups(groups: list[str]) -> list[str]:
    """Validate multi_field groups and return them canonicalized.

    Rules:
    - every token must be a known field (FIELD_ORDER)
    - groups partition the fields — no field appears in two groups
    - the union of all fields must contain 'lot'
    - no empty group
    - two entries must not canonicalize to the same group

    Raises ValueError (Thai message) on any violation.
    """
    canon: list[str] = []
    seen_fields: set[str] = set()
    seen_groups: set[str] = set()
    for g in groups:
        tokens = parse_group(g)
        if not tokens:
            raise ValueError("กลุ่มว่าง — แต่ละกรอบต้องมีอย่างน้อย 1 field")
        for t in tokens:
            if t not in FIELD_ORDER:
                raise ValueError(
                    f"field ไม่รู้จัก '{t}' — ใช้ได้แค่ {', '.join(FIELD_ORDER)}")
            if t in seen_fields:
                raise ValueError(f"field '{t}' อยู่ได้กรอบเดียว — เจอซ้ำข้ามกลุ่ม")
            seen_fields.add(t)
        cg = canonicalize_group(g)
        if cg in seen_groups:
            raise ValueError(f"กลุ่มซ้ำ '{cg}' — รวมเป็นกรอบเดียว")
        seen_groups.add(cg)
        canon.append(cg)
    if "lot" not in seen_fields:
        raise ValueError("multi_field ต้องมี field 'lot' อยู่ในกรอบใดกรอบหนึ่ง")
    return canon
