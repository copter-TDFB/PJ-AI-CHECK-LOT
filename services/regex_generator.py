"""Auto-generate lot regex จากตัวอย่าง — ใช้ใน wizard step 4."""

import re

_IMG_EXT_NONE = ()  # placeholder, not used here


def generate_regex(examples: list[str]) -> str:
    """สร้าง regex ที่ match ทุก example.

    Strategy:
      1. หา alpha-prefix length ที่ common (ขั้นต่ำของทุก example)
      2. หา digit length หลัง alpha prefix
      3. Build pattern: (?i)[A-Z]{n}\\d{m,}[A-Z0-9]*
    """
    cleaned = [e.strip() for e in examples if e and e.strip()]
    if not cleaned:
        return r"^.+$"

    alpha_min = min(_count_leading_alpha(s) for s in cleaned)
    digit_min = min(_count_leading_digit(s[alpha_min:]) for s in cleaned)

    alpha_part = f"[A-Z]{{{alpha_min}}}" if alpha_min > 0 else ""
    digit_part = f"\\d{{{digit_min},}}" if digit_min > 0 else r"\d+"

    return f"(?i){alpha_part}{digit_part}[A-Z0-9]*"


def preview_matches(pattern: str, examples: list[str]) -> list[dict]:
    """Test pattern กับ examples — คืน [{input, match, ok}]."""
    try:
        rx = re.compile(pattern)
    except re.error:
        return [{"input": e, "match": None, "ok": False} for e in examples]

    out = []
    for e in examples:
        m = rx.search(e)
        out.append({
            "input": e,
            "match": m.group(0) if m else None,
            "ok": bool(m),
        })
    return out


def _count_leading_alpha(s: str) -> int:
    n = 0
    for c in s:
        if c.isalpha():
            n += 1
        else:
            break
    return n


def _count_leading_digit(s: str) -> int:
    n = 0
    for c in s:
        if c.isdigit():
            n += 1
        else:
            break
    return n
