import re
from datetime import datetime

# ─── Size patterns ────────────────────────────────────────────────────────────

_SIZE_PATTERN = re.compile(
    r'(?:น้ำหนัก(?:สุทธิ)?|NET\s*WT|NET\s*WEIGHT|ปริมาณ)\s*[:\s]?\s*'
    r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|กรัม|กก\.?|มล\.?|ลิตร)',
    re.IGNORECASE,
)
_SIZE_INLINE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(g|kg|ml|l|กรัม|กก\.?|มล\.?|ลิตร)\b',
    re.IGNORECASE,
)

# keyword ชื่อสินค้าจริงที่ใช้ในระบบ
# เช็ค "X Rich" เป็น phrase ก่อน เพื่อกัน false positive จาก flavor description เช่น "Rich and creamy"
_KW_EXCELLENT_RICH = re.compile(r'\bexcellent\s+rich\b', re.IGNORECASE)
_KW_CLASSIC_RICH   = re.compile(r'\bclassic\s+rich\b',   re.IGNORECASE)
_KW_MEDIUM_RICH    = re.compile(r'\bmedium\s+rich\b',    re.IGNORECASE)
_KW_HOUJICHA_RICH  = re.compile(r'\bhoujicha\s+rich\b',  re.IGNORECASE)
_KW_HOUJICHA       = re.compile(r'\bhoujicha\b',         re.IGNORECASE)
_KW_GENMAICHA      = re.compile(r'\bgenmaicha\b',        re.IGNORECASE)
_KW_EXCELLENT      = re.compile(r'\bexcellent\b',        re.IGNORECASE)
_KW_MEDIUM         = re.compile(r'\bmedium\b',           re.IGNORECASE)
_KW_CLASSIC        = re.compile(r'\bclassic\b',          re.IGNORECASE)


# ─── Generic lot patterns (fallback ทุก class) ────────────────────────────────

_LOT_GENERIC: list[re.Pattern] = [
    re.compile(r'LOT\s*[:\.\-]?\s*([A-Z0-9][A-Z0-9\-_]{3,})', re.IGNORECASE),
    re.compile(r'LOT\s*[:\.\-]?\s*\n\s*([A-Z0-9][A-Z0-9\-_]{3,})', re.IGNORECASE),
    re.compile(
        r'(?:LOT\s*NO|BATCH(?:\s*NO)?|B\s*/\s*N)\s*[:\.\-]?\s*([A-Z0-9][A-Z0-9\-_]{3,})',
        re.IGNORECASE,
    ),
]

# ─── Class-specific lot patterns (ลองก่อน generic) ───────────────────────────
# back_label / container_label / grade_bag ใช้ lot ยาว alpha-prefix รูปแบบเดียวกัน
# เช่น ER0010000019616726P (suffix P/P1/P2 ได้) — ไม่ใช่เลขสั้น
# ฟอร์แมตใหม่แทรกตัวอักษรพิมพ์ใหญ่ 3 ตัวกลางเลข (เช่น ER0010TDF019616726P) จึงใช้
# alternation (?:\d{6,}|\d{4,}[A-Z]{3}\d{2,}) เพื่อรับทั้งเก่าและใหม่

_LOT_BY_CLASS: dict[str, list[re.Pattern]] = {
    # back_label: lot ยาว alpha-prefix (LOT: ER...P) หรือรูป "L XXXXX"
    "back_label": [
        re.compile(r'LOT\s*[:\.\-]?\s*([A-Z]{0,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)', re.IGNORECASE),
        re.compile(r'\bL[:\s]?\s*([A-Z0-9]{5,})', re.IGNORECASE),
    ],
    # container_label: lot ในกล่องขาว
    # - lot สั้น: มี keyword "LOT" นำหน้า เช่น "LOT 6455"
    # - lot ยาว: ตัวอักษรนำ + ตัวเลขยาว บนบรรทัดของตัวเอง เช่น "HR0012000645508526P"
    "container_label": [
        re.compile(r'LOT\s*[:\.\-]?\s*([A-Z0-9\-]{4,})', re.IGNORECASE),
        re.compile(r'^([A-Z]{1,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)$', re.IGNORECASE | re.MULTILINE),
    ],
    # grade_bag: lot ยาว alpha-prefix (เหมือน back_label) — มี LOT นำหน้า หรือยืนเดี่ยว
    "grade_bag": [
        re.compile(r'LOT\s*[:\.\-]?\s*([A-Z]{0,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)', re.IGNORECASE),
        re.compile(r'\b([A-Z]{1,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)\b'),
    ],
    # retail_sachet: lot ด้านหลังซอง — ตัวเลขล้วน 4 ตัว เช่น "0109"
    "retail_sachet": [
        re.compile(r'LOT\s*[:\.\-]?\s*(\d{4})\b', re.IGNORECASE),
    ],
    # capsule_box: lot บนกล่อง capsule — ตัวเลขล้วน 4 ตัว เช่น "0136"
    "capsule_box": [
        re.compile(r'LOT\s*[:\.\-]?\s*(\d{4})\b', re.IGNORECASE),
    ],
}

# ─── Date keyword groups ───────────────────────────────────────────────────────

_EXP_KW = (
    r'(?:BBD|BB|BBE|EXP(?:IRY)?(?:\s*DATE)?|BEST\s*BEFORE'
    r'|USE\s*BY|EXPIRES?|ควรบริโภคก่อน|วันหมดอายุ|หมดอายุ)'
)
_MFG_KW = (
    r'(?:MFG(?:\s*DATE)?|MANUFACTURED(?:\s*ON)?|PRODUCTION(?:\s*DATE)?'
    r'|วันผลิต|ผลิตเมื่อ|ผลิต)'
)

# คั่นวันที่ด้วย / หรือ - (เช่น 02/06/2027 หรือ 30-06-2027)
_DATE_SEP     = r'[/-]'
_DATE_SLASH   = rf'(\d{{1,2}}{_DATE_SEP}\d{{1,2}}{_DATE_SEP}\d{{2,4}})'  # dd/mm/yy(yy) หรือ dd-mm-yyyy
_DATE_COMPACT = r'(\d{8})'                        # ddmmyyyy


def _build_date_patterns(keyword: str) -> list[re.Pattern]:
    """สร้าง regex patterns สำหรับหาวันที่ที่นำด้วย keyword"""
    return [
        re.compile(rf'{keyword}\s*[:\.\-]?\s*{_DATE_SLASH}', re.IGNORECASE),
        re.compile(rf'{keyword}\s*[:\.\-]?\s*{_DATE_COMPACT}', re.IGNORECASE),
        re.compile(
            rf'{keyword}\s*[:\.\-]?\s*\n\s*(?:{_DATE_SLASH[1:-1]}|{_DATE_COMPACT[1:-1]})',
            re.IGNORECASE,
        ),
    ]


EXP_PATTERNS: list[re.Pattern] = _build_date_patterns(_EXP_KW)
MFG_PATTERNS: list[re.Pattern] = _build_date_patterns(_MFG_KW)

# fallback: หา dd/mm/yy(yy) หรือ ddmmyyyy ทั่วไปเมื่อไม่มี keyword นำหน้า
# guard ด้วย (?<![\d-]) / (?![\d-]) กันไป match เศษของเลขยาวที่คั่นด้วย -
# เช่น เลข อย. 11-2-02167-6-0024 หรือเบอร์โทร 02-114-3715
DATE_FALLBACK         = re.compile(rf'(?<![\d-]){_DATE_SLASH}(?![\d-])')
DATE_COMPACT_FALLBACK = re.compile(r'\b(\d{8})\b')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_date(raw: str) -> str | None:
    """แปลงวันที่เป็น ISO format YYYY-MM-DD รับ dd/mm/yy, dd/mm/yyyy หรือ ddmmyyyy"""
    if not raw:
        return None
    s = re.sub(r'\s+', '', raw)
    s = s.replace('-', '/')   # รับ separator แบบ dash (30-06-2027) ให้เหมือน slash

    if re.fullmatch(r'\d{8}', s):
        d, m, y = s[0:2], s[2:4], s[4:8]
    elif re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', s):
        parts = s.split('/')
        d, m, y = parts[0], parts[1], parts[2]
    elif re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2}', s):
        # 2-digit year: 00-49 → 2000s, 50-99 → 1900s
        parts = s.split('/')
        d, m, yy = parts[0], parts[1], parts[2]
        y = str(2000 + int(yy)) if int(yy) < 50 else str(1900 + int(yy))
    else:
        return None

    try:
        # ถ้าปีเกิน 2100 — OCR อ่าน 0 เป็น 9 เช่น 2027→2927
        if int(y) > 2100:
            y = y.replace('9', '0', 1)
        return datetime(int(y), int(m), int(d)).strftime('%Y-%m-%d')
    except ValueError:
        return None


def _search_date(text: str, patterns: list[re.Pattern]) -> str | None:
    """หาวันที่จาก text ด้วย patterns ที่ให้มา"""
    for pat in patterns:
        match = pat.search(text)
        if match:
            return normalize_date(match.group(1))
    return None


def _is_valid_lot(lot: str) -> bool:
    """กรอง false positive — ห้ามเป็นวันที่หรือสั้นเกินไป"""
    if len(lot) < 4:
        return False
    if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', lot):
        return False
    return True


# ─── Lot prefix correction (back_label / grade_bag) ──────────────────────────
# 2 ตัวแรกของ lot ใน back_label และ grade_bag เป็น ENG เสมอ ไม่มีทางเป็นตัวเลข
# OCR มักอ่าน O→0, I→1, S→5, B→8 ในตำแหน่ง prefix

_DIGIT_TO_LETTER_TRANS = str.maketrans('0158', 'OISB')
_ALPHA_PREFIX_CLASSES = frozenset({'back_label', 'grade_bag'})


def _fix_lot_alpha_prefix(lot: str, image_class: str | None) -> str:
    """แก้ digit-lookalikes ใน 2 ตัวแรกของ lot สำหรับ class ที่ prefix เป็น alpha เสมอ"""
    if image_class not in _ALPHA_PREFIX_CLASSES or len(lot) < 2:
        return lot
    fixed = lot[:2].upper().translate(_DIGIT_TO_LETTER_TRANS)
    return fixed + lot[2:]


# ─── OCR correction ───────────────────────────────────────────────────────────

_OCR_FIXES: list[tuple[re.Pattern, str]] = [
    # L?? → LOT: O อ่านผิดเป็น 0/C/Q/G/D, T อ่านผิดเป็น 7/I/1/L/J
    (re.compile(r'\bL[O0CQGD][TI17LJ]\b', re.IGNORECASE), 'LOT'),
    # EXCELLENT/CLASSIC/MEDIUM/HOUJICHA RI?* → RICH — the wildcard covers the 3rd
    # letter too (not just trailing garble), since Vision has misread C as G here
    # (e.g. "MEDIUM RIGH") — safe this broad because only "RICH" ever follows these
    # 4 flavor words in practice.
    (re.compile(r'(\b(?:EXCELLENT|CLASSIC|MEDIUM|HOUJICHA)\s+)RI.\w*\b', re.IGNORECASE), r'\1RICH'),
    (re.compile(r'\bBBl\b',        re.IGNORECASE), 'BBD'),   # BBl → BBD
    (re.compile(r'\bEXl\b',        re.IGNORECASE), 'EXP'),   # EXl → EXP
    # retail_sachet: แก้ตัวอักษรที่ OCR อ่านแทน 0 ใน "LOT XNNN" (X=C,O,D → 0)
    (re.compile(r'(LOT\s*[:\.\-]?\s*)[COD](\d{3})\b', re.IGNORECASE), r'\g<1>0\2'),
    # compact date (ddmmyyyy): ตัวแรกอ่านผิดเป็น Q (0→Q) เช่น Q3062027 → 03062027
    (re.compile(r'\bQ(\d{7})\b'), r'0\1'),
]


def correct_ocr(text: str) -> str:
    """แก้ความผิดพลาด OCR ที่พบบ่อย เช่น LOL→LOT, L0T→LOT"""
    for pat, replacement in _OCR_FIXES:
        text = pat.sub(replacement, text)
    return text


# ─── Public API ───────────────────────────────────────────────────────────────

_UNIT_TO_G = re.compile(r'^(g|กรัม|กรัม\.?)$', re.IGNORECASE)
_UNIT_TO_KG = re.compile(r'^(kg|กก\.?)$', re.IGNORECASE)


def find_size(text: str) -> str | None:
    """ดึงขนาด/น้ำหนักสินค้า คืนในรูปแบบ '100 g' หรือ '1 kg'"""
    for pat in (_SIZE_PATTERN, _SIZE_INLINE):
        m = pat.search(text)
        if m:
            number = m.group(1)
            unit = m.group(2)
            if _UNIT_TO_G.match(unit):
                return f"{number} g"
            if _UNIT_TO_KG.match(unit):
                return f"{number} kg"
            return f"{number} {unit}"
    return None


def _kw_present(kw: str, lower_text: str) -> bool:
    """match keyword แบบ literal ไม่กินคำข้างเคียง (กัน "rich" ไป match ใน "enriched")
    ใช้ lookaround แทน \\b เพราะ keyword อาจลงท้ายด้วยอักขระที่ไม่ใช่ตัวอักษร เช่น "95%"."""
    return re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", lower_text) is not None


def _match_aliases(text: str, aliases: list[dict]) -> str | None:
    """ไล่ aliases ตามลำดับ (priority) — keyword แรกที่เจอใน text → คืน canonical.

    keyword แต่ละตัวเป็นได้ 2 แบบ:
      - string เดี่ยว เช่น "Classic Rich" → เจอตัวเดียวก็ match (OR)
      - list/tuple เช่น ["Excellent", "30 Sachets"] → ต้องเจอ "ครบทุกคำ" (AND)
    """
    lower = text.lower()
    for entry in aliases:
        canonical = (entry.get("canonical") or "").strip()
        for kw in entry.get("keywords") or []:
            if isinstance(kw, (list, tuple)):
                terms = [str(t).strip().lower() for t in kw if str(t).strip()]
                if terms and all(_kw_present(t, lower) for t in terms):
                    return canonical or None
                continue
            kw = str(kw).strip().lower()
            if kw and _kw_present(kw, lower):
                return canonical or None
    return None


def resolve_product_template(template: str | None, size: str | None) -> str | None:
    """ประกอบชื่อ product จาก template — แทน token {size} ด้วยขนาดที่ OCR อ่านได้.

    - ไม่มี {size}        → คืน template ตามเดิม (เช่น 'Houjicha Powder')
    - มี {size} + มี size → แทนค่าแล้วยุบช่องว่างซ้ำ (เช่น 'Medium 40 g')
    - มี {size} + ไม่มี size → คืน None (ยืนยันไม่ได้ → product ไม่ผ่าน)
    """
    if not template:
        return None
    if "{size}" not in template:
        return template
    if not size:
        return None
    resolved = template.replace("{size}", size)
    return " ".join(resolved.split())


def find_product_name(text: str, aliases: list[dict] | None = None) -> str | None:
    """
    ดึงชื่อ product จาก OCR text แล้วคืน canonical name.

    aliases — รายการ {"canonical": str, "keywords": [str]} ต่อ packaging
    (config-driven, ดูฟิลด์ product_aliases ใน YAML). ถ้าให้มา จะใช้แทน
    keyword hardcode; ไม่ให้/ว่าง → fallback hardcode เดิม (class เก่า):
      Rich (ไม่ว่าจะมี Excellent นำหน้าหรือไม่) → 'Excellent Rich 95%'
      Houjicha → 'Houjicha Powder'
      Genmaicha → 'Genmaicha Powder'
      Excellent (ไม่มี Rich) → 'Excellent'
      Medium → 'Medium'
      Classic → 'Classic'
    """
    text = correct_ocr(text)
    if aliases:
        return _match_aliases(text, aliases)
    if _KW_EXCELLENT_RICH.search(text):
        return 'Excellent Rich 95%'
    if _KW_CLASSIC_RICH.search(text):
        return 'Classic Rich 95%'
    if _KW_MEDIUM_RICH.search(text):
        return 'Medium Rich 95%'
    if _KW_HOUJICHA_RICH.search(text):
        return 'Houjicha Rich 95%'
    if _KW_HOUJICHA.search(text):
        return 'Houjicha Powder'
    if _KW_GENMAICHA.search(text):
        return 'Genmaicha Powder'
    if _KW_EXCELLENT.search(text):
        return 'Excellent'
    if _KW_MEDIUM.search(text):
        return 'Medium'
    if _KW_CLASSIC.search(text):
        return 'Classic'
    return None


def find_lot(
    text: str,
    image_class: str | None = None,
    patterns: list[re.Pattern] | None = None,
) -> str | None:
    """
    ค้นหาเลข lot จาก text ดิบ

    Args:
        text: raw OCR text
        image_class: class ของรูป — ใช้ class-specific patterns จาก _LOT_BY_CLASS (backward compat)
        patterns: lot patterns จาก PackagingConfig (ถ้าให้มา จะใช้แทน _LOT_BY_CLASS)
    """
    text = correct_ocr(text)
    # patterns จาก registry มีความสำคัญสูงสุด; fallback ไป _LOT_BY_CLASS (backward compat)
    if patterns is not None:
        class_patterns = patterns
    elif image_class and image_class in _LOT_BY_CLASS:
        class_patterns = _LOT_BY_CLASS[image_class]
    else:
        class_patterns = []

    for pat in class_patterns:
        match = pat.search(text)
        if match:
            # config patterns (wizard-generated) may have no capture group → use full match
            lot = (match.group(1) if match.lastindex else match.group(0)).strip().rstrip('.,;:')
            if _is_valid_lot(lot):
                return _fix_lot_alpha_prefix(lot, image_class)

    # fallback: generic patterns
    for pat in _LOT_GENERIC:
        match = pat.search(text)
        if match:
            lot = match.group(1).strip().rstrip('.,;:')
            if _is_valid_lot(lot):
                return _fix_lot_alpha_prefix(lot, image_class)

    return None


def find_expiry(text: str) -> str | None:
    """ค้นหาวันหมดอายุ คืน ISO format (YYYY-MM-DD) หรือ None"""
    text = correct_ocr(text)
    result = _search_date(text, EXP_PATTERNS)
    if result:
        return result
    # fallback: dd/mm/yy(yy)
    match = DATE_FALLBACK.search(text)
    if match:
        return normalize_date(match.group(1))
    # fallback: ddmmyyyy (เช่น 12032027 บนซองเดี่ยวไม่มี slash)
    match = DATE_COMPACT_FALLBACK.search(text)
    return normalize_date(match.group(1)) if match else None


def find_mfg(text: str) -> str | None:
    """ค้นหาวันผลิต คืน ISO format (YYYY-MM-DD) หรือ None"""
    return _search_date(text, MFG_PATTERNS)
