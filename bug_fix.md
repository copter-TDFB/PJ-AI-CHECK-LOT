สรุป Bug ที่แก้ไป


  ---
  ปัญหาที่เจอ

  ส่งรูป import_sticker ไปที่ production API แล้วได้ lot_number: (01)41580980574830 — ซึ่งผิด ค่าที่ถูกต้องคือ MR0010001012614526R

  ---
  Root Cause (3 ชั้น)

  ชั้นที่ 1 — _find_sticker_crop crop ผิด

  ฟังก์ชันนี้ใช้ contour hierarchy หา QR finder patterns แล้ว crop บริเวณ QR ออกมา แต่บนรูปนี้ (พื้นหลังสแตนเลสมี reflection + รอยขีด) contour detection จับ
  noise ได้เยอะมาก → bounding box รวมทุก contour ออกมาใหญ่เกือบเต็มรูป (1376×2412px จากรูป 4096×3072px)

  ชั้นที่ 2 — zxing ล้มเหลวบน crop ที่ผิด

  crop ที่ได้มามันใหญ่และ context เยอะเกินไป → zxing-cpp decode ไม่ได้ → return ""

  ชั้นที่ 3 — cv2 false positive บน variant ของ crop ผิด

  หลัง zxing ล้มเหลว โค้ดลอง _candidates(s_gray) ซึ่งสร้าง preprocessing variants (Otsu threshold, CLAHE, sharpen ฯลฯ) แล้วส่งให้ cv2.QRCodeDetector
  decode ทีละตัว — บน variant ใดสักตัวของ crop ขนาดใหญ่นั้น cv2 "เห็น" pattern ที่คล้าย QR และ decode ออกมาเป็น (01)41580980574830 ซึ่งเป็นค่าขยะ แต่ cv2 ไม่มี
  confidence score → return ค่านั้นออกไปเลย

  ผลลัพธ์: Phase 0a (sticker crop) return false positive ก่อน → zxing บน full image ซึ่งจะได้ค่าถูกต้อง MR0010001012614526R ไม่ถูก execute เลย

  ---
  วิธีแก้

  สลับลำดับ priority ใน _decode_all_attempts:

  ก่อนแก้:
  Phase 0a: sticker crop → zxing(crop) → cv2 variants(crop) ← false positive ตรงนี้
  Phase 0:  WeChatQRCode full
  Phase 1:  cv2 variants full
  Phase 2:  zxing full ← ถูก แต่ไม่ถึง

  หลังแก้:
  Phase 0:  zxing full image ← รันก่อนเสมอ, แม่น, ไม่ false positive
  Phase 0b: WeChatQRCode full
  Phase 1:  sticker crop → zxing(crop) → cv2 variants(crop)
  Phase 2:  cv2 variants full

  เหตุผลที่ zxing ควรอยู่ก่อนสุด: zxing-cpp ใช้ algorithm ที่แข็งแกร่งกว่า cv2 มาก handle perspective/rotation ได้เอง และแทบไม่มี false positive ต่างจาก cv2
  Phase 0a: sticker crop → zxing(crop) → cv2 variants(crop) ← false positive ตรงนี้
  Phase 0:  WeChatQRCode full
  Phase 1:  cv2 variants full
  Phase 2:  zxing full ← ถูก แต่ไม่ถึง

  หลังแก้:
  Phase 0:  zxing full image ← รันก่อนเสมอ, แม่น, ไม่ false positive
  Phase 0b: WeChatQRCode full
  Phase 1:  sticker crop → zxing(crop) → cv2 variants(crop)
  Phase 2:  cv2 variants full

  เหตุผลที่ zxing ควรอยู่ก่อนสุด: zxing-cpp ใช้ algorithm ที่แข็งแกร่งกว่า cv2 มาก handle perspective/rotation ได้เอง และแทบไม่มี false positive ต่างจาก cv2
  ที่ถ้า pattern "คล้าย" QR ก็คืนค่ามาได้โดยไม่มีการ validate

  ---
  ผลลัพธ์หลังแก้

  ┌─────────────┬───────────────────────────────┬────────────────────────┐
  │             │             ก่อนแก้             │         หลังแก้          │
  ├─────────────┼───────────────────────────────┼────────────────────────┤
  │ lot_number  │ (01)41580980574830 ❌         │ MR0010001012614526R ✅ │
  ├─────────────┼───────────────────────────────┼────────────────────────┤
  │ decode path │ cv2 false positive บน crop ผิด │ zxing full image       │
  └─────────────┴───────────────────────────────┴────────────────────────┘

---

## Fix 2 — OCR อ่าน prefix ของ lot เป็นตัวเลขแทนตัวอักษร (2026-05-28)

**ไฟล์ที่แก้:** `utils/validators.py`

### ปัญหา

`back_label` และ `grade_bag` มี lot ที่ขึ้นต้นด้วยตัวอักษรภาษาอังกฤษ 2 ตัวเสมอ (เช่น `HO`, `HR`, `CR`)  
แต่ OCR อ่าน `O` เป็น `0` (ศูนย์) ทำให้ได้ `H0` แทน `HO`

**ตัวอย่าง:**

| ก่อนแก้ | หลังแก้ |
|---|---|
| `H00005001014612426` ❌ | `HO0005001014612426` ✅ |

### Root Cause

Vision API แยก `O` กับ `0` ได้ยากในฟอนต์แบบ monospace/industrial ที่ใช้บน label
เนื่องจาก 2 ตัวแรกของ lot ไม่มีทางเป็นตัวเลขอยู่แล้ว จึงสามารถแก้ได้โดย deterministic

### วิธีแก้

เพิ่มฟังก์ชัน `_fix_lot_alpha_prefix()` ใน `utils/validators.py`  
map digit lookalikes → letters เฉพาะ 2 ตัวแรกของ lot:

| digit | letter |
|---|---|
| `0` | `O` |
| `1` | `I` |
| `5` | `S` |
| `8` | `B` |

เรียกใช้ใน `find_lot()` ต่อจาก pattern match ทุก path (class-specific + generic fallback)  
ทำงานเฉพาะ `back_label` และ `grade_bag` เท่านั้น — class อื่นไม่กระทบ

### โค้ดที่เพิ่ม

```python
_DIGIT_TO_LETTER_TRANS = str.maketrans('0158', 'OISB')
_ALPHA_PREFIX_CLASSES = frozenset({'back_label', 'grade_bag'})

def _fix_lot_alpha_prefix(lot: str, image_class: str | None) -> str:
    if image_class not in _ALPHA_PREFIX_CLASSES or len(lot) < 2:
        return lot
    fixed = lot[:2].upper().translate(_DIGIT_TO_LETTER_TRANS)
    return fixed + lot[2:]
```

### หมายเหตุ

ถ้าเจอ misread pattern ใหม่ใน prefix เพิ่มใน `_DIGIT_TO_LETTER_TRANS` ได้เลย  
เช่น `6→G` → `str.maketrans('01586', 'OISBG')`

---

## Fix 3 — OCR อ่านตัวเลขนำหน้าของ compact date เป็น Q (2026-06-04)

**ไฟล์ที่แก้:** `utils/validators.py`

### ปัญหา

`container_label` และ `retail_sachet` พิมพ์วันหมดอายุบนซองเป็น compact format `ddmmyyyy` ไม่มี keyword นำหน้า (เช่น `03062027`)  
Vision API อ่าน `0` ตัวแรกเป็น `Q` → ได้ `Q3062027` → `DATE_COMPACT_FALLBACK = re.compile(r'\b(\d{8})\b')` match ไม่ได้เพราะ `Q` ไม่ใช่ digit → `exp_sachet = None`

**ตัวอย่าง:**

| raw text จาก Vision API | ก่อนแก้ | หลังแก้ |
|---|---|---|
| `Q3062027\nLOT 0196` | `exp_sachet = null` ❌ | `exp_sachet = "2027-06-03"` ✅ |

### Root Cause

`Q` และ `0` มีรูปร่างคล้ายกันในฟอนต์ขนาดเล็กบนซอง — Vision API เลือก `Q` แทน `0`  
เนื่องจาก `DATE_COMPACT_FALLBACK` require digit ทุกตัว จึง miss ไปทั้งหมด

### วิธีแก้

เพิ่ม rule ใน `_OCR_FIXES` (`utils/validators.py`):

```python
(re.compile(r'\bQ(\d{7})\b'), r'0\1'),  # Q→0 OCR misread in compact date
```

`correct_ocr()` ถูกเรียกใน `find_expiry()` ซึ่ง run ทุก class → ครอบคลุม `container_label` และ `retail_sachet` โดยอัตโนมัติ  
ความเสี่ยง false positive ต่ำมาก เพราะ `Q` ตามด้วย 7 digits แทบไม่มีความหมายอื่นบนฉลากสินค้า

---

## Fix 5 — `30_sachet` lot_number อ่านได้วันหมดอายุแทนเลข lot จริง (2026-07-02)

### ปัญหา

ส่งรูป `IMG_20260702_091824.jpg` (สติกเกอร์หลังกล่อง MATCHAZUKI รุ่น B-Grade Medium Rich) ผ่าน pipeline จริง (`PipelineRunner` + `PackagingRegistry`, ไม่ใช่ `test_image.py`) ได้ `lot_number: "02-07-2027"` ซึ่งเป็นวันหมดอายุ ไม่ใช่ lot code — ค่าที่ถูกต้องคือ `MR0006HKW028618326R` ที่พิมพ์อยู่บรรทัดถัดไปในฉลากเดียวกัน

### Root Cause (2 บั๊กแยกกัน)

**บั๊กที่ 1 — `_is_valid_lot()` เช็ควันที่ไม่ครบ (`utils/validators.py`)**

`_is_valid_lot()` กัน false positive ด้วย `re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', lot)` — เช็คเฉพาะรูปแบบคั่นด้วย `/` เท่านั้น ตั้งแต่ 2026-06-30 ที่เพิ่มรองรับวันที่คั่นด้วย `-` ให้ `find_expiry`/`normalize_date` (ดู `bug_fix.md` ไม่มีบันทึกไว้ ณ ตอนนั้น) ไม่มีใครอัปเดต `_is_valid_lot` ให้ตามทัน จึงมีช่องโหว่: ค่า `"02-07-2027"` ผ่านเป็น "valid lot" ได้ — กระทบทุก packaging ไม่ใช่แค่ `30_sachet` `tests/test_ocr.py` เดิมมี `test_ignores_date_as_lot` ทดสอบแค่ฟอร์แมต `/` ไม่มี test คู่ฟอร์แมต `-` เลย

**บั๊กที่ 2 — `30_sachet.yaml` lot_patterns ข้ามบรรทัดวันที่ไม่ได้**

ฉลากรุ่น B-Grade รวม 2 label ไว้บรรทัดเดียว (`ควรบริโภคก่อน / Lot :`) แล้วตามด้วยค่า **2 บรรทัด**: วันหมดอายุก่อน แล้วค่อยเป็น lot code จริง —
```
ควรบริโภคก่อน / Lot :
02-07-2027
MR0006HKW028618326R
```
`lot_patterns[0]` (`LOT\s*[:\.\-]?\s*([A-Z]{0,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)`) ออกแบบมาให้จับรูปแบบ `MR...R` นี้เป๊ะ แต่ `\s*` กินแค่ whitespace ข้ามบรรทัดวันที่ (ซึ่งไม่ใช่ whitespace) ไม่ได้ → ไม่แมตช์เลยที่ตำแหน่งนี้ → fallback ไป `lot_patterns[1]` (`[A-Z0-9\-]{4,}` แบบกว้าง) ซึ่งจับ `"02-07-2027"` ได้ทันที (ตัวเลข+ขีดอยู่ใน character class) แล้วหยุดที่ `\n` — ได้ค่าวันที่แทน lot code

ตรวจ training photo จริงทั้ง 13 รูปใน `images/30_sachet/` (ไม่รวม `aug_*`) แล้วพบว่า**ไม่มีรูปไหนเจอ layout นี้เลย** — ทุกรูปเป็นรุ่น "Excellent" ที่ `Lot :` ตามด้วย code `EX...` บรรทัดเดียวกันทันที แปลว่า regex เดิมถูกออกแบบ/เทสต์กับ layout รุ่นเดียว ไม่ครอบคลุมรุ่น B-Grade ที่เพิ่งเจอ

### วิธีแก้

1. `utils/validators.py::_is_valid_lot` — เพิ่ม `-` เป็น separator ที่ยอมรับ: `re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', lot)`
2. `config/packagings/30_sachet.yaml::lot_patterns[0]` — เพิ่ม optional group ให้ข้ามวันที่คั่นด้วย `/` หรือ `-` หนึ่งบรรทัดก่อนถึง code จริง: `LOT\s*[:\.\-]?\s*(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*)?([A-Z]{0,3}(?:\d{6,}|\d{4,}[A-Z]{3}\d{2,})[A-Z0-9]*)` — scope เฉพาะ pattern[0] ของ `30_sachet` เท่านั้น (ไม่แตะ `lot_patterns[1]`, ไม่แตะ `back_label.yaml` หรือ class อื่นที่ใช้ pattern shape เดียวกัน — ยังไม่มีหลักฐานว่ามี layout แบบนี้ที่อื่น)

### เทสต์ที่เพิ่ม (`tests/test_ocr.py`)

- `test_ignores_dash_date_as_lot` — `find_lot("LOT 30-06-2027")` ต้องได้ `None` (คู่ฟอร์แมต `-` ของ `test_ignores_date_as_lot` เดิม)
- `test_30_sachet_lot_skips_interleaved_exp_date` — raw text รูปแบบ B-Grade จริง ต้องได้ `MR0006HKW028618326R`
- `test_30_sachet_lot_same_line_unaffected` — raw text รูปแบบ Excellent (ของจริงจาก training set) ต้องไม่เปลี่ยนพฤติกรรม

### ผลลัพธ์หลังแก้

| | ก่อนแก้ | หลังแก้ |
|---|---|---|
| lot_number | `02-07-2027` ❌ | `MR0006HKW028618326R` ✅ |
| exp_date | `2027-07-02` (ถูกอยู่แล้ว) | `2027-07-02` (ไม่เปลี่ยน) |

Validation: `pytest tests/test_ocr.py` 71 ผ่านหมด, `pytest` เต็ม suite ไม่มี regression ใหม่ (2 failed ใน `test_api_packagings.py` + 3 error ใน `test_classifier.py` เป็นของเดิมอยู่แล้ว ยืนยันด้วย `git stash` แล้วรันซ้ำ), รูป `IMG_20260702_091824.jpg` ผ่าน `PipelineRunner` จริงได้ค่าถูกแล้ว, OCR ซ้ำ 13 รูป training จริงใน `images/30_sachet/` ได้ lot เดิมทุกรูปไม่มี regression

### หมายเหตุ

- `30_sachet` เป็น shipped class ที่ baked เข้า image (ไม่ใช่ GCS-overlay-editable) — ต้อง rebuild image + deploy ถึงจะขึ้น prod ไม่ใช่แค่แก้ YAML/push
- `back_label.yaml` ใช้ pattern shape เดียวกับ `lot_patterns[0]` ของ `30_sachet` (comment เดิมอ้างถึงกันตรงๆ) แต่ยังไม่ได้ตรวจว่ามี layout แบบเดียวกันหรือไม่ — ทิ้งไว้เป็น scope ในอนาคตถ้าเจอหลักฐานจริง
- **Follow-up (ยังไม่ทำ):** กัน TEST_MODE publish ลง dataset folder จริง + เพิ่ม reconcile/GC ให้ `merge_class_names` — ดู spec `docs/superpowers/specs/2026-06-26-detector-dataset-datayaml-cleanup-design.md`
