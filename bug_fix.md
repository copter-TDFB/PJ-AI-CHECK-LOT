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
