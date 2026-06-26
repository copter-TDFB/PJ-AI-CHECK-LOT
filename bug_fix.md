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

## Fix 4 — detector dataset `data.yaml` เพี้ยนจาก current class (2026-06-26)

### ปัญหา

`data.yaml` ของ detector reference dataset บน Drive (`DRIVE_DETECTOR_DATASET_FOLDER_ID`) ประกาศ `nc: 16` แต่ไม่ตรงกับ class ปัจจุบัน (7 packagings → 11 detector classes) และ parse ด้วย PyYAML ไม่ผ่าน — ถ้า retrain detector รอบใหม่จะได้โมเดลที่ label เพี้ยนหรือ training ล้ม

### Root Cause

สแกน label ครบ 565 ไฟล์ (train 451 + val 114) เทียบ id ที่ label ใช้จริง กับชื่อใน `data.yaml`:
- id 0–8 ถูกต้อง (back_label/capsule/container/grade_bag/retail)
- id 11 = `-test_u` (พัง — ไม่มีเว้นวรรคหลัง `-` → ทำ YAML parse ล้ม), 0 refs
- id 12,13 = `-test_y`/`-test_u` (ขยะ) **แต่รูป print_sticker_back ตัวจริงอ้าง id 12,13** → training จะ label print_sticker_back เป็น `-test_y/-test_u`
- id 14,15 = `print_sticker_full` (มี label จริง ~102 ไฟล์ แต่ **ไม่มี config**)
- id 9 = `new_tea_bag_box` (test-draft ที่หลงเหลือ 24 รูป train, ไม่มี config) ไม่ใช่ print_sticker_back

ต้นเหตุระบบ: `services/dataset_publisher.merge_class_names` เป็น **append-only** (numeric label id ห้าม reorder) ไม่มี GC + การ publish ตอน TEST_MODE หลุดขยะ class เข้า dataset prod

### วิธีแก้

สคริปต์ one-off `scripts/cleanup_detector_dataset.py` (dry-run default, `--execute` gate, ลบลง Drive Trash, backup ก่อนแก้, `--verify`):
- remap label print_sticker_back 49 ไฟล์: id `12→9`, `13→10`
- ลบ `print_sticker_full` + `new_tea_bag_box` (151 รายการ images+labels) ลง Trash
- เขียน `data.yaml` ใหม่เป็นขั้นสุดท้าย (commit point): `nc: 11`, 11 names ตรงกับ `models/detector.pt` + `config/packagings/*` เป๊ะ

ผลหลังแก้ (verify live): ids 0–10, id 9/10 = print_sticker_back ล้วน, ไม่มี test junk → **VERIFY OK**

### หมายเหตุ

- `prod detector.pt` ปัจจุบัน (11 คลาส 0–10) ไม่กระทบ — ปัญหานี้มีผลเฉพาะตอน retrain ครั้งหน้า
- Rollback: remap+data.yaml เดิมอยู่ใน `dataset-backup/<ts>/` (local, gitignored); deletes อยู่ใน Drive Trash (30 วัน)
- เจอ side-issue: `drive_client` log อักขระ `→` ทำ Windows cp1252 console crash (`UnicodeEncodeError`) — สคริปต์ harden ด้วย `sys.stdout.reconfigure("utf-8")` (Drive op run ก่อนบรรทัด log นั้นเสมอ จึงสำเร็จ)
- **Follow-up (ยังไม่ทำ):** กัน TEST_MODE publish ลง dataset folder จริง + เพิ่ม reconcile/GC ให้ `merge_class_names` — ดู spec `docs/superpowers/specs/2026-06-26-detector-dataset-datayaml-cleanup-design.md`
