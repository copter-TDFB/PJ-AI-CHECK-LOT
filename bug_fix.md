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

---

## Fix 6 — แก้ `30_sachet` product_aliases แล้ว deploy ไม่มีผลกับ production เลย (2026-07-06)

### ปัญหา

ตั๋ว KAN-43 แก้ `config/packagings/30_sachet.yaml::product_aliases` (ลบ entry เดี่ยว "Medium Rich 95% 100 g" + เพิ่ม "100 g" เป็น AND term ในทั้ง 2 entry ของ B-Grade) — merge PR, build image ใหม่, deploy เป็น candidate revision (0% traffic) เรียบร้อยตามขั้นตอนปกติ แต่ตรวจ `GET /api/packagings/30_sachet` บน candidate แล้วได้ค่าที่**ไม่ตรงกับทั้งของเก่าและของใหม่** — เป็น product_aliases เวอร์ชันที่ 3 ที่ไม่เคยเห็นในโค้ดปัจจุบันเลย

### Root Cause (2 ชั้นซ้อนกัน)

**ชั้นที่ 1 — `config_overrides.json` มี `product_aliases` override เก่าค้าง**

มีคนเคยแก้ผ่าน wizard drawer เมื่อ 2026-07-03 (ดู commit `854292f`) ซึ่ง drawer เขียนได้แค่ AND-group แบบ flatten เป็น OR เท่านั้น (gotcha เดิม, ดู `docs/RUNBOOK.md` "Config vs. code changes" gotcha 2026-07-03) — override ที่เขียนไว้เป็นเวอร์ชันก่อน KAN-43 (5 entries, B-Grade ไม่มี "100 g")

**ชั้นที่ 2 — มีไฟล์ config เต็มก้อนซ้อนอยู่บน GCS ด้วย (ชั้นที่ลึกกว่าและไม่คาดคิด)**

`gs://ocr-lot-checker-config/packagings/30_sachet.yaml` มีไฟล์ snapshot เต็มก้อนของทั้ง config (ไม่ใช่แค่ product_aliases) ค้างอยู่ — ตรงกับเนื้อหา YAML เวอร์ชันก่อน commit `854292f` เป๊ะ (4 entries, B-Grade ไม่มี "100 g", ไม่มี entry "Medium Rich 95% 100 g" เดี่ยวๆ) ตาม design เดิม (`CLAUDE.md`) ไฟล์นี้ควรมีอยู่**เฉพาะ class ที่สร้างผ่าน wizard เท่านั้น** แต่ `30_sachet` ถูกเพิ่มผ่าน git commit ธรรมดา (`d7a00c3`) ไม่เคยผ่าน wizard เลย — คาดว่าไฟล์นี้หลุดมาจากการ hotfix เดียวกันเมื่อ 2026-07-03 (อาจเผลอเรียก `cloudrun_deployer.publish_packaging_to_gcs` แทนที่จะเรียกแค่ `config_overrides.save_product_aliases`)

เพราะ `PackagingRegistry` overlay ชั้นนี้ (full-YAML) มาก่อน `config_overrides.json` (narrow override) ในลำดับการ resolve — ผลคือ **การ rebuild + redeploy image ใหม่ไม่มีผลอะไรกับ production เลย** ตราบใดที่ไฟล์นี้ยังอยู่ ไม่ว่าจะแก้ YAML ในโค้ดถูกแค่ไหนก็ตาม (Fix 5 เคยตั้งสมมติฐานไว้ว่า "`30_sachet` เป็น shipped class... ต้อง rebuild image + deploy ถึงจะขึ้น prod" — สมมติฐานนี้ไม่ครบถ้วน เพราะไม่ครอบคลุมเคสที่มี stray GCS overlay ซ้อนอยู่)

### วิธีแก้

1. ลบ `config_overrides.json` override: `services.config_overrides.delete_product_aliases('30_sachet')` (เรียกผ่าน `DELETE /api/packagings/30_sachet/product-aliases` บน candidate revision เพื่อ trigger reload ในตัวด้วย)
2. ลบไฟล์ overlay เต็มก้อน: `gcloud storage rm gs://ocr-lot-checker-config/packagings/30_sachet.yaml`
3. Verify ด้วย `GET /api/packagings/30_sachet` บน candidate URL ว่ากลับไปอ่านจาก baked YAML ในตัว image ถูกต้อง (4 entries, "100 g" ครบทั้ง 2 entry ของ B-Grade) ก่อนค่อยสลับ traffic 100%

### ผลลัพธ์หลังแก้

| | ก่อนแก้ | หลังแก้ |
|---|---|---|
| `product_aliases` ที่ production เสิร์ฟจริง | 5 entries, B-Grade ไม่ต้องมี "100 g" (จาก `config_overrides.json`) | 4 entries, B-Grade ต้องมี "100 g" ครบ (จาก baked YAML ในตัว image) |
| ผลของการ rebuild+redeploy image เฉยๆ (ไม่แตะ GCS) | ไม่มีผลกับ production เลย | ใช้ได้ตามคาด หลังลบทั้ง 2 overlay แล้ว |

Verify บน production URL จริง (`https://ocr-lot-checker-459907489982.asia-southeast1.run.app`) หลังสลับ traffic 100% แล้ว — `GET /api/packagings/30_sachet` ตรงกับที่ตั้งใจแก้ทุกจุด

### หมายเหตุ

- **CLAUDE.md แก้แล้ว**: บรรทัดที่เคยเขียนว่า "GCS `packagings/` holds ONLY wizard-created/edited classes" ไม่ครบถ้วน — เพิ่ม caveat ว่า shipped class ก็อาจมี stray overlay ได้จริงถ้าเคยมีการ hotfix/publish ผิดที่มาก่อน ต้องเช็คเสมอ ไม่ใช่เชื่อตาม "สร้างยังไง"
- **Follow-up (ยังไม่ทำ)**: เพิ่ม startup check หรือ CI check ที่เตือนเมื่อ shipped class (ไม่เคยผ่าน wizard promote) มี GCS `packagings/<key>.yaml` overlay ค้างอยู่โดยไม่คาดคิด — ป้องกันไม่ให้เคสนี้เงียบซ้ำกับ class อื่น
- ลำดับการแก้สำคัญ: ลบ override/overlay ก่อนที่ image ใหม่จะพร้อม (แต่ยังไม่สลับ traffic) จะทำให้ revision เก่าที่ยังรับ traffic 100% อยู่ fallback ไปที่ baked YAML รุ่นเก่าของมันเอง (เกิด gap ชั่วคราวสำหรับ product ที่เพิ่งถูกลบ entry ไป) — เพื่อลด gap ให้เหลือน้อยที่สุด ควร build+deploy candidate ให้เสร็จก่อน แล้วค่อยลบ override/overlay ผ่าน endpoint ของ candidate เอง (ไม่กระทบ revision เก่าที่ยังรับ traffic อยู่) จากนั้นค่อยสลับ traffic ทันที
