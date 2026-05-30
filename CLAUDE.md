# CLAUDE.md — pj ocr text check lot

## Project Overview
ระบบอ่านและตรวจสอบเลข Lot สินค้าจากรูปภาพอัตโนมัติ พร้อม cross-check กับ Google Sheet

- รับ **binary image** จาก n8n (multipart/form-data) + `sheet_id` + `sheet_gid`
- Classify ชนิดรูปด้วย AI model (**6 class**)
- ดึงเลข lot ตาม class:
  - `import_sticker` → QR Scanner (มี QR code)
  - class อื่น → Detector (crop) → OCR (Google Vision)
- Cross-check ผลกับ Google Sheet (lot/exp/product/sachet)
- ส่งผลลัพธ์ + match flags + verify_message กลับ n8n เป็น JSON
- Deploy บน **Google Cloud Run**

---

## 📋 Development Plan
- **ไฟล์แผนงาน:** `PLAN.md` ที่ root — track progress แต่ละ phase ด้วย checkbox
- **ก่อนเริ่มงานทุกครั้ง:** อ่าน `PLAN.md` ก่อนเพื่อรู้ว่าอยู่ phase ไหน, step ไหนทำเสร็จแล้ว/ยัง
- **เมื่อทำ step ใดเสร็จ:** update checkbox `[ ]` → `[x]`, อัปเดต Progress Overview table, และเพิ่ม entry ใน Change Log
- **เมื่อเปลี่ยน phase:** update status icon (⚪ Not Started → 🟡 In Progress → 🟢 Done) และอัปเดต `Current phase` ด้านบนของไฟล์

---

## Pipeline

```
                                       ┌─ import_sticker ───→ [2a] QR Scanner ─────────────────────┐
n8n (binary + sheet_id) → [1] Classifier ┤                                                          ├→ [4] SheetChecker → JSON
                                       └─ 5 classes อื่น ────→ [2b] Detector → [3] Preprocessor → [3'] OCR ┘
```

**Confidence gate:** ถ้า `class_confidence < 0.6` → skip pipeline, ตอบ `status=low_confidence` + `verify_message="⚠️ ไม่แน่ใจประเภทรูป"`

### 1. Image Classifier (6 class)

| Class | ลักษณะ | Pipeline Path |
|---|---|---|
| `back_label` | ฉลากหลังถุงสินค้า | Detector → OCR |
| `import_sticker` | สติ๊กเกอร์นำเข้า — **มี QR code** | QR Scanner (ข้าม crop/OCR) |
| `container_label` | สติ๊กเกอร์บนภาชนะสแตนเลส (กล่อง + ซอง) | Detector → OCR (แยกทุก crop) |
| `grade_bag` | ถุง MEDIUM / RICH | Detector → OCR |
| `retail_sachet` | ซองเดี่ยว EXCELLENT | Detector → OCR |
| `capsule_box` | กล่อง capsule (ใหม่ — เพิ่มเข้ามาเป็น class ที่ 6) | Detector → OCR |

- **Model:** EfficientNet-V2-S (`models/classifier.pt`) — input 384×384
- **Output:** `(class_name, confidence)` รอบ 4 ตำแหน่งทศนิยม
- **Training:** `train_classifier.py` — Stage 1 head-only → Stage 2 full fine-tune, WeightedRandomSampler + class weights ใน CrossEntropyLoss แก้ imbalance

### 2a. QR Scanner (`pipeline/qr_scanner.py`)
ใช้เฉพาะ `import_sticker` — decode QR หลาย phase (เร็วก่อน → ช้าหลัง):

1. **zxing-cpp** บน full image (แม่นที่สุด, ไม่มี false positive)
2. **WeChatQRCode** full (handle perspective ได้ดี ถ้า opencv build มี contrib)
3. **Sticker crop** (หา QR region ก่อน) → WeChat → zxing → cv2 + variants
4. **cv2.QRCodeDetector** บน preprocessing variants (Otsu, adaptive, CLAHE, sharpen)
5. **Resized** (รูปใหญ่บางครั้ง detector miss) → zxing + cv2
6. **Perspective warp** จาก detected corners (แก้ QR ถ่ายเฉียง)
7. **Scale up** 1.5×/2×/3× (QR เล็กเกินในเฟรม)
8. **Rotation sweep** ทุก 10° + fine sweep รอบมุมเฉียง (45°/135°/225°/315°)

> ลำดับ priority สำคัญ: zxing ต้องอยู่ก่อน cv2 เสมอ เพราะ cv2 มี false positive บน crop ที่ context เยอะ (รายละเอียดดู `bug_fix.md` หัวข้อ Fix 1)

- คืน `lot_number` โดยตรงจาก QR data
- `bbox` = `null` เสมอ (ไม่มี region detection)

### 2b. Region Detector (`pipeline/detector.py`)
ใช้กับ 5 class ที่เหลือ

- **Model:** YOLOv8 (`models/detector.pt`) — multi-class, แยก class ด้วย prefix
- **Class mapping:** YOLO class ที่ขึ้นต้นด้วย `{image_class}_` จะถูก match  
  เช่น `back_label_lot`, `container_label_box`, `container_label_sachet`
- **Output:** `list[DetectionResult]` (`cropped_bytes` + `bbox`) เรียงจาก y1 บน → ล่าง
- **Confidence threshold:** `DETECTOR_CONF` (default `0.25`)
- **Heuristic fallback:** ใช้เมื่อ YOLO miss หรือไม่มี `models/detector.pt`
  - `back_label` / `grade_bag` / `retail_sachet` → crop bottom 30–40%
  - `container_label` → contour-based white-box detection
  - class ที่ไม่มี rule → คืน full image

### 3. Preprocessor (`pipeline/preprocessor.py`)
**Pass-through stub** — Google Cloud Vision จัดการ image enhancement / deskew ได้ดีกว่า manual preprocessing
- คืน `image_bytes` ดิบโดยไม่แก้ไข

### 3'. OCR Engine (`pipeline/ocr_engine.py`)
- **Library:** Google Cloud Vision API (Text Detection)
- **container_label:** OCR **แยกทีละ crop** เพื่อจับคู่ lot↔date ของแต่ละกล่อง/ซองให้ถูก
  - ถ้า OCR ผลลัพธ์ degraded (`raw_text < 8 chars` AND ไม่มี lot/date) → retry ด้วย original bytes
- **class อื่น:** stack crop ทุกอันเป็นรูปเดียวด้วย `stack_images_vertically()` → OCR **1 ครั้ง** (ประหยัด API call)
- แก้ความผิดพลาด OCR ก่อน parse ด้วย `correct_ocr()` ใน `utils/validators.py`
- Extract fields ต่อ class:
  - `find_lot()` — class-specific regex ก่อน, generic fallback หลัง
  - `find_expiry()` / `find_mfg()` — keyword + date pattern
  - `find_product_name()` + `find_size()` (เฉพาะ `back_label` / `grade_bag`)
- ถ้ามีทั้ง product + size จะรวมเป็น `"Houjicha Powder 40 g"` ในฟิลด์ `product_name`

### 4. Sheet Checker (`utils/sheet_checker.py`)
Cross-check ผล OCR/QR กับ Google Sheet (per-request `sheet_id` + `gid`)

- **Auth:** Google ADC (service account บน Cloud Run, `gcloud auth application-default login` ใน local)
- **Scope:** `spreadsheets.readonly`
- **Cache:** TTL 5 นาทีต่อ `(sheet_id, gid)` — ลด API calls
- **Header detection:** หา row แรกที่มี column `Lot.` หรือ `Lot`
- **Match logic ต่อ class:**

| Class | lot_match | exp_match | product_match | sachet_match |
|---|---|---|---|---|
| `import_sticker` | ✅ (full lot, fallback ตัด `[9:13]` ถ้าไม่เจอ) | — | — | — |
| `back_label`, `grade_bag` | ✅ | ✅ | ✅ | — |
| `retail_sachet`, `capsule_box` | ✅ | ✅ | — | — |
| `container_label` | ✅ (จาก `lot_box`) | ✅ (`exp_box`) | — | ✅ (`exp_box == exp_sachet`) |

- ค่า `None` = ไม่เกี่ยวกับ class นั้น | `False` = ไม่ตรง | `True` = ตรง
- ถ้า exception → คืน `None` ทุก field (fail open) + log

---

## Tech Stack
- **Language:** Python 3.11
- **API server:** FastAPI + uvicorn
- **Classifier:** PyTorch + EfficientNet-V2-S (`models/classifier.pt`)
- **Detector:** Ultralytics YOLOv8 (`models/detector.pt`)
- **OCR:** Google Cloud Vision API (Text Detection)
- **QR:** zxing-cpp + cv2.QRCodeDetector + cv2 WeChatQRCode (ถ้ามี)
- **Sheet:** gspread + google-auth (ADC)
- **Image processing:** OpenCV (headless), Pillow, pillow-heif (รองรับ HEIC/HEIF จาก iPhone)
- **Container:** Docker
- **Deploy:** Google Cloud Run @ `asia-southeast1`
- **Registry:** Google Artifact Registry
- **Package manager:** pip

---

## Project Structure

```
pj-ocr-text-check-lot/
├── CLAUDE.md
├── AGENTS.md
├── PLAN.md                    ← phase progress tracker
├── bug_fix.md                 ← บันทึก root cause + fix ของ bug สำคัญ
├── FLOW_PROCESS_PRESENTATION.html  ← deck สำหรับนำเสนอ flow
│
├── main.py                    ← FastAPI app + lifespan + /predict + /health
├── pipeline/
│   ├── classifier.py          ← EfficientNet-V2-S classifier (6 class)
│   ├── qr_scanner.py          ← QR decode หลาย phase (import_sticker)
│   ├── detector.py            ← YOLO detector + heuristic fallback
│   ├── preprocessor.py        ← pass-through stub
│   └── ocr_engine.py          ← Google Vision wrapper + field extraction
│
├── utils/
│   ├── image_utils.py         ← bytes → PIL/numpy, stack_images_vertically()
│   ├── validators.py          ← regex lot/date/size/product + correct_ocr() + _fix_lot_alpha_prefix()
│   └── sheet_checker.py       ← Google Sheet cross-check + cache 5 นาที
│
├── models/                    ← weights (ไม่ commit ลง git)
│   ├── classifier.pt
│   └── detector.pt
│
├── tests/
│   ├── test_classifier.py
│   ├── test_detector.py
│   ├── test_preprocessor.py
│   ├── test_ocr.py
│   └── test_integration.py    ← FastAPI TestClient + mock OcrEngine
│
├── images/                    ← training/eval data (ไม่ commit)
│   ├── back_label/            (162 รูป)
│   ├── capsule_box/           (45 รูป)
│   ├── container_label/       (201 รูป)
│   ├── grade_bag/             (167 รูป)
│   ├── import_sticker/        (301 รูป)
│   └── retail_sachet/         (150 รูป)
│
├── preview/                   ← output ของ preview_pipeline.py
│
├── train_classifier.py        ← train classifier
├── train_detector.py          ← train YOLO detector
├── evaluate.py                ← วัด accuracy ต่อ class
├── confusion_matrix_eval.py   ← confusion matrix @ threshold 0.5/0.6/0.7
├── preview_pipeline.py        ← save pipeline output class ละ N รูป ลง preview/
├── sort_images.py             ← ใช้ classifier แยกรูปดิบเข้า folder
├── colab_classify_training.ipynb  ← train บน Colab GPU
│
├── Dockerfile
├── .dockerignore
├── .gcloudignore
├── .env.example
├── requirements.txt           ← runtime deps
├── requirements-train.txt     ← training deps (ไม่รวมใน production image)
└── skills-lock.json
```

---

## API Contract

### Request
n8n ส่งเป็น **multipart/form-data** + query params:

```http
POST /predict?sheet_id={SHEET_ID}&sheet_gid={GID}
Content-Type: multipart/form-data

file: <binary image>            ← field name ต้องเป็น 'file' เสมอ
```

| Param | Required | Default | Note |
|---|:---:|---|---|
| `file` (multipart) | ✅ | — | ต้องเป็น `image/*` |
| `sheet_id` (query) | ✅ | — | Google Sheet ID |
| `sheet_gid` (query) | ❌ | `0` | tab GID |

### Response (JSON)

**ทุก class คืนฟิลด์เดียวกันหมด** — ฟิลด์ที่ไม่เกี่ยวจะเป็น `null`

```json
{
  "lot_number": "HO0005001014612426",
  "confidence": null,
  "class": "back_label",
  "class_confidence": 0.9631,
  "raw_text": "MATCHAZUKI Houjicha\n...\nBBD:04/05/2027\nLOT:HO0005...",
  "mfg_date": null,
  "exp_date": "2027-05-04",
  "product_name": "Houjicha Powder 40 g",
  "size": "40 g",
  "lot_box": null,
  "lot_sachet": null,
  "exp_box": null,
  "exp_sachet": null,
  "lot_match": true,
  "exp_match": true,
  "product_match": true,
  "sachet_match": null,
  "verify_message": "✅ ตรวจสอบผ่าน",
  "bbox": [120, 80, 340, 160],
  "status": "ok"
}
```

#### Fields ต่อ class

| Class | lot_number | exp_date | product_name | size | lot_box | lot_sachet | exp_box | exp_sachet | sachet_match |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `import_sticker` | ✅ (QR) | — | — | — | — | — | — | — | — |
| `back_label` | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — |
| `grade_bag` | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — |
| `container_label` | — | — | — | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| `retail_sachet` | ✅ | ✅ | — | — | — | — | — | — | — |
| `capsule_box` | ✅ | ✅ | — | — | — | — | — | — | — |

#### `status` values
- `ok` — pipeline สำเร็จ + เจอ lot
- `not_found` — รัน pipeline สำเร็จแต่หา lot ไม่เจอ
- `qr_not_found` — QR decode ไม่ได้ (import_sticker)
- `lot_not_found` — decode QR ได้แต่ extract lot ไม่ได้
- `low_confidence` — class confidence < 0.6 → skip pipeline

#### `verify_message` (สำหรับส่ง Slack ผ่าน n8n)
- `"✅ ตรวจสอบผ่าน"` — match ครบ
- `"❌ ไม่พบ lot ใน sheet"` — lot ไม่ match
- `"❌ exp ไม่ตรง, product ไม่ตรง"` — list errors (concatenated)
- `"❌ กล่องกับซองไม่ตรงกัน"` — เฉพาะ container_label
- `"⚠️ ไม่แน่ใจประเภทรูป กรุณาตรวจสอบด้วยตนเอง"` — low confidence

#### Canonical product_name
`find_product_name()` map keyword → canonical name (จาก `utils/validators.py`):

| Keyword ที่ OCR เจอ | คืนเป็น |
|---|---|
| `Excellent Rich` | `Excellent Rich 95%` |
| `Classic Rich` | `Classic Rich 95%` |
| `Medium Rich` | `Medium Rich 95%` |
| `Houjicha Rich` | `Houjicha Rich 95%` |
| `Houjicha` | `Houjicha Powder` |
| `Genmaicha` | `Genmaicha Powder` |
| `Excellent` (ไม่มี Rich) | `Excellent` |
| `Medium` | `Medium` |
| `Classic` | `Classic` |

ถ้ามี `size` ด้วย จะ concat: `"Houjicha Powder 40 g"`

#### Size format
Normalize เป็น `{number} g` / `{number} kg` เสมอ (รองรับ กรัม / g / กก. / kg)

---

## OCR Corrections (`utils/validators.py`)

### `_OCR_FIXES` — replace common misreads
| ก่อน | หลัง | หมายเหตุ |
|---|---|---|
| `LOL` / `L0L` / `L01` | `LOT` | |
| `L0T` / `LO7` | `LOT` | |
| `LCT` / `LC7` | `LOT` | |
| `EXCELLENT RIC*` | `EXCELLENT RICH` | + CLASSIC/MEDIUM/HOUJICHA |
| `BBl` | `BBD` | |
| `EXl` | `EXP` | |
| `LOT C123` / `LOT O123` / `LOT D123` | `LOT 0123` | retail_sachet only |

### `_fix_lot_alpha_prefix()` — แก้ digit↔letter ใน 2 ตัวแรก
เฉพาะ `back_label` / `grade_bag` — 2 ตัวแรกของ lot เป็นตัวอักษรเสมอ

| digit | → letter |
|---|---|
| `0` | `O` |
| `1` | `I` |
| `5` | `S` |
| `8` | `B` |

ตัวอย่าง: `H00005001014612426` → `HO0005001014612426`

---

## Common Commands (Local Dev)

```bash
# ติดตั้ง deps
pip install -r requirements.txt
# ถ้าจะ train เพิ่ม
pip install -r requirements-train.txt

# auth Vision API + Sheets API (ทำครั้งเดียว)
gcloud auth application-default login

# รัน API server
uvicorn main:app --reload --port 8080

# ทดสอบ predict (PowerShell)
$img = Get-Content "images/back_label/sample.jpg" -Encoding Byte -Raw
Invoke-RestMethod -Uri "http://localhost:8080/predict?sheet_id=YOUR_SHEET_ID&sheet_gid=0" `
  -Method Post -Form @{ file = Get-Item "images/back_label/sample.jpg" }

# รัน tests
python -m pytest tests/ -v

# ประเมิน pipeline
python evaluate.py                          # classifier + detector
python evaluate.py --ocr                    # รวม OCR (ต้องมี ADC)
python evaluate.py --class back_label       # เฉพาะ class

# Confusion matrix
python confusion_matrix_eval.py

# Preview pipeline output ลง preview/
python preview_pipeline.py --n 5

# Build Docker
docker build -t ocr-lot-checker .
docker run -p 8080:8080 ocr-lot-checker
```

---

## Google Cloud Run — Deploy

**Project:** `pj-ai-detect-lot-no`  
**Region:** `asia-southeast1` (Singapore)  
**Production URL:** `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`

```bash
# Build + push image ด้วย Cloud Build
gcloud builds submit \
  --tag asia-southeast1-docker.pkg.dev/pj-ai-detect-lot-no/ocr-repo/ocr-lot-checker:latest \
  --project=pj-ai-detect-lot-no \
  --machine-type=e2-highcpu-8 \
  --timeout=1800 \
  .

# Deploy → Cloud Run
gcloud run deploy ocr-lot-checker \
  --image asia-southeast1-docker.pkg.dev/pj-ai-detect-lot-no/ocr-repo/ocr-lot-checker:latest \
  --region asia-southeast1 \
  --platform managed \
  --service-account ocr-lot-checker-sa@pj-ai-detect-lot-no.iam.gserviceaccount.com \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60 \
  --allow-unauthenticated \
  --project=pj-ai-detect-lot-no

# ดู logs
gcloud run services logs read ocr-lot-checker \
  --region asia-southeast1 --project=pj-ai-detect-lot-no
```

### Service Account Permissions
ที่ `ocr-lot-checker-sa@pj-ai-detect-lot-no.iam.gserviceaccount.com` ต้องมี:
- `roles/visionai.user` (หรือ `roles/serviceusage.serviceUsageConsumer` + Vision API enabled)
- Sheet ที่จะอ่าน — share เป็น Viewer ให้ email ของ SA

> **Vision API & Sheets API:** ใช้ ADC ผ่าน service account ของ Cloud Run runtime — **ไม่ต้องใช้ JSON key** ใน production

---

## Dockerfile (ปัจจุบัน)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps สำหรับ OpenCV (libgl1 ต้องการเพราะ ultralytics ดึง opencv-python non-headless)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch ก่อน (ลด image size ~3.5GB → ~1.3GB)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Environment Variables

```bash
# .env.example
MODEL_CLASSIFIER_PATH=models/classifier.pt
MODEL_DETECTOR_PATH=models/detector.pt
DETECTOR_CONF=0.25
OCR_LANG=th+en
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.7

# Local dev เท่านั้น — ชี้ไป service account JSON
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
```

- ใน local: `python-dotenv` โหลด `.env`
- ใน Cloud Run: ตั้งใน Environment Variables ของ service หรือ Secret Manager
- **ห้าม commit `.env` / `gcp-key.json` ลง git** (อยู่ใน `.gitignore`/`.dockerignore`/`.gcloudignore` แล้ว)

---

## Coding Conventions
- Python 3.11 — type hints ทุก function
- function/variable: `snake_case`
- comment สำคัญ: ภาษาไทยได้
- docstring สำหรับ public function/class
- **binary input** แปลงเป็น numpy/PIL ผ่าน `utils/image_utils.py` เท่านั้น
- log ด้วย `logging` module (ห้ามใช้ `print`)
- error จาก client → `HTTP 400` | error ใน pipeline → `HTTP 500` พร้อม message

---

## Important Notes for Claude

### API & Deploy
- **port:** Cloud Run ต้องการ port `8080` เท่านั้น
- **model weights:** ไม่ commit ลง git — `models/*.pt` อยู่ใน `.gitignore`/`.dockerignore`/`.gcloudignore`
- **memory:** EfficientNet-V2-S + YOLOv8 ใน RAM พร้อมกัน → ขั้นต่ำ 2Gi (ถ้าเพิ่ม model ใหม่ ดูว่า memory เกินหรือไม่)
- **deploy:** ใช้ `gcloud builds submit` เสมอ (Docker local ไม่จำเป็น)
- **field names ใน JSON response:** อย่าลบ/เปลี่ยน เพราะ n8n depend อยู่ — เพิ่มใหม่ได้

### Code maintenance
- ถ้าเพิ่ม dependency → อัปเดต `requirements.txt`
- ถ้าเจอ misread pattern OCR ใหม่ → เพิ่มใน `_OCR_FIXES` (`utils/validators.py`)
- ถ้าเพิ่ม product ใหม่ → เพิ่ม keyword + canonical name ใน `find_product_name()` (`utils/validators.py`)
- ถ้าเพิ่ม class ใหม่ → ต้องอัปเดต:
  1. classifier dataset + retrain (`train_classifier.py`)
  2. detector YOLO class prefix mapping (`pipeline/detector.py`)
  3. routing logic + response fields ใน `main.py:predict()`
  4. `SheetChecker.check()` match logic
  5. `_build_verify_message()` ใน `main.py`
  6. class-specific lot pattern ใน `_LOT_BY_CLASS` (`utils/validators.py`)
- **Sheet schema dependency:** SheetChecker หา column `Lot.` หรือ `Lot` เพื่อระบุ header row — ถ้าจะเปลี่ยน schema sheet ต้องอัปเดต `_get_rows()` + `_find_row_by_lot()` ด้วย

### Debugging
- `bug_fix.md` มี root cause + fix ของ bug สำคัญ (QR false positive, OCR prefix digit→letter)
- container_label มี retry mechanism: ถ้า OCR ผลลัพธ์ degraded (raw_text สั้น + ไม่เจอ lot/date) จะ retry ด้วย original bytes
- import_sticker fallback: ถ้าหา lot ใน sheet ด้วย full lot ไม่เจอ และ lot length ≥ 13 → ลอง short lot `[9:13]` (เช่น `HR0008001014613926R` → `0146`)
