# 📋 PLAN — pj ocr text check lot

> ไฟล์นี้ใช้ track progress การพัฒนา ระบบจะ update checkbox `[x]` เมื่อทำเสร็จ
> **Last updated:** 2026-05-15
> **Current phase:** Phase 7 Done — deployed Cloud Run, production URL พร้อมใช้

---

## 🎯 Progress Overview

| Phase | Status | Progress |
|---|---|---|
| Phase 0 — Setup & Cleanup | 🟢 Done | 6/6 |
| Phase 1 — Refactor app.py | 🟢 Done | 9/9 |
| Phase 2 — Classifier (5 class) | 🟢 Done | 5/5 + retrain ×2 |
| Phase 3 — Region Detector | 🟢 Done | 5/5 |
| Phase 4 — Preprocessor | 🟢 Done | 3/3 |
| Phase 5 — OCR + Validation per class | 🟢 Done | 4/4 |
| Phase 6 — Integration & Testing | 🟢 Done | 4/4 |
| Phase 7 — Deploy Cloud Run | 🟢 Done | 5/5 |
| Phase 8 — Monitoring (optional) | ⚪ Not Started | 0/3 |

**Legend:** ⚪ Not Started · 🟡 In Progress · 🟢 Done · 🔴 Blocked

---

## ⚠️ Open Decisions (ต้องตอบก่อนเริ่ม phase ที่เกี่ยวข้อง)

- [ ] **OCR engine** — เก็บ Google Cloud Vision หรือเปลี่ยนเป็น EasyOCR/PaddleOCR?
- [x] **n8n compatibility** — อัปเดต workflow เป็น `/predict` + response format ใหม่แล้ว
- [ ] **Volume** — คาดว่าจะเรียก/วัน เท่าไหร่?
- [x] **Training data** — images/ มี 709 รูป (back_label×161, container_label×72, grade_bag×124, import_sticker×301, retail_sachet×51)

---

## Phase 0 — Setup & Cleanup 🟢

**เป้าหมาย:** เตรียมโครงสร้างโปรเจกต์และ tooling ให้ตรง CLAUDE.md

- [x] Rename `Dockerfile.txt` → `Dockerfile` *(หมายเหตุ: ไฟล์เก่าหายไปก่อน เลยเขียนใหม่ทั้งหมด)*
- [x] สร้าง project structure (`pipeline/`, `utils/`, `tests/`, `models/`, `images/<class>/`)
- [x] สร้าง `.gitignore`, `.dockerignore`, `.gcloudignore`, `.env.example`
- [x] เปลี่ยน Python base image: `python:3.10-slim` → `python:3.11-slim`
- [x] เพิ่ม system deps ใน Dockerfile (libglib2.0-0, libsm6, libxext6, libxrender-dev, libgomp1) สำหรับ OpenCV
- [x] รวบรวมรูปตัวอย่างใส่ `images/<class>/` class ละ ≥ 20 รูป *(back_label×20, container_label×20, grade_bag×20, import_sticker×20, retail_sachet×20)*

---

## Phase 1 — Refactor app.py เป็นโครงสร้างใหม่ 🟢

**เป้าหมาย:** แยกโค้ดเป็น modules ตาม CLAUDE.md และทำ response/endpoint ให้ตรง spec

- [x] ย้าย `find_lot`, `find_expiry`, `normalize_date` → `utils/validators.py` + เพิ่ม `find_mfg`
- [x] สร้าง `utils/image_utils.py` (binary → numpy/PIL)
- [x] สร้าง `pipeline/ocr_engine.py` (wrap Cloud Vision call)
- [x] สร้าง `main.py` เป็น FastAPI entry point
- [x] เปลี่ยน endpoint `/detect` → `/predict`
- [x] เปลี่ยน response format ให้ตรง spec (`lot_number`, `confidence`, `class`, `mfg_date`, `exp_date`, `bbox`, `status`)
- [x] เพิ่ม `logging` ทุก step (`logging` module ไม่ใช้ print)
- [x] เพิ่ม error handling: 400 (client) / 500 (pipeline)
- [x] อัปเดต n8n workflow: endpoint `/detect` → `/predict`, response format ใหม่

---

## Phase 2 — Classifier (5 class) 🟢

**เป้าหมาย:** จำแนกรูปเป็น 5 class: `back_label`, `import_sticker`, `container_label`, `grade_bag`, `retail_sachet`

**Approach:** ✅ **A: Fine-tune EfficientNet-B0**

- [x] รวบรวม dataset เริ่มต้น (5 class × 20 รูป = 100 รูป)
- [x] เขียน `train_classifier.py` (Stage 1: head only → Stage 2: full fine-tune, บันทึก best model)
- [x] Train รอบ 1 (100 รูป) — val accuracy: **1.000**
- [x] ใช้ classifier แยกรูป 739 ใบจาก `data collect pj check lot/` → เพิ่ม dataset เป็น **709 รูป** (5 class)
- [x] Train รอบ 2 (709 รูป) — val accuracy: **1.000**
- [x] เขียน `pipeline/classifier.py` (load `.pt` + predict class + confidence)
- [x] เขียน `tests/test_classifier.py`

---

## Phase 3 — Region Detector (Crop) 🟡

**เป้าหมาย:** Detect และ crop บริเวณที่มีเลข lot ตาม class

**Approach (เปลี่ยนเป็น):** ✅ **A: YOLOv8 nano detection** *(ตัดสินใจ 2026-05-12)*
- heuristics เดิมยังเก็บไว้เป็น fallback เมื่อ YOLO หา detection ไม่ได้

- [x] เขียน `pipeline/detector.py` (heuristics รุ่นแรก — จะ refactor เป็น YOLO)
- [x] เขียน `tests/test_detector.py`
- [x] Label bbox ของ lot region ด้วย **Roboflow** (709 รูป, 1 class: `lot_region`)
- [x] Train YOLOv8 nano (`train_detector.py`) — export `models/detector.pt`
- [x] Refactor `pipeline/detector.py` ให้ใช้ YOLO inference + heuristic fallback

---

## Phase 4 — Preprocessor 🟢

**เป้าหมาย:** เตรียมรูปก่อนเข้า OCR

- [x] เขียน `pipeline/preprocessor.py`: grayscale → denoise (Gaussian) → threshold (Otsu) → deskew (minAreaRect)
- [x] Special path สำหรับ `container_label`: CLAHE + inpainting glare spot → adaptive threshold
- [x] เขียน `tests/test_preprocessor.py`

---

## Phase 5 — OCR + Validation per class ⚪

**เป้าหมาย:** อ่านเลข lot และ validate ตาม pattern ของแต่ละ class

- [x] ตัดสินใจ OCR engine — คง Google Cloud Vision (import_sticker ใช้ QR scanner แทน)
- [x] ขยาย regex patterns ต่อ class ใน `utils/validators.py` — `_LOT_BY_CLASS` per class + generic fallback
- [x] เพิ่ม MFG date extraction + รองรับ 2-digit year (dd/mm/yy)
- [x] return `confidence` จริง (จาก Vision API page confidence)
- [x] เขียน `tests/test_ocr.py` — 31 tests passed

---

## Phase 6 — Integration & Testing ⚪

**เป้าหมาย:** รันทดสอบ end-to-end และวัด accuracy

- [x] รัน end-to-end ทั้ง 5 class ด้วยรูปจริง (evaluate.py)
- [x] วัด accuracy ต่อ class — classifier ผ่าน 100% ทุก class (20 รูป/class)
- [x] integration test `/predict` ด้วยรูปจริง — 13/13 passed
- [x] ตรวจ `/health` endpoint ใช้งานได้

---

## Phase 7 — Deploy Cloud Run 🟢

**เป้าหมาย:** Deploy production บน Google Cloud Run

- [x] สร้าง Artifact Registry repo: `ocr-repo` (region: `asia-southeast1`)
- [x] Build + push image ด้วย Cloud Build (CPU-only torch, libgl1 fix)
- [x] Deploy ด้วย `--memory 2Gi --cpu 2 --timeout 60` — service account: `ocr-lot-checker-sa`
- [x] ตั้ง env vars — Vision API ใช้ ADC (SA identity บน Cloud Run ไม่ต้อง JSON key)
- [x] ทดสอบ endpoint production — `/health` ✅, `/predict` (back_label + import_sticker) ✅

**Production URL:** `https://ocr-lot-checker-459907489982.asia-southeast1.run.app`

---

## Phase 8 — Monitoring (optional) ⚪

**เป้าหมาย:** สังเกตการใช้งานจริงและ alert เมื่อมีปัญหา

- [ ] Log structured JSON เข้า Cloud Logging
- [ ] Alert เมื่อ `confidence < threshold` หรือ `status != "ok"`
- [ ] Dashboard ดู volume + accuracy ต่อ class

---

## 📝 Change Log

| Date | What |
|---|---|
| 2026-05-12 | สร้างไฟล์ PLAN.md เริ่มต้น |
| 2026-05-12 | เพิ่มหัวข้อ Development Plan ใน CLAUDE.md (ชี้มา PLAN.md) |
| 2026-05-12 | **Phase 0 (5/6):** เขียน Dockerfile ใหม่ (Python 3.11 + OpenCV deps), เขียน requirements.txt ใหม่ (เพิ่ม opencv-python-headless, Pillow, numpy, python-dotenv), สร้าง pipeline/utils/tests/models/images/<5 class>, สร้าง .gitignore/.dockerignore/.gcloudignore/.env.example |
| 2026-05-12 | **Phase 0 (6/6):** user ใส่รูปตัวอย่าง class ละ 20 รูป (back_label, container_label, grade_bag, import_sticker, retail_sachet) |
| 2026-05-12 | **Phase 1 (8/9):** สร้าง utils/validators.py (find_lot, find_expiry, find_mfg, normalize_date→ISO), utils/image_utils.py, pipeline/ocr_engine.py, main.py (POST /predict + logging + error handling) — เปลี่ยน date format เป็น YYYY-MM-DD ตาม spec |
| 2026-05-12 | **Phase 1 (9/9):** user อัปเดต n8n workflow แล้ว |
| 2026-05-12 | **Phase 2 (5/5):** ลบ app.py เก่า, เพิ่ม torch/torchvision, สร้าง train_classifier.py (EfficientNet-B0, 2-stage), pipeline/classifier.py, tests/test_classifier.py — training เสร็จ val_acc=1.000, wire classifier เข้า main.py แล้ว |
| 2026-05-12 | **Phase 3 (3/3):** สร้าง pipeline/detector.py (heuristic crop ต่อ class, contour detection สำหรับ container_label + CLAHE glare removal), tests/test_detector.py, wire detector เข้า main.py — pipeline ครบ Classifier→Detector→OCR |
| 2026-05-12 | **Phase 4 (3/3):** สร้าง pipeline/preprocessor.py (grayscale→denoise→threshold→deskew, container_label ใช้ CLAHE+inpainting+adaptive threshold), tests/test_preprocessor.py, wire เข้า main.py — pipeline ครบ Classifier→Detector→Preprocessor→OCR |
| 2026-05-12 | **Phase 3 — เปลี่ยน approach:** Heuristics → YOLOv8 nano (ตัดสินใจโดย user), refactor pipeline/detector.py ให้ใช้ YOLO inference + heuristic fallback, สร้าง train_detector.py, เพิ่ม ultralytics ใน requirements.txt — รอ user label bbox + train |
| 2026-05-12 | **Dataset expand:** เขียน sort_images.py ใช้ classifier แยกรูป 739 ใบ (3 รอบ threshold 60%→50%), retrain classifier ด้วย 709 รูป (val_acc=1.000), flatten folder structure ใน data collect, images/ พร้อม label ครบ 5 class |
| 2026-05-13 | **train_classifier.py tune:** EPOCHS 30→40, BATCH_SIZE 8→16, เพิ่ม RandomPerspective+RandomGrayscale, แก้ WeightedRandomSampler+class weights (แก้ imbalance retail_sachet vs import_sticker), fix val_ds transforms bug |
| 2026-05-13 | **QR code path:** สร้าง pipeline/qr_scanner.py (OpenCV QRCodeDetector, 3-pass decode), แก้ main.py ให้ branch import_sticker→QrScanner / 4 classes→Detector+OCR (import_sticker คือ class ที่มี QR code ไม่ใช่ class แยก), อัปเดต CLAUDE.md + PLAN.md |
| 2026-05-13 | **Phase 3 Done:** ได้ models/detector.pt (YOLO train ด้วย class names ตรงกับ classifier), refactor detector.py ให้ filter YOLO box ตาม image_class (ไม่ใช่ best box ทุก class), heuristic fallback คงเดิม |
| 2026-05-13 | **Phase 5 Done:** validators.py เพิ่ม _LOT_BY_CLASS per class + generic fallback, รองรับ 2-digit year, ocr_engine.py ส่ง image_class เข้า find_lot, tests/test_ocr.py 31 tests passed |
| 2026-05-13 | **Phase 6 Done:** เขียน evaluate.py (วัด accuracy classifier+detector per class), tests/test_integration.py 13/13 passed (/health, /predict schema, classifier accuracy ≥90% ทุก class), เพิ่ม httpx ใน requirements.txt |
| 2026-05-15 | **Phase 7 Done:** สร้าง Artifact Registry repo `ocr-repo`, fix Dockerfile (CPU-only torch + libgl1), build/push ด้วย Cloud Build, deploy Cloud Run (ocr-lot-checker-sa, 2Gi, 2CPU), ทดสอบ production endpoint ผ่านทั้ง back_label และ import_sticker |
