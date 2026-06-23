# Single-stage training (ตัด seed, prelabel ผ่านปุ่ม)

วันที่: 2026-06-12
สถานะ: อนุมัติดีไซน์แล้ว — รอ review spec

## ปัญหา / เป้าหมาย

ปัจจุบัน wizard เทรนเป็น 2 จังหวะ:

1. **Seed training** (step 3) — label ~20 รูป → อัพ Colab เทรน detector หยาบ → ใช้โมเดลนั้น
   auto-prelabel รูปที่เหลือ (active learning) → ค่อย full train
2. **Full training** (step 5) — publish รูปทั้งหมดขึ้น Drive → Colab เทรน detector ตัวจริง → eval → deploy

seed training มีหน้าที่จริงคือ *ลดงาน annotation* ด้วยการ prelabel ให้อัตโนมัติ และ
full training **ไม่ได้พึ่ง** seed อยู่แล้ว (ต้องการแค่ labels)

ต้องการตัดจังหวะ seed ออก เหลือ **เทรนใหญ่ครั้งเดียว** และให้ prelabel ทำงาน
*เฉพาะตอนเพิ่มรูปให้ class เดิม* (edit-draft) ผ่าน detector ตัวที่ deploy อยู่

## การตัดสินใจ (จาก brainstorming)

| หัวข้อ | สรุป |
|--------|------|
| ขอบเขต prelabel | **edit-draft (class เดิม) เท่านั้น** ผ่าน active detector. draft ใหม่ = label เองทั้งหมด |
| Hard gate full train | **30 รูป** (UI ยังแนะนำ 50) |
| Trigger prelabel | **ปุ่มใน step annotation** เรียก backend รัน detector ฝั่ง server |
| โค้ด seed เดิม | **ลบทิ้งทั้งหมด** — แต่ยก inner routine "detector → save annotation" ออกมาก่อนลบ |
| prelabeled bbox | นับเป็น "labeled" ทันที แก้ได้ในตัว annotator (ไม่เพิ่ม state suggestion ใหม่) |

## Flow ใหม่

```
Class ใหม่ (draft):      อัพ 50 → label เอง ≥30 → [step5] full train → eval → deploy
Class เดิม (edit-draft): อัพรูปใหม่ → กด "Prelabel" (active detector ใส่ bbox ให้)
                         → ตรวจ/แก้ → ≥30 → [step5] full train → deploy & overwrite parent
```

Step 3 เหลือ **annotation อย่างเดียว** — ไม่มี Colab seed อีก. ปุ่ม Prelabel โผล่เฉพาะ edit-draft.

## เปลี่ยนแปลงราย component

### 1. Backend — `api/packagings.py`

**ลบ:**
- `training_seed_start` (`POST /{key}/training/seed/start`)
- `training_seed_done` (`POST /{key}/training/seed/done`)
- ค่าคงที่ `_MIN_LABELED_FOR_SEED`
- import / การใช้งาน `training_bundle`

**ก่อนลบ — ยก helper ออกมา:** routine ภายใน `training_seed_done` ที่ทำ
"รัน detector บนรูป → เลือก best box → เขียน annotation" ย้ายไปเป็นฟังก์ชัน
ใช้ซ้ำได้ (เช่น `services/prelabeler.py` หรือ helper ใน module เดิม) — ใช้ทั้งโดย
endpoint prelabel ใหม่

**เพิ่ม:** `POST /{key}/training/prelabel`
- gate: ต้องเป็น edit-draft (`key.endswith("__edit")`) ไม่งั้น `400`
- parent key = `key[:-len("__edit")]`; อ่าน `detector_yolo_prefixes` ของ parent จาก registry
- ใช้ active detector ที่โหลดอยู่ใน app state (ไม่ดาวน์โหลด/ไม่เทรนใหม่)
- รันเฉพาะรูปที่ **ยังไม่ labeled** → filter detection ด้วย prefixes ของ parent → เขียน bbox annotation
- ตอบ `{prelabeled, skipped_already_labeled, errors}` (รูปแบบเดียวกับ seed/done เดิม)

**แก้ gate:** `training_full_start` เปลี่ยนเงื่อนไขจาก `< 10` เป็น `< 30`
(`need at least 30 labeled images`)

### 2. Services

- ลบ `services/training_bundle.py`
- ลบ `notebook_generator.build_seed_notebook` (เหลือ `build_full_notebook`)
- ถ้าแยก helper ออกมา: `services/prelabeler.py` (run active detector → annotations)

### 3. Draft status state machine

- ลบ status `training_seed` ออกจาก `continueDraft` stepMap (`web/wizard.html` ~บรรทัด 2212)
  และทุกที่ที่อ้างถึง `training_seed`
- progression: `draft → uploading → (annotation; status คงที่ uploading) → configured → training_full → trained`
- resume `uploading → step 3` เหมือนเดิม

### 4. Frontend — `web/wizard.html`

> **ต้องผ่าน skill `frontend-design` และ `web-design-guidelines` ตอน implement**
> (คำสั่งผู้ใช้ 2026-06-12) — ให้ดีไซน์ปุ่ม/สถานะ/ข้อความเป็นไปตามมาตรฐาน 2 skill นี้

- Step 3: ลบ block seed training — `#seed-prog`, ปุ่ม "เริ่ม Seed Training",
  ฟังก์ชัน `trainingSeedStart` / `trainingSeedDone`, ปุ่ม `btn-step3-next` ที่ผูกกับ seed
- เพิ่มปุ่ม **"Prelabel อัตโนมัติ"** ใน step 3 เฉพาะเมื่อ `isEditDraft`
  → เรียก `POST /training/prelabel` → toast สรุปผล (`prelabeled / skipped / errors`)
  → refresh annotator ให้เห็น bbox ที่ถูกใส่
  - ต้องมี hover / focus / active / loading / disabled state ที่ตั้งใจ (web-design-guidelines)
- ปรับ annotation target counter `20` → แนะ `50` / ขั้นต่ำ `30`; อัพเดตข้อความ step desc
- ลบ `trainingSeedStart`, `trainingSeedDone`

### 5. Tests — `tests/test_api_packagings.py`

- ลบ/แก้ test ที่อ้าง seed endpoints
- เพิ่ม:
  - `prelabel` กับ edit-draft → ใส่ bbox ให้รูปที่ยังไม่ label, ข้ามรูปที่ label แล้ว
  - `prelabel` กับ draft ใหม่ (ไม่ใช่ `__edit`) → `400`
  - `full/start` gate: 29 labeled → fail (`400`), 30 labeled → ผ่าน gate

### 6. Docs

- ADR 0001 (seed / active-learning) → mark **superseded**
- ADR ใหม่: "single-stage training + prelabel-on-demand for edit-drafts"
- `CLAUDE.md` ส่วน Wizard API: ลบการอ้าง `training_seed`, อัพเดต status list
  และคำอธิบาย flow

## Error handling

- prelabel: ถ้า active detector ยังไม่พร้อม/โหลดไม่ได้ → `503` ข้อความชัดเจน
- prelabel: รูปที่ detector หา box ไม่เจอ → นับเป็น `errors`/skip, ไม่ crash ทั้ง batch
- full/start: < 30 labeled → `400` พร้อมจำนวนที่มีจริง
- prelabel เรียกบน draft ใหม่ → `400` "prelabel ใช้ได้เฉพาะ edit-draft (class เดิม)"

## นอกขอบเขต (YAGNI)

- ไม่ทำ background job / progress bar ให้ prelabel (รันแบบ synchronous เหมือน seed/done เดิม)
- ไม่เพิ่มสถานะ "suggestion ที่ต้องยืนยัน" แยกจาก labeled
- ไม่แตะ classifier training path
