"""
Integration tests — ทดสอบ API endpoints ผ่าน FastAPI TestClient
(ไม่ต้องรัน server จริง, ไม่ต้องมี Google Cloud credentials)

รัน: python -m pytest tests/test_integration.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

IMAGES_DIR = Path("images")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
MODELS_OK  = (
    Path("models/classifier.pt").exists() and
    Path("models/detector.pt").exists()
)

needs_models = pytest.mark.skipif(not MODELS_OK, reason="models/*.pt not found")


def _first_image(cls: str) -> bytes | None:
    """คืน bytes ของรูปแรกที่เจอใน images/<cls>/"""
    cls_dir = IMAGES_DIR / cls
    if not cls_dir.exists():
        return None
    for p in cls_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            return p.read_bytes()
    return None


# ─── fixture: TestClient ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """สร้าง TestClient โดย mock OcrEngine (ไม่ต้องมี credentials)"""
    mock_ocr_result = {
        "lot_number": "TEST-LOT-001",
        "confidence": 0.99,
        "raw_text":   "LOT TEST-LOT-001 MFG 01/01/2025 EXP 01/01/2027",
        "mfg_date":   "2025-01-01",
        "exp_date":   "2027-01-01",
        "status":     "ok",
    }
    with patch("pipeline.ocr_engine.vision") as mock_vision:
        # mock Vision API response
        mock_annotation = MagicMock()
        mock_annotation.description = mock_ocr_result["raw_text"]
        mock_response = MagicMock()
        mock_response.error.message = ""
        mock_response.text_annotations = [mock_annotation]
        mock_response.full_text_annotation.pages = []
        mock_vision.ImageAnnotatorClient.return_value.text_detection.return_value = mock_response
        mock_vision.Image = MagicMock(side_effect=lambda content: content)

        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as c:
            yield c


# ─── /health ──────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─── /predict — response schema ───────────────────────────────────────────────

REQUIRED_FIELDS = {"lot_number", "confidence", "class", "class_confidence",
                   "raw_text", "mfg_date", "exp_date", "bbox", "status"}


def _assert_schema(body: dict) -> None:
    """ตรวจว่า response มี field ครบและ status ถูกต้อง"""
    missing = REQUIRED_FIELDS - body.keys()
    assert not missing, f"Missing fields: {missing}"
    assert body["status"] in (
        "ok", "not_found", "qr_not_found", "lot_not_found", "low_confidence"
    )


@needs_models
def test_predict_no_file(client):
    r = client.post("/predict")
    assert r.status_code == 422   # FastAPI validation error


_DUMMY_SHEET_PARAMS = {"sheet_id": "DUMMY_SHEET_ID", "sheet_gid": "0"}


@needs_models
def test_predict_invalid_file(client):
    r = client.post(
        "/predict",
        params=_DUMMY_SHEET_PARAMS,
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


@needs_models
@pytest.mark.parametrize("cls", ["back_label", "container_label", "grade_bag", "retail_sachet"])
def test_predict_ocr_classes(client, cls):
    img = _first_image(cls)
    if img is None:
        pytest.skip(f"No images found for {cls}")
    r = client.post(
        "/predict",
        params=_DUMMY_SHEET_PARAMS,
        files={"file": ("test.jpg", img, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    _assert_schema(body)
    assert body["class"] == cls or body["class"] in (
        "back_label", "container_label", "grade_bag", "retail_sachet", "import_sticker"
    )


@needs_models
def test_predict_import_sticker_uses_qr(client):
    """import_sticker ต้องผ่าน QR scanner — bbox จะเป็น null"""
    img = _first_image("import_sticker")
    if img is None:
        pytest.skip("No images found for import_sticker")
    r = client.post(
        "/predict",
        params=_DUMMY_SHEET_PARAMS,
        files={"file": ("test.jpg", img, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    _assert_schema(body)
    # import_sticker ใช้ QR scanner → bbox ต้องเป็น null
    if body["class"] == "import_sticker":
        assert body["bbox"] is None


@needs_models
def test_predict_unconfigured_class_returns_graceful(client, monkeypatch):
    """A high-confidence prediction for a class with no config (e.g. a classifier
    label that ships ahead of its packaging) must return a graceful response,
    not HTTP 500."""
    import main

    class _FakeClf:
        def predict(self, b):
            return ("print_sticker_full", 0.95)

    monkeypatch.setattr(main, "classifier", _FakeClf())
    monkeypatch.setattr(main.registry, "get", lambda k: None)
    monkeypatch.setattr(main.registry, "is_archived", lambda k: False)
    r = client.post(
        "/predict",
        params=_DUMMY_SHEET_PARAMS,
        files={"file": ("x.jpg", b"imgbytes", "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "unconfigured_class"


# ─── Classifier accuracy smoke test ───────────────────────────────────────────

@needs_models
@pytest.mark.parametrize("cls", ["back_label", "container_label", "grade_bag",
                                  "import_sticker", "retail_sachet"])
def test_classifier_accuracy(cls):
    """ตรวจว่า classifier จัดประเภทรูปตัวอย่างอย่างน้อย 90% ถูก"""
    from pipeline.classifier import ImageClassifier
    clf = ImageClassifier("models/classifier.pt")

    cls_dir = IMAGES_DIR / cls
    if not cls_dir.exists():
        pytest.skip(f"images/{cls}/ not found")

    images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        pytest.skip(f"No images in images/{cls}/")

    # ทดสอบแค่ 20 รูปแรกเพื่อให้ test เร็ว
    sample = images[:20]
    correct = sum(1 for p in sample if clf.predict(p.read_bytes())[0] == cls)
    accuracy = correct / len(sample)

    assert accuracy >= 0.90, (
        f"{cls}: classifier accuracy {accuracy:.1%} < 90% ({correct}/{len(sample)})"
    )
