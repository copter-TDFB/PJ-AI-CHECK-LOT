"""
tests สำหรับ ImageClassifier

รัน:
    python -m pytest tests/test_classifier.py -v
"""

import pathlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

CLASSES = ["back_label", "container_label", "grade_bag", "import_sticker", "retail_sachet"]


def _fake_checkpoint():
    """สร้าง checkpoint จำลองสำหรับ test ที่ไม่มี weights จริง"""
    from torchvision import models
    import torch.nn as nn

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return {"model_state": model.state_dict(), "classes": CLASSES}


@pytest.fixture()
def classifier(tmp_path):
    """สร้าง ImageClassifier จาก weights จำลอง"""
    from pipeline.classifier import ImageClassifier

    weights_path = tmp_path / "classifier.pt"
    torch.save(_fake_checkpoint(), weights_path)
    return ImageClassifier(weights_path)


# ─── Unit tests ───────────────────────────────────────────────────────────────

def test_predict_returns_valid_class(classifier):
    """predict คืน class ที่อยู่ใน CLASSES และ confidence ในช่วง [0, 1]"""
    img_bytes = pathlib.Path("images/back_label").glob("*.jpg")
    sample = next(img_bytes, None)
    if sample is None:
        pytest.skip("ไม่มีรูปตัวอย่างใน images/back_label/")

    class_name, confidence = classifier.predict(sample.read_bytes())

    assert class_name in CLASSES
    assert 0.0 <= confidence <= 1.0


def test_predict_output_types(classifier):
    """predict คืน (str, float) เสมอ"""
    import io
    from PIL import Image

    # สร้างรูป dummy 224x224 ขาว
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    class_name, confidence = classifier.predict(buf.getvalue())

    assert isinstance(class_name, str)
    assert isinstance(confidence, float)


def test_missing_weights_raises(tmp_path):
    """FileNotFoundError เมื่อ weights ไม่มีอยู่"""
    from pipeline.classifier import ImageClassifier

    with pytest.raises(FileNotFoundError):
        ImageClassifier(tmp_path / "nonexistent.pt")


def test_all_classes_present(classifier):
    """classes ที่ load มาต้องครบ 5 class"""
    assert sorted(classifier.classes) == sorted(CLASSES)
