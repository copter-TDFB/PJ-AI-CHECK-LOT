"""
tests สำหรับ Preprocessor

รัน:
    python -m pytest tests/test_preprocessor.py -v
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.preprocessor import Preprocessor

CLASSES = ["back_label", "container_label", "grade_bag", "import_sticker", "retail_sachet"]


def _make_jpeg(width=300, height=150, color=(220, 220, 220)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_text_jpeg(text_color=0, bg_color=255) -> bytes:
    """สร้างรูปที่มีเส้น text จำลอง (สำหรับทดสอบ deskew)"""
    img = Image.new("L", (300, 100), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 40, 290, 60], fill=text_color)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def preprocessor():
    return Preprocessor()


# ─── Unit tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("image_class", CLASSES)
def test_run_returns_bytes(preprocessor, image_class):
    """run() ต้องคืน bytes ที่เปิดเป็นรูปได้เสมอ"""
    result = preprocessor.run(_make_jpeg(), image_class)
    img = Image.open(io.BytesIO(result))
    assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.parametrize("image_class", CLASSES)
def test_output_is_grayscale(preprocessor, image_class):
    """output ต้องเป็น grayscale (mode L หรือ channel เดียว)"""
    result = preprocessor.run(_make_jpeg(), image_class)
    img = Image.open(io.BytesIO(result))
    arr = np.array(img)
    # grayscale: array shape คือ (H, W) หรือ (H, W, 1) หรือ RGB ที่ R=G=B
    if arr.ndim == 3:
        assert np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2), "ไม่ใช่ grayscale"
    else:
        assert arr.ndim == 2


def test_run_without_class(preprocessor):
    """ไม่ระบุ class ต้องทำงานได้โดยใช้ Otsu threshold"""
    result = preprocessor.run(_make_jpeg())
    assert len(result) > 0


def test_container_label_path(preprocessor):
    """container_label ต้องผ่าน glare removal path โดยไม่ error"""
    result = preprocessor.run(_make_jpeg(), "container_label")
    img = Image.open(io.BytesIO(result))
    assert img.size[0] > 0


def test_output_size_preserved(preprocessor):
    """ขนาด output ต้องใกล้เคียงกับ input (deskew อาจเปลี่ยนเล็กน้อย)"""
    img_bytes = _make_jpeg(400, 200)
    result = preprocessor.run(img_bytes, "back_label")
    out_img = Image.open(io.BytesIO(result))
    assert out_img.size[0] == 400
    assert out_img.size[1] == 200
