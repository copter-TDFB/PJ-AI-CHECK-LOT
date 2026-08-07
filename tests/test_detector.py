"""
tests สำหรับ RegionDetector

รัน:
    python -m pytest tests/test_detector.py -v
"""

import io
import pathlib

import pytest
from PIL import Image

from pipeline.detector import RegionDetector

CLASSES = ["back_label", "container_label", "grade_bag", "import_sticker", "retail_sachet"]


def _make_jpeg(width: int = 400, height: int = 600, color=(200, 200, 200)) -> bytes:
    """สร้าง JPEG dummy ขนาดที่กำหนด"""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def detector():
    return RegionDetector()


# ─── Unit tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("image_class", CLASSES)
def test_crop_returns_bytes(detector, image_class):
    """crop คืน bytes ที่เปิดเป็น image ได้เสมอ"""
    result = detector.crop(_make_jpeg(), image_class)
    img = Image.open(io.BytesIO(result.cropped_bytes))
    assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.parametrize("image_class", CLASSES)
def test_crop_smaller_than_original(detector, image_class):
    """รูปที่ crop ต้องมีพื้นที่ ≤ รูปต้นฉบับ"""
    original = _make_jpeg(400, 600)
    result = detector.crop(original, image_class)

    orig_img = Image.open(io.BytesIO(original))
    crop_img = Image.open(io.BytesIO(result.cropped_bytes))

    orig_area = orig_img.size[0] * orig_img.size[1]
    crop_area = crop_img.size[0] * crop_img.size[1]
    assert crop_area <= orig_area


@pytest.mark.parametrize("image_class", [c for c in CLASSES if c not in ("container_label", "import_sticker")])
def test_bbox_valid(detector, image_class):
    """bbox ต้อง [x1, y1, x2, y2] โดย x2>x1 และ y2>y1"""
    result = detector.crop(_make_jpeg(), image_class)
    assert result.bbox is not None
    x1, y1, x2, y2 = result.bbox
    assert x2 > x1
    assert y2 > y1


def test_unknown_class_returns_full_image(detector):
    """class ที่ไม่รู้จักต้องคืน full image (bbox=None)"""
    img_bytes = _make_jpeg()
    result = detector.crop(img_bytes, "unknown_class")
    assert result.bbox is None


def test_real_images_per_class(detector):
    """ทดสอบกับรูปจริงใน images/<class>/ (ถ้ามี)"""
    for image_class in CLASSES:
        img_dir = pathlib.Path("images") / image_class
        samples = list(img_dir.glob("*.jpg"))[:2]
        if not samples:
            continue
        for img_path in samples:
            result = detector.crop(img_path.read_bytes(), image_class)
            crop_img = Image.open(io.BytesIO(result.cropped_bytes))
            assert crop_img.size[0] > 0, f"{image_class}: cropped width = 0"
            assert crop_img.size[1] > 0, f"{image_class}: cropped height = 0"


def test_30_sachet_never_crops_to_foreign_class_box(detector):
    """30_sachet has zero trained YOLO boxes of its own — crop_all() must
    always fall through to the full-image heuristic (bbox=None), never
    accept a box trained for a different packaging class (e.g. grade_bag)."""
    if detector._model is None:
        pytest.skip("models/detector.pt not available")

    img_dir = pathlib.Path("images") / "30_sachet"
    samples = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("aug_")]
    if not samples:
        pytest.skip("no real (non-aug_) 30_sachet training photos available")

    offenders = []
    for img_path in samples:
        results = detector.crop_all(img_path.read_bytes(), "30_sachet")
        assert len(results) == 1
        if results[0].bbox is not None:
            offenders.append((img_path.name, results[0].bbox, results[0].class_name))

    assert not offenders, f"crop_all hijacked by a foreign box: {offenders}"


def test_capsule_box_still_crops_to_its_own_class(detector):
    """capsule_box has no '_lot' suffix — its class name matches image_class
    exactly. This must keep working after the 'use every box' fallback is
    removed (it now goes through an explicit exact-match check instead)."""
    if detector._model is None:
        pytest.skip("models/detector.pt not available")

    img_dir = pathlib.Path("images") / "capsule_box"
    samples = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("aug_")]
    if not samples:
        pytest.skip("no real (non-aug_) capsule_box training photos available")

    for img_path in samples:
        results = detector.crop_all(img_path.read_bytes(), "capsule_box")
        assert len(results) == 1
        assert results[0].bbox is not None, f"{img_path.name}: expected a crop, got full image"
        assert results[0].class_name == "capsule_box", f"{img_path.name}: matched {results[0].class_name!r} instead"
