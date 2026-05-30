import io

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


def bytes_to_pil(data: bytes) -> Image.Image:
    """แปลง raw bytes เป็น PIL Image (RGB)"""
    return Image.open(io.BytesIO(data)).convert('RGB')


def bytes_to_numpy(data: bytes) -> np.ndarray:
    """แปลง raw bytes เป็น numpy array (RGB uint8, shape HxWx3)"""
    return np.array(bytes_to_pil(data))


def stack_images_vertically(images: list[bytes], gap: int = 30) -> bytes:
    """
    รวม grayscale JPEG bytes หลายรูปเป็นรูปเดียวในแนวตั้ง
    คั่นด้วย white gap เพื่อไม่ให้ text ของแต่ละ crop ชนกัน
    ถ้า images มีแค่ 1 รูป คืนค่าเดิมโดยไม่แตะ
    """
    if len(images) == 1:
        return images[0]

    arrays = [np.array(Image.open(io.BytesIO(b)).convert('L')) for b in images]
    max_w = max(a.shape[1] for a in arrays)

    padded: list[np.ndarray] = []
    for arr in arrays:
        if arr.shape[1] < max_w:
            pad = np.full((arr.shape[0], max_w - arr.shape[1]), 255, dtype=np.uint8)
            arr = np.hstack([arr, pad])
        padded.append(arr)

    separator = np.full((gap, max_w), 255, dtype=np.uint8)
    stacked = padded[0]
    for arr in padded[1:]:
        stacked = np.vstack([stacked, separator, arr])

    buf = io.BytesIO()
    Image.fromarray(stacked).save(buf, format="JPEG", quality=90)
    return buf.getvalue()
