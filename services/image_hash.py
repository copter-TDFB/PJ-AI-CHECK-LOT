"""Perceptual-hash helpers for detecting visually-duplicate images regardless
of filename (used by scripts/detector_dataset_topup.py's dedup step).
"""
from __future__ import annotations

import io

import imagehash
from PIL import Image


def compute_dhash(image_bytes: bytes) -> str:
    """Return the image's dHash as a hex string."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        return str(imagehash.dhash(im))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit distance between two hex dHash strings."""
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
