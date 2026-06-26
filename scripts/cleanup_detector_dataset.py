"""One-off repair of the Drive detector reference dataset (see
docs/superpowers/specs/2026-06-26-detector-dataset-datayaml-cleanup-design.md).

Dry-run by default; pass --execute to mutate Drive. Deletions go to Trash.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("cleanup_detector_dataset")

TARGET_NAMES: list[str] = [
    "back_label_lot",
    "back_label_name",
    "back_label_size",
    "capsule_box",
    "container_label_box",
    "container_label_sachet",
    "grade_bag_lot",
    "grade_bag_product",
    "retail_sachet_lot",
    "print_sticker_back_lot_exp",
    "print_sticker_back_product_size",
]
REMAP: dict[int, int] = {12: 9, 13: 10}

DET_ROOT = "1xmhCGoUrhPpDOHGdsewusPn57twkQXEr"
TRAIN_LABELS = "14VNccSkH0VU--chNUF_TUqaCMtnEhzXZ"
VAL_LABELS = "1-9ZELumgXRoTbFJI_xpnZUS4BqaNi0Zv"
DELETE_PREFIX = "print_sticker_full"
NEW_PREFIX = "new"


def remap_label_text(text: str, mapping: dict[int, int]) -> str:
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        parts = stripped.split(maxsplit=1)
        try:
            cid = int(parts[0])
        except ValueError:
            out_lines.append(line)
            continue
        if cid in mapping:
            rest = (" " + parts[1]) if len(parts) > 1 else ""
            newline_suffix = line[len(line.rstrip("\r\n")):]
            out_lines.append(f"{mapping[cid]}{rest}{newline_suffix}")
        else:
            out_lines.append(line)
    return "".join(out_lines)


def extract_split_paths(raw_yaml: str) -> tuple[str, str]:
    train = val = ""
    for line in raw_yaml.splitlines():
        if line.startswith("train:"):
            train = line.split(":", 1)[1].strip()
        elif line.startswith("val:"):
            val = line.split(":", 1)[1].strip()
    return train, val


def build_data_yaml(train_path: str, val_path: str, names: list[str]) -> str:
    doc = {"train": train_path, "val": val_path, "nc": len(names), "names": names}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
