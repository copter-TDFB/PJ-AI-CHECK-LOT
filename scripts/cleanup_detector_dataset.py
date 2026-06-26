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
# Test-draft / no-config classes to remove entirely (images + labels + any
# classifier folder). print_sticker_full has no config; new_tea_bag_box is a
# leftover wizard test-draft (no config, no classifier folder).
DELETE_PREFIXES = ["print_sticker_full", "new_tea_bag_box"]
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


def connect():
    from dotenv import load_dotenv
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo, ".env"))
    sys.path.insert(0, repo)
    from services.drive_client import DriveClient
    drive = DriveClient()
    try:
        who = drive._svc.about().get(fields="user").execute().get("user", {})
        logger.info("Drive identity: %s", who.get("emailAddress"))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read Drive identity: %s", e)
    return drive


def resolve_image_folders(drive) -> dict[str, str]:
    out = {}
    for split in ("train", "val"):
        sp = drive.find_in_folder(DET_ROOT, split)
        out[split] = drive.find_in_folder(sp, "images") if sp else None
    return out


def list_txt(drive, folder_id):
    return [f for f in drive.list_folder(folder_id) if f["name"].lower().endswith(".txt")]


def _distinct_ids(text: str) -> set[int]:
    ids = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.add(int(s.split()[0]))
        except ValueError:
            pass
    return ids


def preflight(drive) -> None:
    news = [f for f in list_txt(drive, TRAIN_LABELS) if f["name"].startswith(NEW_PREFIX)][:3]
    logger.info("Pre-flight: inspecting %d 'new*' label files", len(news))
    for f in news:
        ids = _distinct_ids(drive.read_text(f["id"]))
        logger.info("  %s -> ids %s", f["name"], sorted(ids))
        bad = ids - {9}
        if bad:
            raise SystemExit(
                f"ABORT: {f['name']} contains non-9 ids {sorted(bad)} — "
                "'new*' files are not pure print_sticker_back_lot_exp. Decide manually."
            )


def plan_remap(drive):
    plan = []
    for folder in (TRAIN_LABELS, VAL_LABELS):
        for f in list_txt(drive, folder):
            if not f["name"].startswith("print_sticker_back"):
                continue
            text = drive.read_text(f["id"])
            ids = _distinct_ids(text)
            if not (ids & set(REMAP)):
                continue
            plan.append({
                "id": f["id"], "name": f["name"], "folder": folder,
                "old_ids": sorted(ids), "new_text": remap_label_text(text, REMAP),
            })
    return plan


def plan_delete(drive):
    plan = []
    images = resolve_image_folders(drive)
    folders = {
        "train/labels": TRAIN_LABELS, "val/labels": VAL_LABELS,
        "train/images": images["train"], "val/images": images["val"],
    }
    for kind, fid in folders.items():
        if not fid:
            continue
        for f in drive.list_folder(fid):
            if any(f["name"].startswith(p) for p in DELETE_PREFIXES):
                plan.append({"id": f["id"], "name": f["name"], "kind": kind})
    # classifier images/<prefix> folder(s)
    cls_root = os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "")
    if cls_root:
        cls_images = drive.find_in_folder(cls_root, "images")
        for prefix in DELETE_PREFIXES:
            psf = drive.find_in_folder(cls_images, prefix) if cls_images else None
            if psf:
                plan.append({"id": psf, "name": f"images/{prefix}", "kind": "classifier-folder"})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="apply changes (default: dry-run)")
    ap.add_argument("--verify", action="store_true", help="re-scan and validate only")
    args = ap.parse_args()
    drive = connect()
    if args.verify:
        verify(drive)
        return
    preflight(drive)
    remap = plan_remap(drive)
    delete = plan_delete(drive)
    logger.info("REMAP %d label files (12->9, 13->10):", len(remap))
    for p in remap:
        logger.info("  [%s] %s old_ids=%s", p["folder"][:6], p["name"], p["old_ids"])
    logger.info("DELETE %d files/folders (to Trash):", len(delete))
    for p in delete:
        logger.info("  [%s] %s", p["kind"], p["name"])
    if not args.execute:
        logger.info("DRY-RUN. Re-run with --execute to apply.")
        return
    apply_changes(drive, remap, delete)  # defined in Task 3


if __name__ == "__main__":
    main()
