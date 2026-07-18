"""Grow the Drive detector dataset from the local classifier image pool.

Pipeline: dedup (dHash vs Drive's already-published images) -> prelabel
(run models/detector.pt, write YOLO boxes) -> human validates with labelImg
(external, not part of this script) -> publish (upload validated pairs to
Drive train/val, matching the existing detector dataset layout).

Dry-run by default for `publish`; pass --execute to mutate Drive.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("detector_dataset_topup")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOPUP_DIR = REPO / "data" / "detector_topup"
IMAGES_DIR = REPO / "images"
CONFIG_DIR = REPO / "config" / "packagings"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_PRELABEL_CONF = 0.25  # same default as services/active_learning.py


def connect_drive():
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from services.drive_client import DriveClient

    drive = DriveClient()
    try:
        who = drive._svc.about().get(fields="user").execute().get("user", {})
        logger.info("Drive identity: %s", who.get("emailAddress"))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read Drive identity: %s", e)
    return drive


def det_root() -> str:
    root = os.environ.get("DRIVE_DETECTOR_DATASET_FOLDER_ID", "")
    if not root:
        raise SystemExit("DRIVE_DETECTOR_DATASET_FOLDER_ID not set")
    return root


def _images_folder(drive, root: str, split: str) -> str | None:
    sp = drive.find_in_folder(root, split)
    return drive.find_in_folder(sp, "images") if sp else None


def _labels_folder(drive, root: str, split: str) -> str | None:
    sp = drive.find_in_folder(root, split)
    return drive.find_in_folder(sp, "labels") if sp else None


def load_data_yaml(drive, root: str) -> dict:
    yid = drive.find_in_folder(root, "data.yaml")
    if not yid:
        raise SystemExit("data.yaml not found in detector dataset folder")
    return yaml.safe_load(drive.read_text(yid))


def _discover_class_dirs() -> list[Path]:
    if not TOPUP_DIR.exists():
        return []
    return sorted(
        d for d in TOPUP_DIR.iterdir()
        if d.is_dir() and d.name not in ("_all", "_drive_cache") and (d / "images").exists()
    )


def cmd_merge_all(args) -> None:
    """Consolidate every per-class working dir into one shared _all/ folder so
    labelImg can browse all classes in a single pass. Filenames are already
    "{key}_..." (collision-free) and every class shares the same classes.txt
    (the 11 global names), so this is a safe flat merge. Idempotent + additive:
    never overwrites an image/label that's already in _all/ (that could be
    review work already in progress).
    """
    import shutil

    class_dirs = _discover_class_dirs()
    if not class_dirs:
        raise SystemExit(f"no class working dirs found under {TOPUP_DIR} — run 'dedup <key>' first")

    merged_images = TOPUP_DIR / "_all" / "images"
    merged_labels = TOPUP_DIR / "_all" / "labels"
    merged_images.mkdir(parents=True, exist_ok=True)
    merged_labels.mkdir(parents=True, exist_ok=True)

    if not (merged_labels / "classes.txt").exists():
        for class_dir in class_dirs:
            src = class_dir / "labels" / "classes.txt"
            if src.exists():
                shutil.copy2(src, merged_labels / "classes.txt")
                break

    copied_images = copied_labels = 0
    for class_dir in class_dirs:
        img_dir = class_dir / "images"
        labels_dir = class_dir / "labels"
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
                continue
            dest_img = merged_images / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
                copied_images += 1
            label_src = labels_dir / f"{img_path.stem}.txt"
            dest_label = merged_labels / f"{img_path.stem}.txt"
            if label_src.exists() and not dest_label.exists():
                shutil.copy2(label_src, dest_label)
                copied_labels += 1

    total = sum(1 for p in merged_images.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    logger.info("merge-all: %d new images + %d new labels copied in, %d images total -> %s",
                copied_images, copied_labels, total, merged_images)
    logger.info("Review with: labelImg \"%s\" \"%s\" \"%s\"  (switch save format to YOLO)",
                merged_images, merged_labels / "classes.txt", merged_labels)


def class_ids_for_key(key: str, names: list[str]) -> set[int]:
    cfg_path = CONFIG_DIR / f"{key}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    prefixes = cfg.get("detector_yolo_prefixes") or [key]
    return {i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)}


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------
#
# The Drive detector images are NOT reliably named "{key}_...": only
# print_sticker_back (published through dataset_publisher.py) carries that
# prefix. The other 5 classes predate this repo's publish flow (their
# filenames look like Roboflow exports, e.g. "1000017594_jpg.rf.zY...", with
# no class info in the name at all). So "which Drive image belongs to key X"
# can only be answered by reading each label file's class ids — never by
# filename — hence the label-content index below instead of a name filter.

def _hash_cache_path() -> Path:
    return TOPUP_DIR / "_hash_cache.json"


def _load_hash_cache() -> dict:
    p = _hash_cache_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_hash_cache(cache: dict) -> None:
    p = _hash_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_index_path() -> Path:
    return TOPUP_DIR / "_label_index.json"


def _distinct_ids(text: str) -> list[int]:
    ids = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.add(int(s.split()[0]))
        except ValueError:
            pass
    return sorted(ids)


def _read_text_with_retry(drive, file_id: str, attempts: int = 4):
    import time
    last_err = None
    for attempt in range(attempts):
        try:
            return drive.read_text(file_id)
        except Exception as e:  # noqa: BLE001 — Drive downloads flake occasionally over ~500 sequential calls
            last_err = e
            logger.warning("read_text retry %d/%d for %s: %s", attempt + 1, attempts, file_id, e)
            time.sleep(2 * (attempt + 1))
    raise last_err


def _build_label_index(drive, root: str, index: dict) -> dict:
    """Mutates + returns index: {"train/<stem>": {"ids": [...], "image_name": ...,
    "image_id": ...}, "val/<stem>": {...}}. Resumable: entries already present are
    left untouched, so a run interrupted by a transient Drive error can pick up
    where it left off instead of re-downloading everything.
    """
    for split in ("train", "val"):
        img_fid = _images_folder(drive, root, split)
        lbl_fid = _labels_folder(drive, root, split)
        if not img_fid or not lbl_fid:
            continue
        image_by_stem = {Path(f["name"]).stem: f for f in drive.list_folder(img_fid)}
        label_files = [f for f in drive.list_folder(lbl_fid) if f["name"].lower().endswith(".txt")]
        todo = [f for f in label_files if f"{split}/{Path(f['name']).stem}" not in index]
        logger.info("indexing %s: %d/%d label files remaining...", split, len(todo), len(label_files))
        for i, f in enumerate(todo):
            stem = Path(f["name"]).stem
            img = image_by_stem.get(stem)
            if not img:
                logger.warning("label with no matching image, skipping: %s/%s", split, f["name"])
                continue
            ids = _distinct_ids(_read_text_with_retry(drive, f["id"]))
            index[f"{split}/{stem}"] = {"ids": ids, "image_name": img["name"], "image_id": img["id"]}
            if (i + 1) % 50 == 0:
                _label_index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("  %s: %d/%d indexed (checkpoint saved)", split, i + 1, len(todo))
    return index


def _load_or_build_label_index(drive, root: str, refresh: bool = False) -> dict:
    p = _label_index_path()
    index = {}
    if p.exists() and not refresh:
        index = json.loads(p.read_text(encoding="utf-8"))
    index = _build_label_index(drive, root, index)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def cmd_dedup(args) -> None:
    from services.image_hash import compute_dhash, hamming_distance

    key = args.key
    drive = connect_drive()
    root = det_root()
    data_yaml = load_data_yaml(drive, root)
    names = data_yaml.get("names") or []
    key_ids = class_ids_for_key(key, names)

    label_index = _load_or_build_label_index(drive, root, refresh=args.refresh_cache)
    matching = {
        loc: entry for loc, entry in label_index.items()
        if key_ids.intersection(entry["ids"])
    }
    logger.info("Drive has %d images already labeled for '%s' (ids %s)", len(matching), key, sorted(key_ids))

    cache = _load_hash_cache()
    cache_dir = TOPUP_DIR / "_drive_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    drive_hashes = {}
    for loc, entry in matching.items():
        cache_key = f"{key}/{loc}"
        if cache_key not in cache or args.refresh_cache:
            dest = cache_dir / entry["image_name"]
            if not dest.exists():
                drive.download_file(entry["image_id"], dest)
            cache[cache_key] = compute_dhash(dest.read_bytes())
        drive_hashes[loc] = cache[cache_key]
    _save_hash_cache(cache)

    local_dir = IMAGES_DIR / key
    if not local_dir.exists():
        raise SystemExit(f"local images/{key}/ not found")

    out_dir = TOPUP_DIR / key / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = TOPUP_DIR / key / "labels"

    total = dup = new = pruned = 0
    for img_path in sorted(local_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        total += 1
        try:
            h = compute_dhash(img_path.read_bytes())
        except Exception:
            logger.warning("unreadable image, skipping: %s", img_path.name)
            continue
        is_dup = any(hamming_distance(h, dh) <= args.threshold for dh in drive_hashes.values())
        dest = out_dir / f"{key}_{img_path.name}"
        if is_dup:
            dup += 1
            # A prior dedup run (stale cache, or a threshold/logic change) may have
            # already copied this file in before it was recognized as a duplicate.
            # Prune it now so it can't slip into prelabel/publish — unless it already
            # carries a non-empty label, which could mean a human validated it in
            # labelImg; in that case don't silently delete, just flag it.
            if dest.exists():
                stale_label = labels_dir / f"{dest.stem}.txt"
                if stale_label.exists() and stale_label.read_text(encoding="utf-8").strip():
                    logger.warning(
                        "'%s' is now recognized as a Drive duplicate but has a non-empty "
                        "label (possibly human-validated) — leaving in place, review manually: %s",
                        dest.name, stale_label,
                    )
                else:
                    dest.unlink()
                    if stale_label.exists():
                        stale_label.unlink()
                    pruned += 1
            continue
        new += 1
        if not dest.exists():
            dest.write_bytes(img_path.read_bytes())

    logger.info(
        "dedup(%s): %d local, %d duplicate (skipped, %d stale copies pruned), %d new candidates -> %s",
        key, total, dup, pruned, new, out_dir,
    )


# --------------------------------------------------------------------------
# prelabel
# --------------------------------------------------------------------------

def _prelabel_boxes(
    boxes: list[tuple[float, float, float, float, str]],
    class_prefixes: list[str],
) -> list[tuple[float, float, float, float, str]]:
    """Same filter rule as services.active_learning.filter_prelabel_bboxes
    (prefix match + positive area), but keeps the class name instead of
    collapsing it to a generic label — this tool writes real numeric-id YOLO
    lines immediately, with no human relabel-via-wizard-UI step in between.
    """
    kept = []
    for x1, y1, x2, y2, name in boxes:
        if class_prefixes and not any(name.startswith(p) for p in class_prefixes):
            continue
        if x2 > x1 and y2 > y1:
            kept.append((x1, y1, x2, y2, name))
    return kept


def _yolo_line(cid: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = max(0, min(1, ((x1 + x2) / 2) / w))
    cy = max(0, min(1, ((y1 + y2) / 2) / h))
    bw = max(0, min(1, (x2 - x1) / w))
    bh = max(0, min(1, (y2 - y1) / h))
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def cmd_prelabel(args) -> None:
    from PIL import Image

    key = args.key
    cfg_path = CONFIG_DIR / f"{key}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    class_prefixes = cfg.get("detector_yolo_prefixes") or [key]

    model_path = REPO / "models" / "detector.pt"
    if not model_path.exists():
        raise SystemExit(f"detector model not found: {model_path}")

    img_dir = TOPUP_DIR / key / "images"
    if not img_dir.exists():
        raise SystemExit(f"no candidates for '{key}' — run 'dedup {key}' first")
    labels_dir = TOPUP_DIR / key / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    drive = connect_drive()
    data_yaml = load_data_yaml(drive, det_root())
    names = data_yaml.get("names") or []
    name_to_id = {n: i for i, n in enumerate(names)}
    (labels_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    model_names = model.names if hasattr(model, "names") else {}

    prelabeled = no_detection = skipped = unknown_class = 0
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            skipped += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        r = results[0]
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.int().tolist()
            for box, cid in zip(xyxy, cls_ids):
                name = model_names.get(int(cid), "") if isinstance(model_names, dict) else ""
                boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), name))

        kept = _prelabel_boxes(boxes, class_prefixes)
        if not kept:
            no_detection += 1
            continue

        lines = []
        for x1, y1, x2, y2, name in kept:
            if name not in name_to_id:
                unknown_class += 1
                continue
            lines.append(_yolo_line(name_to_id[name], x1, y1, x2, y2, w, h))
        if not lines:
            no_detection += 1
            continue
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        prelabeled += 1

    logger.info(
        "prelabel(%s): %d prelabeled, %d no-detection, %d already-labeled (skipped), "
        "%d boxes with unmapped class name",
        key, prelabeled, no_detection, skipped, unknown_class,
    )
    logger.info(
        "Review with: labelImg \"%s\" \"%s\" \"%s\"  (switch save format to YOLO)",
        img_dir, labels_dir / "classes.txt", labels_dir,
    )


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------

def _class_source_dirs(key: str) -> tuple[Path, Path, str]:
    """Prefer the merged _all/ folder (post merge-all, filtered by "{key}_"
    prefix) if it exists — that's where a merged-review edit actually lands —
    else fall back to the per-class working dir."""
    merged_images = TOPUP_DIR / "_all" / "images"
    if merged_images.exists():
        return merged_images, TOPUP_DIR / "_all" / "labels", f"{key}_"
    return TOPUP_DIR / key / "images", TOPUP_DIR / key / "labels", ""


def _load_confirmed(labels_dir: Path) -> dict:
    """Mirrors detector_annotator.py's confirmed-state file — publish only
    ever uploads images a human explicitly clicked Confirm/Reject on in the
    annotator, never just whatever AI prelabel happened to guess."""
    p = labels_dir / "_confirmed.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def cmd_publish(args) -> None:
    from services.dataset_publisher import _resolve_dest_folders, split_for

    key = args.key
    img_dir, labels_dir, prefix = _class_source_dirs(key)
    if not img_dir.exists():
        raise SystemExit(f"no candidates for '{key}' — run 'dedup {key}' first")

    confirmed = _load_confirmed(labels_dir)

    drive = connect_drive()
    root = det_root()
    data_yaml = load_data_yaml(drive, root)

    dest = _resolve_dest_folders(drive, root, data_yaml)
    existing_files = {fid: {f["name"] for f in drive.list_folder(fid)} for fid in set(dest.values())}

    items = []
    empty = not_confirmed = 0
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        if prefix and not img_path.name.startswith(prefix):
            continue
        if not confirmed.get(img_path.name, False):
            not_confirmed += 1
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
            empty += 1
            continue
        items.append((img_path, label_path))

    logger.info(
        "publish(%s): %d confirmed+validated, %d empty/rejected (excluded), "
        "%d NOT YET confirmed in the annotator (excluded — go confirm them first)",
        key, len(items), empty, not_confirmed,
    )

    uploaded = skipped = 0
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start: batch_start + args.batch_size]
        for img_path, label_path in batch:
            name = img_path.name  # already "{key}_{original_name}" from the dedup step
            lbl_name = f"{Path(name).stem}.txt"
            split = split_for(name)
            img_fid = dest[(split, "images")]
            lbl_fid = dest[(split, "labels")]
            img_exists = name in existing_files[img_fid]
            lbl_exists = lbl_name in existing_files[lbl_fid]
            if img_exists and lbl_exists:
                skipped += 1
                continue
            logger.info("  [%s] %s -> %s/images(+labels)",
                        "UPLOAD" if args.execute else "would upload", name, split)
            if args.execute:
                if not img_exists:
                    drive.upload_file(img_path, parent_id=img_fid, name=name)
                    existing_files[img_fid].add(name)
                if not lbl_exists:
                    drive.upload_bytes(label_path.read_bytes(), name=lbl_name,
                                        parent_id=lbl_fid, mime_type="text/plain")
                    existing_files[lbl_fid].add(lbl_name)
                uploaded += 1
        logger.info("batch %d-%d done (uploaded=%d, skipped-existing=%d so far)",
                    batch_start, batch_start + len(batch), uploaded, skipped)

    if not args.execute:
        logger.info("DRY-RUN. Re-run with --execute to actually upload to Drive.")
    else:
        logger.info("publish(%s) complete: uploaded=%d skipped(existing)=%d", key, uploaded, skipped)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def cmd_status(args) -> None:
    key = args.key
    local_dir = IMAGES_DIR / key
    cand_dir, labels_dir, prefix = _class_source_dirs(key)

    def _matches(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in _IMG_EXTS and (not prefix or p.name.startswith(prefix))

    def _count_images(d: Path, filtered: bool) -> int:
        if not d.exists():
            return 0
        pred = _matches if filtered else (lambda p: p.is_file() and p.suffix.lower() in _IMG_EXTS)
        return sum(1 for p in d.iterdir() if pred(p))

    # local_dir (images/<key>/) is always single-class with unprefixed
    # filenames — never filter it by "{key}_", that only applies to cand_dir
    # once it's the shared _all/ folder holding every class together.
    local_count = _count_images(local_dir, filtered=False)
    cand_count = _count_images(cand_dir, filtered=True)
    confirmed = _load_confirmed(labels_dir)
    labeled = empty = confirmed_count = ready_to_publish = 0
    if cand_dir.exists():
        for p in cand_dir.iterdir():
            if not _matches(p):
                continue
            lp = labels_dir / f"{p.stem}.txt"
            has_label = lp.exists() and bool(lp.read_text(encoding="utf-8").strip())
            if has_label:
                labeled += 1
            elif lp.exists():
                empty += 1
            is_confirmed = confirmed.get(p.name, False)
            if is_confirmed:
                confirmed_count += 1
                if has_label:
                    ready_to_publish += 1

    print(f"{key}: local={local_count}  candidates(post-dedup)={cand_count}  "
          f"has-label={labeled}  empty-label={empty}  not-yet-prelabeled={cand_count - labeled - empty}  "
          f"confirmed-by-human={confirmed_count}/{cand_count}  ready-to-publish={ready_to_publish}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dedup = sub.add_parser("dedup", help="dHash-dedup local images/<key>/ against Drive's published set")
    p_dedup.add_argument("key")
    p_dedup.add_argument("--threshold", type=int, default=5,
                          help="max Hamming distance to call a duplicate (default 5/64 bits)")
    p_dedup.add_argument("--refresh-cache", action="store_true",
                          help="re-scan Drive even if cached hashes already exist for this key")
    p_dedup.set_defaults(func=cmd_dedup)

    p_pre = sub.add_parser("prelabel", help="run models/detector.pt on candidates, write YOLO labels")
    p_pre.add_argument("key")
    p_pre.add_argument("--conf", type=float, default=_PRELABEL_CONF)
    p_pre.set_defaults(func=cmd_prelabel)

    p_pub = sub.add_parser("publish", help="upload validated image+label pairs to Drive train/val")
    p_pub.add_argument("key")
    p_pub.add_argument("--execute", action="store_true", help="apply changes (default: dry-run)")
    p_pub.add_argument("--batch-size", type=int, default=50)
    p_pub.set_defaults(func=cmd_publish)

    p_status = sub.add_parser("status", help="show candidate/label/publish counts for a class")
    p_status.add_argument("key")
    p_status.set_defaults(func=cmd_status)

    p_merge = sub.add_parser("merge-all", help="consolidate every class's candidates into one shared _all/ folder for one-pass labelImg review")
    p_merge.set_defaults(func=cmd_merge_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
