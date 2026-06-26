# Detector Dataset `data.yaml` Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the Drive detector reference dataset so `data.yaml` and the YOLO label files form a clean, internally consistent 11-class set matching `models/detector.pt` and `config/packagings/*`.

**Architecture:** A single one-off script `scripts/cleanup_detector_dataset.py` with pure, unit-tested helpers (label-id remap, data.yaml builder) plus Drive-side actions guarded behind an explicit `--execute` flag (dry-run is the default). All mutations are preceded by a local backup; deletions go to Drive Trash (recoverable), never permanent. A `verify` mode re-scans the dataset afterward.

**Tech Stack:** Python 3.11, `services.drive_client.DriveClient` (SA key via `GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json`), PyYAML, pytest.

## Global Constraints

- Run from repo root; `load_dotenv(os.path.join(repo, ".env"))` explicitly (not bare `load_dotenv()`).
- Use the module `logger`, never `print()` in `services/`; the script itself may use `logging`/its own CLI output but MUST NOT use `print()` if a hook forbids it — use `logging` configured to stdout.
- Drive identity must be the SA / OAuth user with Editor on the dataset folders (`gcp-key.json`).
- Default mode is **dry-run**. Mutations happen ONLY with `--execute`.
- Deletions use Trash (`files().update(fileId, body={"trashed": True})`), never permanent delete.
- Known folder ids: detector root `1xmhCGoUrhPpDOHGdsewusPn57twkQXEr`; train/labels `14VNccSkH0VU--chNUF_TUqaCMtnEhzXZ`; val/labels `1-9ZELumgXRoTbFJI_xpnZUS4BqaNi0Zv`. Classifier root from `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`.
- Target class set (must match `models/detector.pt` names 0–10 exactly):
  `back_label_lot, back_label_name, back_label_size, capsule_box, container_label_box, container_label_sachet, grade_bag_lot, grade_bag_product, retail_sachet_lot, print_sticker_back_lot_exp, print_sticker_back_product_size`
- Label id remap: `{12: 9, 13: 10}`. Delete (`DELETE_PREFIXES`): all `print_sticker_full_*` AND all `new_tea_bag_box_*` (leftover wizard test-draft, no config/classifier folder — confirmed 24 train image+label pairs at id 9, 0 in val; controller decision 2026-06-26 after dry-run revealed id-9 `new*` files are `new_tea_bag_box`, not `print_sticker_back`).

---

### Task 1: Pure helpers + unit tests

**Files:**
- Create: `scripts/cleanup_detector_dataset.py`
- Create: `tests/test_cleanup_detector_dataset.py`

**Interfaces:**
- Produces:
  - `TARGET_NAMES: list[str]` — the 11 class names in order.
  - `REMAP: dict[int, int]` — `{12: 9, 13: 10}`.
  - `remap_label_text(text: str, mapping: dict[int, int]) -> str` — rewrites the leading integer class id of each non-empty line per `mapping`; lines whose id is not in `mapping` are unchanged; preserves the rest of each line and trailing newline shape.
  - `extract_split_paths(raw_yaml: str) -> tuple[str, str]` — returns `(train_path, val_path)` by reading the `train:`/`val:` lines from raw text (the file does not parse as YAML, so read line-wise).
  - `build_data_yaml(train_path: str, val_path: str, names: list[str]) -> str` — returns a YAML string with keys `train`, `val`, `nc: len(names)`, `names` (block list), parseable by `yaml.safe_load`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cleanup_detector_dataset.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from scripts.cleanup_detector_dataset import (
    TARGET_NAMES, REMAP, remap_label_text, extract_split_paths, build_data_yaml,
)


def test_target_names_has_11_entries():
    assert len(TARGET_NAMES) == 11
    assert TARGET_NAMES[9] == "print_sticker_back_lot_exp"
    assert TARGET_NAMES[10] == "print_sticker_back_product_size"


def test_remap_rewrites_only_leading_id():
    text = "12 0.5 0.5 0.2 0.2\n13 0.1 0.1 0.3 0.3\n"
    out = remap_label_text(text, REMAP)
    assert out == "9 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.3 0.3\n"


def test_remap_leaves_unmapped_ids_untouched():
    text = "0 0.5 0.5 0.2 0.2\n8 0.1 0.1 0.3 0.3\n"
    assert remap_label_text(text, REMAP) == text


def test_remap_skips_blank_lines():
    text = "12 0.5 0.5 0.2 0.2\n\n"
    assert remap_label_text(text, REMAP) == "9 0.5 0.5 0.2 0.2\n\n"


def test_extract_split_paths():
    raw = (
        "train: /content/drive/MyDrive/data check lot/train/images\n"
        "val: /content/drive/MyDrive/data check lot/val/images\n"
        "nc: 16\nnames:\n- back_label_lot\n-test_u\n"
    )
    train, val = extract_split_paths(raw)
    assert train == "/content/drive/MyDrive/data check lot/train/images"
    assert val == "/content/drive/MyDrive/data check lot/val/images"


def test_build_data_yaml_roundtrips():
    out = build_data_yaml("t/images", "v/images", TARGET_NAMES)
    parsed = yaml.safe_load(out)
    assert parsed["nc"] == 11
    assert parsed["names"] == TARGET_NAMES
    assert parsed["train"] == "t/images"
    assert parsed["val"] == "v/images"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cleanup_detector_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (script/functions not defined yet).

- [ ] **Step 3: Write the script with the pure helpers**

```python
# scripts/cleanup_detector_dataset.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cleanup_detector_dataset.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/cleanup_detector_dataset.py tests/test_cleanup_detector_dataset.py
git commit -m "feat(scripts): pure helpers for detector dataset cleanup"
```

---

### Task 2: Drive scan + dry-run plan reporting

**Files:**
- Modify: `scripts/cleanup_detector_dataset.py`

**Interfaces:**
- Consumes: `TARGET_NAMES`, `REMAP`, `remap_label_text`, folder-id constants from Task 1.
- Produces:
  - `connect() -> DriveClient` — loads `.env` from repo root, returns a `DriveClient`, logs the authenticated email.
  - `resolve_image_folders(drive) -> dict[str, str]` — `{"train": <train/images id>, "val": <val/images id>}` via `find_in_folder(DET_ROOT, "train"|"val")` then `find_in_folder(..., "images")`.
  - `list_txt(drive, folder_id) -> list[dict]` — label files (`.txt`) in a folder.
  - `plan_remap(drive) -> list[dict]` — for each `print_sticker_back_*.txt` in train+val labels, return `{id, name, folder, old_ids, new_text}` where `old_ids` are the distinct ids found and `new_text` is the remapped content; skips files already free of remap ids.
  - `plan_delete(drive) -> list[dict]` — all `print_sticker_full_*` files across train/val images+labels (and classifier `images/print_sticker_full` folder id if present) as `{id, name, kind}`.
  - `preflight(drive) -> None` — read up to 3 `new*.txt` files in train labels, log their distinct class ids; assert each contains only id `9`. Raise `SystemExit` with a clear message if any `new*` file contains an id other than 9 (so a human decides before merge).

- [ ] **Step 1: Add Drive helpers and the dry-run planner**

```python
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
            if f["name"].startswith(DELETE_PREFIX):
                plan.append({"id": f["id"], "name": f["name"], "kind": kind})
    # classifier images/<prefix> folder
    cls_root = os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "")
    if cls_root:
        cls_images = drive.find_in_folder(cls_root, "images")
        psf = drive.find_in_folder(cls_images, DELETE_PREFIX) if cls_images else None
        if psf:
            plan.append({"id": psf, "name": f"images/{DELETE_PREFIX}", "kind": "classifier-folder"})
    return plan
```

- [ ] **Step 2: Add a `main()` CLI that runs preflight + prints the dry-run plan**

```python
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
```

- [ ] **Step 3: Run the dry-run and eyeball the plan**

Run: `python scripts/cleanup_detector_dataset.py`
Expected:
- Drive identity logged (a real `@...` email, not blank).
- Pre-flight logs ≤3 `new*` files each with `ids [9]` (no abort).
- `REMAP ~98 label files` listed, every `old_ids` ⊆ `[12, 13]`.
- `DELETE ~102` entries, every name starts with `print_sticker_full`, plus one `classifier-folder` entry if present.
- Ends with `DRY-RUN. Re-run with --execute to apply.`

If any line violates the above (e.g. a remap file with an unexpected id, or a delete entry not prefixed `print_sticker_full`), STOP and report — do not proceed to Task 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleanup_detector_dataset.py
git commit -m "feat(scripts): dry-run planner + preflight for dataset cleanup"
```

---

### Task 3: Backup + apply (remap, trash, rewrite data.yaml)

**Files:**
- Modify: `scripts/cleanup_detector_dataset.py`
- Modify: `.gitignore` (add `dataset-backup/`)

**Interfaces:**
- Consumes: `plan_remap`, `plan_delete`, `build_data_yaml`, `extract_split_paths`, `TARGET_NAMES`.
- Produces:
  - `backup(drive, remap, delete) -> str` — creates `dataset-backup/<stamp>/` (stamp passed in / derived from an arg, NOT `datetime.now()` inside a tested pure fn — here it is fine to use `time.strftime` in this I/O function), downloads current `data.yaml` and every remap+delete file into it, writes a `manifest.json` listing them. Returns the backup dir path.
  - `apply_changes(drive, remap, delete) -> None` — calls `backup`, then for each remap file `update_file_content(id, new_text.encode())`; for each delete entry `drive._svc.files().update(fileId=id, body={"trashed": True}).execute()`; finally rewrites `data.yaml` (the commit point) via `build_data_yaml` using paths from the current `data.yaml` raw text.

- [ ] **Step 1: Add `.gitignore` entry**

Add this line to `.gitignore`:
```
dataset-backup/
```

- [ ] **Step 2: Implement `backup` and `apply_changes`**

```python
def _data_yaml_id(drive):
    yid = drive.find_in_folder(DET_ROOT, "data.yaml")
    if not yid:
        raise SystemExit("ABORT: data.yaml not found in detector root")
    return yid


def backup(drive, remap, delete) -> str:
    import json
    import time
    from pathlib import Path
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bdir = Path(repo) / "dataset-backup" / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    yid = _data_yaml_id(drive)
    drive.download_file(yid, bdir / "data.yaml")
    manifest = {"data_yaml_id": yid, "remap": [], "delete": []}
    for p in remap:
        drive.download_file(p["id"], bdir / "remap" / p["name"])
        manifest["remap"].append({"id": p["id"], "name": p["name"], "folder": p["folder"]})
    for p in delete:
        if p["kind"] != "classifier-folder":  # files only; folder restored from Trash
            drive.download_file(p["id"], bdir / "delete" / f'{p["kind"].replace("/", "_")}__{p["name"]}')
        manifest["delete"].append(p)
    (bdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Backup written to %s", bdir)
    return str(bdir)


def apply_changes(drive, remap, delete) -> None:
    backup(drive, remap, delete)
    for p in remap:
        drive.update_file_content(p["id"], p["new_text"].encode("utf-8"), mime_type="text/plain")
    logger.info("Remapped %d label files", len(remap))
    for p in delete:
        drive._svc.files().update(fileId=p["id"], body={"trashed": True}).execute()
    logger.info("Trashed %d files/folders", len(delete))
    yid = _data_yaml_id(drive)
    raw = drive.read_text(yid)
    train, val = extract_split_paths(raw)
    if not train or not val:
        raise SystemExit(f"ABORT: could not extract train/val from data.yaml (train={train!r} val={val!r})")
    new_yaml = build_data_yaml(train, val, TARGET_NAMES)
    yaml.safe_load(new_yaml)  # sanity: must parse
    drive.update_file_content(yid, new_yaml.encode("utf-8"), mime_type="text/yaml")
    logger.info("Rewrote data.yaml (nc=11). Done.")
```

- [ ] **Step 3: Execute for real**

Run: `python scripts/cleanup_detector_dataset.py --execute`
Expected:
- `Backup written to dataset-backup/<stamp>` (verify the dir exists locally with `data.yaml`, `remap/`, `delete/`, `manifest.json`).
- `Remapped ~98 label files`
- `Trashed ~102 files/folders`
- `Rewrote data.yaml (nc=11). Done.`

- [ ] **Step 4: Commit**

```bash
git add scripts/cleanup_detector_dataset.py .gitignore
git commit -m "feat(scripts): backup + apply for detector dataset cleanup"
```

---

### Task 4: Verify the repaired dataset

**Files:**
- Modify: `scripts/cleanup_detector_dataset.py`

**Interfaces:**
- Consumes: `connect`, `list_txt`, `_distinct_ids`, `resolve_image_folders`, `TARGET_NAMES`, label folder constants.
- Produces:
  - `verify(drive) -> None` — re-scans train+val labels, asserts: every class id is in `0..10`; no file name starts with `print_sticker_full`; `data.yaml` parses with `yaml.safe_load`, `nc == 11`, `names == TARGET_NAMES`. Logs a per-id → packaging-prefix summary. Raises `SystemExit` on any violation.

- [ ] **Step 1: Implement `verify`**

```python
def verify(drive) -> None:
    from collections import defaultdict
    seen = defaultdict(set)
    leftover_full = []
    for folder in (TRAIN_LABELS, VAL_LABELS):
        for f in list_txt(drive, folder):
            if f["name"].startswith(DELETE_PREFIX):
                leftover_full.append(f["name"])
            pfx = f["name"].split("_aug")[0].split(".")[0]
            for cid in _distinct_ids(drive.read_text(f["id"])):
                seen[cid].add(pfx[:24])
    bad_ids = [c for c in seen if c < 0 or c > 10]
    yid = _data_yaml_id(drive)
    doc = yaml.safe_load(drive.read_text(yid))
    problems = []
    if bad_ids:
        problems.append(f"label ids outside 0..10: {sorted(bad_ids)}")
    if leftover_full:
        problems.append(f"print_sticker_full files still present: {leftover_full[:5]}")
    if doc.get("nc") != 11:
        problems.append(f"nc != 11 (got {doc.get('nc')})")
    if doc.get("names") != TARGET_NAMES:
        problems.append("names != TARGET_NAMES")
    for cid in sorted(seen):
        logger.info("  id %2d <- %s", cid, sorted(seen[cid]))
    if problems:
        raise SystemExit("VERIFY FAILED:\n  - " + "\n  - ".join(problems))
    logger.info("VERIFY OK: 11-class dataset is clean and consistent.")
```

- [ ] **Step 2: Run verification**

Run: `python scripts/cleanup_detector_dataset.py --verify`
Expected:
- Per-id summary `id 0..10` each mapping to the expected packaging prefix (9 → print_sticker_back + new, 10 → print_sticker_back).
- No id > 10, no `print_sticker_full` leftovers.
- `VERIFY OK: 11-class dataset is clean and consistent.`

- [ ] **Step 3: Cross-check against the deployed model**

Run:
```bash
python -c "from ultralytics import YOLO; import yaml; n=YOLO('models/detector.pt').names; print('model:', [n[i] for i in range(len(n))])"
```
Expected: the printed list equals `TARGET_NAMES` (model names 0–10 align 1:1 with the new `data.yaml`).

- [ ] **Step 4: Commit**

```bash
git add scripts/cleanup_detector_dataset.py
git commit -m "feat(scripts): verify mode for detector dataset cleanup"
```

---

## Self-Review Notes

- **Spec coverage:** preflight (spec step 0) → Task 2; backup (step 1) → Task 3; remap (step 2) → Tasks 1+3; delete print_sticker_full (step 3) → Tasks 2+3; rewrite data.yaml (step 4) → Task 3; verify (step 5) → Task 4. Rollback → backup dir (Task 3) + Drive Trash. Decisions (delete print_sticker_full, rebuild 11-class) reflected in `DELETE_PREFIX`/`TARGET_NAMES`.
- **Type consistency:** `remap_label_text(text, mapping)`, `build_data_yaml(train, val, names)`, `extract_split_paths(raw)`, `plan_remap`/`plan_delete` dict shapes, and `_distinct_ids` are used consistently across tasks.
- **Out of scope:** root-cause prevention (TEST_MODE publish guard / GC) — separate spec.
