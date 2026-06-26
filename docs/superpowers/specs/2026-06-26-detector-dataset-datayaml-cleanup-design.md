# Detector Dataset `data.yaml` Cleanup — Design

**Date:** 2026-06-26
**Status:** Approved
**Scope:** Repair the detector training dataset on Google Drive so `data.yaml` and the
YOLO label files are internally consistent and match the current packaging config
(7 packagings → 11 detector classes). This is a **data-repair task on the Drive
reference dataset only**; it does not touch the running production model.

## Problem

The detector reference dataset lives in Drive under `DRIVE_DETECTOR_DATASET_FOLDER_ID`
(`data check lot`, folder id `1xmhCGoUrhPpDOHGdsewusPn57twkQXEr`). Its `data.yaml`
declares `nc: 16` with this `names` list (raw, as stored):

```
0  back_label_lot
1  back_label_name
2  back_label_size
3  capsule_box
4  container_label_box
5  container_label_sachet
6  grade_bag_lot
7  grade_bag_product
8  retail_sachet_lot
9  print_sticker_back_lot_exp
10 print_sticker_back_product_size
-test_u           # malformed: no space after "-"  (would be index 11)
-test_y           # malformed                       (index 12)
-test_u           # malformed, duplicate            (index 13)
14 print_sticker_full_lot_exp
15 print_sticker_full_product_size
```

The three `-test_*` lines are not valid YAML list items (missing space after `-`),
so the file **fails to parse with PyYAML** (`ScannerError`).

A full scan of all 565 label files (451 train + 114 val) maps each numeric class id
to the packaging its label files actually belong to:

| id | name in `data.yaml` | label files actually using id | verdict |
|----|---------------------|-------------------------------|---------|
| 0–2 | back_label_lot/name/size | back_label | correct |
| 3 | capsule_box | capsule_box | correct |
| 4–5 | container_label_box/sachet | container_label | correct |
| 6–7 | grade_bag_lot/product | grade_bag | correct |
| 8 | retail_sachet_lot | retail_sachet | correct |
| 9 | print_sticker_back_lot_exp | old `new*` files (24) | name correct, old upload set |
| 10 | print_sticker_back_product_size | — (0 refs) | dead class |
| 11 | `-test_u` (malformed) | — (0 refs) | junk |
| 12 | `-test_y` (malformed) | **print_sticker_back** (49) | **WRONG NAME** — really print_sticker_back |
| 13 | `-test_u` (malformed, dup) | **print_sticker_back** (49) | **WRONG NAME** — really print_sticker_back |
| 14–15 | print_sticker_full_lot_exp/size | print_sticker_full (51 each) | name correct, **no config exists** |

**Core breakage:** the real print_sticker_back images (ids 12,13, ~98 files) are named
as test garbage (`-test_y` / `-test_u`) in `data.yaml`. A future detector retrain would
label print_sticker_back regions as `-test_y`/`-test_u`, and the YAML would not even parse.

**Root cause:** `services/dataset_publisher.merge_class_names` is append-only (numeric
label ids forbid reordering). When print_sticker_back was re-published (via edit-draft),
it appended new ids (12,13) instead of reusing 9,10, and earlier TEST_MODE publishes
leaked junk classes (`-test_*`, indices 11–13) into the production dataset. There is no
garbage-collection / reconcile path.

**Risk surface:** the deployed `detector.pt` (11 classes, 0–10, correct) is unaffected —
it only matters at the next retrain. So this repair carries no production-serving risk.

## Decisions (confirmed with user)

1. **print_sticker_full** (ids 14,15, ~102 label files, no `config/packagings` entry) →
   **delete** (move to Trash). It is not usable without a config; promoting it later is a
   separate effort.
2. **Cleanup approach** → **Rebuild to a clean 11-class dataset** that matches the deployed
   `detector.pt` and the current configs exactly (not a name-only patch, which would leave
   duplicate `print_sticker_back_lot_exp` at ids 9 and 12 plus index gaps).

## Target `data.yaml`

```yaml
train: /content/drive/MyDrive/data check lot/train/images
val: /content/drive/MyDrive/data check lot/val/images
nc: 11
names:
- back_label_lot            # 0
- back_label_name           # 1
- back_label_size           # 2
- capsule_box               # 3
- container_label_box       # 4
- container_label_sachet    # 5
- grade_bag_lot             # 6
- grade_bag_product         # 7
- retail_sachet_lot         # 8
- print_sticker_back_lot_exp       # 9
- print_sticker_back_product_size  # 10
```

(Existing top-level keys other than `names`/`nc` are preserved verbatim.)

## Operations

All operations run from the repo root via a one-off Python script using
`services.drive_client.DriveClient` with the SA key
(`GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json`, `load_dotenv(<repo>/.env)`).
Known folder ids: detector root `1xmhCGoUrhPpDOHGdsewusPn57twkQXEr`;
train/labels `14VNccSkH0VU--chNUF_TUqaCMtnEhzXZ`;
val/labels `1-9ZELumgXRoTbFJI_xpnZUS4BqaNi0Zv`.

0. **Pre-flight verify** — read 2–3 `new*` label files (id 9) and confirm they are genuine
   print_sticker_back lot/exp crops (single box). If they look unrelated, STOP and surface
   to the user before merging them into class 9.
1. **Backup** — download the current `data.yaml` and every label file that will be modified
   or deleted into a local `dataset-backup/<timestamp>/` tree (gitignored) before any
   mutation. Record the touched file list.
2. **Remap print_sticker_back labels** — for each `print_sticker_back_*.txt` in train+val
   labels (~98 files; scan confirms they contain only ids 12,13), rewrite each line's leading
   class id `12→9`, `13→10`, then `update_file_content`. This merges them with the `new*`
   files already at id 9, giving 9 = lot_exp, 10 = product_size.
3. **Delete print_sticker_full** — move to Trash (recoverable 30 days, not permanent delete)
   all `print_sticker_full_*` images+labels in train and val, plus the classifier folder
   `data classify check lot/images/print_sticker_full/` if present.
4. **Rewrite `data.yaml`** LAST (the commit point) — 11 names, `nc: 11`, must parse with
   PyYAML. Preserve `train`/`val`/other keys.
5. **Verify** — re-run the id→prefix scan. Expect only ids 0–10, each mapping to exactly one
   packaging, no print_sticker_full, no `-test_*`. Confirm `data.yaml` parses and
   `nc == len(names) == 11`.

## Rollback

- `data.yaml` and all modified labels are in the local backup → re-upload to restore.
- print_sticker_full files are in Drive Trash → restore from Trash.

## Out of Scope (follow-up spec)

Root-cause prevention is deferred: guard so TEST_MODE never publishes into the real dataset
folders, and/or a reconcile/GC capability for `merge_class_names`. Tracked separately.

## Verification / Success Criteria

- `data.yaml` parses cleanly; `nc: 11`, 11 names matching the target list above.
- Label scan shows class ids 0–10 only; each id maps to exactly one packaging.
- No `print_sticker_full_*` files remain in detector train/val or classifier images.
- `models/detector.pt` (11 classes 0–10) names align 1:1 with the new `data.yaml` names.
