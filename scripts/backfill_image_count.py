"""Backfill `image_count` into packaging YAMLs so the dashboard list/detail no longer
counts dataset images via Drive on every request (the cold-start dashboard stall).

  local (default) : count images/<key>/ and append `image_count: N` to each
                    config/packagings/<key>.yaml that lacks it. Non-destructive
                    (appends a root key; keeps comments). Use --force to overwrite.
  gcs             : copy each LOCAL yaml's image_count into its GCS counterpart
                    (packagings/<key>.yaml) so PROD picks it up. Pure text op —
                    no Drive, no model. Needs GCS access (GCS_CONFIG_BUCKET).

Run from the repo root:
    python scripts/backfill_image_count.py            # local YAMLs
    python scripts/backfill_image_count.py --force     # re-count + overwrite local
    python scripts/backfill_image_count.py gcs         # propagate local values to GCS

print() is avoided per repo convention; output goes through sys.stdout/sys.stderr.
"""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PKG_DIR = REPO / "config" / "packagings"
IMAGES_DIR = REPO / "images"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _count_images(key: str) -> int | None:
    d = IMAGES_DIR / key
    if not d.exists():
        return None
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def backfill_local(force: bool) -> None:
    for path in sorted(PKG_DIR.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key = data.get("key", path.stem)
        if data.get("image_count") is not None and not force:
            sys.stderr.write(f"  {key}: already has image_count={data['image_count']} — skip\n")
            continue
        count = _count_images(key)
        if count is None:
            sys.stderr.write(f"  {key}: no images/{key}/ dir — skip\n")
            continue
        if force and data.get("image_count") is not None:
            # rewrite in place via regex-free reparse is risky for comments; do a
            # line replace instead so formatting/comments survive.
            lines = path.read_text(encoding="utf-8").splitlines()
            out = [f"image_count: {count}" if line.strip().startswith("image_count:") else line
                   for line in lines]
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\nimage_count: {count}\n")
        sys.stderr.write(f"  {key}: image_count = {count}\n")


def backfill_gcs() -> None:
    from services import gcs_store

    store = gcs_store.get_store()
    if store is None:
        sys.stderr.write("no GCS store (GCS_CONFIG_BUCKET unset) — nothing to do\n")
        return

    for path in sorted(PKG_DIR.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        local = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key = local.get("key", path.stem)
        count = local.get("image_count")
        if count is None:
            sys.stderr.write(f"  {key}: local yaml has no image_count — run local mode first; skip\n")
            continue
        gcs_path = f"{gcs_store.PACKAGINGS_PREFIX}{key}.yaml"
        text = store.get_text(gcs_path)
        if text is None:
            sys.stderr.write(f"  {key}: not published to GCS — skip\n")
            continue
        gdata = yaml.safe_load(text) or {}
        if gdata.get("image_count") == count:
            sys.stderr.write(f"  {key}: GCS already image_count={count} — skip\n")
            continue
        if "image_count" in gdata:
            lines = [f"image_count: {count}" if ln.strip().startswith("image_count:") else ln
                     for ln in text.splitlines()]
            new_text = "\n".join(lines) + "\n"
        else:
            new_text = text.rstrip("\n") + f"\nimage_count: {count}\n"
        store.put_text(gcs_path, new_text, content_type="text/yaml")
        sys.stderr.write(f"  {key}: GCS image_count = {count} (written)\n")


def main() -> None:
    args = sys.argv[1:]
    if "gcs" in args:
        backfill_gcs()
    else:
        backfill_local(force="--force" in args)


if __name__ == "__main__":
    main()
