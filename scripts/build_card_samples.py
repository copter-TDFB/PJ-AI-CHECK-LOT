"""Generate small card thumbnails for the wizard dashboard from local images/<key>/.

Writes web/samples/<key>/0.jpg.. (downscaled) and emits a `BAKED_SAMPLES` JS const
on stdout to paste into web/wizard.html. Run from the repo root:

    python scripts/build_card_samples.py > /tmp/baked.js

print() is avoided per repo convention; output goes through sys.stdout/sys.stderr.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

SRC = REPO / "images"
OUT = REPO / "web" / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_W = 320
PER_CLASS = 3


def main() -> None:
    manifest: dict[str, list[str]] = {}
    if not SRC.exists():
        sys.stderr.write(f"no local images dir at {SRC}\n")
        return
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        key = class_dir.name
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )[:PER_CLASS]
        if not imgs:
            continue
        dest_dir = OUT / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, src in enumerate(imgs):
            im = Image.open(src).convert("RGB")
            if im.width > MAX_W:
                im = im.resize((MAX_W, round(im.height * MAX_W / im.width)))
            dest = dest_dir / f"{i}.jpg"
            im.save(dest, "JPEG", quality=70)
            paths.append(f"samples/{key}/{i}.jpg")
        manifest[key] = paths
        sys.stderr.write(f"  {key}: {len(paths)} thumbs\n")

    sys.stdout.write(
        "const BAKED_SAMPLES = "
        + json.dumps(manifest, indent=2, ensure_ascii=False)
        + ";\n"
    )


if __name__ == "__main__":
    main()
