"""Bake the drawer's "Sample images" (originals + detector crops + bbox) into the
frontend so opening a packaging card on prod needs NO Drive download and NO model.

For each active packaging's first PER_CLASS images it:
  - reuses the card original  web/samples/<key>/<i>.jpg  (regenerates if missing),
  - runs the REAL detector once offline → writes crop JPEGs web/samples/<key>/<i>.crop<j>.jpg,
  - scales each bbox into the *baked* original's pixel space (originals are downscaled),
  - emits a `BAKED_DRAWER_SAMPLES` JS const on stdout to paste into web/wizard.html.

Mirrors the live path (api/packagings.get_samples → _resolve_pipeline + detector.crop_all):
qr_scanner classes get regions=[] (no crop), exactly like the drawer's live response.

Run from the repo root (needs torch/ultralytics + models/detector.pt):

    python scripts/build_drawer_samples.py > scratchpad/baked_drawer.js

print() is avoided per repo convention; output goes through sys.stdout/sys.stderr.
"""

import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from pipeline.detector import RegionDetector  # noqa: E402
from pipeline.packaging_registry import PackagingRegistry  # noqa: E402

SRC = REPO / "images"
OUT = REPO / "web" / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_W = 320        # must match build_card_samples.py so baked originals line up
CROP_MAX_W = 240
PER_CLASS = 3


def _save_downscaled(im: Image.Image, dest: Path, max_w: int) -> int:
    """Save RGB JPEG capped at max_w. Returns the written width."""
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=72)
    return im.width


def main() -> None:
    if not SRC.exists():
        sys.stderr.write(f"no local images dir at {SRC}\n")
        return

    detector = RegionDetector(os.getenv("MODEL_DETECTOR_PATH", "models/detector.pt"))
    registry = PackagingRegistry()

    manifest: dict[str, list[dict]] = {}
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        key = class_dir.name
        cfg = registry.get(key)
        if cfg is None:
            sys.stderr.write(f"  {key}: no active config — skip\n")
            continue
        pipeline = cfg.pipeline
        sub_regions = cfg.sub_regions or []

        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )[:PER_CLASS]
        if not imgs:
            continue

        dest_dir = OUT / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []

        for i, src in enumerate(imgs):
            full = Image.open(src).convert("RGB")
            full_w = full.width

            # Reuse the card original; regenerate if missing so coords are known.
            orig_dest = dest_dir / f"{i}.jpg"
            if orig_dest.exists():
                with Image.open(orig_dest) as o:
                    baked_w = o.width
            else:
                baked_w = _save_downscaled(full, orig_dest, MAX_W)

            scale = baked_w / full_w
            regions: list[dict] = []

            if pipeline != "qr_scanner":
                try:
                    dets = detector.crop_all(src.read_bytes(), key)
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"  {key}/{i}: detector failed: {e}\n")
                    dets = []
                for j, det in enumerate(dets):
                    crop_dest = dest_dir / f"{i}.crop{j}.jpg"
                    with Image.open(io.BytesIO(det.cropped_bytes)) as cim:
                        _save_downscaled(cim.convert("RGB"), crop_dest, CROP_MAX_W)
                    bbox = None
                    if det.bbox:
                        x1, y1, x2, y2 = det.bbox
                        bbox = [round(x1 * scale), round(y1 * scale),
                                round(x2 * scale), round(y2 * scale)]
                    regions.append({
                        "src": f"samples/{key}/{i}.crop{j}.jpg",
                        "label": sub_regions[j] if j < len(sub_regions) else None,
                        "bbox": bbox,
                    })

            entries.append({"orig": f"samples/{key}/{i}.jpg", "regions": regions})

        manifest[key] = entries
        sys.stderr.write(f"  {key}: {len(entries)} samples, "
                         f"{sum(len(e['regions']) for e in entries)} crops\n")

    sys.stdout.write(
        "const BAKED_DRAWER_SAMPLES = "
        + json.dumps(manifest, indent=2, ensure_ascii=False)
        + ";\n"
    )


if __name__ == "__main__":
    main()
