"""Build the combined Full Training notebook (detector + classifier) from the
two proven local notebooks, apply the wizard fixes, upload to Drive, print the
file_id. Run once (and again whenever the source notebooks change).

Fixes baked in:
- Detector: drop the cell that overwrites data.yaml (use publisher-maintained).
- Classifier: auto-discover CLASSES from images/<class>/ instead of a hardcoded list.

Usage: python scripts/build_full_training_notebook.py
"""

import json
import logging
import re
import sys
from pathlib import Path

# Run from repo root regardless of cwd — `python scripts/...` puts scripts/ on
# sys.path, not the repo root, so `services` would otherwise be unimportable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DET_NB = Path("ai_crop_lot.ipynb")
CLS_NB = Path("colab_classify_training.ipynb")
OUT = Path("lot-checker-full-training.ipynb")

CLASSES_RE = re.compile(r"CLASSES\s*=\s*sorted\(\[.*?\]\)", re.DOTALL)
CLASSES_REPLACEMENT = (
    "CLASSES      = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])"
)


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def detector_cells() -> list[dict]:
    nb = json.loads(DET_NB.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _src(c)
        if "yaml_content" in src and "data.yaml" in src:
            continue  # drop the data.yaml-overwrite cell
        out.append(_code(src))
    return out


def classifier_cells() -> list[dict]:
    nb = json.loads(CLS_NB.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = _src(c)
        if "drive.mount" in src or src.strip().startswith("!pip install"):
            continue  # detector section already mounts + installs
        if CLASSES_RE.search(src):
            src = CLASSES_RE.sub(CLASSES_REPLACEMENT, src)
        out.append(_code(src))
    return out


def main() -> None:
    load_dotenv()
    det = detector_cells()
    cls = classifier_cells()
    assert len(det) >= 2, f"expected >=2 detector cells, got {len(det)}"
    assert len(cls) >= 4, f"expected >=4 classifier cells, got {len(cls)}"
    assert any("iterdir()" in _src(c) for c in cls), "CLASSES auto-discover not applied"

    cells = (
        [_md("# Full Training — Detector + Classifier\n\n"
             "กด `Runtime → Run all`. Detector (~60 min) แล้วต่อ Classifier (~30 min).\n"
             "Models เซฟลง Drive: `data check lot/<run>.pt` + "
             "`data classify check lot/models/classifier.pt`.")]
        + [_md("## ── Detector (YOLO) ──")] + det
        + [_md("## ── Classifier (EfficientNet-V2-S) ──")] + cls
    )
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("wrote %s (%d cells)", OUT, len(cells))

    from services.drive_client import DriveClient
    drive = DriveClient()
    file_id = drive.upload_bytes(
        OUT.read_bytes(), name=OUT.name, parent_id="root",
        mime_type="application/vnd.google.colaboratory",
    )
    logger.info("UPLOADED — file_id=%s", file_id)
    logger.info("colab: https://colab.research.google.com/drive/%s", file_id)


if __name__ == "__main__":
    main()
