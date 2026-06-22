"""Guards that the Full Training notebook source emits eval.json + a fixed
detector filename so the /sync endpoint can find them."""

import json
from pathlib import Path


def test_detector_notebook_writes_eval_json_and_fixed_name():
    nb = json.loads(Path("ai_crop_lot.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"]
                    if c["cell_type"] == "code")
    assert "full_detector.pt" in src, "detector must save to fixed full_detector.pt"
    assert "eval.json" in src, "notebook must write eval.json"
    assert "detector_mAP_50" in src, "eval.json must carry detector_mAP_50"
