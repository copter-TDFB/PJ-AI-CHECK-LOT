"""Run a single image through the REAL /predict code path (classifier -> registry ->
PipelineRunner), unlike test_image.py which bypasses the registry and skips config=.

Usage: python scripts/run_real_pipeline.py <image_path>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from pipeline.classifier import ImageClassifier
from pipeline.detector import RegionDetector
from pipeline.ocr_engine import OcrEngine
from pipeline.packaging_registry import PackagingRegistry
from pipeline.pipeline_runner import PipelineRunner
from pipeline.preprocessor import Preprocessor
from services import config_overrides, model_registry

_DEFAULT_CONF_THRESHOLD = 0.6


def main(image_path: str) -> None:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    registry = PackagingRegistry(overrides=config_overrides.load())
    clf_path, det_path = model_registry.sync()
    classifier = ImageClassifier(clf_path)
    detector = RegionDetector(det_path)
    preprocessor = Preprocessor()
    ocr_engine = OcrEngine()
    runner = PipelineRunner(detector, preprocessor, ocr_engine, qr_scanner=None)

    image_class, class_confidence = classifier.predict(image_bytes)
    print(f"class={image_class} class_confidence={class_confidence:.4f}")

    config = registry.get(image_class)
    if config is None:
        archived = registry.is_archived(image_class)
        print(f"NO CONFIG for class '{image_class}' (archived={archived}) — pipeline not run")
        return

    conf_threshold = config.conf_threshold if config.conf_threshold else _DEFAULT_CONF_THRESHOLD
    print(f"conf_threshold={conf_threshold} detection_mode={config.detection_mode}")
    if class_confidence < conf_threshold:
        force = "--force" in sys.argv
        print("LOW CONFIDENCE — production would skip the pipeline here" + (" (forcing anyway)" if force else ""))
        if not force:
            return

    result, bbox = runner.run(image_bytes, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("bbox:", bbox)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_real_pipeline.py <image_path> [--force]")
        sys.exit(1)
    main(sys.argv[1])
