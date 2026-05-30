import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from utils.image_utils import bytes_to_pil

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageClassifier:
    """โหลด EfficientNet-V2-S weights และ predict class ของรูปภาพ"""

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.classes: list[str] = checkpoint["classes"]

        self._model = self._load_model(len(self.classes))
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

        logger.info("ImageClassifier loaded: %s (%d classes)", model_path, len(self.classes))

    @staticmethod
    def _load_model(num_classes: int) -> nn.Module:
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Classify รูปภาพ

        Args:
            image_bytes: raw image bytes

        Returns:
            (class_name, confidence) เช่น ("import_sticker", 0.9821)
        """
        img: Image.Image = bytes_to_pil(image_bytes)
        tensor = _TRANSFORM(img).unsqueeze(0)  # shape: (1, 3, 384, 384)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        class_idx = int(probs.argmax())
        confidence = round(float(probs[class_idx]), 4)
        class_name = self.classes[class_idx]

        logger.info("Classified as %s (confidence=%.4f)", class_name, confidence)
        return class_name, confidence
