import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class PackagingConfig:
    key: str
    display_name: str
    pipeline: str                    # 'detector_ocr' | 'qr_scanner'
    lot_patterns: list[re.Pattern]
    fields_extracted: list[str]      # ['lot', 'exp', 'product', 'size']
    sheet_checks: list[str]          # ['lot', 'exp', 'product', 'sachet']
    post_ocr_fixes: list[str]        # ['lot_alpha_prefix']
    message_template_key: str
    model_classifier_label: str
    detector_yolo_prefixes: list[str]
    conf_threshold: float
    accuracy: float | None           # measured top-1 accuracy on test set (Phase 4 auto-populates)
    gate_on_lot: bool
    lot_short_fallback: bool
    sub_regions: list[str]           # ['box', 'sachet'] for container_label


@dataclass
class MessageTemplate:
    key: str
    ok_message: str
    fail_template: str
    field_labels: dict[str, str]


class PackagingRegistry:
    """Loads and caches packaging configs + message templates from config/."""

    def __init__(self) -> None:
        self._configs: dict[str, PackagingConfig] = {}
        self._templates: dict[str, MessageTemplate] = {}
        self._load()

    def _load(self) -> None:
        pkg_dir = _CONFIG_DIR / "packagings"
        # *.yaml glob skips *.yaml.archived and *.yaml.bak-* by extension
        for path in sorted(pkg_dir.glob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            patterns = [re.compile(p) for p in data.get("lot_patterns", [])]
            cfg = PackagingConfig(
                key=data["key"],
                display_name=data.get("display_name", data["key"]),
                pipeline=data.get("pipeline", "detector_ocr"),
                lot_patterns=patterns,
                fields_extracted=data.get("fields_extracted", []),
                sheet_checks=data.get("sheet_checks", []),
                post_ocr_fixes=data.get("post_ocr_fixes", []),
                message_template_key=data.get("message_template_key", "default_full"),
                model_classifier_label=data.get("model_classifier_label", data["key"]),
                detector_yolo_prefixes=data.get("detector_yolo_prefixes", []),
                conf_threshold=float(data.get("conf_threshold", 0.6)),
                accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
                gate_on_lot=bool(data.get("gate_on_lot", True)),
                lot_short_fallback=bool(data.get("lot_short_fallback", False)),
                sub_regions=data.get("sub_regions", []),
            )
            self._configs[cfg.key] = cfg
            logger.debug("Loaded packaging config: %s", cfg.key)

        tmpl_dir = _CONFIG_DIR / "message_templates"
        for path in sorted(tmpl_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            tmpl = MessageTemplate(
                key=data["key"],
                ok_message=data["ok_message"],
                fail_template=data["fail_template"],
                field_labels=data.get("field_labels", {}),
            )
            self._templates[tmpl.key] = tmpl

        logger.info(
            "PackagingRegistry loaded — %d packagings, %d templates",
            len(self._configs),
            len(self._templates),
        )

    def get(self, key: str) -> PackagingConfig | None:
        return self._configs.get(key)

    def get_template(self, template_key: str) -> MessageTemplate | None:
        return self._templates.get(template_key)

    def all_keys(self) -> list[str]:
        return list(self._configs.keys())

    def is_archived(self, key: str) -> bool:
        """True if there's an archived YAML for this key (active YAML must NOT exist)."""
        if key in self._configs:
            return False
        return (_CONFIG_DIR / "packagings" / f"{key}.yaml.archived").exists()

    def archived_keys(self) -> list[str]:
        """List packagings that are currently archived."""
        pkg_dir = _CONFIG_DIR / "packagings"
        if not pkg_dir.exists():
            return []
        return sorted(
            p.name[: -len(".yaml.archived")]
            for p in pkg_dir.glob("*.yaml.archived")
        )
