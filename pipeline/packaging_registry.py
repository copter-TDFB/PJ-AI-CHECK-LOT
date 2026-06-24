import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _config_dir() -> Path:
    """Config root — `OCR_CONFIG_DIR` env (test harness) or the repo `config/`."""
    override = os.getenv("OCR_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR


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
    detection_mode: str = "single"   # 'single' | 'cross_check' | 'multi_field'
    product_aliases: list[dict] = field(default_factory=list)
    # [{'canonical': str, 'keywords': [str]}] — config-driven product matching
    image_count: int | None = None   # dataset image count snapshot — avoids a per-request
    # Drive lookup on the dashboard list/detail; None → endpoints fall back to a live count


@dataclass
class MessageTemplate:
    key: str
    ok_message: str
    fail_template: str
    field_labels: dict[str, str]


class PackagingRegistry:
    """Loads and caches packaging configs + message templates from config/."""

    def __init__(self, overrides: dict[str, dict] | None = None) -> None:
        """`overrides` — runtime tuning values merged over the YAML
        (`conf_threshold`, `product_aliases`), see ADR 0004."""
        self._configs: dict[str, PackagingConfig] = {}
        self._templates: dict[str, MessageTemplate] = {}
        self._load(overrides or {})

    @staticmethod
    def _merged_conf_threshold(data: dict, override: object) -> float:
        yaml_value = float(data.get("conf_threshold", 0.6))
        if not isinstance(override, dict) or "conf_threshold" not in override:
            return yaml_value
        try:
            return float(override["conf_threshold"])
        except (TypeError, ValueError):
            logger.warning(
                "Invalid conf_threshold override for %s: %r — using YAML value",
                data.get("key"), override["conf_threshold"],
            )
            return yaml_value

    @staticmethod
    def _merged_product_aliases(data: dict, override: object) -> list:
        yaml_value = data.get("product_aliases", [])
        if not isinstance(override, dict) or "product_aliases" not in override:
            return yaml_value
        ov = override["product_aliases"]
        if not isinstance(ov, list):
            logger.warning(
                "Invalid product_aliases override for %s: %r — using YAML value",
                data.get("key"), ov,
            )
            return yaml_value
        return ov

    def _config_from_data(self, data: dict, override: object) -> PackagingConfig:
        patterns = [re.compile(p) for p in data.get("lot_patterns", [])]
        return PackagingConfig(
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
            conf_threshold=self._merged_conf_threshold(data, override),
            accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
            gate_on_lot=bool(data.get("gate_on_lot", True)),
            lot_short_fallback=bool(data.get("lot_short_fallback", False)),
            sub_regions=data.get("sub_regions", []),
            detection_mode=data.get("detection_mode", "single"),
            product_aliases=self._merged_product_aliases(data, override),
            image_count=int(data["image_count"]) if data.get("image_count") is not None else None,
        )

    def _overlay_from_gcs(self, overrides: dict[str, dict]) -> None:
        """Overlay durable config from GCS on top of the baked-in image config.

        GCS is authoritative when present: per-key YAMLs override/add classes,
        and `archived` keys are tombstoned (removed). Never raises — a GCS
        problem must not take the service down; we keep the image config.
        """
        from services import gcs_store

        store = gcs_store.get_store()
        if store is None:
            return
        try:
            manifest = store.read_json(gcs_store.MANIFEST_PATH) or {}
            for key in manifest.get("archived", []):
                self._configs.pop(key, None)
            for key in manifest.get("packagings", {}):
                text = store.get_text(f"{gcs_store.PACKAGINGS_PREFIX}{key}.yaml")
                if text is None:
                    logger.warning("GCS manifest lists %s but YAML missing — skipping", key)
                    continue
                data = yaml.safe_load(text)
                cfg = self._config_from_data(data, overrides.get(data["key"]))
                if cfg.key != key:
                    logger.warning(
                        "GCS YAML at packagings/%s.yaml declares key=%s — using declared key",
                        key, cfg.key,
                    )
                self._configs[cfg.key] = cfg
                logger.info("Overlaid packaging config from GCS: %s", cfg.key)
        except Exception as e:
            logger.warning("GCS config overlay failed — using image config only: %s", e)

    def _load(self, overrides: dict[str, dict]) -> None:
        pkg_dir = _config_dir() / "packagings"
        # *.yaml glob skips *.yaml.archived and *.yaml.bak-* by extension
        for path in sorted(pkg_dir.glob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg = self._config_from_data(data, overrides.get(data["key"]))
            self._configs[cfg.key] = cfg
            logger.debug("Loaded packaging config: %s", cfg.key)

        self._overlay_from_gcs(overrides)

        tmpl_dir = _config_dir() / "message_templates"
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
        return (_config_dir() / "packagings" / f"{key}.yaml.archived").exists()

    def archived_keys(self) -> list[str]:
        """List packagings that are currently archived."""
        pkg_dir = _config_dir() / "packagings"
        if not pkg_dir.exists():
            return []
        return sorted(
            p.name[: -len(".yaml.archived")]
            for p in pkg_dir.glob("*.yaml.archived")
        )
