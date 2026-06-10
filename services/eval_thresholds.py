"""Load + check hard-floor eval thresholds (admin-controlled)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PATH = Path("config/eval_thresholds.yaml")
_cache: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        if not _PATH.exists():
            _cache = {"hard_floor": {}, "recommended": {}}
        else:
            _cache = yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}
    return _cache


def reload() -> dict[str, Any]:
    """Force re-read (useful in tests)."""
    global _cache
    _cache = None
    return load()


def check_hard_floor(eval_data: dict[str, Any]) -> dict[str, Any]:
    """Compare eval metrics vs hard_floor — return {passed, failures, thresholds}."""
    cfg = load()
    floor = cfg.get("hard_floor", {})
    failures: list[dict[str, Any]] = []
    for key, min_val in floor.items():
        actual = eval_data.get(key)
        if actual is None:
            failures.append({"metric": key, "actual": None, "required": min_val, "reason": "missing"})
        elif float(actual) < float(min_val):
            failures.append({"metric": key, "actual": float(actual), "required": min_val})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "hard_floor": floor,
        "recommended": cfg.get("recommended", {}),
    }
