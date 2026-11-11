from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _default_config_path() -> Path:
    packaged = Path(__file__).with_name("default.yaml")
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return loaded


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration and merge it recursively over defaults."""

    default_path = _default_config_path()
    config = _read_yaml(default_path) if default_path.exists() else {}
    if path is not None:
        config = _deep_merge(config, _read_yaml(Path(path)))
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected configuration section '{key}' to be a mapping")
    return value
