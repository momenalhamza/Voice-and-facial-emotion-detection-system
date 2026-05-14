"""Tiny config loader.

`load_config()` reads configs/config.yaml from the project root and returns a
nested dict. Sub-sections are accessible via attribute style (`cfg.face.lr`)
through the `Box` wrapper for ergonomic training scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Box(dict):
    """dict that also supports attribute access (one level deep, recursive)."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        for k, v in data.items():
            self[k] = Box(v) if isinstance(v, dict) else v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


# Project root = parent of `src/`
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> Box:
    """Load YAML config from disk and wrap in a Box for dotted access."""
    path = Path(path) if path else CONFIG_PATH
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    return Box(raw)


def resolve_path(relative: str | Path) -> Path:
    """Resolve any config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
