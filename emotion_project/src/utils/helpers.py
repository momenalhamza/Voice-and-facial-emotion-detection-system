"""Shared helpers: device selection, seeding, checkpoint I/O, early stopping."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(preference: str = "auto") -> torch.device:
    """Pick GPU when available unless explicitly forced to CPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Make a training run as reproducible as PyTorch allows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str,
) -> Path:
    """Persist a model + optimizer + epoch state to checkpoints/."""
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    torch.save(state, out_path)
    return out_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a saved checkpoint dict."""
    return torch.load(Path(path), map_location=map_location)


@dataclass
class EarlyStopper:
    """Classic patience-based early stopping on a 'higher is better' metric.

    Returns True from `step()` when training should stop.
    """

    patience: int
    min_delta: float = 0.0
    _best: float = float("-inf")
    _bad_epochs: int = 0

    def step(self, metric: float) -> bool:
        if metric > self._best + self.min_delta:
            self._best = metric
            self._bad_epochs = 0
            return False
        self._bad_epochs += 1
        return self._bad_epochs >= self.patience

    @property
    def best(self) -> float:
        return self._best


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
