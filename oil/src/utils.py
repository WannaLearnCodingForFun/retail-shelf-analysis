"""Shared helpers: device selection, seeding, JSON I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """Use CUDA GPU when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Reproducible randomness for splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_class_names(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)


def load_class_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["class_names"]
