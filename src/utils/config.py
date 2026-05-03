import os
from pathlib import Path
from typing import Any

import yaml

from .paths import repo_root, resolve_under


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("paths", {})
    cfg["paths"]["root"] = str(root)
    env_dev = os.environ.get("SHAMPOO_DEVICE")
    if env_dev:
        cfg.setdefault("device", {})["prefer"] = env_dev
    return cfg


def data_path(cfg: dict[str, Any], key: str) -> Path:
    rel = cfg["paths"][key]
    return resolve_under(Path(cfg["paths"]["root"]), rel)
