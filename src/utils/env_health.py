import sys
from pathlib import Path

import numpy as np
import torch


def preferred_device(spec: str) -> str:
    s = spec.lower().strip()
    if s == "cuda":
        return "0" if torch.cuda.is_available() else "cpu"
    if s == "cpu":
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def ensure_layout(root: Path, cfg_paths: dict) -> None:
    keys = ("data_raw", "data_processed", "data_external", "models_weights", "models_configs", "outputs_predictions", "outputs_metrics")
    for k in keys:
        (root / cfg_paths[k]).mkdir(parents=True, exist_ok=True)


def run_health_check() -> int:
    from src.utils.config import load_config

    cfg = load_config()
    root = Path(cfg["paths"]["root"])
    ensure_layout(root, cfg["paths"])

    errors: list[str] = []

    torch_ver = getattr(torch, "__version__", "unknown")
    cuda_avail = torch.cuda.is_available()
    device_str = preferred_device(cfg.get("device", {}).get("prefer", "auto"))
    try:
        from ultralytics import YOLO
    except ImportError as e:
        errors.append(f"ultralytics import failed: {e}")

    print(f"torch {torch_ver} | cuda_available={cuda_avail} | yolo_device={device_str}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    # Minimal forward pass validates CPU/GPU exec path without project weights
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model = YOLO("yolov8n.pt")
    _ = model.predict(source=dummy, verbose=False, device=device_str, imgsz=320)
    print("YOLOv8 smoke predict: ok")
    return 0


if __name__ == "__main__":
    sys.exit(run_health_check())
