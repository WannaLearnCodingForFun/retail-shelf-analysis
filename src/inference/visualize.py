from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils.env_health import preferred_device


@dataclass(frozen=True)
class PredictSpec:
    weights: Path
    image: Path
    out_path: Path
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device_prefer: str = "auto"


def predict_and_render(spec: PredictSpec) -> Path:
    if not spec.weights.exists():
        raise FileNotFoundError(f"Missing weights: {spec.weights}")
    if not spec.image.exists():
        raise FileNotFoundError(f"Missing image: {spec.image}")

    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    device = preferred_device(spec.device_prefer)
    model = YOLO(str(spec.weights))
    res = model.predict(
        source=str(spec.image),
        conf=spec.conf,
        iou=spec.iou,
        imgsz=spec.imgsz,
        device=device,
        verbose=False,
    )[0]

    img = res.plot()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(spec.out_path), bgr)
    return spec.out_path

