from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    conf: float
    color: tuple[int, int, int]  # BGR


def _brand_color(name: str) -> tuple[int, int, int]:
    palette = {
        "head_shoulders": (255, 128, 0),
        "pantene": (40, 40, 200),
        "dove": (200, 200, 60),
        "sunsilk": (0, 215, 255),
        "others": (140, 140, 140),
    }
    return palette.get(name, palette["others"])


def render_boxes(image_bgr: np.ndarray, detections: list[dict], line: int | None = None) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    lw = line if line is not None else max(2, int(round(min(h, w) / 450)))

    for d in detections:
        brand = str(d.get("detected_brand") or d.get("class_name") or "others")
        conf = float(d.get("confidence", 0.0))
        x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
        color = _brand_color(brand)

        p1 = (max(0, int(x1)), max(0, int(y1)))
        p2 = (min(w - 1, int(x2)), min(h - 1, int(y2)))
        cv2.rectangle(out, p1, p2, color, lw)

        txt = f"{brand} | {int(round(conf * 100))}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_txt = max(th + 6, p1[1])
        cv2.rectangle(out, (p1[0], y_txt - th - 6), (p1[0] + tw + 8, y_txt), color, -1)
        cv2.putText(out, txt, (p1[0] + 4, y_txt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out

