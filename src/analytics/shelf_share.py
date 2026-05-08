from __future__ import annotations

from collections import defaultdict


def compute_shelf_share(detections: list[dict], class_names: list[str]) -> dict[str, float]:
    area_by_class: dict[str, float] = defaultdict(float)
    total = 0.0
    for d in detections:
        cls_id = int(d["class_id"])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else "others"
        w = float(d["width"])
        h = float(d["height"])
        area = max(0.0, w * h)
        area_by_class[cls_name] += area
        total += area
    if total <= 0:
        return {c: 0.0 for c in class_names}
    return {c: round((area_by_class.get(c, 0.0) / total) * 100.0, 2) for c in class_names}


def compute_per_detection_share(detections: list[dict]) -> list[dict]:
    total = sum(max(0.0, float(d.get("width", 0.0)) * float(d.get("height", 0.0))) for d in detections)
    if total <= 0:
        for d in detections:
            d["shelf_share_area"] = 0.0
        return detections
    for d in detections:
        area = max(0.0, float(d.get("width", 0.0)) * float(d.get("height", 0.0)))
        d["shelf_share_area"] = round((area / total) * 100.0, 2)
    return detections

