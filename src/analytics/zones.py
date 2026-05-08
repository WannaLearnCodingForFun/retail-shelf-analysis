from __future__ import annotations


def shelf_zone(y_center: float, image_h: float) -> str:
    if image_h <= 0:
        return "middle"
    t1 = image_h / 3.0
    t2 = 2.0 * image_h / 3.0
    if y_center < t1:
        return "top"
    if y_center < t2:
        return "middle"
    return "bottom"


def zone_stats(detections: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for d in detections:
        b = str(d.get("detected_brand") or d.get("class_name") or "others")
        z = str(d.get("shelf_zone") or "middle")
        out.setdefault(b, {"top": 0, "middle": 0, "bottom": 0})
        out[b][z] = int(out[b].get(z, 0)) + 1
    return out

