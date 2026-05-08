from __future__ import annotations

from collections import Counter


def compute_facings(detections: list[dict], brands: list[str]) -> dict[str, int]:
    c = Counter()
    for d in detections:
        b = str(d.get("detected_brand") or d.get("class_name") or "others")
        if b not in brands:
            b = "others"
        c[b] += 1
    return {b: int(c.get(b, 0)) for b in brands}

