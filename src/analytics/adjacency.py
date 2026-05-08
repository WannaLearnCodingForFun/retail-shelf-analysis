from __future__ import annotations

from collections import Counter


def _center(d: dict) -> tuple[float, float]:
    return (float(d["x1"]) + float(d["x2"])) / 2.0, (float(d["y1"]) + float(d["y2"])) / 2.0


def adjacency_counts(detections: list[dict], max_neighbors: int = 1, dist_norm: float = 0.75) -> dict[str, dict[str, int]]:
    if len(detections) < 2:
        return {}
    pairs = Counter()
    centers = [_center(d) for d in detections]
    sizes = [max(float(d["width"]), float(d["height"]), 1.0) for d in detections]

    for i, di in enumerate(detections):
        bi = str(di.get("detected_brand") or di.get("class_name") or "others")
        cxi, cyi = centers[i]
        neigh: list[tuple[float, int]] = []
        for j, dj in enumerate(detections):
            if i == j:
                continue
            cxj, cyj = centers[j]
            dx = cxi - cxj
            dy = cyi - cyj
            d2 = dx * dx + dy * dy
            thresh = (dist_norm * (sizes[i] + sizes[j]) / 2.0) ** 2
            if d2 <= thresh:
                neigh.append((d2, j))
        neigh.sort(key=lambda x: x[0])
        for _, j in neigh[:max_neighbors]:
            bj = str(detections[j].get("detected_brand") or detections[j].get("class_name") or "others")
            if bi == bj:
                continue
            a, b = sorted([bi, bj])
            pairs[(a, b)] += 1

    out: dict[str, dict[str, int]] = {}
    for (a, b), n in pairs.items():
        out.setdefault(a, {})
        out[a][b] = int(n)
    return out

