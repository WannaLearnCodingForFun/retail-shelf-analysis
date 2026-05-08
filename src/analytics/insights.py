from __future__ import annotations


def generate_insights(
    shelf_share: dict[str, float],
    facings: dict[str, int] | None = None,
    zone_stats: dict[str, dict[str, int]] | None = None,
    adjacency: dict[str, dict[str, int]] | None = None,
    avg_conf: float | None = None,
    low_conf_count: int | None = None,
) -> dict:
    ordered = sorted(shelf_share.items(), key=lambda x: x[1], reverse=True)
    non_zero = [(k, v) for k, v in ordered if v > 0]
    dominant_brand, dominant_share = non_zero[0] if non_zero else ("others", 0.0)

    observations: list[str] = []
    if len(non_zero) >= 2:
        gap = round(non_zero[0][1] - non_zero[1][1], 2)
        observations.append(f"{non_zero[0][0]} leads by {gap} pts over {non_zero[1][0]}.")

    if dominant_share < 30:
        observations.append("Category is fragmented (no clear shelf leader).")
    elif dominant_share > 60:
        observations.append("High dominance: one brand controls most visible shelf space.")
    else:
        observations.append("Moderate dominance with meaningful competition.")

    if facings:
        top_f = max(facings.items(), key=lambda x: x[1])[0] if facings else "others"
        observations.append(f"Most facings: {top_f} ({facings.get(top_f, 0)}).")

    middle_leader = None
    if zone_stats:
        middle_counts = {b: z.get("middle", 0) for b, z in zone_stats.items()}
        if middle_counts:
            middle_leader = max(middle_counts.items(), key=lambda x: x[1])[0]
            observations.append(f"Middle shelf leader: {middle_leader}.")

    clustered = None
    if adjacency:
        flat = [(a, b, n) for a, m in adjacency.items() for b, n in m.items()]
        if flat:
            a, b, n = max(flat, key=lambda t: t[2])
            clustered = a
            observations.append(f"Frequent adjacency: {a} ↔ {b} ({n}).")

    if avg_conf is not None:
        observations.append(f"Average confidence: {avg_conf:.2f}.")
    if low_conf_count is not None and low_conf_count > 0:
        observations.append(f"Low-confidence detections: {low_conf_count}.")

    summary = f"Dominant brand: {dominant_brand} ({dominant_share:.2f}% shelf share)."
    return {
        "dominant_brand": dominant_brand,
        "dominant_share": dominant_share,
        "middle_shelf_leader": middle_leader,
        "most_clustered_brand": clustered,
        "distribution": ordered,
        "observations": observations,
        "summary": summary,
    }

