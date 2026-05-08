from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

from src.analytics.adjacency import adjacency_counts
from src.analytics.facings import compute_facings
from src.analytics.insights import generate_insights
from src.analytics.shelf_share import compute_per_detection_share, compute_shelf_share
from src.analytics.zones import shelf_zone, zone_stats
from src.inference.annotate import render_boxes
from src.utils.config import load_config
from src.utils.paths import repo_root
from src.utils.env_health import preferred_device

_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _clip_assets(device: str):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _CLIP_MODEL.eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _presence_prompt_bank() -> dict[str, list[str]]:
    return {
        "head_shoulders": [
            "a bottle of Head & Shoulders shampoo",
            "Head & Shoulders shampoo products on a shelf",
            "a supermarket shelf containing Head & Shoulders shampoo",
        ],
        "pantene": [
            "a bottle of Pantene shampoo",
            "Pantene shampoo products on a shelf",
            "a supermarket shelf containing Pantene shampoo",
        ],
        "dove": [
            "a bottle of Dove shampoo",
            "Dove shampoo products on a shelf",
            "a supermarket shelf containing Dove shampoo",
        ],
        "sunsilk": [
            "a bottle of Sunsilk shampoo",
            "Sunsilk shampoo products on a shelf",
            "a supermarket shelf containing Sunsilk shampoo",
        ],
        "others": [
            "a bottle of miscellaneous shampoo",
            "miscellaneous shampoo products on a shelf",
            "a supermarket shelf containing miscellaneous shampoo brands",
        ],
    }


def _clip_scores_for_image(img_bgr: np.ndarray, labels: list[str], prompt_bank: dict[str, list[str]], device: str) -> dict[str, float]:
    model, proc = _clip_assets(device)
    texts: list[str] = []
    owners: list[str] = []
    for label in labels:
        for t in prompt_bank[label]:
            texts.append(t)
            owners.append(label)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ins = proc(text=texts, images=rgb, return_tensors="pt", padding=True)
    ins = {k: v.to(device) if hasattr(v, "to") else v for k, v in ins.items()}
    with torch.inference_mode():
        out = model(**ins)
        logits = out.logits_per_image[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    by_label: dict[str, list[float]] = {l: [] for l in labels}
    for i, p in enumerate(probs.tolist()):
        by_label[owners[i]].append(float(p))
    return {l: float(np.mean(v)) if v else 0.0 for l, v in by_label.items()}


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, s) for s in scores.values()))
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: float(max(0.0, v) / total) for k, v in scores.items()}


def _softmax_normalize(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    keys = list(scores.keys())
    if not keys:
        return {}
    vals = np.array([float(scores[k]) for k in keys], dtype=np.float64)
    # per-brand balancing: remove global level/scale bias before softmax
    vals = vals - float(vals.mean())
    std = float(vals.std())
    if std > 1e-8:
        vals = vals / std
    t = max(1e-4, float(temperature))
    vals = vals / t
    vals = vals - float(vals.max())
    ex = np.exp(vals)
    den = float(ex.sum())
    if den <= 0:
        return {k: 0.0 for k in keys}
    probs = ex / den
    return {k: float(p) for k, p in zip(keys, probs)}


def _expanded_crop(img_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float, expand_ratio: float) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    ex = bw * expand_ratio
    ey = bh * expand_ratio
    x1i = max(0, int(x1 - ex))
    y1i = max(0, int(y1 - ey))
    x2i = min(w, int(x2 + ex))
    y2i = min(h, int(y2 + ey))
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = img_bgr[y1i:y2i, x1i:x2i]
    return crop if crop.size > 0 else None


def _is_blurry(crop_bgr: np.ndarray, blur_var_threshold: float) -> bool:
    if crop_bgr.size == 0:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var < blur_var_threshold


def _grid_regions(img_bgr: np.ndarray) -> list[np.ndarray]:
    h, w = img_bgr.shape[:2]
    ys = [0, h // 2, h]
    xs = [0, w // 2, w]
    regions: list[np.ndarray] = []
    for yi in range(2):
        for xi in range(2):
            r = img_bgr[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            if r.size > 0:
                regions.append(r)
    return regions


def analyze_image(
    image_path: Path,
    weights_path: Path,
    class_names: list[str],
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    max_det: int | None = None,
) -> dict:
    cfg = load_config()
    d = cfg.get("inference", {}).get("defaults", {})
    conf = float(d.get("confidence_threshold", 0.45)) if conf is None else float(conf)
    iou = float(d.get("iou_threshold", 0.5)) if iou is None else float(iou)
    imgsz = int(d.get("image_size", 960)) if imgsz is None else int(imgsz)
    max_det = int(d.get("max_detections", 100)) if max_det is None else int(max_det)
    detector_min_conf = float(d.get("detector_min_conf", 0.35))
    clip_min_conf = float(d.get("clip_min_conf", 0.28))
    min_box_area_ratio = float(d.get("min_box_area_ratio", 0.0015))
    expand_ratio = float(d.get("context_expand_ratio", 0.35))
    blur_var_threshold = float(d.get("blur_var_threshold", 45.0))
    assign_min_conf = float(d.get("assign_min_conf", 0.33))
    clip_temperature = float(d.get("clip_temperature", 1.15))

    device = preferred_device("auto")
    model = YOLO(str(weights_path))
    schedule = [conf, max(0.35, conf * 0.8), 0.25, 0.18]
    res = model.predict(source=str(image_path), conf=schedule[0], iou=iou, imgsz=imgsz, device=device, verbose=False, max_det=max_det)[0]
    if res.boxes is None or len(res.boxes) == 0:
        for c_try in schedule[1:]:
            res = model.predict(
                source=str(image_path),
                conf=float(c_try),
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
                max_det=max_det,
            )[0]
            if res.boxes is not None and len(res.boxes) > 0:
                break

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    h, w = img_bgr.shape[:2]

    labels = [b for b in class_names if b in _presence_prompt_bank()]
    if "others" not in labels:
        labels.append("others")
    prompt_bank = _presence_prompt_bank()

    full_scores = _clip_scores_for_image(img_bgr, labels, prompt_bank, device)
    full_norm = _softmax_normalize(full_scores, temperature=clip_temperature)

    regional_scores_accum: list[dict[str, float]] = []
    for region in _grid_regions(img_bgr):
        if region.size == 0:
            continue
        regional_scores_accum.append(_clip_scores_for_image(region, labels, prompt_bank, device))

    regional_mean = {
        k: float(np.mean([m.get(k, 0.0) for m in regional_scores_accum])) if regional_scores_accum else 0.0 for k in labels
    }
    regional_norm = _softmax_normalize(regional_mean, temperature=clip_temperature)

    detections: list[dict] = []
    expanded_scores_accum: list[dict[str, float]] = []
    local_debug: list[dict] = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            c = float(b.conf[0].item())
            if c < detector_min_conf:
                continue
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw * bh < min_box_area_ratio * (w * h):
                continue

            context_crop = _expanded_crop(img_bgr, x1, y1, x2, y2, expand_ratio=expand_ratio)
            if context_crop is None:
                continue
            if _is_blurry(context_crop, blur_var_threshold=blur_var_threshold):
                continue

            local_scores = _clip_scores_for_image(context_crop, labels, prompt_bank, device)
            expanded_scores_accum.append(local_scores)
            local_norm = _softmax_normalize(local_scores, temperature=clip_temperature)

            # Detection-level labeling is local-first to prevent global brand forcing.
            brand = max(local_norm.items(), key=lambda kv: kv[1])[0]
            brand_conf = float(local_norm.get(brand, 0.0))

            if brand_conf < max(clip_min_conf, assign_min_conf):
                brand = "others"
            local_debug.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": c,
                    "local_scores_raw": local_scores,
                    "local_scores_norm": local_norm,
                    "assigned_brand": brand,
                    "assigned_brand_conf": brand_conf,
                }
            )
            cls_id = class_names.index(brand) if brand in class_names else class_names.index("others")
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": brand,
                    "detected_brand": brand,
                    "confidence": c,
                    "brand_confidence": brand_conf,
                    "label_source": "contextual_clip" if brand != "others" else "fallback_others",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": bw,
                    "height": bh,
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "shelf_zone": shelf_zone(cy, float(h)),
                }
            )

    expanded_mean = {
        k: float(np.mean([m.get(k, 0.0) for m in expanded_scores_accum])) if expanded_scores_accum else 0.0 for k in labels
    }
    expanded_norm = _softmax_normalize(expanded_mean, temperature=clip_temperature)
    # Mandatory fusion: 0.25 full + 0.50 contextual + 0.25 expanded
    fused_pre = {k: 0.25 * full_norm.get(k, 0.0) + 0.50 * regional_norm.get(k, 0.0) + 0.25 * expanded_norm.get(k, 0.0) for k in labels}
    brand_presence = _softmax_normalize(fused_pre, temperature=1.0)
    if sum(brand_presence.values()) <= 0:
        brand_presence = _normalize({k: full_scores.get(k, 0.0) for k in labels})
    if sum(brand_presence.values()) <= 0:
        uniform = 1.0 / max(1, len(labels))
        brand_presence = {k: uniform for k in labels}
    presence_ranking = sorted(brand_presence.items(), key=lambda kv: kv[1], reverse=True)

    compute_per_detection_share(detections)
    shelf_share = compute_shelf_share(detections, class_names)
    facings = compute_facings(detections, class_names)
    zones = zone_stats(detections)
    adjacency = adjacency_counts(detections)

    confs = [float(d["confidence"]) for d in detections]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    low_conf = sum(1 for c in confs if c < detector_min_conf)

    brand_conf_acc: dict[str, list[float]] = {k: [] for k in class_names}
    for d_det in detections:
        b = str(d_det.get("detected_brand", "others"))
        if b not in brand_conf_acc:
            b = "others"
        brand_conf_acc[b].append(float(d_det.get("brand_confidence", 0.0)))
    avg_brand_conf = {
        b: (float(np.mean(v)) if v else 0.0) for b, v in brand_conf_acc.items()
    }
    total_facings = max(1, int(sum(facings.values())))
    dominance_scores = {
        b: (
            0.5 * float(shelf_share.get(b, 0.0)) / 100.0
            + 0.3 * float(facings.get(b, 0)) / float(total_facings)
            + 0.2 * float(avg_brand_conf.get(b, 0.0))
        )
        for b in class_names
    }

    insights = generate_insights(
        shelf_share,
        brand_presence=brand_presence,
        dominance_scores=dominance_scores,
        facings=facings,
        zone_stats=zones,
        adjacency=adjacency,
        avg_conf=avg_conf,
        low_conf_count=low_conf,
    )

    annotated = render_boxes(img_bgr, detections)
    root = repo_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = root / "outputs" / "predictions" / f"annotated_{ts}.jpg"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pred_path), annotated)

    metrics_dir = root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    debug_payload = {
        "image": str(Path(image_path).resolve()),
        "full_scores_raw": full_scores,
        "full_scores_norm": full_norm,
        "regional_scores_raw": regional_scores_accum,
        "regional_scores_mean_raw": regional_mean,
        "regional_scores_norm": regional_norm,
        "expanded_scores_raw": expanded_scores_accum,
        "expanded_scores_mean_raw": expanded_mean,
        "expanded_scores_norm": expanded_norm,
        "fused_scores_pre_norm": fused_pre,
        "brand_presence_final": brand_presence,
        "brand_presence_ranking": presence_ranking,
        "crop_level_predictions": local_debug,
        "dominance_scores": dominance_scores,
    }
    (metrics_dir / "debug_scores.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    print("debug_scores:", json.dumps({"full": full_scores, "regional_mean": regional_mean, "expanded_mean": expanded_mean, "presence": brand_presence}, indent=2))

    payload = {
        "image": str(Path(image_path).resolve()),
        "weights": str(Path(weights_path).resolve()),
        "total_products": len(detections),
        "avg_confidence": avg_conf,
        "low_confidence_count": low_conf,
        "brand_presence": brand_presence,
        "brand_presence_ranking": presence_ranking,
        "dominance_scores": dominance_scores,
        "debug_score_summary": {
            "full_scores": full_scores,
            "regional_scores_mean": regional_mean,
            "expanded_scores_mean": expanded_mean,
            "fused_scores_pre_norm": fused_pre,
            "brand_presence": brand_presence,
        },
        "shelf_share": shelf_share,
        "facings": facings,
        "zone_stats": zones,
        "adjacency": adjacency,
        "insights": insights,
        "detections": detections,
        "annotated_image": str(pred_path.resolve()),
        "debug_scores_path": str((metrics_dir / "debug_scores.json").resolve()),
    }
    (metrics_dir / f"analysis_{ts}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload | {"rendered_bgr": annotated}

