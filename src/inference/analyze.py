from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
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


def _classify_crop_clip(crop_bgr, class_names: list[str], device: str) -> tuple[str, float]:
    model, proc = _clip_assets(device)
    prompts = {
        "head_shoulders": "a bottle of Head & Shoulders shampoo",
        "pantene": "a bottle of Pantene shampoo",
        "dove": "a bottle of Dove shampoo",
        "sunsilk": "a bottle of Sunsilk shampoo",
        "others": "other shampoo product",
    }
    labels = [c for c in class_names if c in prompts]
    if "others" not in labels:
        labels.append("others")
    texts = [prompts[l] for l in labels]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    ins = proc(text=texts, images=rgb, return_tensors="pt", padding=True)
    ins = {k: v.to(device) if hasattr(v, "to") else v for k, v in ins.items()}
    with torch.inference_mode():
        out = model(**ins)
        logits = out.logits_per_image[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx])


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

    device = preferred_device("auto")
    model = YOLO(str(weights_path))
    res = model.predict(source=str(image_path), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, max_det=max_det)[0]

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    h, w = img_bgr.shape[:2]

    detections: list[dict] = []
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

            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w, int(x2)), min(h, int(y2))
            crop = img_bgr[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            brand, brand_conf = _classify_crop_clip(crop, class_names, device=device)
            if brand_conf < clip_min_conf:
                brand = "others"
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

    compute_per_detection_share(detections)
    shelf_share = compute_shelf_share(detections, class_names)
    facings = compute_facings(detections, class_names)
    zones = zone_stats(detections)
    adjacency = adjacency_counts(detections)

    confs = [float(d["confidence"]) for d in detections]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    low_conf = sum(1 for c in confs if c < detector_min_conf)

    insights = generate_insights(shelf_share, facings=facings, zone_stats=zones, adjacency=adjacency, avg_conf=avg_conf, low_conf_count=low_conf)

    annotated = render_boxes(img_bgr, detections)
    root = repo_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = root / "outputs" / "predictions" / f"annotated_{ts}.jpg"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pred_path), annotated)

    metrics_dir = root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(Path(image_path).resolve()),
        "weights": str(Path(weights_path).resolve()),
        "total_products": len(detections),
        "avg_confidence": avg_conf,
        "low_confidence_count": low_conf,
        "shelf_share": shelf_share,
        "facings": facings,
        "zone_stats": zones,
        "adjacency": adjacency,
        "insights": insights,
        "detections": detections,
        "annotated_image": str(pred_path.resolve()),
    }
    (metrics_dir / f"analysis_{ts}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload | {"rendered_bgr": annotated}

