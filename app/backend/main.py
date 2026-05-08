from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.analyze import analyze_image
from src.utils.config import load_config
from src.utils.paths import repo_root

app = FastAPI(title="Retail Shelf Analysis API", version="1.0.0")


def _model_path(cfg: dict) -> Path:
    root = repo_root()
    det = root / cfg.get("inference", {}).get("weights", {}).get("detector", "models/weights/shampoo_detector.pt")
    if det.exists():
        return det
    pre = root / "models/weights/detector_pretrained.pt"
    if pre.exists():
        return pre
    return Path("yolov8m.pt")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    cfg = load_config()
    classes = cfg["classes"]["names"]
    try:
        weights = _model_path(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    out = analyze_image(tmp_path, weights, classes)
    preview_path = repo_root() / cfg["paths"]["outputs_predictions"] / "api_last_result.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), out.pop("rendered_bgr"))
    out["preview_image"] = str(preview_path.resolve())
    return out

