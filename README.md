# Retail Shelf Intelligence (Shampoo)

Stable demo pipeline:

Shelf image -> YOLO bottle detector -> bottle crops -> CLIP zero-shot brand classification -> shelf analytics -> Streamlit/FastAPI output.

Supported brands:

- `head_shoulders`
- `pantene`
- `dove`
- `sunsilk`
- `others`

## Quick Start (First Run)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/frontend/app.py
```

## Backend API

```bash
uvicorn app.backend.main:app --reload --port 8000
```

`POST /analyze` with an image file returns:

- detections
- shelf share and facings
- shelf zones
- insights summary
- preview image path

## Runtime Notes

- Detector defaults are fixed internally for stability:
  - confidence `0.45`
  - IoU `0.5`
  - image size `960`
  - max detections `100`
- Detections below `0.35` are discarded.
- If `models/weights/shampoo_detector.pt` is unavailable, inference falls back to available local detector weights, then `yolov8m.pt`.
- Outputs are written to:
  - `outputs/predictions/`
  - `outputs/metrics/`