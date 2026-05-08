#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.visualize import PredictSpec, predict_and_render
from src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Run detector inference and render boxes.")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    cfg = load_config()
    device_pref = cfg.get("device", {}).get("prefer", "auto")
    out = predict_and_render(
        PredictSpec(
            weights=Path(args.weights).resolve(),
            image=Path(args.image).resolve(),
            out_path=Path(args.out).resolve(),
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            device_prefer=str(device_pref),
        )
    )
    print(f"rendered={out}")


if __name__ == "__main__":
    main()

