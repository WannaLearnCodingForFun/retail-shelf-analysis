#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.grozi120 import prepare_grozi120
from src.data.sku110k import prepare_sku110k
from src.data.custom_shampoo import CustomPrepSpec, prepare_custom_dataset
from src.inference.preview_labels import preview_labeled_samples
from src.training.yolo_train import TrainSpec, train_detector
from src.utils.config import load_config
from src.utils.env_health import preferred_device, run_health_check
from src.utils.paths import repo_root


def main() -> None:
    p = argparse.ArgumentParser(description="Retail shelf shampoo CV pipeline entrypoint.")
    sub = p.add_subparsers(dest="cmd", required=True)
    hp = sub.add_parser("health", help="environment and YOLOv8 smoke check")
    hp.set_defaults(_fn=lambda: run_health_check())

    dp = sub.add_parser("prepare", help="download/prepare datasets")
    dp_sub = dp.add_subparsers(dest="ds", required=True)
    dp_sub.add_parser("sku110k")
    g = dp_sub.add_parser("grozi120")
    g.add_argument("--api-key", default=None, help="Roboflow API key (else uses ROBOFLOW_API_KEY env var)")
    dp_sub.add_parser("custom-shampoo")

    tp = sub.add_parser("train-detector", help="train/finetune detector")
    tp.add_argument("--data", required=True)
    tp.add_argument("--init", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--epochs", type=int, default=1)
    tp.add_argument("--imgsz", type=int, default=640)
    tp.add_argument("--batch", type=int, default=8)
    tp.add_argument("--fraction", type=float, default=1.0)
    tp.add_argument("--workers", type=int, default=4)
    tp.add_argument("--name", default="detector_run")
    tp.add_argument("--project", default="outputs/metrics/detector")
    tp.add_argument("--skip-if-exists", action="store_true")

    vf = sub.add_parser("visualize-labels", help="preview custom dataset labels")
    vf.add_argument("--data-yaml", default=None)
    vf.add_argument("--out-dir", default="outputs/predictions/custom_label_preview")
    vf.add_argument("--count", type=int, default=5)

    args = p.parse_args()

    if args.cmd == "health":
        sys.exit(args._fn())

    cfg = load_config()
    root = repo_root()

    if args.cmd == "prepare":
        if args.ds == "sku110k":
            p1 = cfg["phase1"]["sku110k"]
            dataset, yaml_path = prepare_sku110k(
                url=p1["url"],
                external_dir=(root / p1["external_dir"]).resolve(),
                processed_dir=(root / p1["processed_dir"]).resolve(),
            )
            print(f"prepared_root={dataset.root}")
            print(f"yolo_yaml={yaml_path}")
            return

        if args.ds == "grozi120":
            p2 = cfg["phase2"]["grozi120"]
            rf = p2["roboflow"]
            out = prepare_grozi120(
                external_dir=(root / p2["external_dir"]).resolve(),
                processed_dir=(root / p2["processed_dir"]).resolve(),
                workspace=rf["workspace"],
                project=rf["project"],
                version=int(rf["version"]),
                fmt=rf.get("format", "yolov8"),
                api_key=args.api_key,
            )
            print(f"extracted_dir={out.extracted_dir}")
            print(f"yolo_yaml={out.processed_yaml}")
            return

        p3 = cfg["phase3"]["custom_dataset"]
        class_cfg = cfg["classes"]
        weights = root / cfg["phase4"]["final_training"]["init_weights"]
        if not weights.exists():
            fallback = root / "models/weights/detector_pretrained.pt"
            weights = fallback if fallback.exists() else Path("yolov8n.pt")
        data_yaml = prepare_custom_dataset(
            CustomPrepSpec(
                source_dir=(root / p3["source_dir"]).resolve(),
                fallback_source_dir=(root / p3["fallback_source_dir"]).resolve(),
                dataset_dir=(root / p3["dataset_dir"]).resolve(),
                classes=list(class_cfg["names"]),
                alias_map={k: list(v) for k, v in class_cfg["alias_map"].items()},
                weights=Path(weights),
                conf=float(p3["detect_conf"]),
                iou=float(p3["detect_iou"]),
                imgsz=int(p3["detect_imgsz"]),
                split_train=float(p3["split"]["train"]),
                split_val=float(p3["split"]["val"]),
                split_test=float(p3["split"]["test"]),
                seed=int(cfg["training"]["seed"]),
            )
        )
        print(f"yolo_yaml={data_yaml}")
        return

    if args.cmd == "visualize-labels":
        p3 = cfg["phase3"]["custom_dataset"]
        data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else (root / p3["data_yaml"]).resolve()
        saved = preview_labeled_samples(data_yaml, (root / args.out_dir).resolve(), sample_count=int(args.count))
        for p in saved:
            print(f"preview={p}")
        return

    device = preferred_device(cfg.get("device", {}).get("prefer", "auto"))
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    if args.skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
        print(f"weights={out_path}")
        return
    spec = TrainSpec(
        data_yaml=Path(args.data).resolve(),
        weights_out=out_path,
        model=str(Path(args.init)),
        pretrained=True,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        fraction=float(args.fraction),
        workers=int(args.workers),
        device=device,
        project_dir=(root / args.project).resolve(),
        name=str(args.name),
    )
    out = train_detector(spec)
    print(f"weights={out}")
    return


if __name__ == "__main__":
    main()
