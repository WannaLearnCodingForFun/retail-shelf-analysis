#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sku110k import prepare_sku110k
from src.utils.config import load_config
from src.utils.paths import repo_root


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1: download and prepare SKU-110K (YOLO labels + yaml).")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    p1 = cfg["phase1"]["sku110k"]

    external_dir = (root / p1["external_dir"]).resolve()
    processed_dir = (root / p1["processed_dir"]).resolve()
    url = p1["url"]

    dataset, yaml_path = prepare_sku110k(url=url, external_dir=external_dir, processed_dir=processed_dir)
    print(f"prepared_root={dataset.root}")
    print(f"yolo_yaml={yaml_path}")


if __name__ == "__main__":
    main()

