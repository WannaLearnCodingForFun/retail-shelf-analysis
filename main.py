#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env_health import run_health_check


def main() -> None:
    p = argparse.ArgumentParser(description="Retail shelf shampoo CV pipeline entrypoint.")
    sub = p.add_subparsers(dest="cmd", required=True)
    hp = sub.add_parser("health", help="environment and YOLOv8 smoke check")
    hp.set_defaults(_fn=lambda: run_health_check())
    args = p.parse_args()
    sys.exit(args._fn())


if __name__ == "__main__":
    main()
