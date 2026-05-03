#!/usr/bin/env python3
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env_health import run_health_check


if __name__ == "__main__":
    sys.exit(run_health_check())
