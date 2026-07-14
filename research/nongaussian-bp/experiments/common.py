"""Shared experiment scaffolding: paths, argument parsing, provenance."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def experiment_parser(name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=name, description=description)
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced settings for a fast smoke run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / name,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    return parser


def provenance() -> dict[str, str]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
