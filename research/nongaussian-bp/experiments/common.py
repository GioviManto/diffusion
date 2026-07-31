"""Shared experiment scaffolding: paths, argument parsing, provenance."""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument(
        "--only", default=None,
        help=(
            "Comma-separated part names to run (default: all). Parts are "
            "independent and write disjoint CSVs, so a scheduler can run them "
            "as parallel array tasks into one --output-dir with no merge step. "
            "Use --list-parts to see the names."
        ),
    )
    parser.add_argument(
        "--list-parts", action="store_true",
        help="Print the part names this experiment defines, then exit.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Override a setting, repeatable, e.g. --set n_rep=64 "
            "--set sizes='(64,256,1024,4096)'. VALUE is parsed as a Python "
            "literal. Overrides land in params.json, so a scaled-up run stays "
            "self-describing."
        ),
    )
    return parser


def apply_overrides(settings: dict, assignments) -> dict:
    """Apply `--set KEY=VALUE` assignments to a settings dict.

    Unknown keys are rejected rather than silently added: a typo in a job
    script would otherwise run the default configuration on the cluster and
    look like a completed experiment.
    """
    import ast

    updated = dict(settings)
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in updated:
            raise SystemExit(
                f"--set: unknown key {key!r}. Known keys: {sorted(updated)}"
            )
        try:
            updated[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"--set {key}: cannot parse {raw!r} ({exc})") from exc
    return updated


def select_parts(parts: dict, only: str | None) -> dict:
    """Filter an ordered {name: callable} mapping by a --only specification."""
    if not only:
        return parts
    wanted = [p.strip() for p in only.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in parts]
    if unknown:
        raise SystemExit(
            f"--only: unknown part(s) {unknown}. Known parts: {list(parts)}"
        )
    return {name: parts[name] for name in wanted}


def provenance() -> dict[str, str]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cpu_count": str(os.cpu_count()),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
