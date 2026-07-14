"""Reproducibility and I/O helpers.

Every experiment writes (a) its full parameter set as JSON, (b) tabular results as
CSV, and (c) figures as PNG. No hidden state: all randomness flows through
`numpy.random.Generator` objects derived from explicit seeds via `rng_for`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def rng_for(*keys: object) -> np.random.Generator:
    """Deterministic RNG from a tuple of hashable keys (common-random-numbers).

    Using `rng_for(experiment, rho, t, trial)` guarantees that the same clean chain
    and noise are drawn for every *method or discretization* compared at that
    setting, so method comparisons are paired and never confounded by Monte Carlo
    noise (audit finding F3).
    """
    seed_seq = np.random.SeedSequence(
        [abs(hash(k)) % (2**31) for k in ("bp-markov-2026",) + keys]
    )
    return np.random.default_rng(seed_seq)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path | str, payload: Mapping[str, Any] | Any) -> None:
    """Write experiment parameters/metadata as pretty JSON."""

    def default(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
