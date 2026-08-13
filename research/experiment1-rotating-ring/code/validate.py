#!/usr/bin/env python3
"""Reproducibility and numerical sanity checks for the final Experiment 1 bundle."""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from surrogate_model import (  # noqa: E402
    SurrogateConfig,
    add_ou_noise,
    clean_score,
    exact_score_and_jacobian,
    rotation,
    sample_clean_trajectories,
)


def require_files(paths: list[Path]) -> None:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing or empty files: " + ", ".join(missing))


def finite_numeric_csv(path: Path) -> None:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite numerical entry in {path}")


def surrogate_checks() -> dict[str, float]:
    rng = np.random.default_rng(20260725)
    cfg = SurrogateConfig(T=8, lam=0.05, sigma=0.15, psi=2.0 * math.pi / 8.0,
                          quadrature_nodes=220, posterior_grid_size=1200)
    y, z = sample_clean_trajectories(rng, cfg, 1)

    # Gauge round trip: de-rotate Z and recover the co-rotating Y.
    y_from_z = np.empty_like(y)
    z_roundtrip = np.empty_like(z)
    for k in range(cfg.T):
        y_from_z[:, k] = z[:, k] @ rotation(-k * cfg.psi).T
        z_roundtrip[:, k] = y_from_z[:, k] @ rotation(k * cfg.psi).T
    gauge_error = float(max(np.max(np.abs(y_from_z - y)), np.max(np.abs(z_roundtrip - z))))

    # A perfect co-rotating trajectory on the unit circle is a clean stationary point.
    perfect = np.zeros((1, cfg.T, 2), dtype=float)
    perfect[..., 0] = 1.0
    perfect_score_error = float(np.max(np.abs(clean_score(perfect, cfg))))

    # Analytic block Jacobian against centered finite differences.
    t = 0.4
    x = add_ou_noise(rng, y, t)
    indices = np.arange(cfg.T)
    cache: dict = {}
    score, jac, _ = exact_score_and_jacobian(x, indices, t, cfg, cache)
    eps = 2.0e-5
    max_abs = 0.0
    for j in (0, cfg.T // 2, cfg.T - 1):
        for b in range(2):
            xp = x.copy(); xm = x.copy()
            xp[0, j, b] += eps
            xm[0, j, b] -= eps
            sp, _, _ = exact_score_and_jacobian(xp, indices, t, cfg, cache)
            sm, _, _ = exact_score_and_jacobian(xm, indices, t, cfg, cache)
            fd = (sp - sm) / (2.0 * eps)
            for k in (0, cfg.T // 2, cfg.T - 1):
                analytic = jac[0, k, j, :, b]
                max_abs = max(max_abs, float(np.max(np.abs(fd[0, k] - analytic))))

    return {
        "surrogate_jacobian_fd_max_abs": max_abs,
        "gauge_roundtrip_max_abs": gauge_error,
        "perfect_clean_path_score_max_abs": perfect_score_error,
    }


def main() -> None:
    figure_files = [ROOT / "figures" / f"fig{i:02d}_{name}.pdf" for i, name in [
        (1, "problem_setup"), (2, "two_clocks"), (3, "score_graphs"),
        (4, "metric_construction"), (5, "rotation"), (6, "gauge"),
        (7, "clean_score"), (8, "surrogate_response"),
        (9, "surrogate_diagnostics"), (10, "transition_kernels"),
        (11, "joint_score_fields"), (12, "joint_vs_marginal_fields"),
        (13, "taylor"), (14, "response_profiles"), (15, "range_intensity"),
        (16, "receptive_field"), (17, "parameter_sweeps"), (18, "validation"),
    ]]
    table_files = sorted((ROOT / "tables").glob("*.tex"))
    data_files = sorted((ROOT / "data").rglob("*.csv"))
    require_files([ROOT / "main.tex", *figure_files, *table_files, *data_files])
    for path in data_files:
        finite_numeric_csv(path)

    fd = pd.read_csv(ROOT / "data/polar/fluctuation_response_validation.csv")
    grid = pd.read_csv(ROOT / "data/polar/grid_convergence.csv")
    checks = surrogate_checks()
    checks.update({
        "polar_fdt_relative_error_max": float(fd["relative_error"].max()),
        "polar_response_grid_relative_error_max": float(grid["response_grid_relative_error"].max()),
        "figure_pdf_count": float(len(figure_files)),
        "csv_count": float(len(data_files)),
    })

    thresholds = {
        "surrogate_jacobian_fd_max_abs": 1.0e-4,
        "gauge_roundtrip_max_abs": 1.0e-12,
        "perfect_clean_path_score_max_abs": 1.0e-12,
        "polar_fdt_relative_error_max": 1.0e-7,
        "polar_response_grid_relative_error_max": 1.0e-7,
    }
    failures = {k: (checks[k], tol) for k, tol in thresholds.items() if checks[k] > tol}
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "thresholds": thresholds,
        "failures": failures,
    }
    text_lines = [
        "Experiment 1 validation",
        "=======================",
        f"status: {payload['status']}",
        "",
    ]
    for key, value in checks.items():
        if key in thresholds:
            text_lines.append(f"{key}: {value:.12g}  (threshold {thresholds[key]:.3g})")
        else:
            text_lines.append(f"{key}: {value:.12g}")
    text_lines.extend(["", "All CSV inputs contain finite numerical values.",
                       f"Verified {len(figure_files)} curated figure PDFs and {len(data_files)} CSV data files."])
    (ROOT / "validation.txt").write_text("\n".join(text_lines) + "\n")
    (ROOT / "validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\n".join(text_lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
