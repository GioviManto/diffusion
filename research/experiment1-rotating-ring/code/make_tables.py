#!/usr/bin/env python3
"""Generate every LaTeX result table of the note from the machine-readable data.

No number is typed by hand: each table is produced from the CSV files in
``data/`` or from ``validation.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import diagnostics_table  # noqa: E402

TABLES = ROOT / "tables"
DATA = ROOT / "data"
TABLES.mkdir(exist_ok=True)


def write_tabular(name: str, spec: str, header: str, rows: list[str]) -> None:
    body = [
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (TABLES / name).write_text("\n".join(body) + "\n")


def nearest(frame: pd.DataFrame, t: float) -> pd.Series:
    return frame.iloc[(frame["t"] - t).abs().argmin()]


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


# ---------------------------------------------------------------- diagnostics
def table_weighted_diagnostics() -> None:
    """Chapter 3: the four diagnostics side by side, original polar model."""
    profiles = pd.read_csv(DATA / "polar/baseline_response_profiles.csv")
    radial = diagnostics_table(profiles, "rr_rms")
    tangential = diagnostics_table(profiles, "tt_rms")
    total = diagnostics_table(profiles, "fro_rms")

    rows = []
    for t in (0.03, 0.10, 0.40, 0.70, 1.00, 1.50, 2.00):
        a, b, c = nearest(radial, t), nearest(tangential, t), nearest(total, t)
        rows.append(
            f"{c.t:.2f} & {c.intensity:.2f} & {c.mean_lag:.2f} & "
            f"{c.normalised_range:.2f} & "
            f"{a.relative_reach:.3f} & {b.relative_reach:.3f} & {c.relative_reach:.3f} \\\\"
        )
    ceiling = total.iloc[0]
    rows.append("\\midrule")
    rows.append(
        f"\\emph{{flat profile}} & --- & {ceiling.flat_mean_lag:.2f} & "
        f"{ceiling.flat_normalised_range:.2f} & --- & --- & --- \\\\"
    )
    write_tabular(
        "weighted_diagnostics.tex",
        "rrrrrrr",
        r"$t$ & $I_{\rm off}$ & $\bar\ell$ & $\xi^{\rm norm}$ "
        r"& $\widetilde\Xi_r$ & $\widetilde\Xi_\theta$ & $\widetilde\Xi$",
        rows,
    )


def table_weighted_surrogate() -> None:
    """Chapter 3: the same diagnostics for the surrogate, plus window radius."""
    profiles = pd.read_csv(DATA / "surrogate/influence_profiles.csv")
    diag = diagnostics_table(profiles, "mean_block_norm")
    windows = pd.read_csv(DATA / "surrogate/window_errors_full.csv")
    summary = pd.read_csv(DATA / "surrogate/summary.csv")

    rows = []
    for t in (0.05, 0.20, 0.70, 1.50):
        d = nearest(diag, t)
        s = nearest(summary, t)
        group = windows[windows["t"] == t]
        within = group[group["relative_rmse"] <= 0.05]
        radius = (
            str(int(within["window_radius"].min()))
            if len(within)
            else f">{int(group['window_radius'].max())}"
        )
        marginal = float(s["joint_vs_marginal_relative_mse"]) ** 0.5
        rows.append(
            f"{d.t:.2f} & {marginal:.3f} & {d.intensity:.3f} & {d.mean_lag:.2f} & "
            f"{d.normalised_range:.2f} & {d.relative_reach:.3f} & {radius} \\\\"
        )
    ceiling = diag.iloc[0]
    rows.append("\\midrule")
    rows.append(
        f"\\emph{{flat profile}} & --- & --- & {ceiling.flat_mean_lag:.2f} & "
        f"{ceiling.flat_normalised_range:.2f} & --- & --- \\\\"
    )
    write_tabular(
        "surrogate_results.tex",
        "rrrrrrr",
        r"$t$ & marginal RMSE & $I_{\rm off}$ & $\bar\ell$ & $\xi^{\rm norm}$ "
        r"& $\widetilde\Xi$ & $L_{5\%}$",
        rows,
    )


# --------------------------------------------------------------- polar model
def table_polar_response() -> None:
    profiles = pd.read_csv(DATA / "polar/baseline_response_profiles.csv")
    radial = diagnostics_table(profiles, "rr_rms")
    tangential = diagnostics_table(profiles, "tt_rms")
    windows = pd.read_csv(DATA / "polar/window_receptive_summary.csv")
    marginal = pd.read_csv(DATA / "polar/joint_vs_marginal.csv")

    rows = []
    for t in (0.03, 0.10, 0.40, 0.70, 1.00, 1.50, 2.00):
        a, b = nearest(radial, t), nearest(tangential, t)
        w = nearest(windows, t)
        m = nearest(marginal, t)
        rows.append(
            f"{a.t:.2f} & {m.relative_marginal_score_error:.3f} & "
            f"{a.mean_lag:.2f} & {b.mean_lag:.2f} & "
            f"{a.relative_reach:.3f} & {b.relative_reach:.3f} & "
            f"{int(w.radial_L_5pct)} & {int(w.tangential_L_5pct)} \\\\"
        )
    write_tabular(
        "polar_response_results.tex",
        "rrrrrrrr",
        r"$t$ & marg.\ RMSE & $\bar\ell_r$ & $\bar\ell_\theta$ & "
        r"$\widetilde\Xi_r$ & $\widetilde\Xi_\theta$ & $L_r^{5\%}$ & $L_\theta^{5\%}$",
        rows,
    )


def table_clean_scales() -> None:
    """Low-noise exponential fits, next to the clean prediction."""
    summary = pd.read_csv(DATA / "polar/baseline_response_summary.csv")
    rows = []
    for t in (0.03, 0.06, 0.10):
        a = nearest(summary, t)
        rows.append(
            f"{a.t:.2f} & {a.rr_exp_fit_length:.2f} & {a.rr_exp_fit_r2:.3f} & "
            f"{a.tt_exp_fit_length:.2f} & {a.tt_exp_fit_r2:.3f} \\\\"
        )
    write_tabular(
        "exp_fit_results.tex",
        "rrrrr",
        r"$t$ & $\xi_r^{\rm exp}$ & $R_r^2$ & $\xi_\theta^{\rm exp}$ & $R_\theta^2$",
        rows,
    )


def table_taylor() -> None:
    taylor = pd.read_csv(DATA / "taylor/taylor_vs_exact.csv")
    rows = []
    for t in (0.03, 0.15, 0.40, 0.70, 1.00, 1.50, 2.00):
        a = nearest(taylor, t)
        rows.append(
            f"{a.t:.2f} & {a.central_relative_error:.3f} & "
            f"{a.trajectory_relative_error:.3f} & {a.mean_cosine_similarity:.3f} \\\\"
        )
    write_tabular(
        "taylor_results.tex",
        "rrrr",
        r"$t$ & central RMSE & trajectory RMSE & mean cosine",
        rows,
    )


def table_validation() -> None:
    report = json.loads((ROOT / "validation.json").read_text())
    checks = report["checks"]
    thresholds = report["thresholds"]
    labels = {
        "polar_fdt_relative_error_max":
            "posterior covariance vs finite differences (polar)",
        "surrogate_jacobian_fd_max_abs":
            "analytic surrogate Jacobian vs finite differences",
        "polar_response_grid_relative_error_max":
            r"$41\times128$ vs $51\times192$ response grid",
        "gauge_roundtrip_max_abs":
            "rotation gauge round trip",
        "perfect_clean_path_score_max_abs":
            "clean score on an exact co-rotating ring path",
    }
    rows = []
    for key, label in labels.items():
        value = checks[key]
        shown = "$0$ (exact)" if value == 0 else sci(value)
        rows.append(f"{label} & {shown} & {sci(thresholds[key])} \\\\")
    write_tabular(
        "validation.tex", "lrr", "check & discrepancy & tolerance", rows
    )


def main() -> None:
    table_weighted_diagnostics()
    table_weighted_surrogate()
    table_polar_response()
    table_clean_scales()
    table_taylor()
    table_validation()
    print(f"wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
