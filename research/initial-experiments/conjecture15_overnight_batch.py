"""
Overnight companion experiments for Conjecture 15 (Laplace AR(1), K=2).

This script is intentionally lighter than the full grid paper notebook and is
designed to run in parallel while heavier jobs are executing.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from scores_exact import LaplaceAR1_K2


OUT_DIR = Path("../figures/conjecture15/overnight_batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.3, 0.8, 0.95]
TS = [0.1, 0.5, 1.5]
H_STEPS = [1e-2, 5e-3, 2.5e-3]
FOURIER_N = 25
FOURIER_KMAX = 9.0
N_RANDOM = 180
RADIUS = 4.0
SEED = 20260422


@dataclass
class PointEval:
    sigma: np.ndarray
    ratio31: float


def hessian_from_score(model: LaplaceAR1_K2, x: np.ndarray, t: float, h: float) -> np.ndarray:
    hess = np.zeros((3, 3), dtype=float)
    for j in range(3):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        sp = model.score(xp, t=t, h=h, N=FOURIER_N, K_max=FOURIER_KMAX)
        sm = model.score(xm, t=t, h=h, N=FOURIER_N, K_max=FOURIER_KMAX)
        hess[:, j] = -(sp - sm) / (2.0 * h)
    return 0.5 * (hess + hess.T)


def eval_point(model: LaplaceAR1_K2, x: np.ndarray, t: float, h: float) -> PointEval:
    hess = hessian_from_score(model, x, t=t, h=h)
    sigma = np.linalg.svd(hess, compute_uv=False)
    ratio = float(sigma[2] / max(sigma[0], 1e-12))
    return PointEval(sigma=sigma, ratio31=ratio)


def random_cloud_experiment(rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for alpha in ALPHAS:
        model = LaplaceAR1_K2(alpha=alpha, b=1.0, mu0=0.0, sigma0=1.0)
        for t in TS:
            points = rng.uniform(-RADIUS, RADIUS, size=(N_RANDOM, 3))
            ratios = []
            print(f"[cloud] alpha={alpha}, t={t} ...")
            for i, x in enumerate(points):
                if i % max(N_RANDOM // 6, 1) == 0:
                    print(f"  progress {i}/{N_RANDOM}")
                pe = eval_point(model, x, t=t, h=1e-2)
                ratios.append(pe.ratio31)
            ratios = np.asarray(ratios)
            rows.append(
                {
                    "alpha": alpha,
                    "t": t,
                    "n_points": int(N_RANDOM),
                    "ratio_median": float(np.median(ratios)),
                    "ratio_mean": float(np.mean(ratios)),
                    "ratio_q90": float(np.quantile(ratios, 0.90)),
                    "ratio_q95": float(np.quantile(ratios, 0.95)),
                    "ratio_max": float(np.max(ratios)),
                }
            )
    return rows


def centerline_experiment() -> list[dict]:
    rows: list[dict] = []
    s_vals = np.linspace(-4.0, 4.0, 81)
    for alpha in ALPHAS:
        model = LaplaceAR1_K2(alpha=alpha, b=1.0, mu0=0.0, sigma0=1.0)
        fig, axes = plt.subplots(1, len(TS), figsize=(16, 4), constrained_layout=True)
        for ax, t in zip(axes, TS):
            ratios = []
            for s in s_vals:
                x = np.array([s, alpha * s, alpha * alpha * s], dtype=float)
                ratios.append(eval_point(model, x, t=t, h=1e-2).ratio31)
            ratios = np.asarray(ratios)
            ax.plot(s_vals, ratios, lw=2)
            ax.set_title(f"alpha={alpha}, t={t}")
            ax.set_xlabel("s (x0=s, x1=alpha*s, x2=alpha^2*s)")
            ax.set_ylabel("sigma3/sigma1")
            ax.grid(alpha=0.3)
            rows.append(
                {
                    "alpha": alpha,
                    "t": t,
                    "curve_ratio_median": float(np.median(ratios)),
                    "curve_ratio_q95": float(np.quantile(ratios, 0.95)),
                    "curve_ratio_max": float(np.max(ratios)),
                }
            )
        fig.savefig(OUT_DIR / f"centerline_ratio_alpha_{alpha}.png", bbox_inches="tight")
        plt.close(fig)
    return rows


def h_convergence_experiment() -> list[dict]:
    rows: list[dict] = []
    anchor_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([2.0, 1.0, 0.5]),
        np.array([-2.0, -1.0, -0.5]),
    ]
    for alpha in ALPHAS:
        model = LaplaceAR1_K2(alpha=alpha, b=1.0, mu0=0.0, sigma0=1.0)
        for t in TS:
            for p_idx, x in enumerate(anchor_points):
                vals = []
                for h in H_STEPS:
                    vals.append(eval_point(model, x, t=t, h=h).ratio31)
                vals = np.asarray(vals)
                rel12 = float(abs(vals[0] - vals[1]) / max(abs(vals[0]), 1e-12))
                rel23 = float(abs(vals[1] - vals[2]) / max(abs(vals[1]), 1e-12))
                rows.append(
                    {
                        "alpha": alpha,
                        "t": t,
                        "point_index": p_idx,
                        "x": x.tolist(),
                        "ratio_h_1e-2": float(vals[0]),
                        "ratio_h_5e-3": float(vals[1]),
                        "ratio_h_2p5e-3": float(vals[2]),
                        "rel_diff_1e-2_to_5e-3": rel12,
                        "rel_diff_5e-3_to_2p5e-3": rel23,
                    }
                )
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("Starting overnight companion experiments...")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Settings: FOURIER_N={FOURIER_N}, K_MAX={FOURIER_KMAX}, N_RANDOM={N_RANDOM}")

    cloud = random_cloud_experiment(rng)
    centerline = centerline_experiment()
    hconv = h_convergence_experiment()

    payload = {
        "meta": {
            "seed": SEED,
            "alphas": ALPHAS,
            "times": TS,
            "h_steps": H_STEPS,
            "fourier_n": FOURIER_N,
            "fourier_kmax": FOURIER_KMAX,
            "n_random": N_RANDOM,
            "radius": RADIUS,
        },
        "random_cloud_summary": cloud,
        "centerline_summary": centerline,
        "h_convergence_summary": hconv,
    }

    out_json = OUT_DIR / "overnight_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")
    print("Overnight companion experiments completed.")


if __name__ == "__main__":
    main()
