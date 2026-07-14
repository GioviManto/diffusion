"""
Medium-resolution Conjecture 15 grid suite.

This script runs the same core diagnostics as the notebook, but at a larger
grid than the quick profile and with direct JSON/PNG outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d

from scores_exact import LaplaceAR1_K2


ALPHAS = [0.3, 0.8, 0.95]
TS = [0.1, 0.5, 1.5]
H_STEPS = [1e-2, 5e-3]
X_MAX = 3.5
N_GRID = 9
FOURIER_N = 31
FOURIER_KMAX = 10.0

OUT_DIR = Path("../figures/conjecture15/grid_suite")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GridResult:
    alpha: float
    t: float
    h: float
    x_grid: np.ndarray
    points: np.ndarray
    svs: np.ndarray
    angle_deg: np.ndarray


def predicted_subspace(alpha: float) -> np.ndarray:
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = np.array([0.0, 1.0, alpha])
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - np.dot(u2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    return np.stack([u1, u2], axis=1)


def principal_angle_deg(U_num: np.ndarray, U_pred: np.ndarray) -> float:
    sval = np.linalg.svd(U_num.T @ U_pred, compute_uv=False)
    sval = np.clip(sval, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.min(sval))))


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


def run_grid(alpha: float, t: float, h: float) -> GridResult:
    model = LaplaceAR1_K2(alpha=alpha, b=1.0, mu0=0.0, sigma0=1.0)
    pred = predicted_subspace(alpha)

    xs = np.linspace(-X_MAX, X_MAX, N_GRID)
    mesh = np.meshgrid(xs, xs, xs, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, 3)

    svs = np.zeros((points.shape[0], 3), dtype=float)
    angles = np.zeros(points.shape[0], dtype=float)

    total = points.shape[0]
    for i, x in enumerate(points):
        if i % max(total // 12, 1) == 0:
            print(f"[grid] alpha={alpha}, t={t}, h={h:g}: {i}/{total}")
        hess = hessian_from_score(model, x, t=t, h=h)
        _, sigma, vt = np.linalg.svd(hess)
        svs[i] = sigma
        angles[i] = principal_angle_deg(vt[:2].T, pred)

    return GridResult(alpha=alpha, t=t, h=h, x_grid=xs, points=points, svs=svs, angle_deg=angles)


def residual_coordinates(points: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, x2 = points[:, 0], points[:, 1], points[:, 2]
    return x1 - alpha * x0, x2 - alpha * x1


def save_figures(result: GridResult) -> dict[str, str]:
    alpha, t, h = result.alpha, result.t, result.h
    cfg_dir = OUT_DIR / f"alpha_{alpha}_t_{t}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    n = result.x_grid.size
    sigma3 = result.svs[:, 2].reshape(n, n, n)
    xs = result.x_grid
    mid = n // 2

    fig1, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im = axes[0].imshow(sigma3[mid], origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], aspect="auto")
    axes[0].set_title("x0=0")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig1.colorbar(im, ax=axes[0], shrink=0.9)
    im = axes[1].imshow(sigma3[:, mid, :], origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], aspect="auto")
    axes[1].set_title("x1=0")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x2")
    fig1.colorbar(im, ax=axes[1], shrink=0.9)
    im = axes[2].imshow(sigma3[:, :, mid], origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], aspect="auto")
    axes[2].set_title("x2=0")
    axes[2].set_xlabel("x0")
    axes[2].set_ylabel("x1")
    fig1.colorbar(im, ax=axes[2], shrink=0.9)
    fig1.suptitle(f"sigma3 slices alpha={alpha}, t={t}, h={h:g}")
    p1 = cfg_dir / f"sigma3_slices_h_{h:g}.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    ratio = result.svs[:, 2] / np.maximum(result.svs[:, 0], 1e-12)
    r1, r2 = residual_coordinates(result.points, alpha)
    stat_ratio, xe, ye, _ = binned_statistic_2d(r1, r2, ratio, statistic="mean", bins=35)
    stat_angle, _, _, _ = binned_statistic_2d(r1, r2, result.angle_deg, statistic="mean", bins=35)

    fig2, ax2 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im2 = ax2.imshow(stat_ratio.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto")
    ax2.set_title(f"sigma3/sigma1 map alpha={alpha}, t={t}, h={h:g}")
    ax2.set_xlabel("r1")
    ax2.set_ylabel("r2")
    cb2 = fig2.colorbar(im2, ax=ax2)
    cb2.set_label("mean sigma3/sigma1")
    p2 = cfg_dir / f"ratio_map_h_{h:g}.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im3 = ax3.imshow(stat_angle.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto")
    ax3.set_title(f"angle map alpha={alpha}, t={t}, h={h:g}")
    ax3.set_xlabel("r1")
    ax3.set_ylabel("r2")
    cb3 = fig3.colorbar(im3, ax=ax3)
    cb3.set_label("mean principal angle (deg)")
    p3 = cfg_dir / f"angle_map_h_{h:g}.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)

    return {"sigma3_fig": str(p1), "ratio_fig": str(p2), "angle_fig": str(p3)}


def summarize(result: GridResult) -> dict:
    ratio = result.svs[:, 2] / np.maximum(result.svs[:, 0], 1e-12)
    return {
        "alpha": result.alpha,
        "t": result.t,
        "h": result.h,
        "sigma3_mean": float(np.mean(result.svs[:, 2])),
        "sigma3_median": float(np.median(result.svs[:, 2])),
        "sigma3_q95": float(np.quantile(result.svs[:, 2], 0.95)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_q95": float(np.quantile(ratio, 0.95)),
        "angle_mean_deg": float(np.mean(result.angle_deg)),
        "angle_q95_deg": float(np.quantile(result.angle_deg, 0.95)),
    }


def main() -> None:
    print("Starting medium-resolution Conjecture 15 grid suite...")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Grid settings: N_GRID={N_GRID}, X_MAX={X_MAX}, FOURIER_N={FOURIER_N}, K_MAX={FOURIER_KMAX}")

    results: dict[tuple[float, float, float], GridResult] = {}
    rows = []
    for alpha in ALPHAS:
        for t in TS:
            for h in H_STEPS:
                res = run_grid(alpha=alpha, t=t, h=h)
                results[(alpha, t, h)] = res
                rec = summarize(res)
                rec.update(save_figures(res))
                rows.append(rec)

    stability = []
    for alpha in ALPHAS:
        for t in TS:
            r1 = results[(alpha, t, H_STEPS[0])]
            r2 = results[(alpha, t, H_STEPS[1])]
            ratio1 = r1.svs[:, 2] / np.maximum(r1.svs[:, 0], 1e-12)
            ratio2 = r2.svs[:, 2] / np.maximum(r2.svs[:, 0], 1e-12)
            rel = np.abs(ratio1 - ratio2) / np.maximum(np.abs(ratio1), 1e-12)
            stability.append(
                {
                    "alpha": alpha,
                    "t": t,
                    "median_rel_diff_ratio": float(np.median(rel)),
                    "q95_rel_diff_ratio": float(np.quantile(rel, 0.95)),
                }
            )

    out = {
        "meta": {
            "alphas": ALPHAS,
            "times": TS,
            "h_steps": H_STEPS,
            "x_max": X_MAX,
            "n_grid": N_GRID,
            "fourier_n": FOURIER_N,
            "fourier_kmax": FOURIER_KMAX,
        },
        "rows": rows,
        "stability": stability,
    }
    out_path = OUT_DIR / "grid_suite_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")
    print("Medium-resolution grid suite completed.")


if __name__ == "__main__":
    main()
