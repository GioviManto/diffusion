#!/usr/bin/env python3
"""Exact experiments for the rotating-ring surrogate with known rotation.

Model in the co-rotating frame:
    Y_0 ~ p_lambda(a) proportional to exp(-( ||a|| - 1 )^2 / (2 lambda))
    Y_k = Y_0 + sigma * sum_{r=1}^k eta_r, eta_r ~ N(0, I_2)

OU corruption of the whole trajectory:
    X_t = m_t Y + sqrt(Delta_t) xi,
    m_t = exp(-t), Delta_t = 1-exp(-2t).

Conditional on the ring anchor A=Y_0, X_t is Gaussian. Integrating over A gives
an exact location mixture. The score and its Jacobian are available in closed form
up to one-dimensional radial quadrature.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ive
from numpy.polynomial.legendre import leggauss

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    seed: int = 7
    T: int = 30
    lam: float = 0.05
    sigma: float = 0.15
    psi: float = 2.0 * math.pi / 30.0
    n_samples: int = 350
    t_values: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0)
    selected_heatmap_t: Tuple[float, ...] = (0.02, 0.2, 0.7, 2.0)
    max_window_radius: int = 10
    quadrature_nodes: int = 160
    posterior_grid_size: int = 450


class RingPosteriorLookup:
    """Lookup E[A|h] and Cov(A|h) for a fixed beta.

    Posterior density on A in R^2:
        p(a|h) proportional to exp(-(r-1)^2/(2 lam) - beta r^2/2 + h dot a).
    By rotational symmetry, the mean is parallel to h and the covariance has
    parallel and perpendicular eigenvalues.
    """

    def __init__(self, lam: float, beta: float, h_max: float, n_h: int, n_quad: int):
        if lam <= 0 or beta < 0 or h_max <= 0:
            raise ValueError("Invalid posterior lookup parameters")
        self.lam = float(lam)
        self.beta = float(beta)
        self.h_grid = np.linspace(0.0, float(h_max), int(n_h))

        precision = 1.0 / lam + beta
        mode_at_hmax = (1.0 / lam + h_max) / precision
        r_max = max(4.0, mode_at_hmax + 10.0 / math.sqrt(precision))
        nodes, weights = leggauss(int(n_quad))
        r = 0.5 * (nodes + 1.0) * r_max
        w = 0.5 * r_max * weights

        H = self.h_grid[:, None]
        R = r[None, :]
        Q = H * R
        base = -((R - 1.0) ** 2) / (2.0 * lam) - 0.5 * beta * R**2
        # ive(n,q) = exp(-q) I_n(q) for q >= 0. Add q back in exponent.
        expo = base + Q
        shift = np.max(expo, axis=1, keepdims=True)
        common = np.exp(expo - shift) * w[None, :]

        i0e = ive(0, Q)
        i1e = ive(1, Q)
        i2e = ive(2, Q)

        den = np.sum(common * R * i0e, axis=1)
        num_mean = np.sum(common * R**2 * i1e, axis=1)
        num_par2 = 0.5 * np.sum(common * R**3 * (i0e + i2e), axis=1)
        num_perp2 = 0.5 * np.sum(common * R**3 * (i0e - i2e), axis=1)

        if not np.all(np.isfinite(den)) or np.any(den <= 0):
            raise FloatingPointError("Posterior quadrature failed")

        mean = num_mean / den
        e_par2 = num_par2 / den
        e_perp2 = num_perp2 / den
        var_par = np.maximum(e_par2 - mean**2, 0.0)
        var_perp = np.maximum(e_perp2, 0.0)

        # Enforce the isotropic h=0 limit numerically.
        iso = 0.5 * (var_par[0] + var_perp[0])
        mean[0] = 0.0
        var_par[0] = iso
        var_perp[0] = iso

        self.mean_grid = mean
        self.var_par_grid = var_par
        self.var_perp_grid = var_perp

    def moments(self, h: Array) -> Tuple[Array, Array]:
        """Return posterior mean and covariance for h vectors of shape (n,2)."""
        h = np.asarray(h, dtype=float)
        if h.ndim != 2 or h.shape[1] != 2:
            raise ValueError("h must have shape (n,2)")
        rho = np.linalg.norm(h, axis=1)
        if np.max(rho) > self.h_grid[-1] * (1.0 + 1e-10):
            raise ValueError(f"h magnitude {np.max(rho):.3g} exceeds lookup range {self.h_grid[-1]:.3g}")
        mean_mag = np.interp(rho, self.h_grid, self.mean_grid)
        var_par = np.interp(rho, self.h_grid, self.var_par_grid)
        var_perp = np.interp(rho, self.h_grid, self.var_perp_grid)

        unit = np.zeros_like(h)
        nz = rho > 1e-12
        unit[nz] = h[nz] / rho[nz, None]
        mean = mean_mag[:, None] * unit

        cov = np.zeros((h.shape[0], 2, 2), dtype=float)
        eye = np.eye(2)
        for n in range(h.shape[0]):
            if nz[n]:
                uu = np.outer(unit[n], unit[n])
                cov[n] = var_perp[n] * eye + (var_par[n] - var_perp[n]) * uu
            else:
                cov[n] = var_perp[n] * eye
        return mean, cov


def sample_ring_anchor(rng: np.random.Generator, n: int, lam: float) -> Array:
    """Sample from the 2D density proportional to exp(-(r-1)^2/(2 lam))."""
    # Build a high-resolution inverse CDF for radial marginal r*exp(...), r>=0.
    r_max = 1.0 + 10.0 * math.sqrt(lam) + 1.0
    r = np.linspace(0.0, r_max, 200_000)
    pdf = r * np.exp(-0.5 * (r - 1.0) ** 2 / lam)
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * np.diff(r))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    radii = np.interp(rng.random(n), cdf, r)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def sample_clean_trajectories(rng: np.random.Generator, cfg: Config) -> Tuple[Array, Array]:
    """Return co-rotating Y and laboratory-frame Z, both shape (n,T,2)."""
    anchors = sample_ring_anchor(rng, cfg.n_samples, cfg.lam)
    increments = cfg.sigma * rng.standard_normal((cfg.n_samples, cfg.T - 1, 2))
    y = np.empty((cfg.n_samples, cfg.T, 2), dtype=float)
    y[:, 0] = anchors
    y[:, 1:] = anchors[:, None, :] + np.cumsum(increments, axis=1)

    z = np.empty_like(y)
    for k in range(cfg.T):
        c, s = math.cos(k * cfg.psi), math.sin(k * cfg.psi)
        rot = np.array([[c, -s], [s, c]])
        z[:, k] = y[:, k] @ rot.T
    return y, z


def ou_corrupt(rng: np.random.Generator, y: Array, t: float) -> Array:
    m = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    return m * y + math.sqrt(delta) * rng.standard_normal(y.shape)


def covariance_random_walk(indices: Array, sigma: float) -> Array:
    idx = np.asarray(indices, dtype=float)
    return sigma**2 * np.minimum.outer(idx, idx)


def exact_score_and_jacobian(
    x: Array,
    indices: Array,
    t: float,
    lam: float,
    sigma: float,
    lookup_cache: Dict[Tuple[Tuple[int, ...], float], RingPosteriorLookup],
    cfg: Config,
) -> Tuple[Array, Array]:
    """Exact score and Jacobian for a trajectory subset.

    x shape: (n,L,2), indices are the absolute internal-time indices represented in x.
    Returns score shape (n,L,2), Jacobian shape (n,L,L,2,2).
    """
    indices = np.asarray(indices, dtype=int)
    n, L, d = x.shape
    if d != 2 or len(indices) != L:
        raise ValueError("Shape/index mismatch")
    m = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    K = covariance_random_walk(indices, sigma)
    A = m * m * K + delta * np.eye(L)
    Ainv = np.linalg.inv(A)
    ones = np.ones(L)
    q = Ainv @ ones
    beta = m * m * float(ones @ q)
    h = m * np.einsum("l,nld->nd", q, x)

    key = (tuple(indices.tolist()), round(float(t), 12))
    if key not in lookup_cache:
        h_mag = np.linalg.norm(h, axis=1)
        # Generous headroom also supports finite-difference validation.
        h_max = max(5.0, float(np.max(h_mag)) * 1.20 + 2.0)
        lookup_cache[key] = RingPosteriorLookup(
            lam=lam,
            beta=beta,
            h_max=h_max,
            n_h=cfg.posterior_grid_size,
            n_quad=cfg.quadrature_nodes,
        )
    mean_a, cov_a = lookup_cache[key].moments(h)

    score = -np.einsum("ij,njd->nid", Ainv, x) + m * q[None, :, None] * mean_a[:, None, :]

    jac = np.empty((n, L, L, 2, 2), dtype=float)
    eye = np.eye(2)
    for i in range(L):
        for j in range(L):
            jac[:, i, j] = -Ainv[i, j] * eye + (m * m * q[i] * q[j]) * cov_a
    return score, jac


def block_frobenius(jac: Array) -> Array:
    return np.sqrt(np.sum(jac * jac, axis=(-2, -1)))


def clean_score(y: Array, lam: float, sigma: float) -> Array:
    """Exact clean score at t=0 in the co-rotating frame."""
    n, T, _ = y.shape
    out = np.zeros_like(y)
    r0 = np.linalg.norm(y[:, 0], axis=1)
    unit0 = y[:, 0] / np.maximum(r0[:, None], 1e-12)
    ring = ((1.0 - r0) / lam)[:, None] * unit0
    out[:, 0] = ring + (y[:, 1] - y[:, 0]) / sigma**2
    if T > 2:
        out[:, 1:-1] = (y[:, :-2] - 2.0 * y[:, 1:-1] + y[:, 2:]) / sigma**2
    out[:, -1] = (y[:, -2] - y[:, -1]) / sigma**2
    return out


def rotate_blocks(arr: Array, psi: float, inverse: bool) -> Array:
    """Rotate vector blocks by +/- k psi. arr shape (...,T,2)."""
    out = np.empty_like(arr)
    T = arr.shape[-2]
    sign = -1.0 if inverse else 1.0
    for k in range(T):
        angle = sign * k * psi
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        out[..., k, :] = arr[..., k, :] @ rot.T
    return out


def finite_difference_validation(
    x_one: Array,
    indices: Array,
    t: float,
    cfg: Config,
    cache: Dict,
) -> float:
    score, jac = exact_score_and_jacobian(x_one[None], indices, t, cfg.lam, cfg.sigma, cache, cfg)
    eps = 2e-5
    max_err = 0.0
    for j in [0, len(indices) // 2, len(indices) - 1]:
        for d in [0, 1]:
            xp = x_one.copy(); xm = x_one.copy()
            xp[j, d] += eps; xm[j, d] -= eps
            sp, _ = exact_score_and_jacobian(xp[None], indices, t, cfg.lam, cfg.sigma, cache, cfg)
            sm, _ = exact_score_and_jacobian(xm[None], indices, t, cfg.lam, cfg.sigma, cache, cfg)
            fd = (sp[0] - sm[0]) / (2.0 * eps)
            analytic = jac[0, :, j, :, d]
            max_err = max(max_err, float(np.max(np.abs(fd - analytic))))
    return max_err


def plot_example_trajectory(y: Array, z: Array, cfg: Config, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    theta = np.linspace(0, 2 * math.pi, 400)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    for ax, traj, title in [
        (axes[0], z[0], "Laboratory frame: coherent rotation"),
        (axes[1], y[0], "Co-rotating frame: anchored random walk"),
    ]:
        ax.plot(circle[:, 0], circle[:, 1], linestyle="--", linewidth=1)
        ax.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], s=50, label="first frame")
        ax.set_aspect("equal")
        ax.set_xlabel("first coordinate")
        ax.set_ylabel("second coordinate")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_trajectory_and_gauge.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(heatmaps: Dict[float, Array], outdir: Path) -> None:
    for t, mat in heatmaps.items():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(mat, origin="lower", aspect="auto")
        ax.set_xlabel("perturbed frame j")
        ax.set_ylabel("score block k")
        ax.set_title(f"Average block-Jacobian norm, t={t:g}")
        fig.colorbar(im, ax=ax, label="E ||d S_k / d x_j||_F")
        fig.tight_layout()
        fig.savefig(outdir / f"02_jacobian_heatmap_t_{t:.2f}.png", dpi=180)
        plt.close(fig)


def plot_influence_profiles(profile_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for t, group in profile_df.groupby("t"):
        if t in (0.02, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 3.0):
            ax.semilogy(group["lag"], group["mean_block_norm"], marker="o", markersize=3, label=f"t={t:g}")
    ax.set_xlabel("temporal lag |k-j|")
    ax.set_ylabel("mean block-Jacobian norm")
    ax.set_title("How cross-frame influence decays with lag")
    ax.legend(ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "03_influence_vs_lag.png", dpi=180)
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary_df["t"], summary_df["cross_influence_mass"], marker="o")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel("mean total cross-frame Jacobian mass")
    ax.set_title("Total cross-frame influence weakens as diffusion noise grows")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "04_cross_influence_vs_t.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary_df["t"], summary_df["joint_vs_marginal_relative_mse"], marker="o")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel("relative MSE of marginal score")
    ax.set_title("Error from ignoring the rest of the trajectory")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "05_joint_vs_marginal.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary_df["t"], summary_df["cross_radius_95"], marker="o")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel("lag capturing 95% of cross-frame influence")
    ax.set_title("95% radius of the remaining cross-frame influence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "06_receptive_field_vs_t.png", dpi=180)
    plt.close(fig)


def plot_window_errors(window_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for t, group in window_df.groupby("t"):
        if t in (0.05, 0.2, 0.7, 1.5):
            ax.semilogy(group["window_radius"], group["relative_mse"], marker="o", label=f"t={t:g}")
    ax.set_xlabel("window radius L")
    ax.set_ylabel("relative MSE for central score block")
    ax.set_title("How much context is needed for the central score?")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "07_window_approximation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    bundle_root = Path(__file__).resolve().parents[1]
    root = bundle_root / "raw_reproduction" / "surrogate"
    figdir = root / "figures"
    datadir = root / "data"
    figdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    y, z = sample_clean_trajectories(rng, cfg)
    plot_example_trajectory(y, z, cfg, figdir)

    cache: Dict[Tuple[Tuple[int, ...], float], RingPosteriorLookup] = {}
    indices = np.arange(cfg.T)
    center = cfg.T // 2

    summary_rows = []
    profile_rows = []
    heatmaps: Dict[float, Array] = {}
    window_rows = []

    # Use independent corruption noise at each t, but fixed clean trajectories.
    for t in cfg.t_values:
        x = ou_corrupt(rng, y, t)
        full_score, full_jac = exact_score_and_jacobian(x, indices, t, cfg.lam, cfg.sigma, cache, cfg)
        norms = block_frobenius(full_jac)
        mean_norm = np.mean(norms, axis=0)

        if any(abs(t - hs) < 1e-12 for hs in cfg.selected_heatmap_t):
            heatmaps[t] = mean_norm

        # Lag profile averaged over all pairs with the same lag.
        for lag in range(cfg.T):
            vals = [mean_norm[i, j] for i in range(cfg.T) for j in range(cfg.T) if abs(i - j) == lag]
            profile_rows.append({"t": t, "lag": lag, "mean_block_norm": float(np.mean(vals))})

        row = mean_norm[center]
        off = row.copy(); off[center] = 0.0
        total_cross = float(np.sum(off))
        cumulative = 0.0
        radius95 = 0
        if total_cross > 1e-10:
            for L in range(1, cfg.T):
                lo, hi = max(0, center - L), min(cfg.T, center + L + 1)
                cumulative = float(np.sum(off[lo:hi]))
                if cumulative / total_cross >= 0.95:
                    radius95 = L
                    break

        # Compare the central full-joint score to the exact one-frame marginal score.
        # This directly tests whether the rest of the trajectory helps denoise frame k.
        marginal_center, _ = exact_score_and_jacobian(
            x[:, center : center + 1], np.array([center]), t, cfg.lam, cfg.sigma, cache, cfg
        )
        target_center = full_score[:, center]
        mse_gap = float(np.mean((target_center - marginal_center[:, 0]) ** 2))
        score_power = float(np.mean(target_center**2))
        rel_gap = mse_gap / max(score_power, 1e-14)

        summary_rows.append(
            {
                "t": t,
                "cross_influence_mass": total_cross,
                "cross_radius_95": radius95,
                "joint_vs_marginal_relative_mse": rel_gap,
                "mean_score_power": score_power,
            }
        )

        # Window approximation for selected t values.
        if t in (0.05, 0.2, 0.7, 1.5):
            target = full_score[:, center]
            target_power = float(np.mean(target**2))
            for L in range(cfg.max_window_radius + 1):
                lo, hi = max(0, center - L), min(cfg.T, center + L + 1)
                sub_idx = np.arange(lo, hi)
                sub_score, _ = exact_score_and_jacobian(
                    x[:, lo:hi], sub_idx, t, cfg.lam, cfg.sigma, cache, cfg
                )
                approx = sub_score[:, center - lo]
                rel = float(np.mean((approx - target) ** 2) / max(target_power, 1e-14))
                window_rows.append({"t": t, "window_radius": L, "relative_mse": rel})

    summary_df = pd.DataFrame(summary_rows)
    profile_df = pd.DataFrame(profile_rows)
    window_df = pd.DataFrame(window_rows)

    summary_df.to_csv(datadir / "summary.csv", index=False)
    profile_df.to_csv(datadir / "influence_profiles.csv", index=False)
    window_df.to_csv(datadir / "window_errors.csv", index=False)

    plot_heatmaps(heatmaps, figdir)
    plot_influence_profiles(profile_df, figdir)
    plot_summary(summary_df, figdir)
    plot_window_errors(window_df, figdir)

    # Validations.
    t_val = 0.4
    x_val = ou_corrupt(rng, y[:1], t_val)[0]
    fd_err = finite_difference_validation(x_val, indices, t_val, cfg, cache)

    # Gauge validation: de-rotate a lab-frame noisy sample, compute score, rotate back,
    # and verify the score norm is unchanged blockwise.
    x_y = ou_corrupt(rng, y[:1], 0.7)
    x_z = rotate_blocks(x_y, cfg.psi, inverse=False)
    x_back = rotate_blocks(x_z, cfg.psi, inverse=True)
    s_y, _ = exact_score_and_jacobian(x_y, indices, 0.7, cfg.lam, cfg.sigma, cache, cfg)
    s_z_from_gauge = rotate_blocks(s_y, cfg.psi, inverse=False)
    gauge_x_err = float(np.max(np.abs(x_back - x_y)))
    gauge_norm_err = float(
        np.max(np.abs(np.linalg.norm(s_z_from_gauge, axis=-1) - np.linalg.norm(s_y, axis=-1)))
    )

    # Clean score stationarity check on a perfect co-rotating path lying on the ring.
    theta0 = 0.37
    perfect = np.tile(np.array([math.cos(theta0), math.sin(theta0)]), (cfg.T, 1))[None]
    clean_zero = float(np.max(np.abs(clean_score(perfect, cfg.lam, cfg.sigma))))

    validation = {
        "finite_difference_max_abs_error": fd_err,
        "gauge_roundtrip_max_abs_error": gauge_x_err,
        "gauge_score_block_norm_max_abs_error": gauge_norm_err,
        "clean_perfect_path_score_max_abs": clean_zero,
    }
    (root / "validation.json").write_text(json.dumps(validation, indent=2))
    (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nWINDOW ERRORS")
    print(window_df.to_string(index=False))
    print("\nVALIDATION")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
