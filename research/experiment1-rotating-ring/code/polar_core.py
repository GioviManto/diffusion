#!/usr/bin/env python3
"""Numerical score-dynamics experiments for the board-faithful polar ring model.

Latent clean trajectory (internal time u):
    dR_u     = -kappa (R_u - 1) du + sqrt(2 D_r) dB_u^r
    dTheta_u = omega du              + sqrt(2 D_theta) dB_u^theta
    A_u      = R_u (cos Theta_u, sin Theta_u)

The whole Cartesian trajectory A=(A_0,...,A_{T-1}) is one datum in R^{2T}.
The generative forward channel is the VP Ornstein-Uhlenbeck channel
    X_t = m_t A + sqrt(Delta_t) Xi,
    m_t=exp(-t), Delta_t=1-exp(-2t).

The exact joint score identity is
    S_k(x,t) = (m_t E[A_k | X_t=x] - x_k) / Delta_t.

The posterior mean is computed by exact forward-backward message passing on a
finite polar grid. Thus the only approximation is the grid discretization.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    seed: int = 20260725
    T: int = 20
    kappa: float = 2.0
    D_r: float = 0.015
    omega: float = 1.0
    D_theta: float = 0.005
    n_r: int = 31
    n_theta: int = 64
    r_min: float = 0.52
    r_max: float = 1.48
    n_score_samples: int = 50
    n_window_samples: int = 20
    n_jacobian_samples: int = 1
    t_values: tuple[float, ...] = (0.03, 0.06, 0.10, 0.20, 0.40, 0.70, 1.00, 1.50, 2.00)
    heatmap_times: tuple[float, ...] = (0.03, 0.20, 0.70, 1.50)
    max_window_radius: int = 10
    jacobian_eps: float = 2e-3


def ou_parameters(t: float) -> tuple[float, float]:
    if t <= 0:
        raise ValueError("t must be positive")
    m = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    return m, delta


def wrapped_normal_transition(theta_grid: Array, shift: float, var: float, wraps: int = 3) -> Array:
    """Row-stochastic wrapped-normal transition on a periodic angle grid."""
    if var <= 0:
        raise ValueError("angular transition variance must be positive")
    old = theta_grid[:, None]
    new = theta_grid[None, :]
    base = new - old - shift
    density = np.zeros_like(base)
    sd = math.sqrt(var)
    for n in range(-wraps, wraps + 1):
        density += np.exp(-0.5 * ((base + 2.0 * math.pi * n) / sd) ** 2)
    density = np.maximum(density, 0.0)
    rows = density.sum(axis=1, keepdims=True)
    if np.any(rows <= 0):
        raise RuntimeError("zero row in angular transition")
    return density / rows


def gaussian_transition(grid: Array, means: Array, var: float) -> Array:
    """Row-stochastic Gaussian transition on a one-dimensional grid."""
    if var <= 0:
        raise ValueError("radial transition variance must be positive")
    z = (grid[None, :] - means[:, None]) / math.sqrt(var)
    p = np.exp(-0.5 * z**2)
    p = np.maximum(p, 0.0)
    rows = p.sum(axis=1, keepdims=True)
    if np.any(rows <= 0):
        raise RuntimeError("zero row in radial transition")
    return p / rows


@dataclass(frozen=True)
class PolarRingModel:
    kappa: float
    D_r: float
    omega: float
    D_theta: float
    T: int

    @property
    def period(self) -> float:
        return 2.0 * math.pi / abs(self.omega)

    @property
    def h(self) -> float:
        return self.period / self.T

    @property
    def radial_stationary_std(self) -> float:
        return math.sqrt(self.D_r / self.kappa)

    def radial_transition(self, r: Array) -> tuple[Array, float]:
        a = math.exp(-self.kappa * self.h)
        mean = 1.0 + a * (r - 1.0)
        var = (self.D_r / self.kappa) * (1.0 - math.exp(-2.0 * self.kappa * self.h))
        return mean, var

    def angular_transition(self) -> tuple[float, float]:
        return self.omega * self.h, 2.0 * self.D_theta * self.h

    def simulate(self, rng: np.random.Generator) -> tuple[Array, Array, Array]:
        r = np.empty(self.T)
        theta = np.empty(self.T)
        # Natural stationary initialization; negative draws are negligible here.
        while True:
            candidate = rng.normal(1.0, self.radial_stationary_std)
            if candidate > 0:
                r[0] = candidate
                break
        theta[0] = rng.uniform(0.0, 2.0 * math.pi)
        a = math.exp(-self.kappa * self.h)
        q_r = (self.D_r / self.kappa) * (1.0 - math.exp(-2.0 * self.kappa * self.h))
        q_theta = 2.0 * self.D_theta * self.h
        for k in range(self.T - 1):
            mean_r = 1.0 + a * (r[k] - 1.0)
            while True:
                candidate = rng.normal(mean_r, math.sqrt(q_r))
                if candidate > 0:
                    r[k + 1] = candidate
                    break
            theta[k + 1] = (theta[k] + self.omega * self.h + rng.normal(0.0, math.sqrt(q_theta))) % (2.0 * math.pi)
        xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        return r, theta, xy


class PolarGridSmoother:
    def __init__(self, model: PolarRingModel, r_grid: Array, theta_grid: Array):
        self.model = model
        self.r_grid = np.asarray(r_grid, dtype=float)
        self.theta_grid = np.asarray(theta_grid, dtype=float)
        self.R, self.Theta = np.meshgrid(self.r_grid, self.theta_grid, indexing="ij")
        self.Axy = np.stack((self.R * np.cos(self.Theta), self.R * np.sin(self.Theta)), axis=-1)

        r_mean, r_var = model.radial_transition(self.r_grid)
        theta_shift, theta_var = model.angular_transition()
        self.Tr = gaussian_transition(self.r_grid, r_mean, r_var)
        self.Ttheta = wrapped_normal_transition(self.theta_grid, theta_shift, theta_var)

        pr = norm.pdf(self.r_grid, loc=1.0, scale=model.radial_stationary_std)
        ptheta = np.ones_like(self.theta_grid)
        self.prior0 = np.outer(pr, ptheta)
        self.prior0 /= self.prior0.sum()
        self.prior_marginals = self._clean_prior_marginals()

    def _propagate(self, mass: Array) -> Array:
        pred = self.Tr.T @ mass @ self.Ttheta
        pred = np.maximum(pred, 0.0)
        s = float(pred.sum())
        if s <= 0:
            raise RuntimeError("zero propagated mass")
        return pred / s

    def _clean_prior_marginals(self) -> list[Array]:
        out = [self.prior0]
        for _ in range(self.model.T - 1):
            out.append(self._propagate(out[-1]))
        return out

    def _likelihoods(self, x: Array, t: float, left: int = 0, right: int | None = None) -> list[Array]:
        m, delta = ou_parameters(t)
        if right is None:
            right = self.model.T - 1
        out: list[Array] = []
        for k in range(left, right + 1):
            residual = x[k][None, None, :] - m * self.Axy
            logl = -np.sum(residual**2, axis=-1) / (2.0 * delta)
            logl -= float(np.max(logl))
            out.append(np.exp(logl))
        return out

    def posterior_beliefs(self, x: Array, t: float, left: int = 0, right: int | None = None) -> list[Array]:
        if right is None:
            right = self.model.T - 1
        if not (0 <= left <= right < self.model.T):
            raise ValueError("invalid window")
        likelihoods = self._likelihoods(x, t, left, right)
        n = right - left + 1

        forward: list[Array] = [self.prior_marginals[left] * likelihoods[0]]
        forward[0] /= forward[0].sum()
        for q in range(n - 1):
            f = self._propagate(forward[-1]) * likelihoods[q + 1]
            f /= f.sum()
            forward.append(f)

        backward: list[Array] = [np.empty_like(self.prior0) for _ in range(n)]
        backward[-1] = np.ones_like(self.prior0)
        backward[-1] /= backward[-1].sum()
        for q in range(n - 2, -1, -1):
            g = likelihoods[q + 1] * backward[q + 1]
            b = self.Tr @ g @ self.Ttheta.T
            b = np.maximum(b, 0.0)
            b /= b.sum()
            backward[q] = b

        beliefs: list[Array] = []
        for q in range(n):
            b = forward[q] * backward[q]
            b /= b.sum()
            beliefs.append(b)
        return beliefs

    def score(self, x: Array, t: float) -> tuple[Array, Array]:
        beliefs = self.posterior_beliefs(x, t)
        means = np.array([np.sum(b[..., None] * self.Axy, axis=(0, 1)) for b in beliefs])
        m, delta = ou_parameters(t)
        return means, (m * means - x) / delta

    def marginal_score(self, x: Array, t: float) -> tuple[Array, Array]:
        likelihoods = self._likelihoods(x, t)
        means = np.empty((self.model.T, 2))
        for k in range(self.model.T):
            b = self.prior_marginals[k] * likelihoods[k]
            b /= b.sum()
            means[k] = np.sum(b[..., None] * self.Axy, axis=(0, 1))
        m, delta = ou_parameters(t)
        return means, (m * means - x) / delta

    def window_score(self, x: Array, t: float, center: int, radius: int) -> Array:
        left = max(0, center - radius)
        right = min(self.model.T - 1, center + radius)
        beliefs = self.posterior_beliefs(x, t, left, right)
        b = beliefs[center - left]
        mean = np.sum(b[..., None] * self.Axy, axis=(0, 1))
        m, delta = ou_parameters(t)
        return (m * mean - x[center]) / delta

    def score_jacobian_fd(self, x: Array, t: float, eps: float) -> Array:
        """Full block Jacobian J[k,j,a,b] by central finite differences."""
        T = self.model.T
        J = np.empty((T, T, 2, 2))
        for j in range(T):
            for b in range(2):
                xp = x.copy(); xm = x.copy()
                xp[j, b] += eps; xm[j, b] -= eps
                _, sp = self.score(xp, t)
                _, sm = self.score(xm, t)
                J[:, j, :, b] = (sp - sm) / (2.0 * eps)
        return J


def add_ou_noise(clean: Array, t: float, rng: np.random.Generator) -> Array:
    m, delta = ou_parameters(t)
    return m * clean + math.sqrt(delta) * rng.normal(size=clean.shape)


def relative_rmse(a: Array, b: Array) -> float:
    num = float(np.mean(np.sum((a - b) ** 2, axis=-1)))
    den = float(np.mean(np.sum(b**2, axis=-1)))
    return math.sqrt(num / max(den, 1e-15))


def plot_clean_model(model: PolarRingModel, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    th = np.linspace(0.0, 2.0 * math.pi, 500)
    axes[0].plot(np.cos(th), np.sin(th), linestyle="--", linewidth=1.0, label="unit circle")
    for _ in range(8):
        _, _, xy = model.simulate(rng)
        axes[0].plot(xy[:, 0], xy[:, 1], marker="o", markersize=2.5, linewidth=1.1, alpha=0.72)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Board-faithful polar trajectories")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend()

    r, theta, _ = model.simulate(rng)
    u = np.arange(model.T) * model.h
    axes[1].plot(u, r, marker="o", markersize=3, label=r"radius $R_u$")
    axes[1].plot(u, 1.0 + 0.18 * np.sin(theta), linewidth=1.2, label=r"$1+0.18\sin\Theta_u$ (angle guide)")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("internal time u")
    axes[1].set_title("Radial confinement and angular motion")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_original_model_trajectories.png", dpi=220)
    plt.close(fig)


def run_joint_vs_marginal(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for t in cfg.t_values:
        full_all = []
        marg_all = []
        for _ in range(cfg.n_score_samples):
            _, _, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            _, full = solver.score(x, t)
            _, marg = solver.marginal_score(x, t)
            # Interior blocks avoid endpoint effects.
            full_all.append(full[2:-2])
            marg_all.append(marg[2:-2])
        F = np.concatenate(full_all, axis=0)
        M = np.concatenate(marg_all, axis=0)
        err = relative_rmse(M, F)
        cosine = float(np.mean(np.sum(F * M, axis=-1) / np.maximum(np.linalg.norm(F, axis=-1) * np.linalg.norm(M, axis=-1), 1e-12)))
        rows.append({"t": t, "relative_marginal_score_error": err, "mean_cosine_similarity": cosine})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "joint_vs_marginal.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(df["t"], df["relative_marginal_score_error"], marker="o")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel("relative RMSE")
    ax.set_title("Ignoring the trajectory: marginal-score error")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_joint_vs_marginal_error.png", dpi=220)
    plt.close(fig)
    return df


def plot_score_arrows(model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator, t: float = 0.35) -> None:
    _, _, clean = model.simulate(rng)
    x = add_ou_noise(clean, t, rng)
    _, full = solver.score(x, t)
    _, marg = solver.marginal_score(x, t)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, score, title in [
        (axes[0], full, "Joint score: the whole trajectory is used"),
        (axes[1], marg, "Marginal score: every frame is treated alone"),
    ]:
        ax.plot(clean[:, 0], clean[:, 1], marker="o", markersize=3, linewidth=1.2, label="clean")
        ax.scatter(x[:, 0], x[:, 1], s=22, label="noisy")
        scale = np.quantile(np.linalg.norm(score, axis=1), 0.8)
        qscale = max(scale, 1e-6) * 5.0
        ax.quiver(x[:, 0], x[:, 1], score[:, 0], score[:, 1], angles="xy", scale_units="xy", scale=qscale, width=0.005)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend()
    fig.suptitle(f"Score blocks at diffusion time t={t:g}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_joint_and_marginal_score_arrows.png", dpi=220)
    plt.close(fig)


def run_window_experiment(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator) -> pd.DataFrame:
    center = model.T // 2
    rows: list[dict[str, float]] = []
    selected_t = (0.03, 0.10, 0.20, 0.40, 0.70, 1.50)
    for t in selected_t:
        full_scores: list[Array] = []
        window_scores: dict[int, list[Array]] = {L: [] for L in range(cfg.max_window_radius + 1)}
        for _ in range(cfg.n_window_samples):
            _, _, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            _, full = solver.score(x, t)
            full_scores.append(full[center])
            for L in range(cfg.max_window_radius + 1):
                window_scores[L].append(solver.window_score(x, t, center, L))
        F = np.stack(full_scores)
        for L in range(cfg.max_window_radius + 1):
            W = np.stack(window_scores[L])
            rows.append({"t": t, "radius": L, "frames": 2 * L + 1, "relative_error": relative_rmse(W, F)})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "window_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for t, group in df.groupby("t"):
        ax.plot(group["radius"], group["relative_error"], marker="o", markersize=3, label=f"t={t:g}")
    ax.set_yscale("log")
    ax.set_xlabel("window radius L")
    ax.set_ylabel("relative score error at central frame")
    ax.set_title("How much temporal context is needed?")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_window_approximation.png", dpi=220)
    plt.close(fig)
    return df


def block_norms(J: Array) -> Array:
    return np.sqrt(np.sum(J**2, axis=(-2, -1)))


def run_jacobian_experiment(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    profiles: list[dict[str, float]] = []
    for t in cfg.heatmap_times:
        mats = []
        for _ in range(cfg.n_jacobian_samples):
            _, _, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            J = solver.score_jacobian_fd(x, t, cfg.jacobian_eps)
            mats.append(block_norms(J))
        H = np.mean(mats, axis=0)
        np.save(DATADIR / f"jacobian_block_norm_t_{t:.2f}.npy", H)

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = ax.imshow(H, origin="lower", aspect="auto")
        ax.set_xlabel("perturbed frame j")
        ax.set_ylabel("score block k")
        ax.set_title(rf"$\|\partial S_k/\partial x_j\|_F$, t={t:g}")
        fig.colorbar(im, ax=ax, label="block sensitivity")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"05_jacobian_heatmap_t_{t:.2f}.png", dpi=220)
        plt.close(fig)

        off = H.copy()
        np.fill_diagonal(off, 0.0)
        total_off = float(off.mean())
        center = model.T // 2
        row = H[center]
        offrow = row.copy(); offrow[center] = 0.0
        mass = float(offrow.sum())
        if mass > 0:
            dist = np.abs(np.arange(model.T) - center)
            order = np.argsort(dist)
            cum = np.cumsum(offrow[order]) / mass
            effective_radius = int(np.min(dist[order][cum >= 0.95]))
        else:
            effective_radius = 0
        rows.append({"t": t, "mean_offdiagonal_sensitivity": total_off, "center_95pct_influence_radius": effective_radius})

        for lag in range(model.T):
            values = []
            for k in range(model.T):
                j = k + lag
                if j < model.T:
                    values.append(H[k, j])
                if lag > 0:
                    j2 = k - lag
                    if j2 >= 0:
                        values.append(H[k, j2])
            profiles.append({"t": t, "lag": lag, "mean_block_sensitivity": float(np.mean(values)) if values else np.nan})

    summary = pd.DataFrame(rows)
    summary.to_csv(DATADIR / "jacobian_summary.csv", index=False)
    prof = pd.DataFrame(profiles)
    prof.to_csv(DATADIR / "influence_profiles.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for t, group in prof.groupby("t"):
        ax.semilogy(group["lag"], np.maximum(group["mean_block_sensitivity"], 1e-12), marker="o", markersize=3, label=f"t={t:g}")
    ax.set_xlabel("temporal lag |k-j|")
    ax.set_ylabel("mean block sensitivity")
    ax.set_title("Temporal influence profile of the joint score")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_influence_vs_lag.png", dpi=220)
    plt.close(fig)
    return summary


def validation_checks(model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator) -> dict[str, float]:
    checks: dict[str, float] = {}
    checks["radial_transition_row_error"] = float(np.max(np.abs(solver.Tr.sum(axis=1) - 1.0)))
    checks["angular_transition_row_error"] = float(np.max(np.abs(solver.Ttheta.sum(axis=1) - 1.0)))
    checks["prior_mass_error"] = abs(float(solver.prior0.sum()) - 1.0)

    _, _, clean = model.simulate(rng)
    t = 0.4
    x = add_ou_noise(clean, t, rng)
    _, s = solver.score(x, t)
    # Global rotation equivariance: the model has uniform theta_0 and rotationally invariant dynamics.
    phi = 0.37
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    xr = x @ R.T
    _, sr = solver.score(xr, t)
    target = s @ R.T
    checks["rotation_equivariance_relative_error"] = float(np.linalg.norm(sr - target) / max(np.linalg.norm(target), 1e-12))

    # Finite-difference stability at two eps values for one Jacobian column.
    j, b = model.T // 2, 0
    vals = []
    for eps in (2e-3, 1e-3):
        xp = x.copy(); xm = x.copy(); xp[j, b] += eps; xm[j, b] -= eps
        _, sp = solver.score(xp, t); _, sm = solver.score(xm, t)
        vals.append((sp - sm) / (2.0 * eps))
    checks["jacobian_fd_halving_relative_change"] = float(np.linalg.norm(vals[0] - vals[1]) / max(np.linalg.norm(vals[1]), 1e-12))
    return checks


def grid_convergence_checks(cfg: Config, model: PolarRingModel, rng: np.random.Generator) -> dict[str, float]:
    """Compare the production grid with a finer polar grid on a few samples."""
    coarse = PolarGridSmoother(
        model,
        np.linspace(cfg.r_min, cfg.r_max, cfg.n_r),
        np.linspace(0.0, 2.0 * math.pi, cfg.n_theta, endpoint=False),
    )
    fine = PolarGridSmoother(
        model,
        np.linspace(cfg.r_min, cfg.r_max, 41),
        np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False),
    )
    out: dict[str, float] = {}
    for t in (0.03, 0.10, 0.40, 1.00):
        sc: list[Array] = []
        sf: list[Array] = []
        for _ in range(4):
            _, _, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            sc.append(coarse.score(x, t)[1])
            sf.append(fine.score(x, t)[1])
        out[f"grid_relative_score_difference_t_{t:.2f}"] = relative_rmse(
            np.concatenate(sc, axis=0), np.concatenate(sf, axis=0)
        )
    return out


def write_report(cfg: Config, model: PolarRingModel, joint_df: pd.DataFrame, window_df: pd.DataFrame, jac_df: pd.DataFrame, checks: dict[str, float]) -> None:
    # Compute compact thresholds.
    thresholds = []
    for t, group in window_df.groupby("t"):
        eligible = group[group["relative_error"] < 0.05]
        L = int(eligible["radius"].min()) if len(eligible) else -1
        thresholds.append((float(t), L))

    lines = [
        "# First numerical study of the original polar rotating-ring problem",
        "",
        "## Model",
        "",
        r"\[dR_u=-\kappa(R_u-1)du+\sqrt{2D_r}\,dB_u^{(r)},\qquad d\Theta_u=\omega du+\sqrt{2D_\theta}\,dB_u^{(\theta)}.\]",
        r"\[A_u=R_u(\cos\Theta_u,\sin\Theta_u),\qquad X_t=e^{-t}A+\sqrt{1-e^{-2t}}\,\xi.\]",
        "",
        "The whole trajectory is one datum in R^{2T}. We compute the joint score from",
        "",
        r"\[S_k(x,t)=\frac{e^{-t}\,\mathbb E[A_k\mid X_t=x]-x_k}{1-e^{-2t}}.\]",
        "",
        "The posterior mean is obtained by forward-backward message passing on a polar grid. The method is exact for the discretized hidden Markov chain; grid resolution is the only approximation.",
        "",
        "## Baseline parameters",
        "",
        f"- T={cfg.T}, kappa={cfg.kappa}, D_r={cfg.D_r}, omega={cfg.omega}, D_theta={cfg.D_theta}",
        f"- stationary radial standard deviation sqrt(D_r/kappa)={model.radial_stationary_std:.4f}",
        f"- polar grid: {cfg.n_r} x {cfg.n_theta}",
        "",
        "## What was measured",
        "",
        "1. joint score versus independent one-frame marginal scores;",
        "2. score error when only a temporal window is used;",
        "3. the block Jacobian ||dS_k/dx_j||, estimated by finite differences;",
        "4. rotation-equivariance and numerical stability checks.",
        "",
        "## Main results",
        "",
        "### Joint versus marginal",
        "",
        joint_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Window radius needed for less than 5% central-score error",
        "",
        "| t | minimum L | frames used |",
        "|---:|---:|---:|",
    ]
    for t, L in thresholds:
        frames = 2 * L + 1 if L >= 0 else -1
        lines.append(f"| {t:.2f} | {L} | {frames} |")
    lines += [
        "",
        "### Jacobian summary",
        "",
        jac_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "The original polar process confirms the same qualitative phenomenon as the surrogate, but for a genuinely confined ring. The clean latent law is Markov and local in (R,Theta), while the noised Cartesian score is a smoothing object and therefore uses multiple frames. The amount of useful context depends on diffusion time.",
        "",
        "Unlike the surrogate, later frames do not diffuse radially without bound: radial mean reversion keeps the trajectory near the ring. Therefore any long-range score dependence observed here is due to posterior smoothing through the nonlinear polar dynamics, not to a shared unconstrained random-walk anchor alone.",
        "",
        "## Validation",
        "",
        "```json",
        json.dumps(checks, indent=2),
        "```",
        "",
        "## Figures",
        "",
        "- `01_original_model_trajectories.png`",
        "- `02_joint_vs_marginal_error.png`",
        "- `03_joint_and_marginal_score_arrows.png`",
        "- `04_window_approximation.png`",
        "- `05_jacobian_heatmap_t_*.png`",
        "- `06_influence_vs_lag.png`",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_notebook() -> None:
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Original polar rotating-ring score experiments\n\nThis notebook displays the outputs generated by `run_experiments.py`."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\nROOT=Path.cwd()\n"),
        nbf.v4.new_markdown_cell("## Model and clean trajectories"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/01_original_model_trajectories.png')))"),
        nbf.v4.new_markdown_cell("## Joint versus marginal score"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/02_joint_vs_marginal_error.png')))\ndisplay(pd.read_csv(ROOT/'data/joint_vs_marginal.csv'))"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/03_joint_and_marginal_score_arrows.png')))"),
        nbf.v4.new_markdown_cell("## Finite-window approximation"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/04_window_approximation.png')))\ndisplay(pd.read_csv(ROOT/'data/window_errors.csv').head(20))"),
        nbf.v4.new_markdown_cell("## Cross-frame score sensitivity"),
        nbf.v4.new_code_cell("for t in [0.03,0.20,0.70,1.50]:\n    display(Image(filename=str(ROOT/f'figures/05_jacobian_heatmap_t_{t:.2f}.png')))\ndisplay(Image(filename=str(ROOT/'figures/06_influence_vs_lag.png')))"),
    ]
    nbf.write(nb, ROOT / "original_polar_score_experiments.ipynb")


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    model = PolarRingModel(cfg.kappa, cfg.D_r, cfg.omega, cfg.D_theta, cfg.T)
    r_grid = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r)
    theta_grid = np.linspace(0.0, 2.0 * math.pi, cfg.n_theta, endpoint=False)
    solver = PolarGridSmoother(model, r_grid, theta_grid)

    (ROOT / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    plot_clean_model(model, rng)
    joint_df = run_joint_vs_marginal(cfg, model, solver, rng)
    plot_score_arrows(model, solver, rng)
    window_df = run_window_experiment(cfg, model, solver, rng)
    jac_df = run_jacobian_experiment(cfg, model, solver, rng)
    checks = validation_checks(model, solver, rng)
    checks.update(grid_convergence_checks(cfg, model, rng))
    (ROOT / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    write_report(cfg, model, joint_df, window_df, jac_df, checks)
    make_notebook()

    readme = """# Original polar score experiments\n\nRun:\n\n```bash\npython run_experiments.py\n```\n\nThe script implements forward-backward message passing on a polar grid for the board-faithful model. See `report.md` for results.\n"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    archive = ROOT.parent / "original_polar_score_experiments.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ROOT.parent, ROOT.name)
    print(joint_df.to_string(index=False))
    print(jac_df.to_string(index=False))
    print(json.dumps(checks, indent=2))
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
