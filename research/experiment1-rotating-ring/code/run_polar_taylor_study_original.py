#!/usr/bin/env python3
"""Exact-grid and Taylor-linearized score study for the polar rotating-ring model."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polar_core import PolarRingModel, PolarGridSmoother, add_ou_noise, ou_parameters

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ROOT = BUNDLE_ROOT / "raw_reproduction" / "polar_taylor"
FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    seed: int = 20260726
    T: int = 20
    kappa: float = 2.0
    D_r: float = 0.015
    omega: float = 1.0
    D_theta: float = 0.005
    n_r: int = 31
    n_theta: int = 64
    r_min: float = 0.52
    r_max: float = 1.48
    field_n: int = 23
    field_extent: float = 1.55
    field_times: tuple[float, ...] = (0.05, 0.25, 0.70, 1.50)
    metric_times: tuple[float, ...] = (0.03, 0.08, 0.15, 0.25, 0.40, 0.70, 1.00, 1.50, 2.00)
    metric_samples: int = 32


def rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def estimate_initial_phase(x: np.ndarray, model: PolarRingModel) -> float:
    """Estimate the common phase by de-rotating and averaging the noisy frames."""
    aligned = np.empty_like(x)
    for k in range(model.T):
        aligned[k] = rotation(-model.omega * model.h * k) @ x[k]
    v = aligned.sum(axis=0)
    if np.linalg.norm(v) < 1e-12:
        return 0.0
    return math.atan2(v[1], v[0])


def linearized_gaussian(model: PolarRingModel, t: float, theta0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, covariance, and precision of the first-order phase-conditioned model.

    Let rho_k=R_k-1 and phi_k=Theta_k-(theta0+k omega h). Around rho=phi=0,
        A_k = (1+rho_k)(cos(bar_theta_k+phi_k), sin(...))
            ~= bar_a_k + e_r,k rho_k + e_theta,k phi_k.
    The latent dynamics for rho and phi are linear Gaussian, so the stacked
    Cartesian approximation is Gaussian.
    """
    T = model.T
    a = math.exp(-model.kappa * model.h)
    var_r = model.D_r / model.kappa
    q_theta = 2.0 * model.D_theta * model.h

    # Latent covariance in ordering (rho_0, phi_0, rho_1, phi_1, ...).
    Cz = np.zeros((2 * T, 2 * T), dtype=float)
    for i in range(T):
        for j in range(T):
            Cz[2 * i, 2 * j] = var_r * (a ** abs(i - j))
            # Phase-conditioned branch: phi_0=0, then a Brownian random walk.
            Cz[2 * i + 1, 2 * j + 1] = q_theta * min(i, j)

    H = np.zeros((2 * T, 2 * T), dtype=float)
    mean_clean = np.empty((T, 2), dtype=float)
    for k in range(T):
        th = theta0 + model.omega * model.h * k
        er = np.array([math.cos(th), math.sin(th)])
        et = np.array([-math.sin(th), math.cos(th)])
        mean_clean[k] = er
        H[2 * k:2 * k + 2, 2 * k] = er
        H[2 * k:2 * k + 2, 2 * k + 1] = et

    C_clean = H @ Cz @ H.T
    m, delta = ou_parameters(t)
    mean_noisy = m * mean_clean.reshape(-1)
    Sigma = m * m * C_clean + delta * np.eye(2 * T)
    precision = np.linalg.inv(Sigma)
    return mean_noisy, Sigma, precision


def linearized_score(x: np.ndarray, model: PolarRingModel, t: float, theta0: float | None = None) -> np.ndarray:
    if theta0 is None:
        theta0 = estimate_initial_phase(x, model)
    mean, _, precision = linearized_gaussian(model, t, theta0)
    return (-precision @ (x.reshape(-1) - mean)).reshape(model.T, 2)


def cavity_mass(solver: PolarGridSmoother, x: np.ndarray, t: float, k: int) -> np.ndarray:
    """Latent mass p(H_k | X_{-k}=x_{-k}) on the polar grid."""
    likelihoods = solver._likelihoods(x, t)

    left = solver.prior0.copy()
    for j in range(k):
        left *= likelihoods[j]
        left /= left.sum()
        left = solver._propagate(left)

    right = np.ones_like(solver.prior0)
    right /= right.sum()
    for j in range(solver.model.T - 1, k, -1):
        g = likelihoods[j] * right
        right = solver.Tr @ g @ solver.Ttheta.T
        right = np.maximum(right, 0.0)
        right /= right.sum()

    cav = left * right
    cav /= cav.sum()
    return cav


def exact_score_slice(solver: PolarGridSmoother, x: np.ndarray, t: float, k: int,
                      points: np.ndarray, cavity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate conditional log density, posterior mean and exact joint score over x_k points."""
    if cavity is None:
        cavity = cavity_mass(solver, x, t, k)
    m, delta = ou_parameters(t)
    residual = points[:, None, None, :] - m * solver.Axy[None, ...]
    logl = -np.sum(residual * residual, axis=-1) / (2.0 * delta)
    maxlog = np.max(logl, axis=(1, 2), keepdims=True)
    weights = cavity[None, ...] * np.exp(logl - maxlog)
    z = weights.sum(axis=(1, 2))
    means = np.einsum("mij,ijc->mc", weights, solver.Axy) / z[:, None]
    scores = (m * means - points) / delta
    log_density = np.log(z) + maxlog[:, 0, 0]
    return log_density, means, scores


def marginal_score_slice(solver: PolarGridSmoother, t: float, k: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prior = solver.prior_marginals[k]
    m, delta = ou_parameters(t)
    residual = points[:, None, None, :] - m * solver.Axy[None, ...]
    logl = -np.sum(residual * residual, axis=-1) / (2.0 * delta)
    maxlog = np.max(logl, axis=(1, 2), keepdims=True)
    weights = prior[None, ...] * np.exp(logl - maxlog)
    z = weights.sum(axis=(1, 2))
    means = np.einsum("mij,ijc->mc", weights, solver.Axy) / z[:, None]
    return np.log(z) + maxlog[:, 0, 0], (m * means - points) / delta


def linearized_slice(x: np.ndarray, model: PolarRingModel, t: float, k: int,
                     points: np.ndarray, theta0: float) -> tuple[np.ndarray, np.ndarray]:
    mean, Sigma, Q = linearized_gaussian(model, t, theta0)
    base = x.reshape(-1).copy()
    scores = np.empty_like(points)
    logd = np.empty(points.shape[0])
    sl = slice(2 * k, 2 * k + 2)
    for n, p in enumerate(points):
        y = base.copy()
        y[sl] = p
        d = y - mean
        s = -Q @ d
        scores[n] = s[sl]
        logd[n] = -0.5 * d @ Q @ d
    return logd, scores


def draw_field(ax, xx, yy, logd, scores, title: str, observed: np.ndarray | None = None,
               clean: np.ndarray | None = None) -> None:
    n = xx.shape[0]
    ld = logd.reshape(n, n)
    sc = scores.reshape(n, n, 2)
    ld = ld - np.nanmax(ld)
    ax.contourf(xx, yy, ld, levels=18)
    norms = np.linalg.norm(sc, axis=-1, keepdims=True)
    display = sc / (1.0 + norms)
    stride = 2
    ax.quiver(xx[::stride, ::stride], yy[::stride, ::stride],
              display[::stride, ::stride, 0], display[::stride, ::stride, 1],
              angles="xy", scale_units="xy", scale=1.25, width=0.004)
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(np.cos(th), np.sin(th), linestyle="--", linewidth=1.0)
    if observed is not None:
        ax.scatter([observed[0]], [observed[1]], marker="x", s=45, label="observed")
    if clean is not None:
        ax.scatter([clean[0]], [clean[1]], marker="o", s=30, label="clean")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel(r"candidate $x_k^{(1)}$")
    ax.set_ylabel(r"candidate $x_k^{(2)}$")


def plot_taylor_geometry() -> None:
    theta = 0.65
    er = np.array([math.cos(theta), math.sin(theta)])
    et = np.array([-math.sin(theta), math.cos(theta)])
    phis = np.linspace(-0.75, 0.75, 200)
    exact = np.column_stack((np.cos(theta + phis), np.sin(theta + phis)))
    tangent = er[None, :] + phis[:, None] * et[None, :]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    th = np.linspace(0, 2 * math.pi, 400)
    axes[0].plot(np.cos(th), np.sin(th), linestyle="--", linewidth=1.0)
    axes[0].plot(exact[:, 0], exact[:, 1], linewidth=2.0, label="exact arc")
    axes[0].plot(tangent[:, 0], tangent[:, 1], linewidth=1.5, label="first-order tangent")
    axes[0].quiver([er[0], er[0]], [er[1], er[1]], [0.35 * er[0], 0.35 * et[0]],
                   [0.35 * er[1], 0.35 * et[1]], angles="xy", scale_units="xy", scale=1)
    axes[0].scatter([er[0]], [er[1]], s=35)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Taylor expansion around one point of the ring")
    axes[0].legend()

    rho = np.linspace(-0.25, 0.25, 121)
    phi = np.linspace(-0.75, 0.75, 151)
    rr, pp = np.meshgrid(rho, phi, indexing="ij")
    exact_map = (1 + rr)[..., None] * np.stack((np.cos(theta + pp), np.sin(theta + pp)), axis=-1)
    linear_map = er + rr[..., None] * er + pp[..., None] * et
    err = np.linalg.norm(exact_map - linear_map, axis=-1)
    im = axes[1].pcolormesh(phi, rho, err)
    axes[1].set_xlabel(r"angular deviation $\phi$")
    axes[1].set_ylabel(r"radial deviation $\rho$")
    axes[1].set_title("Norm of the first-order Cartesian error")
    fig.colorbar(im, ax=axes[1], label="absolute error")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_taylor_geometry.png", dpi=220)
    plt.close(fig)


def plot_posteriors_on_circle(model: PolarRingModel, solver: PolarGridSmoother,
                              clean: np.ndarray, x_by_t: dict[float, np.ndarray], k: int) -> None:
    fig, axes = plt.subplots(1, len(x_by_t), figsize=(16.0, 4.0), sharex=True, sharey=True)
    for ax, (t, x) in zip(axes, x_by_t.items()):
        beliefs = solver.posterior_beliefs(x, t)
        b = beliefs[k]
        vals = b / b.max()
        ax.scatter(solver.Axy[..., 0].ravel(), solver.Axy[..., 1].ravel(),
                   c=vals.ravel(), s=10)
        th = np.linspace(0, 2 * math.pi, 400)
        ax.plot(np.cos(th), np.sin(th), linestyle="--", linewidth=1.0)
        mean = np.sum(b[..., None] * solver.Axy, axis=(0, 1))
        ax.scatter([mean[0]], [mean[1]], marker="x", s=45, label="posterior mean")
        ax.scatter([clean[k, 0]], [clean[k, 1]], marker="o", s=28, label="clean")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t={t:.2f}")
        ax.set_xlabel("latent x")
    axes[0].set_ylabel("latent y")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Exact smoothed posterior of latent frame k={k} on the polar state grid")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_posterior_on_circle.png", dpi=220)
    plt.close(fig)


def plot_score_fields(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother,
                      clean: np.ndarray, x_by_t: dict[float, np.ndarray], k: int) -> pd.DataFrame:
    axis = np.linspace(-cfg.field_extent, cfg.field_extent, cfg.field_n)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    fig_exact, axes_exact = plt.subplots(1, len(cfg.field_times), figsize=(16.5, 4.1), sharex=True, sharey=True)
    fig_lin, axes_lin = plt.subplots(1, len(cfg.field_times), figsize=(16.5, 4.1), sharex=True, sharey=True)
    fig_cmp, axes_cmp = plt.subplots(2, len(cfg.field_times), figsize=(16.5, 8.0), sharex=True, sharey=True)
    rows = []
    for col, t in enumerate(cfg.field_times):
        x = x_by_t[t]
        cav = cavity_mass(solver, x, t, k)
        loge, _, se = exact_score_slice(solver, x, t, k, points, cav)
        theta0 = estimate_initial_phase(x, model)
        logl, sl = linearized_slice(x, model, t, k, points, theta0)
        logm, sm = marginal_score_slice(solver, t, k, points)

        draw_field(axes_exact[col], xx, yy, loge, se, f"Exact joint, t={t:.2f}", x[k], clean[k])
        draw_field(axes_lin[col], xx, yy, logl, sl, f"Taylor branch, t={t:.2f}", x[k], clean[k])
        draw_field(axes_cmp[0, col], xx, yy, loge, se, f"Joint, t={t:.2f}", x[k], clean[k])
        draw_field(axes_cmp[1, col], xx, yy, logm, sm, f"Marginal, t={t:.2f}", x[k], clean[k])

        # Compare only near the high-density exact conditional region.
        mask = loge >= loge.max() - 4.0
        rel = np.sqrt(np.mean(np.sum((sl[mask] - se[mask]) ** 2, axis=1)) /
                      max(np.mean(np.sum(se[mask] ** 2, axis=1)), 1e-14))
        cos = np.sum(sl[mask] * se[mask], axis=1) / (
            np.linalg.norm(sl[mask], axis=1) * np.linalg.norm(se[mask], axis=1) + 1e-12)
        rows.append({"t": t, "local_field_relative_error": rel,
                     "local_field_mean_cosine": float(np.mean(cos)),
                     "phase_estimate": theta0})

    fig_exact.suptitle("Exact joint-score slices: all other noisy frames are fixed")
    fig_exact.tight_layout(); fig_exact.savefig(FIGDIR / "03_exact_joint_score_fields.png", dpi=220); plt.close(fig_exact)
    fig_lin.suptitle("First-order Taylor / linear-Gaussian score slices")
    fig_lin.tight_layout(); fig_lin.savefig(FIGDIR / "04_taylor_score_fields.png", dpi=220); plt.close(fig_lin)
    fig_cmp.suptitle("Joint score versus one-frame marginal score")
    fig_cmp.tight_layout(); fig_cmp.savefig(FIGDIR / "05_joint_vs_marginal_fields.png", dpi=220); plt.close(fig_cmp)
    return pd.DataFrame(rows)


def run_metric_experiment(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother,
                          rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for t in cfg.metric_times:
        exact_all, lin_all = [], []
        central_exact, central_lin = [], []
        for _ in range(cfg.metric_samples):
            _, _, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            _, exact = solver.score(x, t)
            lin = linearized_score(x, model, t)
            exact_all.append(exact)
            lin_all.append(lin)
            central_exact.append(exact[model.T // 2])
            central_lin.append(lin[model.T // 2])
        exact_arr = np.concatenate(exact_all, axis=0)
        lin_arr = np.concatenate(lin_all, axis=0)
        ce = np.asarray(central_exact); cl = np.asarray(central_lin)
        rel = math.sqrt(np.mean(np.sum((lin_arr - exact_arr) ** 2, axis=1)) /
                        max(np.mean(np.sum(exact_arr ** 2, axis=1)), 1e-14))
        relc = math.sqrt(np.mean(np.sum((cl - ce) ** 2, axis=1)) /
                         max(np.mean(np.sum(ce ** 2, axis=1)), 1e-14))
        cosine = np.sum(lin_arr * exact_arr, axis=1) / (
            np.linalg.norm(lin_arr, axis=1) * np.linalg.norm(exact_arr, axis=1) + 1e-12)
        rows.append({"t": t, "trajectory_relative_error": rel,
                     "central_relative_error": relc,
                     "mean_cosine_similarity": float(np.mean(cosine))})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "taylor_vs_exact.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(df["t"], df["trajectory_relative_error"], marker="o", label="whole trajectory")
    ax.plot(df["t"], df["central_relative_error"], marker="s", label="central frame")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel("relative score error")
    ax.set_title("Accuracy of the first-order local Taylor score")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGDIR / "06_taylor_accuracy_vs_time.png", dpi=220); plt.close(fig)
    return df


def write_report(cfg: Config, model: PolarRingModel, field_df: pd.DataFrame, metric_df: pd.DataFrame) -> None:
    lines = [
        "# Original polar model: exact score evaluation and Taylor linearization",
        "",
        "## 1. Exact nonlinear problem",
        "",
        r"The latent state is $H_k=(R_k,\Theta_k)$ and $A_k=R_k(\cos\Theta_k,\sin\Theta_k)$. The whole trajectory is noised by $X_t=e^{-t}A+\sqrt{1-e^{-2t}}\,\Xi$.",
        "",
        r"The exact score is $S_k(x,t)=[e^{-t}\mathbb E(A_k\mid X_t=x)-x_k]/(1-e^{-2t})$. We evaluate the posterior mean by forward-backward messages on a polar grid.",
        "",
        "## 2. First-order Taylor model",
        "",
        r"Choose one rotating branch $\bar\theta_k=\theta_0+k\omega h$, write $R_k=1+\rho_k$ and $\Theta_k=\bar\theta_k+\phi_k$, and define $e_{r,k}=(\cos\bar\theta_k,\sin\bar\theta_k)$ and $e_{\theta,k}=(-\sin\bar\theta_k,\cos\bar\theta_k)$.",
        "",
        r"Then $A_k=(1+\rho_k)[e_{r,k}\cos\phi_k+e_{\theta,k}\sin\phi_k]$ and",
        "",
        r"$$A_k=e_{r,k}+e_{r,k}\rho_k+e_{\theta,k}\phi_k-\tfrac12e_{r,k}\phi_k^2+e_{\theta,k}\rho_k\phi_k+\cdots.$$",
        r"Keeping only first order gives $A_k\simeq e_{r,k}+e_{r,k}\rho_k+e_{\theta,k}\phi_k$. Since $\rho$ is an AR(1) process and $\phi$ is a Gaussian random walk, the approximation is a linear Gaussian state-space model. Its noised trajectory is Gaussian and its score is $S^{\rm lin}(x,t)=-\Sigma_t^{-1}(x-\mu_t)$.",
        "",
        "This is a local, phase-conditioned approximation. It cannot represent the full rotational mixture when the posterior spreads over a large part of the circle.",
        "",
        "## 3. Parameters",
        "",
        f"- T={cfg.T}, kappa={cfg.kappa}, D_r={cfg.D_r}, D_theta={cfg.D_theta}, omega={cfg.omega}",
        f"- internal step h={model.h:.5f}",
        f"- radial stationary standard deviation={model.radial_stationary_std:.5f}",
        f"- exact polar grid={cfg.n_r} x {cfg.n_theta}",
        "",
        "## 4. Local vector-field comparison",
        "",
        field_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The local error is measured only where the exact conditional log-density is within four nats of its maximum. This avoids judging a local Taylor approximation in regions carrying essentially no conditional probability.",
        "",
        "## 5. Monte Carlo comparison on noisy trajectories",
        "",
        metric_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Interpretation",
        "",
        "The grid smoother evaluates the exact nonlinear score up to polar-grid resolution. The Taylor model is useful near a single inferred rotating branch: it separates radial and tangential fluctuations and turns smoothing into linear algebra. Its failure mode is also informative: when angular uncertainty becomes broad or multimodal, one tangent plane cannot represent the ring.",
        "",
        "## Figures",
        "",
        "- `01_taylor_geometry.png`: geometry and truncation error of the Taylor expansion.",
        "- `02_posterior_on_circle.png`: exact smoothed latent posterior at several diffusion times.",
        "- `03_exact_joint_score_fields.png`: exact conditional score slices.",
        "- `04_taylor_score_fields.png`: linearized score slices.",
        "- `05_joint_vs_marginal_fields.png`: joint versus marginal fields.",
        "- `06_taylor_accuracy_vs_time.png`: quantitative Taylor error.",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global model
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    model = PolarRingModel(cfg.kappa, cfg.D_r, cfg.omega, cfg.D_theta, cfg.T)
    r_grid = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r)
    theta_grid = np.linspace(0.0, 2.0 * math.pi, cfg.n_theta, endpoint=False)
    solver = PolarGridSmoother(model, r_grid, theta_grid)

    # One common clean trajectory and one common Gaussian direction across t.
    _, _, clean = model.simulate(rng)
    xi = rng.normal(size=clean.shape)
    x_by_t = {}
    for t in cfg.field_times:
        m, delta = ou_parameters(t)
        x_by_t[t] = m * clean + math.sqrt(delta) * xi

    k = model.T // 2
    plot_taylor_geometry()
    plot_posteriors_on_circle(model, solver, clean, x_by_t, k)
    field_df = plot_score_fields(cfg, model, solver, clean, x_by_t, k)
    field_df.to_csv(DATADIR / "local_field_comparison.csv", index=False)
    metric_df = run_metric_experiment(cfg, model, solver, rng)
    write_report(cfg, model, field_df, metric_df)
    (ROOT / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# Polar Taylor score study\n\nRun `python run_study.py`. The exact nonlinear score is evaluated by polar-grid forward-backward smoothing; the comparison model is the first-order phase-conditioned Taylor approximation.\n",
        encoding="utf-8",
    )

    archive = ROOT.parent / "polar_taylor_score_study.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ROOT.parent, ROOT.name)
    print("Local field comparison")
    print(field_df.to_string(index=False))
    print("\nMonte Carlo comparison")
    print(metric_df.to_string(index=False))
    print(f"\nWrote {archive}")


if __name__ == "__main__":
    main()
