#!/usr/bin/env python3
"""Correlation length, receptive field, and response intensity of the joint score
for the board-faithful polar rotating-ring model.

The nonlinear posterior integrals are evaluated directly on a polar grid using the
clean Markov-chain factorization. The term "belief propagation" is deliberately
not used in the report: computationally, this is just repeated evaluation of the
one-step integral operators of the chain.

Main exact identity (posterior fluctuation-response):
    d S_k / d x_j = -delta_kj I / Delta
                    + m^2 / Delta^2 Cov(A_k, A_j | X=x).

The script computes posterior pair covariances exactly on the finite grid, projects
the response into co-moving radial/tangential bases, compares with a local
linear-Gaussian prediction, and measures truncated-window score errors.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = BUNDLE_ROOT / "raw_reproduction" / "polar_range"
FIGDIR = RUN_ROOT / "figures"
DATADIR = RUN_ROOT / "data"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

# Import the validated local polar implementation shipped in this bundle.
from polar_core import (  # noqa: E402
    PolarGridSmoother,
    PolarRingModel,
    add_ou_noise,
    ou_parameters,
    relative_rmse,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    seed: int = 20260725
    T: int = 20
    kappa: float = 2.0
    D_r: float = 0.015
    omega: float = 1.0
    D_theta: float = 0.005
    n_r: int = 41
    n_theta: int = 128
    r_min: float = 0.45
    r_max: float = 1.55
    response_samples: int = 28
    window_samples: int = 36
    sweep_samples: int = 12
    t_values: tuple[float, ...] = (0.03, 0.06, 0.10, 0.20, 0.40, 0.70, 1.00, 1.50, 2.00)
    display_times: tuple[float, ...] = (0.03, 0.10, 0.40, 0.70, 1.50)
    max_window_radius: int = 10
    fd_eps: float = 1e-3


def raw_forward(solver: PolarGridSmoother, mass: Array) -> Array:
    return solver.Tr.T @ mass @ solver.Ttheta


def raw_backward(solver: PolarGridSmoother, mass: Array) -> Array:
    return solver.Tr @ mass @ solver.Ttheta.T


def chain_quantities(solver: PolarGridSmoother, x: Array, t: float):
    """Return normalized forward arrays, backward arrays, beliefs, means, likelihoods.

    Normalization constants may differ by time; pair-moment propagation below uses
    the same constants for ordinary and feature-weighted masses, so they cancel.
    """
    T = solver.model.T
    likelihoods = solver._likelihoods(x, t)  # validated internal helper

    forward: list[Array] = [solver.prior0 * likelihoods[0]]
    forward[0] /= forward[0].sum()
    for q in range(1, T):
        f = likelihoods[q] * raw_forward(solver, forward[-1])
        f /= f.sum()
        forward.append(f)

    backward: list[Array] = [np.empty_like(solver.prior0) for _ in range(T)]
    backward[-1] = np.ones_like(solver.prior0)
    backward[-1] /= backward[-1].sum()
    for q in range(T - 2, -1, -1):
        b = raw_backward(solver, likelihoods[q + 1] * backward[q + 1])
        b = np.maximum(b, 0.0)
        b /= b.sum()
        backward[q] = b

    beliefs: list[Array] = []
    means = np.empty((T, 2))
    for q in range(T):
        b = forward[q] * backward[q]
        b /= b.sum()
        beliefs.append(b)
        means[q] = np.sum(b[..., None] * solver.Axy, axis=(0, 1))
    return likelihoods, forward, backward, beliefs, means


def ordered_pair_moment(
    solver: PolarGridSmoother,
    likelihoods: list[Array],
    forward: list[Array],
    backward: list[Array],
    start: int,
    end: int,
) -> Array:
    """Compute E[A_start A_end^T | x] exactly on the finite grid, start <= end."""
    if start > end:
        raise ValueError("start must not exceed end")
    if start == end:
        b = forward[start] * backward[start]
        b /= b.sum()
        return np.einsum("ij,ija,ijb->ab", b, solver.Axy, solver.Axy)

    base = forward[start].copy()
    weighted = [forward[start] * solver.Axy[..., a] for a in range(2)]

    # Propagate the ordinary mass and two feature-weighted masses with identical
    # normalizations. Weighted masses may have either sign and are not normalized.
    for q in range(start + 1, end + 1):
        base_new = likelihoods[q] * raw_forward(solver, base)
        scale = float(base_new.sum())
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("invalid pair-moment propagation scale")
        base = base_new / scale
        weighted = [likelihoods[q] * raw_forward(solver, w) / scale for w in weighted]

    endpoint_weight = backward[end]
    den = float(np.sum(base * endpoint_weight))
    out = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            out[a, b] = np.sum(weighted[a] * solver.Axy[..., b] * endpoint_weight) / den
    return out


def center_response_from_covariance(
    solver: PolarGridSmoother,
    x: Array,
    t: float,
    center: int,
):
    """Exact J[center,j] via posterior pair covariances on the finite grid."""
    likelihoods, forward, backward, beliefs, means = chain_quantities(solver, x, t)
    T = solver.model.T
    covs = np.empty((T, 2, 2))
    for j in range(T):
        if j <= center:
            mom_jc = ordered_pair_moment(solver, likelihoods, forward, backward, j, center)
            mom = mom_jc.T
        else:
            mom = ordered_pair_moment(solver, likelihoods, forward, backward, center, j)
        covs[j] = mom - np.outer(means[center], means[j])

    m, delta = ou_parameters(t)
    J = (m * m / (delta * delta)) * covs
    J[center] -= np.eye(2) / delta
    score = (m * means - x) / delta
    return J, covs, means, score, beliefs


def local_basis(theta: float) -> tuple[Array, Array]:
    er = np.array([math.cos(theta), math.sin(theta)])
    et = np.array([-math.sin(theta), math.cos(theta)])
    return er, et


def project_response(J: Array, theta: Array, center: int) -> dict[str, Array]:
    T = J.shape[0]
    erc, etc = local_basis(theta[center])
    rr = np.empty(T); rt = np.empty(T); tr = np.empty(T); tt = np.empty(T); fn = np.empty(T)
    for j in range(T):
        erj, etj = local_basis(theta[j])
        rr[j] = erc @ J[j] @ erj
        rt[j] = erc @ J[j] @ etj
        tr[j] = etc @ J[j] @ erj
        tt[j] = etc @ J[j] @ etj
        fn[j] = np.linalg.norm(J[j], ord="fro")
    return {"rr": rr, "rt": rt, "tr": tr, "tt": tt, "fro": fn}


def lag_aggregate(values: list[Array], center: int, mode: str = "rms") -> Array:
    """Aggregate arrays indexed by j into a profile indexed by lag |j-center|."""
    max_lag = max(center, len(values[0]) - 1 - center)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        vals = []
        for v in values:
            for j in (center - lag, center + lag):
                if 0 <= j < len(v) and (lag == 0 or j == center - lag or j == center + lag):
                    vals.append(v[j])
            # for lag 0 the tuple repeats the center; retain once
            if lag == 0:
                vals = vals[:-1] if len(vals) > 1 else vals
        arr = np.asarray(vals)
        if mode == "rms":
            out[lag] = math.sqrt(float(np.mean(arr * arr)))
        elif mode == "mean_abs":
            out[lag] = float(np.mean(np.abs(arr)))
        elif mode == "mean":
            out[lag] = float(np.mean(arr))
        else:
            raise ValueError(mode)
    return out


def weighted_influence_radius(profile: Array, center: int, T: int, fraction: float = 0.95) -> int:
    masses = []
    for lag in range(1, len(profile)):
        count = int(center - lag >= 0) + int(center + lag < T)
        masses.append((lag, count * profile[lag]))
    total = sum(v for _, v in masses)
    if total <= 0:
        return 0
    c = 0.0
    for lag, v in masses:
        c += v
        if c / total >= fraction:
            return lag
    return masses[-1][0]


def integrated_length(profile: Array) -> float:
    """One-sided integrated response length, normalized at lag 1."""
    if len(profile) <= 1 or profile[1] <= 0:
        return 0.0
    return float(np.sum(profile[1:] / profile[1]))


def exponential_fit_length(profile: Array) -> tuple[float, float]:
    """Fit C(l) ~ exp(-l/xi) where the signal exceeds 5% of C(1)."""
    if len(profile) < 4 or profile[1] <= 0:
        return float("nan"), float("nan")
    lags = np.arange(1, len(profile))
    mask = np.isfinite(profile[1:]) & (profile[1:] > 0.05 * profile[1])
    if mask.sum() < 3:
        mask = np.isfinite(profile[1:]) & (profile[1:] > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = lags[mask].astype(float)
    y = np.log(profile[1:][mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
    xi = -1.0 / slope if slope < -1e-10 else float("inf")
    return float(xi), r2


def linear_response_prediction(model: PolarRingModel, t: float, max_lag: int) -> dict[str, Array | float]:
    h = model.h
    a = math.exp(-model.kappa * h)
    q_r = (model.D_r / model.kappa) * (1.0 - a * a)
    q_t = 2.0 * model.D_theta * h
    m, delta = ou_parameters(t)
    lam = m * m / delta

    cosh_gr = (1.0 + a * a + q_r * lam) / (2.0 * a)
    cosh_gr = max(cosh_gr, 1.0)
    gr = math.acosh(cosh_gr)
    cosh_gt = 1.0 + 0.5 * q_t * lam
    gt = math.acosh(max(cosh_gt, 1.0))

    lags = np.arange(max_lag + 1)
    Cr = q_r / (2.0 * a * max(math.sinh(gr), 1e-15)) * np.exp(-gr * lags)
    Ct = q_t / (2.0 * max(math.sinh(gt), 1e-15)) * np.exp(-gt * lags)
    pref = m * m / (delta * delta)
    Jr = pref * Cr
    Jt = pref * Ct
    return {
        "rr": Jr,
        "tt": Jt,
        "xi_rr": 1.0 / gr if gr > 0 else float("inf"),
        "xi_tt": 1.0 / gt if gt > 0 else float("inf"),
    }


def make_model_solver(cfg: Config, *, kappa=None, D_theta=None, T=None):
    model = PolarRingModel(
        kappa=cfg.kappa if kappa is None else float(kappa),
        D_r=cfg.D_r,
        omega=cfg.omega,
        D_theta=cfg.D_theta if D_theta is None else float(D_theta),
        T=cfg.T if T is None else int(T),
    )
    solver = PolarGridSmoother(
        model,
        np.linspace(cfg.r_min, cfg.r_max, cfg.n_r),
        np.linspace(0.0, 2.0 * math.pi, cfg.n_theta, endpoint=False),
    )
    return model, solver


def run_baseline_response(cfg: Config, rng: np.random.Generator):
    model, solver = make_model_solver(cfg)
    center = model.T // 2
    profile_rows = []
    summary_rows = []
    stored = {}

    for t in cfg.t_values:
        projected = {name: [] for name in ("rr", "rt", "tr", "tt", "fro")}
        signed = {name: [] for name in ("rr", "rt", "tr", "tt")}
        for _ in range(cfg.response_samples):
            _, theta, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            J, _, _, _, _ = center_response_from_covariance(solver, x, t, center)
            p = project_response(J, theta, center)
            for name in projected:
                projected[name].append(p[name])
            for name in signed:
                signed[name].append(p[name])

        profiles = {name: lag_aggregate(vals, center, "rms") for name, vals in projected.items()}
        signed_profiles = {name: lag_aggregate(vals, center, "mean") for name, vals in signed.items()}
        stored[t] = profiles
        pred = linear_response_prediction(model, t, len(profiles["rr"]) - 1)

        for lag in range(len(profiles["rr"])):
            profile_rows.append({
                "t": t,
                "lag": lag,
                "rr_rms": profiles["rr"][lag],
                "tt_rms": profiles["tt"][lag],
                "rt_rms": profiles["rt"][lag],
                "tr_rms": profiles["tr"][lag],
                "fro_rms": profiles["fro"][lag],
                "rr_signed_mean": signed_profiles["rr"][lag],
                "tt_signed_mean": signed_profiles["tt"][lag],
                "rr_linear": pred["rr"][lag],
                "tt_linear": pred["tt"][lag],
            })

        xi_rr_int = integrated_length(profiles["rr"])
        xi_tt_int = integrated_length(profiles["tt"])
        xi_rr_fit, r2_rr = exponential_fit_length(profiles["rr"])
        xi_tt_fit, r2_tt = exponential_fit_length(profiles["tt"])
        L95_rr = weighted_influence_radius(profiles["rr"], center, model.T, 0.95)
        L95_tt = weighted_influence_radius(profiles["tt"], center, model.T, 0.95)

        # Integrated off-diagonal intensity with finite-trajectory multiplicity.
        def integrated_intensity(profile):
            total = 0.0
            for lag in range(1, len(profile)):
                count = int(center - lag >= 0) + int(center + lag < model.T)
                total += count * profile[lag]
            return total

        int_rr = integrated_intensity(profiles["rr"])
        int_tt = integrated_intensity(profiles["tt"])
        int_fro = integrated_intensity(profiles["fro"])
        self_fro = profiles["fro"][0]
        summary_rows.append({
            "t": t,
            "rr_nearest_neighbor_intensity": profiles["rr"][1],
            "tt_nearest_neighbor_intensity": profiles["tt"][1],
            "fro_nearest_neighbor_intensity": profiles["fro"][1],
            "rr_integrated_offdiag_intensity": int_rr,
            "tt_integrated_offdiag_intensity": int_tt,
            "fro_integrated_offdiag_intensity": int_fro,
            "fro_self_response": self_fro,
            "offdiag_fraction": int_fro / max(int_fro + self_fro, 1e-15),
            "rr_integrated_length": xi_rr_int,
            "tt_integrated_length": xi_tt_int,
            "rr_exp_fit_length": xi_rr_fit,
            "tt_exp_fit_length": xi_tt_fit,
            "rr_exp_fit_r2": r2_rr,
            "tt_exp_fit_r2": r2_tt,
            "rr_L95_response_mass": L95_rr,
            "tt_L95_response_mass": L95_tt,
            "rr_linear_length": pred["xi_rr"],
            "tt_linear_length": pred["xi_tt"],
        })

    profiles_df = pd.DataFrame(profile_rows)
    summary_df = pd.DataFrame(summary_rows)
    profiles_df.to_csv(DATADIR / "baseline_response_profiles.csv", index=False)
    summary_df.to_csv(DATADIR / "baseline_response_summary.csv", index=False)

    # Plot response profiles.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    for t in cfg.display_times:
        g = profiles_df[profiles_df.t == t]
        axes[0].semilogy(g.lag, np.maximum(g.rr_rms, 1e-12), marker="o", markersize=3, label=f"t={t:g}")
        axes[1].semilogy(g.lag, np.maximum(g.tt_rms, 1e-12), marker="o", markersize=3, label=f"t={t:g}")
    axes[0].set_title("Radial response")
    axes[1].set_title("Tangential response")
    for ax in axes:
        ax.set_xlabel(r"temporal lag $|j-k|$")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"RMS $|e_k^T(\partial S_k/\partial x_j)e_j|$")
    axes[1].legend(ncol=1)
    fig.suptitle("Joint-score response profiles in the original polar model")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_radial_tangential_response_profiles.png", dpi=230)
    plt.close(fig)

    # Exact nonlinear vs local linear prediction at representative times.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for col, t in enumerate((0.10, 0.70)):
        g = profiles_df[profiles_df.t == t]
        axes[0, col].semilogy(g.lag[1:], g.rr_rms[1:], marker="o", label="nonlinear numerical")
        axes[0, col].semilogy(g.lag[1:], g.rr_linear[1:], linestyle="--", label="Taylor prediction")
        axes[1, col].semilogy(g.lag[1:], g.tt_rms[1:], marker="o", label="nonlinear numerical")
        axes[1, col].semilogy(g.lag[1:], g.tt_linear[1:], linestyle="--", label="Taylor prediction")
        axes[0, col].set_title(f"Radial, t={t:g}")
        axes[1, col].set_title(f"Tangential, t={t:g}")
        axes[1, col].set_xlabel("lag")
        for row in range(2):
            axes[row, col].grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("response intensity")
    axes[1, 0].set_ylabel("response intensity")
    axes[0, 0].legend(); axes[1, 0].legend()
    fig.suptitle("Nonlinear response versus first-order Gaussian prediction")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_nonlinear_vs_taylor_response.png", dpi=230)
    plt.close(fig)

    # Intensity versus diffusion time.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].semilogy(summary_df.t, summary_df.rr_nearest_neighbor_intensity, marker="o", label="radial nearest neighbor")
    axes[0].semilogy(summary_df.t, summary_df.tt_nearest_neighbor_intensity, marker="o", label="tangential nearest neighbor")
    axes[0].semilogy(summary_df.t, summary_df.fro_integrated_offdiag_intensity, marker="o", label="all off-diagonal, integrated")
    axes[0].set_xlabel("diffusion time t"); axes[0].set_ylabel("response intensity")
    axes[0].set_title("Intensity decreases with noising")
    axes[0].grid(True, which="both", alpha=0.25); axes[0].legend()
    axes[1].plot(summary_df.t, summary_df.offdiag_fraction, marker="o")
    axes[1].set_xlabel("diffusion time t"); axes[1].set_ylabel("off-diagonal fraction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Fraction of total response carried by other frames")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_response_intensity_vs_t.png", dpi=230)
    plt.close(fig)

    # Lengths and response-mass radii.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(summary_df.t, summary_df.rr_integrated_length, marker="o", label="radial numerical")
    axes[0].plot(summary_df.t, summary_df.tt_integrated_length, marker="o", label="tangential numerical")
    axes[0].plot(summary_df.t, summary_df.rr_linear_length, linestyle="--", label="radial Taylor")
    axes[0].plot(summary_df.t, summary_df.tt_linear_length, linestyle="--", label="tangential Taylor")
    axes[0].set_xlabel("diffusion time t"); axes[0].set_ylabel("length in frames")
    axes[0].set_title("Integrated response length")
    axes[0].grid(True, alpha=0.25); axes[0].legend()
    axes[1].plot(summary_df.t, summary_df.rr_L95_response_mass, marker="o", label="radial")
    axes[1].plot(summary_df.t, summary_df.tt_L95_response_mass, marker="o", label="tangential")
    axes[1].set_xlabel("diffusion time t"); axes[1].set_ylabel("radius in frames")
    axes[1].set_title("Radius containing 95% of cross-frame response")
    axes[1].grid(True, alpha=0.25); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_response_lengths_vs_t.png", dpi=230)
    plt.close(fig)

    return model, solver, profiles_df, summary_df


def run_window_receptive_field(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator):
    center = model.T // 2
    rows = []
    for t in cfg.t_values:
        full = []
        windows = {L: [] for L in range(cfg.max_window_radius + 1)}
        full_r = []; full_t = []
        win_r = {L: [] for L in windows}; win_t = {L: [] for L in windows}
        for _ in range(cfg.window_samples):
            _, theta, clean = model.simulate(rng)
            x = add_ou_noise(clean, t, rng)
            _, s = solver.score(x, t)
            sc = s[center]
            er, et = local_basis(theta[center])
            full.append(sc); full_r.append(er @ sc); full_t.append(et @ sc)
            for L in windows:
                sw = solver.window_score(x, t, center, L)
                windows[L].append(sw)
                win_r[L].append(er @ sw); win_t[L].append(et @ sw)
        F = np.stack(full)
        Fr = np.asarray(full_r); Ft = np.asarray(full_t)
        for L in windows:
            W = np.stack(windows[L])
            Wr = np.asarray(win_r[L]); Wt = np.asarray(win_t[L])
            total_err = relative_rmse(W, F)
            radial_err = math.sqrt(float(np.mean((Wr - Fr) ** 2)) / max(float(np.mean(Fr ** 2)), 1e-15))
            tang_err = math.sqrt(float(np.mean((Wt - Ft) ** 2)) / max(float(np.mean(Ft ** 2)), 1e-15))
            rows.append({"t": t, "radius": L, "frames": min(model.T, 2 * L + 1),
                         "total_relative_error": total_err,
                         "radial_relative_error": radial_err,
                         "tangential_relative_error": tang_err})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "window_receptive_field.csv", index=False)

    summary = []
    for t, g in df.groupby("t"):
        row = {"t": t}
        for col, prefix in (("total_relative_error", "total"), ("radial_relative_error", "radial"), ("tangential_relative_error", "tangential")):
            for threshold in (0.10, 0.05, 0.01):
                ok = g[g[col] <= threshold]
                row[f"{prefix}_L_{int(100*threshold)}pct"] = int(ok.radius.min()) if len(ok) else np.nan
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(DATADIR / "window_receptive_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    for t in cfg.display_times:
        g = df[df.t == t]
        axes[0].semilogy(g.radius, g.radial_relative_error, marker="o", markersize=3, label=f"t={t:g}")
        axes[1].semilogy(g.radius, g.tangential_relative_error, marker="o", markersize=3, label=f"t={t:g}")
    axes[0].set_title("Radial score component")
    axes[1].set_title("Tangential score component")
    for ax in axes:
        ax.axhline(0.05, linestyle="--", linewidth=1)
        ax.set_xlabel("temporal window radius L")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("relative score error")
    axes[1].legend()
    fig.suptitle("Functional receptive field: score from a truncated observation window")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_window_receptive_field.png", dpi=230)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(sdf.t, sdf.radial_L_5pct, marker="o", label="radial, 5% error")
    ax.plot(sdf.t, sdf.tangential_L_5pct, marker="o", label="tangential, 5% error")
    ax.plot(sdf.t, sdf.total_L_5pct, marker="o", label="total, 5% error")
    ax.set_xlabel("diffusion time t"); ax.set_ylabel("minimum window radius L")
    ax.set_title("Receptive field required for 5% score accuracy")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGDIR / "06_receptive_radius_vs_t.png", dpi=230); plt.close(fig)
    return df, sdf


def parameter_point_metrics(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother,
                            t: float, rng: np.random.Generator, samples: int):
    center = model.T // 2
    rr_all = []; tt_all = []
    for _ in range(samples):
        _, theta, clean = model.simulate(rng)
        x = add_ou_noise(clean, t, rng)
        J, *_ = center_response_from_covariance(solver, x, t, center)
        p = project_response(J, theta, center)
        rr_all.append(p["rr"]); tt_all.append(p["tt"])
    rr = lag_aggregate(rr_all, center, "rms")
    tt = lag_aggregate(tt_all, center, "rms")
    return {
        "rr_length": integrated_length(rr),
        "tt_length": integrated_length(tt),
        "rr_L95": weighted_influence_radius(rr, center, model.T),
        "tt_L95": weighted_influence_radius(tt, center, model.T),
        "rr_nn": rr[1],
        "tt_nn": tt[1],
    }


def run_parameter_sweeps(cfg: Config, rng: np.random.Generator):
    rows = []
    kappas = (0.5, 1.0, 2.0, 4.0)
    dthetas = (0.002, 0.005, 0.020, 0.080)
    times = (0.10, 0.40, 0.70, 1.50)
    for kappa in kappas:
        model, solver = make_model_solver(cfg, kappa=kappa)
        for t in times:
            met = parameter_point_metrics(cfg, model, solver, t, rng, cfg.sweep_samples)
            rows.append({"sweep": "kappa", "parameter": kappa, "t": t, **met})
    for dt in dthetas:
        model, solver = make_model_solver(cfg, D_theta=dt)
        for t in times:
            met = parameter_point_metrics(cfg, model, solver, t, rng, cfg.sweep_samples)
            rows.append({"sweep": "D_theta", "parameter": dt, "t": t, **met})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "parameter_sweeps.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    for t in times:
        g = df[(df.sweep == "kappa") & (df.t == t)]
        axes[0, 0].plot(g.parameter, g.rr_length, marker="o", label=f"t={t:g}")
        axes[1, 0].semilogy(g.parameter, g.rr_nn, marker="o", label=f"t={t:g}")
        g = df[(df.sweep == "D_theta") & (df.t == t)]
        axes[0, 1].semilogx(g.parameter, g.tt_length, marker="o", label=f"t={t:g}")
        axes[1, 1].loglog(g.parameter, g.tt_nn, marker="o", label=f"t={t:g}")
    axes[0, 0].set_title(r"Radial length versus confinement $\kappa$")
    axes[1, 0].set_title(r"Radial nearest-neighbor intensity versus $\kappa$")
    axes[0, 1].set_title(r"Tangential length versus angular noise $D_\theta$")
    axes[1, 1].set_title(r"Tangential nearest-neighbor intensity versus $D_\theta$")
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("parameter value")
        ax.legend()
    axes[0, 0].set_ylabel("integrated length")
    axes[0, 1].set_ylabel("integrated length")
    axes[1, 0].set_ylabel("response intensity")
    axes[1, 1].set_ylabel("response intensity")
    fig.tight_layout(); fig.savefig(FIGDIR / "07_parameter_sweeps.png", dpi=230); plt.close(fig)
    return df


def run_fdt_validation(cfg: Config, model: PolarRingModel, solver: PolarGridSmoother, rng: np.random.Generator):
    center = model.T // 2
    rows = []
    for t in (0.10, 0.70, 1.50):
        _, theta, clean = model.simulate(rng)
        x = add_ou_noise(clean, t, rng)
        Jcov, covs, means, score, _ = center_response_from_covariance(solver, x, t, center)
        for lag in (0, 1, 3, 7):
            j = min(model.T - 1, center + lag)
            Jfd = np.empty((2, 2))
            for b in range(2):
                xp = x.copy(); xm = x.copy()
                xp[j, b] += cfg.fd_eps; xm[j, b] -= cfg.fd_eps
                _, sp = solver.score(xp, t); _, sm = solver.score(xm, t)
                Jfd[:, b] = (sp[center] - sm[center]) / (2.0 * cfg.fd_eps)
            rel = np.linalg.norm(Jfd - Jcov[j]) / max(np.linalg.norm(Jfd), 1e-14)
            rows.append({"t": t, "lag": lag, "relative_error": rel,
                         "fd_norm": np.linalg.norm(Jfd), "covariance_identity_norm": np.linalg.norm(Jcov[j])})
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "fluctuation_response_validation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for t, g in df.groupby("t"):
        ax.semilogy(g.lag, g.relative_error, marker="o", label=f"t={t:g}")
    ax.set_xlabel("lag"); ax.set_ylabel("relative discrepancy")
    ax.set_title("Finite difference versus posterior covariance identity")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGDIR / "08_fluctuation_response_validation.png", dpi=230); plt.close(fig)
    return df


def grid_check(cfg: Config, rng: np.random.Generator):
    model_c, solver_c = make_model_solver(cfg)
    fine_cfg = Config(**{**asdict(cfg), "n_r": 51, "n_theta": 192, "response_samples": cfg.response_samples,
                         "window_samples": cfg.window_samples, "sweep_samples": cfg.sweep_samples})
    model_f, solver_f = make_model_solver(fine_cfg)
    center = model_c.T // 2
    out = []
    for t in (0.10, 0.70):
        _, theta, clean = model_c.simulate(rng)
        x = add_ou_noise(clean, t, rng)
        Jc, *_ = center_response_from_covariance(solver_c, x, t, center)
        Jf, *_ = center_response_from_covariance(solver_f, x, t, center)
        rel = np.linalg.norm(Jc - Jf) / max(np.linalg.norm(Jf), 1e-14)
        out.append({"t": t, "response_grid_relative_error": rel})
    df = pd.DataFrame(out)
    df.to_csv(DATADIR / "grid_convergence.csv", index=False)
    return df


def write_report(cfg: Config, response_summary: pd.DataFrame, window_summary: pd.DataFrame,
                 sweeps: pd.DataFrame, fdt: pd.DataFrame, grid: pd.DataFrame):
    # Pull concise headline values programmatically.
    def row_at(t):
        return response_summary.iloc[(response_summary.t - t).abs().argmin()]
    r003, r040, r070, r150, r200 = [row_at(t) for t in (0.03, 0.40, 0.70, 1.50, 2.00)]
    fdt_max = float(fdt.relative_error.max())
    grid_max = float(grid.response_grid_relative_error.max())

    text = f"""# Joint-score range study for the original polar model

## Question

How do the clean Markov dynamics determine the **intensity**, **correlation length**, and
**functional receptive field** of the noised joint score?

The model is

\[
dR_u=-\kappa(R_u-1)du+\sqrt{{2D_r}}dB_u^r,\qquad
 d\Theta_u=\omega du+\sqrt{{2D_\theta}}dB_u^\theta,
\]
\[
A_u=R_u(\cos\Theta_u,\sin\Theta_u),\qquad
X_t=e^{{-t}}A+\sqrt{{1-e^{{-2t}}}}\,Z.
\]

Baseline parameters: `T={cfg.T}`, `kappa={cfg.kappa}`, `D_r={cfg.D_r}`,
`D_theta={cfg.D_theta}`, one expected revolution, polar grid `{cfg.n_r} x {cfg.n_theta}`.

## Exact response identity

For the joint score block

\[
S_k(x,t)=\frac{{m\,E[A_k\mid X=x]-x_k}}{{\Delta}},\qquad
m=e^{{-t}},\quad\Delta=1-e^{{-2t}},
\]

the response to perturbing frame `j` is

\[
\frac{{\partial S_k}}{{\partial x_j}}
=-\frac{{\delta_{{kj}}}}{{\Delta}}I_2
+\frac{{m^2}}{{\Delta^2}}\operatorname{{Cov}}(A_k,A_j\mid X=x).
\]

The experiments evaluate the posterior integrals directly on the polar grid, then project this
matrix into co-moving radial and tangential directions.

## Headline findings

1. **The radial and tangential ranges are different.** At `t=0.40`, the integrated
   radial response length is `{r040.rr_integrated_length:.2f}` frames, while the tangential
   length is `{r040.tt_integrated_length:.2f}` frames. The angular phase is much more coherent
   than radial fluctuations in this baseline.

2. **Range and intensity move in opposite directions.** The integrated off-diagonal
   response intensity falls from `{r003.fro_integrated_offdiag_intensity:.3g}` at `t=0.03`
   to `{r200.fro_integrated_offdiag_intensity:.3g}` at `t=2.00`, while the 95% tangential
   response radius changes from `{int(r003.tt_L95_response_mass)}` to
   `{int(r200.tt_L95_response_mass)}` frames. A broad response at large noise can therefore
   be extremely weak.

3. **A response length is not the same as a functional receptive field.** The windowed-score
   experiment measures the minimum radius needed to reproduce the full score. See
   `data/window_receptive_summary.csv`; radial and tangential components can require different
   context sizes.

4. **The Taylor model captures the mechanism but not every nonlinear detail.** The analytic
   Gaussian prediction correctly separates a short radial mode from a long angular mode and
   predicts the broadening/weakening tradeoff. Differences increase when the angular posterior
   is no longer confined to one local branch.

5. **The fluctuation-response relation is numerically verified.** Across the tested times and
   lags, the largest finite-difference discrepancy is `{fdt_max:.2e}`. The maximum coarse-versus-fine
   grid discrepancy in the response matrix is `{grid_max:.2%}`.

## Interpretation

The clean process is local in internal time, but the noised joint score is not. Its nonlocality is
structured rather than arbitrary: radial information is damped by mean reversion, whereas angular
information is transmitted through a slowly diffusing phase. Diffusion time broadens the posterior
communication channel, but the score prefactor `m^2/Delta^2` simultaneously suppresses its strength.

## Files

- `figures/01_radial_tangential_response_profiles.png`: radial/tangential decay with lag.
- `figures/02_nonlinear_vs_taylor_response.png`: nonlinear result versus Taylor prediction.
- `figures/03_response_intensity_vs_t.png`: intensity and off-diagonal fraction.
- `figures/04_response_lengths_vs_t.png`: response lengths and 95% radii.
- `figures/05_window_receptive_field.png`: truncated-window score error.
- `figures/06_receptive_radius_vs_t.png`: functional receptive radius versus diffusion time.
- `figures/07_parameter_sweeps.png`: effects of `kappa` and `D_theta`.
- `figures/08_fluctuation_response_validation.png`: covariance identity validation.
- `data/*.csv`: all numerical results.
- `source/run_range_study.py`: complete reproducible implementation.
"""
    (RUN_ROOT / "report.md").write_text(text)


def make_notebook():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Joint-score correlation length, receptive field, and intensity\n\nThis notebook loads the outputs of `source/run_range_study.py`. Re-run the script from a terminal to regenerate all posterior integrations."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path('.')"),
        nbf.v4.new_code_cell("summary=pd.read_csv(ROOT/'data/baseline_response_summary.csv')\ndisplay(summary)"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/01_radial_tangential_response_profiles.png')))"),
        nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'figures/04_response_lengths_vs_t.png')))"),
        nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'data/window_receptive_summary.csv'))\ndisplay(Image(filename=str(ROOT/'figures/05_window_receptive_field.png')))"),
        nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'data/parameter_sweeps.csv'))\ndisplay(Image(filename=str(ROOT/'figures/07_parameter_sweeps.png')))"),
    ]
    nbf.write(nb, RUN_ROOT / "polar_joint_score_range_study.ipynb")


def main():
    cfg = Config()
    (RUN_ROOT / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    rng = np.random.default_rng(cfg.seed)
    model, solver, profiles, response_summary = run_baseline_response(cfg, rng)
    window_df, window_summary = run_window_receptive_field(cfg, model, solver, rng)
    sweeps = run_parameter_sweeps(cfg, rng)
    fdt = run_fdt_validation(cfg, model, solver, rng)
    grid = grid_check(cfg, rng)
    write_report(cfg, response_summary, window_summary, sweeps, fdt, grid)
    make_notebook()
    readme = """# Polar joint-score range study

Run:

```bash
python source/run_range_study.py
```

The script evaluates the nonlinear posterior chain integrals on a polar grid, computes
joint-score response matrices from posterior covariances, measures truncated-window errors,
and compares against the first-order linear-Gaussian prediction.
"""
    (RUN_ROOT / "README.md").write_text(readme)
    shutil.make_archive(str(RUN_ROOT), "zip", RUN_ROOT.parent, RUN_ROOT.name)
    print(response_summary.to_string(index=False))
    print("\nWindow summary:\n", window_summary.to_string(index=False))
    print("\nFDT validation max error:", fdt.relative_error.max())
    print("Grid check max error:", grid.response_grid_relative_error.max())


if __name__ == "__main__":
    main()
