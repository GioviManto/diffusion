#!/usr/bin/env python3
"""Analysis utilities for the original polar rotating-ring model.

The clean latent state is H_k=(R_k,Theta_k). The whole Cartesian trajectory is
corrupted by the VP OU channel. Posterior integrals are evaluated on a finite
polar grid by sequential forward and backward integration along the Markov chain.
"""
from __future__ import annotations

import math
import numpy as np

from polar_core import PolarGridSmoother, PolarRingModel, ou_parameters

Array = np.ndarray


def raw_forward(solver: PolarGridSmoother, mass: Array) -> Array:
    return solver.Tr.T @ mass @ solver.Ttheta


def raw_backward(solver: PolarGridSmoother, mass: Array) -> Array:
    return solver.Tr @ mass @ solver.Ttheta.T


def chain_quantities(solver: PolarGridSmoother, x: Array, t: float):
    """Normalized forward arrays, backward arrays, beliefs, means, likelihoods."""
    T = solver.model.T
    likelihoods = solver._likelihoods(x, t)
    forward: list[Array] = [solver.prior0 * likelihoods[0]]
    forward[0] /= forward[0].sum()
    for k in range(1, T):
        f = likelihoods[k] * raw_forward(solver, forward[-1])
        f /= f.sum()
        forward.append(f)

    backward: list[Array] = [np.empty_like(solver.prior0) for _ in range(T)]
    backward[-1] = np.ones_like(solver.prior0)
    backward[-1] /= backward[-1].sum()
    for k in range(T - 2, -1, -1):
        b = raw_backward(solver, likelihoods[k + 1] * backward[k + 1])
        b = np.maximum(b, 0.0)
        b /= b.sum()
        backward[k] = b

    beliefs: list[Array] = []
    means = np.empty((T, 2), dtype=float)
    for k in range(T):
        b = forward[k] * backward[k]
        b /= b.sum()
        beliefs.append(b)
        means[k] = np.sum(b[..., None] * solver.Axy, axis=(0, 1))
    return likelihoods, forward, backward, beliefs, means


def ordered_pair_moment(
    solver: PolarGridSmoother,
    likelihoods: list[Array],
    forward: list[Array],
    backward: list[Array],
    start: int,
    end: int,
) -> Array:
    """Compute E[A_start A_end^T | x] on the finite grid, start <= end."""
    if start > end:
        raise ValueError("start must not exceed end")
    if start == end:
        b = forward[start] * backward[start]
        b /= b.sum()
        return np.einsum("ij,ija,ijb->ab", b, solver.Axy, solver.Axy)

    base = forward[start].copy()
    weighted = [forward[start] * solver.Axy[..., a] for a in range(2)]
    for k in range(start + 1, end + 1):
        base_new = likelihoods[k] * raw_forward(solver, base)
        scale = float(base_new.sum())
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("invalid propagation scale")
        base = base_new / scale
        weighted = [likelihoods[k] * raw_forward(solver, w) / scale for w in weighted]

    endpoint_weight = backward[end]
    den = float(np.sum(base * endpoint_weight))
    out = np.empty((2, 2), dtype=float)
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
    """J[center,j] from the exact posterior covariance identity on the grid."""
    likelihoods, forward, backward, beliefs, means = chain_quantities(solver, x, t)
    T = solver.model.T
    covs = np.empty((T, 2, 2), dtype=float)
    for j in range(T):
        if j <= center:
            mom_jc = ordered_pair_moment(solver, likelihoods, forward, backward, j, center)
            mom = mom_jc.T
        else:
            mom = ordered_pair_moment(solver, likelihoods, forward, backward, center, j)
        covs[j] = mom - np.outer(means[center], means[j])

    m, delta = ou_parameters(t)
    J = (m * m / delta**2) * covs
    J[center] -= np.eye(2) / delta
    score = (m * means - x) / delta
    return J, covs, means, score, beliefs


def local_basis(theta: float) -> tuple[Array, Array]:
    er = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    et = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    return er, et


def project_response(J: Array, theta: Array, center: int) -> dict[str, Array]:
    """Project J[center,j] into oracle clean radial/tangential bases."""
    T = J.shape[0]
    erc, etc = local_basis(theta[center])
    out = {name: np.empty(T, dtype=float) for name in ("rr", "rt", "tr", "tt", "fro")}
    for j in range(T):
        erj, etj = local_basis(theta[j])
        out["rr"][j] = erc @ J[j] @ erj
        out["rt"][j] = erc @ J[j] @ etj
        out["tr"][j] = etc @ J[j] @ erj
        out["tt"][j] = etc @ J[j] @ etj
        out["fro"][j] = np.linalg.norm(J[j], ord="fro")
    return out


def estimate_initial_phase(x: Array, model: PolarRingModel) -> float:
    aligned = np.empty_like(x)
    for k in range(model.T):
        angle = -model.omega * model.h * k
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        aligned[k] = R @ x[k]
    v = aligned.sum(axis=0)
    return 0.0 if np.linalg.norm(v) < 1e-12 else math.atan2(v[1], v[0])


def linearized_gaussian(model: PolarRingModel, t: float, theta0: float):
    """Mean, covariance and precision of the first-order phase-conditioned model."""
    T = model.T
    a = math.exp(-model.kappa * model.h)
    var_r = model.D_r / model.kappa
    q_theta = 2.0 * model.D_theta * model.h

    Cz = np.zeros((2 * T, 2 * T), dtype=float)
    for i in range(T):
        for j in range(T):
            Cz[2 * i, 2 * j] = var_r * a ** abs(i - j)
            # Local unwrapped branch, conditioned on phi_0=0.
            Cz[2 * i + 1, 2 * j + 1] = q_theta * min(i, j)

    H = np.zeros((2 * T, 2 * T), dtype=float)
    mean_clean = np.empty((T, 2), dtype=float)
    for k in range(T):
        th = theta0 + model.omega * model.h * k
        er, et = local_basis(th)
        mean_clean[k] = er
        H[2 * k:2 * k + 2, 2 * k] = er
        H[2 * k:2 * k + 2, 2 * k + 1] = et

    C_clean = H @ Cz @ H.T
    m, delta = ou_parameters(t)
    mean_noisy = m * mean_clean.reshape(-1)
    Sigma = m * m * C_clean + delta * np.eye(2 * T)
    Q = np.linalg.inv(Sigma)
    return mean_noisy, Sigma, Q


def linearized_score(x: Array, model: PolarRingModel, t: float, theta0: float | None = None) -> Array:
    if theta0 is None:
        theta0 = estimate_initial_phase(x, model)
    mean, _, Q = linearized_gaussian(model, t, theta0)
    return (-Q @ (x.reshape(-1) - mean)).reshape(model.T, 2)


def cavity_mass(solver: PolarGridSmoother, x: Array, t: float, k: int) -> Array:
    """p(H_k | X_{-k}) on the grid, used for a 2D conditional score slice."""
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


def exact_score_slice(
    solver: PolarGridSmoother,
    x: Array,
    t: float,
    k: int,
    points: Array,
    cavity: Array | None = None,
) -> tuple[Array, Array, Array]:
    if cavity is None:
        cavity = cavity_mass(solver, x, t, k)
    m, delta = ou_parameters(t)
    residual = points[:, None, None, :] - m * solver.Axy[None, ...]
    logl = -np.sum(residual**2, axis=-1) / (2.0 * delta)
    maxlog = np.max(logl, axis=(1, 2), keepdims=True)
    weights = cavity[None, ...] * np.exp(logl - maxlog)
    z = weights.sum(axis=(1, 2))
    means = np.einsum("mij,ijc->mc", weights, solver.Axy) / z[:, None]
    scores = (m * means - points) / delta
    log_density = np.log(z) + maxlog[:, 0, 0]
    return log_density, means, scores


def marginal_score_slice(solver: PolarGridSmoother, t: float, k: int, points: Array):
    prior = solver.prior_marginals[k]
    m, delta = ou_parameters(t)
    residual = points[:, None, None, :] - m * solver.Axy[None, ...]
    logl = -np.sum(residual**2, axis=-1) / (2.0 * delta)
    maxlog = np.max(logl, axis=(1, 2), keepdims=True)
    weights = prior[None, ...] * np.exp(logl - maxlog)
    z = weights.sum(axis=(1, 2))
    means = np.einsum("mij,ijc->mc", weights, solver.Axy) / z[:, None]
    return np.log(z) + maxlog[:, 0, 0], (m * means - points) / delta


def linearized_slice(x: Array, model: PolarRingModel, t: float, k: int, points: Array, theta0: float):
    mean, _, Q = linearized_gaussian(model, t, theta0)
    base = x.reshape(-1).copy()
    scores = np.empty_like(points)
    logd = np.empty(len(points), dtype=float)
    sl = slice(2 * k, 2 * k + 2)
    for n, p in enumerate(points):
        y = base.copy()
        y[sl] = p
        d = y - mean
        s = -Q @ d
        scores[n] = s[sl]
        logd[n] = -0.5 * d @ Q @ d
    return logd, scores


def linear_response_prediction(model: PolarRingModel, t: float, max_lag: int):
    """Infinite-chain response prediction of the local Taylor model."""
    h = model.h
    a = math.exp(-model.kappa * h)
    q_r = (model.D_r / model.kappa) * (1.0 - a * a)
    q_t = 2.0 * model.D_theta * h
    m, delta = ou_parameters(t)
    lam = m * m / delta

    gamma_r = math.acosh(max((1 + a * a + q_r * lam) / (2 * a), 1.0))
    gamma_t = math.acosh(max(1 + 0.5 * q_t * lam, 1.0))
    lag = np.arange(max_lag + 1)
    Cr = q_r / (2 * a * max(math.sinh(gamma_r), 1e-15)) * np.exp(-gamma_r * lag)
    Ct = q_t / (2 * max(math.sinh(gamma_t), 1e-15)) * np.exp(-gamma_t * lag)
    pref = m * m / delta**2
    return {
        "rr": pref * Cr,
        "tt": pref * Ct,
        "xi_rr": 1 / gamma_r if gamma_r > 0 else float("inf"),
        "xi_tt": 1 / gamma_t if gamma_t > 0 else float("inf"),
    }
