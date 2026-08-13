#!/usr/bin/env python3
"""Exact rotating-ring surrogate used in Experiment 1.

The known rotation is removed first. In the co-rotating frame,

    Y_0 = A,  A ~ p_lambda(a) proportional to exp(-(||||a||||-1)^2/(2 lambda)),
    Y_k = A + sigma * sum_{r=1}^k eta_r,

and the whole trajectory is corrupted by the VP Ornstein-Uhlenbeck channel

    X_t = exp(-t) Y + sqrt(1-exp(-2t)) Xi.

Conditional on A, X_t is Gaussian. The only remaining nonlinear integral is the
2D posterior of A, which reduces by rotational symmetry to 1D radial quadrature.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ive

Array = np.ndarray


@dataclass(frozen=True)
class SurrogateConfig:
    T: int = 30
    lam: float = 0.05
    sigma: float = 0.15
    psi: float = 2.0 * math.pi / 30.0
    quadrature_nodes: int = 180
    posterior_grid_size: int = 600


def ou_parameters(t: float) -> tuple[float, float]:
    if t <= 0:
        raise ValueError("t must be positive")
    m = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    return m, delta


def rotation(angle: float) -> Array:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def sample_ring_anchor(rng: np.random.Generator, n: int, lam: float) -> Array:
    """Sample the 2D density proportional to exp(-(r-1)^2/(2 lambda))."""
    r_max = 2.0 + 10.0 * math.sqrt(lam)
    r = np.linspace(0.0, r_max, 200_000)
    radial_pdf = r * np.exp(-0.5 * (r - 1.0) ** 2 / lam)
    cdf = np.cumsum((radial_pdf[:-1] + radial_pdf[1:]) * 0.5 * np.diff(r))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    radii = np.interp(rng.random(n), cdf, r)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def sample_clean_trajectories(
    rng: np.random.Generator,
    cfg: SurrogateConfig,
    n: int,
) -> tuple[Array, Array]:
    """Return co-rotating Y and laboratory Z, each shape (n,T,2)."""
    anchors = sample_ring_anchor(rng, n, cfg.lam)
    increments = cfg.sigma * rng.standard_normal((n, cfg.T - 1, 2))
    y = np.empty((n, cfg.T, 2), dtype=float)
    y[:, 0] = anchors
    y[:, 1:] = anchors[:, None, :] + np.cumsum(increments, axis=1)
    z = np.empty_like(y)
    for k in range(cfg.T):
        z[:, k] = y[:, k] @ rotation(k * cfg.psi).T
    return y, z


def add_ou_noise(rng: np.random.Generator, y: Array, t: float) -> Array:
    m, delta = ou_parameters(t)
    return m * y + math.sqrt(delta) * rng.standard_normal(y.shape)


def random_walk_covariance(indices: Array, sigma: float) -> Array:
    idx = np.asarray(indices, dtype=float)
    return sigma**2 * np.minimum.outer(idx, idx)


class RingPosteriorLookup:
    """Posterior moments of the shared ring anchor for fixed beta.

    p(a | h) proportional to exp[-(r-1)^2/(2 lambda) - beta r^2/2 + h dot a].
    The mean is parallel to h. The covariance has one eigenvalue parallel to h
    and one perpendicular to it.
    """

    def __init__(self, lam: float, beta: float, h_max: float, n_h: int, n_quad: int):
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
        # ive(n,q) = exp(-q) I_n(q), q >= 0. Add q to the exponent.
        expo = base + Q
        shift = np.max(expo, axis=1, keepdims=True)
        common = np.exp(expo - shift) * w[None, :]

        i0e = ive(0, Q)
        i1e = ive(1, Q)
        i2e = ive(2, Q)
        den = np.sum(common * R * i0e, axis=1)
        mean_num = np.sum(common * R**2 * i1e, axis=1)
        par2_num = 0.5 * np.sum(common * R**3 * (i0e + i2e), axis=1)
        perp2_num = 0.5 * np.sum(common * R**3 * (i0e - i2e), axis=1)

        mean = mean_num / den
        e_par2 = par2_num / den
        e_perp2 = perp2_num / den
        var_par = np.maximum(e_par2 - mean**2, 0.0)
        var_perp = np.maximum(e_perp2, 0.0)
        iso = 0.5 * (var_par[0] + var_perp[0])
        mean[0] = 0.0
        var_par[0] = iso
        var_perp[0] = iso
        self.mean_grid = mean
        self.var_par_grid = var_par
        self.var_perp_grid = var_perp

    def moments(self, h: Array) -> tuple[Array, Array]:
        h = np.asarray(h, dtype=float)
        rho = np.linalg.norm(h, axis=1)
        if np.max(rho) > self.h_grid[-1] * (1 + 1e-10):
            raise ValueError("posterior field exceeds lookup range")
        mean_mag = np.interp(rho, self.h_grid, self.mean_grid)
        var_par = np.interp(rho, self.h_grid, self.var_par_grid)
        var_perp = np.interp(rho, self.h_grid, self.var_perp_grid)
        unit = np.zeros_like(h)
        nz = rho > 1e-12
        unit[nz] = h[nz] / rho[nz, None]
        mean = mean_mag[:, None] * unit
        cov = np.empty((len(h), 2, 2), dtype=float)
        eye = np.eye(2)
        for i in range(len(h)):
            if nz[i]:
                uu = np.outer(unit[i], unit[i])
                cov[i] = var_perp[i] * eye + (var_par[i] - var_perp[i]) * uu
            else:
                cov[i] = var_perp[i] * eye
        return mean, cov


def exact_score_and_jacobian(
    x: Array,
    indices: Array,
    t: float,
    cfg: SurrogateConfig,
    cache: Dict[Tuple[Tuple[int, ...], float], RingPosteriorLookup] | None = None,
) -> tuple[Array, Array, dict[str, Array]]:
    """Exact score and Jacobian for selected trajectory indices.

    Parameters
    ----------
    x : array, shape (n,L,2)
    indices : absolute internal-time indices, shape (L,)

    Returns
    -------
    score : (n,L,2)
    jacobian : (n,L,L,2,2)
    pieces : useful matrices C, C^{-1}, q, posterior covariance.
    """
    if cache is None:
        cache = {}
    x = np.asarray(x, dtype=float)
    indices = np.asarray(indices, dtype=int)
    n, L, d = x.shape
    if d != 2 or len(indices) != L:
        raise ValueError("shape/index mismatch")
    m, delta = ou_parameters(t)
    K = random_walk_covariance(indices, cfg.sigma)
    C = m * m * K + delta * np.eye(L)
    Cinv = np.linalg.inv(C)
    ones = np.ones(L)
    q = Cinv @ ones
    beta = m * m * float(ones @ q)
    h = m * np.einsum("l,nld->nd", q, x)

    key = (tuple(indices.tolist()), round(float(t), 12))
    if key not in cache:
        h_max = max(5.0, 1.2 * float(np.max(np.linalg.norm(h, axis=1))) + 2.0)
        cache[key] = RingPosteriorLookup(
            cfg.lam,
            beta,
            h_max,
            cfg.posterior_grid_size,
            cfg.quadrature_nodes,
        )
    mean_a, cov_a = cache[key].moments(h)

    score = -np.einsum("ij,njd->nid", Cinv, x) + m * q[None, :, None] * mean_a[:, None, :]
    jac = np.empty((n, L, L, 2, 2), dtype=float)
    eye = np.eye(2)
    for i in range(L):
        for j in range(L):
            jac[:, i, j] = -Cinv[i, j] * eye + m * m * q[i] * q[j] * cov_a
    return score, jac, {"K": K, "C": C, "Cinv": Cinv, "q": q, "mean_a": mean_a, "cov_a": cov_a}


def clean_score(y: Array, cfg: SurrogateConfig) -> Array:
    """Clean joint score in the co-rotating frame."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    r0 = np.linalg.norm(y[..., 0, :], axis=-1)
    unit0 = y[..., 0, :] / np.maximum(r0[..., None], 1e-12)
    ring = ((1.0 - r0) / cfg.lam)[..., None] * unit0
    out[..., 0, :] = ring + (y[..., 1, :] - y[..., 0, :]) / cfg.sigma**2
    if cfg.T > 2:
        out[..., 1:-1, :] = (y[..., :-2, :] - 2.0 * y[..., 1:-1, :] + y[..., 2:, :]) / cfg.sigma**2
    out[..., -1, :] = (y[..., -2, :] - y[..., -1, :]) / cfg.sigma**2
    return out
