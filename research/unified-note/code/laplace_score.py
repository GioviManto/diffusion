"""
laplace_score.py
================

Laplace K = 1 warm-up: a single Laplace variable

    a ~ Laplace(0, b),    p_0(a) = (1 / (2 b)) exp(- |a| / b),    Var(a) = 2 b^2

corrupted by the OU forward channel

    x = exp(-t) a + sqrt(Delta_t) xi,    xi ~ N(0, 1).

This is the smallest case in which the noised marginal ceases to be
Gaussian and the joint score becomes nonlinear in x. We compute every
object by 1D quadrature -- the closed-form truncated-Gaussian
representation is intentionally avoided here, both because it is the
section we agreed to cut and because quadrature is fully sufficient
for the figures.

Tweedie identity (single-variable form):
    S(x, t) = (exp(-t) E[a | x] - x) / Delta_t.

References (Established):
    [K1] session_summary_K1_warmup.pdf, sections 3.1-3.2, 3.4-3.5
    [K1] Theorem 2 (Tweedie identity)
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from ar1_utils import ou_params


# ---------------------------------------------------------------------------
# Variance-matching helper (equal-variance comparison with Gaussian innovation)
# ---------------------------------------------------------------------------

def variance_matched_b(sigma_eta: float) -> float:
    """Laplace scale b such that Var(Laplace(0, b)) == sigma_eta^2.

    Var(Laplace(0, b)) = 2 b^2, so b = sigma_eta / sqrt(2).
    Used to compare the Laplace innovation against a Gaussian innovation
    of the same variance.
    """
    return float(sigma_eta) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Densities and score
# ---------------------------------------------------------------------------

def _laplace_pdf(a: np.ndarray | float, b: float) -> np.ndarray | float:
    return np.exp(-np.abs(a) / b) / (2.0 * b)


def _gaussian_pdf(z: np.ndarray | float, mean: float, var: float
                  ) -> np.ndarray | float:
    return np.exp(-0.5 * (z - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def laplace_marginal(x: float, t: float, b: float = 1.0,
                     limit: int = 200) -> float:
    """Return the noised marginal p_t(x) by 1D quadrature.

    p_t(x) = int_R p_0(a) N(x ; exp(-t) a, Delta_t) da.
    """
    mu, Delta = ou_params(t)

    def integrand(a: float) -> float:
        return _laplace_pdf(a, b) * _gaussian_pdf(x, mu * a, Delta)

    val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=limit)
    return val


def laplace_posterior_mean(x: float, t: float, b: float = 1.0,
                           limit: int = 200) -> float:
    """Return E[a | x] by 1D quadrature."""
    mu, Delta = ou_params(t)

    def joint(a: float) -> float:
        return _laplace_pdf(a, b) * _gaussian_pdf(x, mu * a, Delta)

    def first_moment(a: float) -> float:
        return a * joint(a)

    norm, _ = integrate.quad(joint, -np.inf, np.inf, limit=limit)
    num, _ = integrate.quad(first_moment, -np.inf, np.inf, limit=limit)
    return num / norm


def laplace_score(x: float, t: float, b: float = 1.0,
                  limit: int = 200) -> float:
    """Return the joint score S(x, t) via Tweedie's identity."""
    mu, Delta = ou_params(t)
    e_a = laplace_posterior_mean(x, t, b, limit=limit)
    return (mu * e_a - x) / Delta


# Vectorised convenience wrappers
def _vectorise(fn, x_array, *args, **kwargs):
    out = np.empty_like(np.asarray(x_array, dtype=float))
    for i, x in enumerate(np.atleast_1d(x_array)):
        out[i] = fn(float(x), *args, **kwargs)
    return out


def laplace_marginal_v(x_array, t: float, b: float = 1.0) -> np.ndarray:
    return _vectorise(laplace_marginal, x_array, t, b)


def laplace_score_v(x_array, t: float, b: float = 1.0) -> np.ndarray:
    return _vectorise(laplace_score, x_array, t, b)


# ---------------------------------------------------------------------------
# Fourier representation (K = 1)
#
# Setup: y = mu * a + sqrt(Delta_t) * xi, a ~ Laplace(0, b), xi ~ N(0, 1).
# Independence factorises the characteristic function:
#     phi_t(k) := E[exp(i k y)]
#               = E[exp(i k mu a)] * E[exp(i k sqrt(Delta_t) xi)]
#               = 1 / (1 + b^2 mu^2 k^2) * exp(-Delta_t k^2 / 2).
# The first factor is the Laplace c.f. (rescaled by mu), the second is
# the Gaussian noise c.f.  Multiplying both sides by (1 + b^2 mu^2 k^2)
# and inverse-transforming gives the screened Poisson equation
#     (1 - b^2 mu^2 d^2/dy^2) p_t(y) = G_{Delta_t}(y),
# whose Green's function is the Laplace kernel of width b * mu.
# ---------------------------------------------------------------------------

def laplace_marginal_cf(k: float | np.ndarray, t: float,
                        b: float = 1.0) -> float | np.ndarray:
    """Closed-form characteristic function of p_t (K = 1, Laplace prior).

        phi_t(k) = 1 / (1 + b^2 mu^2 k^2) * exp(-Delta_t k^2 / 2).
    """
    mu, Delta = ou_params(t)
    laplace_factor = 1.0 / (1.0 + (b * mu * k) ** 2)
    gauss_factor = np.exp(-0.5 * Delta * np.asarray(k) ** 2)
    return laplace_factor * gauss_factor


def laplace_marginal_cf_quadrature(k: float, t: float, b: float = 1.0,
                                   limit: int = 400) -> complex:
    """Evaluate phi_t(k) = int p_t(y) exp(i k y) dy by quadrature on the
    closed-form density laplace_marginal.  Used by the audit only.
    """
    def re(y: float) -> float:
        return laplace_marginal(y, t, b) * np.cos(k * y)

    def im(y: float) -> float:
        return laplace_marginal(y, t, b) * np.sin(k * y)

    re_val, _ = integrate.quad(re, -np.inf, np.inf, limit=limit)
    im_val, _ = integrate.quad(im, -np.inf, np.inf, limit=limit)
    return complex(re_val, im_val)


def laplace_marginal_via_green(y: float, t: float, b: float = 1.0,
                               limit: int = 400) -> float:
    """Reconstruct p_t(y) from the screened-Poisson representation
    p_t = G_ODE * G_{Delta_t}, where G_ODE(z) = exp(-|z|/(b mu)) / (2 b mu).
    Used by the audit to verify the screened Poisson identity numerically.
    """
    mu, Delta = ou_params(t)
    bmu = b * mu

    def G_ode(z: float) -> float:
        return np.exp(-abs(z) / bmu) / (2.0 * bmu)

    def G_gauss(z: float) -> float:
        return np.exp(-0.5 * z * z / Delta) / np.sqrt(2.0 * np.pi * Delta)

    def integrand(z: float) -> float:
        return G_ode(y - z) * G_gauss(z)

    val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=limit)
    return val


# ---------------------------------------------------------------------------
# Gaussian K = 1 reference for side-by-side comparison
# ---------------------------------------------------------------------------

def gaussian_K1_score(x_array, t: float, sigma_a: float = 1.0) -> np.ndarray:
    """Joint score of Gaussian a ~ N(0, sigma_a^2) corrupted by OU at t.

    Sigma_t = exp(-2 t) sigma_a^2 + Delta_t,  S(x, t) = - x / Sigma_t.
    """
    mu, Delta = ou_params(t)
    Sigma_t = (mu * mu) * sigma_a * sigma_a + Delta
    return -np.asarray(x_array, dtype=float) / Sigma_t


def gaussian_K1_marginal(x_array, t: float, sigma_a: float = 1.0
                         ) -> np.ndarray:
    mu, Delta = ou_params(t)
    Sigma_t = (mu * mu) * sigma_a * sigma_a + Delta
    return _gaussian_pdf(np.asarray(x_array, dtype=float), 0.0, Sigma_t)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity: at t -> 0 the Laplace score should approach -sign(x) / Delta_t
    # interpretation, but Delta_t -> 0 makes the score blow up; we instead
    # check (i) the Tweedie identity and (ii) a finite-difference check
    # of S = d_x log p_t.
    b = 1.0
    t = 0.4
    x = 0.7
    # Finite difference of log p_t
    h = 1e-4
    lp_plus = np.log(laplace_marginal(x + h, t, b))
    lp_minus = np.log(laplace_marginal(x - h, t, b))
    s_fd = (lp_plus - lp_minus) / (2 * h)
    s_tweedie = laplace_score(x, t, b)
    print(f"|S_tweedie - S_fd|  = {abs(s_tweedie - s_fd):.2e}")

    # Check the Gaussian limit: a Gaussian variable of variance 2 b^2
    # gives a linear score; the Laplace score, while nonlinear, must
    # still be antisymmetric in x.
    s_pos = laplace_score(0.5, 0.3, b)
    s_neg = laplace_score(-0.5, 0.3, b)
    print(f"|S(x) + S(-x)|       = {abs(s_pos + s_neg):.2e}")
