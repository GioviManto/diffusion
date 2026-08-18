"""Shared error metrics. All take (estimate, reference) in that order."""

from __future__ import annotations

import numpy as np


def rel_l2(est: np.ndarray, ref: np.ndarray) -> float:
    """||est - ref||_2 / ||ref||_2."""
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(est - ref)) / denom if denom > 0 else np.nan


def mse(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((est - ref) ** 2))


def max_abs(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(est - ref)))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v)) / denom if denom > 0 else np.nan


def transition_hellinger(
    log_k_est: np.ndarray,
    log_k_ref: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    interior_frac: float = 0.5,
) -> dict:
    """Hellinger distance between two transition kernels, at the density level.

    Everything else in this package measures the fitted kernel through what it
    does -- the score it induces, the posterior mean it produces. Those are the
    quantities the diffusion model actually uses, but they are also forgiving:
    two visibly different innovation densities can induce scores that agree to
    the third decimal at moderate noise, because the channel has already blurred
    the difference away. A claim that the transition itself was recovered has to
    be stated on the transition.

    For each parent state u_j the columns K(.|u_j) are densities on the grid, so

        H^2(p, q) = 1 - sum_k w_k sqrt(p_k q_k)

    with w the quadrature weights. H is in [0, 1]; 0 is identical, 1 is disjoint
    support. Reported as a summary over parent states rather than a single
    number, because the tails are where a fitted mixture goes wrong and averaging
    over columns hides it.

    Returns median, mean, p90 and max over parent states, plus the same over
    interior parents only. The interior restriction matters for the same reason
    it does in the quadrature diagnostic: at |u| near the half-width the true
    transition density has a tail outside the grid, so its column is not
    normalised there and the distance picks up truncation rather than fit error.
    `interior_frac` is the fraction of the half-width kept, matching
    `bp_grid`'s convention.

    RESOLUTION FLOOR, ~4e-8. H is the square root of 1 - BC, and BC is computed
    to about 1e-15, so the root amplifies that to 1e-8 -- measured at 3.7e-8 for a
    kernel against itself at M = 401. Two densities closer than that are
    indistinguishable here, and a reported Hellinger of 1e-8 means "identical to
    arithmetic", not "identical". This is inherent to the metric, not to the
    implementation: nothing in double precision recovers the difference between
    BC = 1 and BC = 1 - 1e-17. Score-level metrics do NOT have this floor, which
    is one reason to keep reporting both.
    """
    # Column-normalise under the SAME quadrature both kernels are represented
    # in. Comparing an analytically normalised density against a grid-normalised
    # one measures the normalisation convention, not the fit -- a real hazard
    # here, where the analytic kernels normalise on the continuum and the MDN
    # kernel normalises on the grid.
    #
    # This is also why there is no "are these normalised?" guard: after this step
    # they are, by construction, so such a check could never fire.
    def cols(log_k):
        k = np.exp(log_k - log_k.max(axis=0, keepdims=True))
        mass = (k * weights[:, None]).sum(axis=0)
        return k / np.maximum(mass, 1e-300)

    p, q = cols(log_k_est), cols(log_k_ref)
    overlap = (weights[:, None] * np.sqrt(p * q)).sum(axis=0)
    # BC <= 1 by Cauchy-Schwarz; anything above is arithmetic, and clipping the
    # negative that would otherwise reach the sqrt is the only reason to clamp.
    h = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(overlap, 1.0)))

    interior = np.abs(grid) <= interior_frac * float(np.max(np.abs(grid)))
    return {
        "hellinger_median": float(np.median(h)),
        "hellinger_mean": float(h.mean()),
        "hellinger_p90": float(np.quantile(h, 0.90)),
        "hellinger_max": float(h.max()),
        "hellinger_median_interior": float(np.median(h[interior])),
        "hellinger_max_interior": float(h[interior].max()),
    }


def score_mean_identity_residual(
    s_est: np.ndarray,
    s_ref: np.ndarray,
    m_est: np.ndarray,
    m_ref: np.ndarray,
    alpha: float,
    delta: float,
) -> float:
    """Relative residual of the exact identity  s_est - s_ref = (a/d)(m_est - m_ref).

    Any score produced through `score_from_posterior_mean` satisfies this to
    machine precision; a large residual indicates an implementation bug.
    """
    direct = s_est - s_ref
    predicted = (alpha / delta) * (m_est - m_ref)
    return float(
        np.linalg.norm(direct - predicted) / (np.linalg.norm(direct) + 1e-300)
    )
