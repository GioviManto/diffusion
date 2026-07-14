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
