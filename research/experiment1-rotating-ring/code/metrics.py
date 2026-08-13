#!/usr/bin/env python3
"""Response diagnostics for the joint diffusion score.

Given a cross-frame response profile

    C_t(l) = E_x || dS_k / dx_{k+l} ||_F ,   l = 0, 1, ..., Lmax,

four scalars summarise it. They are deliberately kept separate because they
answer different questions and, as the measurements show, move in different
directions with diffusion time.

    I_off(t)   = sum_{l>=1} C_t(l)                 how much context matters
    lbar(t)    = sum_{l>=1} l C_t(l) / I_off(t)    how far it comes from
    Xi(t)      = I_off(t) * lbar(t)                weighted reach (absolute)
    Xi_rel(t)  = Xi(t) / C_t(0)                    weighted reach (relative)

`lbar` and `Xi_rel` both have a finite-chain ceiling, reported alongside them,
which is the value a completely structureless (flat) profile would produce.
Reading a value near that ceiling as a long correlation length is the main
interpretation error these definitions are designed to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass(frozen=True)
class ResponseDiagnostics:
    """Summary of one cross-frame response profile."""

    self_response: float
    intensity: float
    mean_lag: float
    weighted_reach: float
    relative_reach: float
    normalised_range: float
    radius_95: float
    max_lag: int

    @property
    def flat_mean_lag(self) -> float:
        """Mean lag a structureless (flat) profile would give."""
        return 0.5 * (1.0 + self.max_lag)

    @property
    def flat_normalised_range(self) -> float:
        """Normalised range a structureless profile would give."""
        return float(self.max_lag)

    def as_dict(self) -> dict[str, float]:
        return {
            "self_response": self.self_response,
            "intensity": self.intensity,
            "mean_lag": self.mean_lag,
            "weighted_reach": self.weighted_reach,
            "relative_reach": self.relative_reach,
            "normalised_range": self.normalised_range,
            "radius_95": self.radius_95,
            "max_lag": float(self.max_lag),
            "flat_mean_lag": self.flat_mean_lag,
            "flat_normalised_range": self.flat_normalised_range,
        }


def diagnostics(lags: Array, profile: Array) -> ResponseDiagnostics:
    """Compute all response diagnostics from one profile.

    Parameters
    ----------
    lags : integer lags, must contain 0 and be sorted ascending.
    profile : non-negative response magnitudes at those lags.
    """
    lags = np.asarray(lags, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if lags.shape != profile.shape:
        raise ValueError("lags and profile must have the same shape")
    if not np.all(np.diff(lags) > 0):
        raise ValueError("lags must be strictly increasing")
    if lags[0] != 0:
        raise ValueError("profile must include lag 0 (the self response)")
    if np.any(profile < 0):
        raise ValueError("response magnitudes must be non-negative")

    self_response = float(profile[0])
    off_lags, off = lags[1:], profile[1:]
    intensity = float(off.sum())
    if intensity <= 0:
        raise ValueError("off-diagonal response mass vanishes")

    mean_lag = float((off_lags * off).sum() / intensity)
    weighted_reach = intensity * mean_lag
    cumulative = np.cumsum(off) / intensity
    radius_95 = float(off_lags[int(np.searchsorted(cumulative, 0.95))])

    return ResponseDiagnostics(
        self_response=self_response,
        intensity=intensity,
        mean_lag=mean_lag,
        weighted_reach=weighted_reach,
        relative_reach=weighted_reach / self_response,
        normalised_range=float((off / off[0]).sum()),
        radius_95=radius_95,
        max_lag=int(off_lags[-1]),
    )


def diagnostics_table(
    frame: pd.DataFrame,
    value_column: str,
    time_column: str = "t",
    lag_column: str = "lag",
) -> pd.DataFrame:
    """Apply :func:`diagnostics` to every diffusion time in a tidy profile table."""
    records = []
    for time, group in frame.groupby(time_column, sort=True):
        ordered = group.sort_values(lag_column)
        summary = diagnostics(
            ordered[lag_column].to_numpy(),
            ordered[value_column].to_numpy(),
        ).as_dict()
        records.append({time_column: float(time), **summary})
    return pd.DataFrame.from_records(records)
