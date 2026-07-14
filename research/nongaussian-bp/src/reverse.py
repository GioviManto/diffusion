"""Reverse-time samplers for the OU forward process.

Convention (matches src/noising.py)
-----------------------------------
Forward SDE:   dX_t = -X_t dt + sqrt(2) dW_t,  t in [0, T],  X_0 ~ p_0.
Marginals:     X_t ~ Law(alpha_t a + sqrt(Delta_t) z).

Reverse-time SDE (Anderson 1982), integrated from t = T down to t = t_min:

    dX = [ -X - 2 s(X, t) ] dt + sqrt(2) dW_bar        (dt < 0),

implemented in Euler-Maruyama form with step h > 0:

    x <- x + h [ x + 2 s(x, t) ] + sqrt(2 h) xi,   xi ~ N(0, I).

Probability-flow ODE (same marginals, deterministic):

    dX/dt = -X - s(X, t)    =>    x <- x + h [ x + s(x, t) ]   (backwards),

with an optional Heun (trapezoidal) corrector.

Because unit-variance data keep Var(X_t) = 1, we initialize at t = T with
X_T ~ N(0, I) (the T -> infinity limit; at T = 3 the bias is alpha^2 ~ 2.5e-3),
or from a forward-noised known sample for reconstruction experiments.

The final output can be read out either as x(t_min) or through the posterior
mean ("denoising readout")  m = (x + Delta_t s(x, t)) / alpha_t, which is exact
for BP-based scores since they are built from posterior means.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

ScoreFn = Callable[[np.ndarray, float], np.ndarray]
Callback = Callable[[float, np.ndarray, np.ndarray], None]


def time_grid(t_max: float, t_min: float, n_steps: int) -> np.ndarray:
    """Geometric time grid from t_max down to t_min (denser at small t,
    where the score stiffens as alpha/Delta ~ 1/(2t))."""
    return np.geomspace(t_max, t_min, n_steps + 1)


def reverse_sde(
    x_init: np.ndarray,
    score_fn: ScoreFn,
    times: np.ndarray,
    rng: np.random.Generator,
    callback: Callback | None = None,
) -> np.ndarray:
    """Euler-Maruyama reverse SDE from times[0] down to times[-1].

    All stochasticity comes from `rng`; passing generators with identical state
    to different score functions yields common-noise paired trajectories.
    """
    x = x_init.copy()
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s = score_fn(x, float(t_now))
        if callback is not None:
            callback(float(t_now), x, s)
        x = x + h * (x + 2.0 * s) + np.sqrt(2.0 * h) * rng.standard_normal(x.shape)
    if callback is not None:
        s = score_fn(x, float(times[-1]))
        callback(float(times[-1]), x, s)
    return x


def probability_flow_ode(
    x_init: np.ndarray,
    score_fn: ScoreFn,
    times: np.ndarray,
    callback: Callback | None = None,
    heun: bool = True,
) -> np.ndarray:
    """Deterministic probability-flow ODE from times[0] down to times[-1]."""
    x = x_init.copy()
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s = score_fn(x, float(t_now))
        if callback is not None:
            callback(float(t_now), x, s)
        drift = x + s
        x_euler = x + h * drift
        if heun:
            s2 = score_fn(x_euler, float(t_next))
            drift2 = x_euler + s2
            x = x + 0.5 * h * (drift + drift2)
        else:
            x = x_euler
    if callback is not None:
        s = score_fn(x, float(times[-1]))
        callback(float(times[-1]), x, s)
    return x


def denoising_readout(x: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
    """Posterior-mean readout  E[a | x] = (x + Delta_t s) / alpha_t."""
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    return (x + delta * s) / alpha
