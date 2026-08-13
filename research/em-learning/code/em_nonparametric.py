"""Rungs 2 and 3 of the EM ladder: learn the innovation *shape*, not just its variance.

Motivation
----------
Rung 1 (``em_parametric``) uses a Gaussian E-step, which by the closure/LMMSE proposition
depends on the innovation only through its variance.  It is therefore structurally
incapable of learning the shape of the innovation law, no matter how much data it sees.
This module removes that limitation by running the E-step with *grid* BP, which carries the
whole message function, and by making the innovation density itself the object being
estimated.

This is the literal form of Marc's suggestion: BP is run without being told the transition
kernel.  We assume only that the chain is additive,

    a_{i+1} = rho a_i + eps,      eps ~ p_eps  (unknown),

and estimate ``rho`` together with the tabulated density ``p_eps``.

The M-step
----------
The expected complete-data log-likelihood contributed by the transitions is

    Q(rho, p_eps) = sum_i E_{xi_i} [ log p_eps(a_{i+1} - rho a_i) ] ,

with ``xi_i`` the pairwise posterior from the E-step.  Two consequences:

* **For fixed rho, the maximiser over p_eps is available in closed form.**  It is the
  distribution of the residual ``a_{i+1} - rho a_i`` under the pairwise posterior --
  obtained by "shearing" each ``xi_i`` along the line ``a' - rho a = const`` and
  marginalising.  (Proof: writing ``g`` for the residual mass and ``N`` for its total,
  ``Q(p) = int g log p``, and Gibbs' inequality gives ``Q(p) <= Q(g/N)`` with equality only
  at ``p = g/N``.)  We deposit mass with linear interpolation so the operation conserves
  probability exactly rather than rounding to the nearest bin.

  Two caveats that the phrase "closed form" would otherwise hide.  First, the splat is the
  exact maximiser of the *interpolated-log* objective, whereas the code's ``Q`` evaluates the
  log of the *interpolated* density; by concavity the former is a minorant of the latter, and
  the measured gap is about 0.1 nats.  Second, linear deposition convolves the residual law
  with a triangular kernel of half-width ``h``, which **adds h^2/6 to the estimated
  variance**: measured excess 1.59e-3 at h=0.1 against a predicted 1.67e-3, and 3.94e-4 at
  h=0.05 against 4.17e-4, i.e. clean second-order scaling.  At the resolutions used here that
  is a +0.4% bias on ``q``.  It is the same mechanism as the smoothing bias discussed below,
  and it is present even at ``smooth_sigma = 0``.

* **For fixed p_eps there is no closed form for rho**, so it is maximised by a direct scan
  of ``Q`` over a bracket around the current value, with the incumbent always included as a
  candidate so the step can never move backwards.

How rigorous is the ascent guarantee?
--------------------------------------
This is a *heuristic* generalised EM, and the honest position is to say so rather than to
claim monotone convergence of the likelihood.  Four places where the guarantee is only
approximate:

1. The ``rho`` scan is exact only up to the resolution of the scan grid.  It is now safe in
   the sense that it cannot decrease ``Q`` (the incumbent is a candidate), which was **not**
   true before: with the bracket clipped at ``rho_max`` the incumbent fell off the grid for
   ``rho >= 0.90``, and on a rho=0.95 chain ``Q`` decreased at three separate iterations.
2. Smoothing, when enabled, is applied *after* the maximisation and strictly decreases ``Q``
   (measured -0.23, -1.99, -19.73 nats at sigma = 0.5h, 1.0h, 2.0h).  This is one more reason
   the default is zero; any claim of monotonicity is conditional on ``smooth_sigma = 0``.
3. The zero-mean reprojection is likewise applied after the maximiser and costs ~0.02 nats
   when it fires.
4. ``Q`` as reported here covers only the transition term.  The complete-data objective also
   contains ``E[log pi(a_1)]``, which depends on the parameters through the stationary
   density; that term is about 4% of ``|Q|`` at convergence and is omitted.

Consequently ``q_trace`` is a diagnostic, not the EM objective, and its monotonicity implies
nothing rigorous.  A better monitor is available essentially for free and is worth adding:
the exact observed-data log-likelihood is the sum of the log normalising constants that
``normalize_density`` currently discards in the forward pass.  Accumulating
``log sum(L_i * ell_i * w)`` would give the true likelihood to watch instead.

The zero-mean constraint on p_eps
---------------------------------
We constrain ``p_eps`` to have zero mean after each M-step.  It is worth being precise about
why, because the obvious justification is wrong.

There is *no* shift degeneracy.  If ``p'(e) = p(e - c)`` then ``K'(a'|a) = p(a' - rho a - c)``,
and ``K' == K`` forces ``c = 0``: the kernel determines ``(rho, p_eps)`` uniquely.  A shift is
directly observable, because it moves the stationary mean to ``c/(1-rho)`` and hence moves
``E[x] = alpha_t E[a]`` with ``alpha_t`` known.  (Checked: shifting ``p_eps`` by 0.5 at
rho=0.8 moves the stationary mean from 0.0000 to +2.4967 = 0.5/0.2, and no ``rho'`` in
[0,0.99] brings ``K'`` back to ``K`` -- the best residual is 0.32 against a kernel peak of
0.66.)

The constraint is therefore a *true restriction satisfied by the data-generating process*,
imposed to improve conditioning rather than to resolve an ambiguity.  The profiled objective
``max_p Q(rho,p)`` equals ``-N * H(residual)``, and differential entropy is shift invariant,
so an unconstrained nonparametric ``p`` simply discards the first-moment equation
``E[a'] = rho E[a]``; centring puts it back.  For a stationary zero-mean chain that equation
is nearly vacuous, and empirically the reprojection is close to inert: from a centred start
it never fires, and from a start deliberately off-centre by +0.8 the M-step recentres itself
to +0.0009 with ``rho`` converging to 0.8000 either way.

Two honest caveats about the implementation.  The correction only fires when the mean exceeds
half a grid spacing, so a systematic bias below that is left in place.  And a rigid
translation is not the exact constrained maximiser: that would be
``p_j = n_j / (w_j (N + nu u_j))`` with ``nu`` the root of ``sum_j u_j n_j/(N + nu u_j) = 0``.
The rigid shift gives away about 0.02 nats relative to that, which is why the reprojection is
listed below as one of the places the ascent guarantee is only approximate.

Regularisation, and why it defaults to *off*
--------------------------------------------
A tabulated density estimated from finite data is noisy, and that noise feeds back into the
next E-step through the kernel, so smoothing is tempting.  It is also, at the bandwidths one
might naively pick, actively harmful.  Measured on a Laplace chain (rho=0.80, q=0.36,
b=0.424, 400 chains, t=0.15, grid spacing h=0.1):

    smooth_sigma    rho_hat   var_hat   p(0)     L1 to true Laplace
    0               0.800     0.373     0.753    0.192
    0.5 h           0.800     0.379     0.731    0.208
    1.0 h           0.790     0.408     0.678    0.251
    2.0 h           0.770     0.497     0.590    0.344

Smoothing with a Gaussian of width sigma adds sigma^2 to the variance and flattens the peak,
and both errors propagate into ``rho``.  The default is therefore ``smooth_sigma = 0``.  The
option remains for cases where the raw estimate is visibly unstable, but any non-zero value
must be reported alongside the result, which is why it is stored on the result object.

Resolution requirement
----------------------
The binding constraint is grid resolution, not smoothing.  The innovation density must be
resolved on the same grid that carries the messages, so the grid spacing has to be small
compared with the innovation scale.  Measured on the same problem, with the number of grid
points per innovation standard deviation:

    pts/scale   rho_hat   var_hat   L1 to Laplace   L1 to Gaussian
    4.2         0.790     0.374     0.197           0.125      <- WRONG family
    8.5         0.800     0.361     0.166           0.185      <- correct
    14.1        0.800     0.359     0.159           0.202      <- correct

Below roughly eight points per innovation standard deviation the learned density is closer
to a Gaussian than to the truth, and the whole point of the exercise is lost.  ``em_nonparametric_fit``
therefore checks this and warns.  Note also what is *not* recovered even at high resolution:
the peak value at zero is around 0.81 against a true 1.18, so the cusp of the Laplace is
systematically under-estimated.  We can claim the families are distinguished; we cannot
claim the density is recovered pointwise.

Cost
----
The E-step is ``O(n_chains * n * M^2)`` per iteration, which dominates everything else here.
Nonparametric EM is genuinely more expensive than rung 1 (which is ``O(n_chains * n)``), and
that difference is itself part of the story: the cheap estimator buys you two numbers, the
expensive one buys you a function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bp_core import (
    Grid,
    grid_bp,
    likelihood_matrix,
    normalize_density,
    transition_matrix_from_innovation,
)


# -----------------------------------------------------------------------------

@dataclass
class NonparametricEMResult:
    rho: float
    innovation_density: np.ndarray
    grid: Grid
    n_iter: int
    converged: bool
    q_trace: list[float] = field(default_factory=list)
    rho_trace: list[float] = field(default_factory=list)
    smooth_sigma: float = 0.0
    density_floor: float = 0.0
    pts_per_innovation_scale: float = 0.0

    @property
    def innovation_variance(self) -> float:
        p = self.innovation_density
        w = self.grid.weights
        u = self.grid.points
        mean = float(np.sum(u * p * w))
        return float(np.sum((u - mean) ** 2 * p * w))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def stationary_density(
    K: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 20000,
    rtol: float = 1e-12,
) -> np.ndarray:
    """Stationary law of the chain, by power iteration on the tabulated kernel.

    With a nonparametric innovation there is no closed-form stationary distribution, so we
    obtain it as the fixed point of ``pi = K pi``.  This is used as the prior on ``a_1``;
    getting it wrong biases the first site and, through it, ``rho``.

    Convergence rate.  The subdominant eigenvalue of the kernel is ``rho``, so the error
    decays like ``rho^n_iter``.  The previous fixed budget of 300 iterations was adequate at
    moderate correlation but badly short at the top of the range: at ``rho = 0.985``,
    ``0.985^300 = e^{-4.5}``, roughly 1% error.  Measured there, 300 iterations returned a
    density with L1 error 1.1e-3 and variance 1.00231 instead of 1.0, still drifting by
    4.6e-4 between 300 and 20000 iterations.  The old absolute stopping test (1e-13 on
    O(1) density values) sat at the double-precision floor and never fired, so the shortfall
    was silent.  We now use a *relative* tolerance and a budget large enough for
    ``rho -> 0.985``.

    Note also that ``transition_matrix_from_innovation`` truncates the kernel at the grid
    edges, so ``K`` is slightly sub-stochastic and this returns the Perron (quasi-stationary)
    vector rather than the exact stationary law.  On a grid wide enough for the boundary-mass
    diagnostic to pass, the difference is negligible.
    """
    m = K.shape[0]
    pi = np.full(m, 1.0 / float(np.sum(weights)))
    for _ in range(n_iter):
        new = normalize_density(K @ (pi * weights), weights)
        if np.max(np.abs(new - pi)) < rtol * max(float(np.max(new)), 1e-300):
            return new
        pi = new
    return pi


def _gaussian_smooth(p: np.ndarray, grid: Grid, sigma: float) -> np.ndarray:
    """Convolve a tabulated density with a Gaussian of width ``sigma`` (same units as grid)."""
    if sigma <= 0.0:
        return p
    d = grid.points[:, None] - grid.points[None, :]
    ker = np.exp(-0.5 * (d / sigma) ** 2)
    ker /= ker.sum(axis=1, keepdims=True)
    return ker @ p


def _residual_distribution(
    xi_sum: np.ndarray, grid: Grid, rho: float
) -> np.ndarray:
    """Distribution of ``a' - rho a`` under the aggregated pairwise posterior.

    ``xi_sum[l, k]`` is the total posterior mass on ``(a_i = u_l, a_{i+1} = u_k)``.  The
    residual for that pair is ``u_k - rho u_l``; we deposit its mass onto the two
    neighbouring grid nodes with linear weights, which conserves total mass and avoids the
    quantisation ripple that nearest-bin assignment produces.
    """
    u = grid.points
    lo = float(u[0])
    h = grid.spacing

    resid = u[None, :] - rho * u[:, None]          # (l, k)

    # Clip rather than discard: residuals outside the grid are rare if the grid is wide
    # enough, and silently dropping them would break normalisation.
    pos = np.clip((resid - lo) / h, 0.0, (u.size - 1) - 1e-12)
    idx = np.floor(pos).astype(np.int64).ravel()
    frac = (pos - np.floor(pos)).ravel()
    mass = xi_sum.ravel()

    out = np.zeros(u.size)
    np.add.at(out, idx, mass * (1.0 - frac))
    np.add.at(out, idx + 1, mass * frac)

    # ``out`` holds probability mass per node; convert to a density before normalising.
    return normalize_density(out / grid.weights, grid.weights)


def _expected_transition_loglik(
    xi_sum: np.ndarray, grid: Grid, rho: float, p_eps: np.ndarray, floor: float
) -> float:
    """Q(rho) = sum over pairs of xi * log p_eps(a' - rho a), by linear interpolation."""
    u = grid.points
    lo, hi = float(u[0]), float(u[-1])
    h = grid.spacing
    resid = u[None, :] - rho * u[:, None]
    pos = np.clip((resid - lo) / h, 0.0, (u.size - 1) - 1e-12)
    idx = np.floor(pos).astype(np.int64)
    frac = pos - idx
    safe = np.maximum(p_eps, floor)
    vals = (1.0 - frac) * safe[idx] + frac * safe[idx + 1]
    return float(np.sum(xi_sum * np.log(vals)))


# -----------------------------------------------------------------------------
# The EM loop
# -----------------------------------------------------------------------------

def em_nonparametric_fit(
    x: np.ndarray,
    alpha: float,
    delta: float,
    grid: Grid,
    *,
    rho_init: float = 0.5,
    max_iter: int = 40,
    tol: float = 1e-7,
    smooth_sigma: float | None = None,
    density_floor: float = 1e-12,
    rho_scan_halfwidth: float = 0.12,
    rho_scan_points: int = 25,
    rho_max: float = 0.985,
    verbose: bool = False,
) -> NonparametricEMResult:
    """Learn ``(rho, p_eps)`` from noisy chains with no parametric family assumed.

    ``x`` has shape ``(n_chains, n)``.  ``grid`` must be wide enough to contain both the
    state and the innovation; the boundary diagnostic in ``grid_bp`` will complain if not.
    """
    n_chains, n = x.shape
    u = grid.points
    w = grid.weights
    if smooth_sigma is None:
        # Off by default: see the module docstring. Non-zero bandwidths inflate the variance
        # by sigma^2 and flatten the peak, and both errors propagate into rho.
        smooth_sigma = 0.0

    rho = float(rho_init)
    # Initialise from a Gaussian whose variance matches the observed lag-0/lag-1 structure,
    # after removing the known OU noise contribution. A deliberately crude start: the point
    # is to show the shape is *learned*, not supplied.
    var_x = float(np.mean(x ** 2))
    var_a0 = max((var_x - delta) / max(alpha ** 2, 1e-12), 1e-3)
    q0 = max(var_a0 * (1.0 - rho ** 2), 1e-3)
    p_eps = normalize_density(np.exp(-0.5 * u ** 2 / q0), w)

    # Resolution guard. Below ~8 grid points per innovation standard deviation the learned
    # density comes out closer to a Gaussian than to the truth, which silently defeats the
    # whole purpose of the nonparametric rung. Warn rather than fail: the caller may be
    # deliberately exploring the under-resolved regime.
    pts_per_scale = np.sqrt(q0) / grid.spacing
    if pts_per_scale < 8.0:
        import warnings

        warnings.warn(
            f"grid resolves the innovation with only {pts_per_scale:.1f} points per standard "
            f"deviation (spacing {grid.spacing:.3g}, initial innovation sd {np.sqrt(q0):.3g}). "
            "Below about 8 the estimated density is biased towards a Gaussian and the "
            "family cannot be identified. Refine the grid or narrow its extent.",
            RuntimeWarning,
            stacklevel=2,
        )
    res_pts_per_scale = float(pts_per_scale)

    res = NonparametricEMResult(rho, p_eps, grid, 0, False,
                                smooth_sigma=smooth_sigma, density_floor=density_floor,
                                pts_per_innovation_scale=res_pts_per_scale)
    prev_q = -np.inf

    for it in range(1, max_iter + 1):
        K = transition_matrix_from_innovation(grid, rho, p_eps)
        # Guard: a degenerate kernel (all-zero column) would poison the E-step.
        colmass = K.T @ w
        if np.any(colmass <= 0.0):
            K = K + 1e-300
        pi = stationary_density(K, w)

        # ---- E-step: accumulate pairwise posteriors over all chains ----
        xi_sum = np.zeros((grid.size, grid.size))
        for c in range(n_chains):
            ell = likelihood_matrix(grid, x[c], alpha, delta)
            _, xi = grid_bp(grid, K, ell, pi, want_pairwise=True)
            xi_sum += xi.sum(axis=0)

        # ---- M-step: rho by scan, then p_eps in closed form ----
        lo = max(0.0, rho - rho_scan_halfwidth)
        hi = min(rho_max, rho + rho_scan_halfwidth)
        # The incumbent must be a candidate, otherwise this "maximisation" can return a
        # WORSE rho than it started with and the generalised-EM ascent guarantee is void.
        # When the bracket clips against rho_max the linspace no longer contains rho: at
        # rho=0.90 the nearest candidate is 4.2e-4 away, at rho=0.95 it is 2.7e-3 away.
        # Measured effect before this fix, on a rho=0.95 chain: Q decreased at iterations
        # 1, 3 and 5 by 0.111, 0.015 and 0.137 nats.
        cands = np.unique(np.append(np.linspace(lo, hi, rho_scan_points), rho))
        qs = [
            _expected_transition_loglik(xi_sum, grid, r, p_eps, density_floor)
            for r in cands
        ]
        rho = float(cands[int(np.argmax(qs))])

        p_new = _residual_distribution(xi_sum, grid, rho)
        p_new = _gaussian_smooth(p_new, grid, smooth_sigma)
        p_new = normalize_density(np.maximum(p_new, density_floor), w)

        # Fix the shift degeneracy: the innovation must have zero mean, otherwise rho and a
        # constant offset trade off against each other and the scan drifts.
        mean_eps = float(np.sum(u * p_new * w))
        if abs(mean_eps) > 0.5 * grid.spacing:
            p_new = np.interp(u + mean_eps, u, p_new, left=0.0, right=0.0)
            p_new = normalize_density(np.maximum(p_new, density_floor), w)
        p_eps = p_new

        q_now = _expected_transition_loglik(xi_sum, grid, rho, p_eps, density_floor)
        res.q_trace.append(q_now)
        res.rho_trace.append(rho)
        if verbose:
            print(f"    iter {it:3d}  rho={rho:.4f}  Q={q_now:.6f}", flush=True)

        if np.isfinite(prev_q) and abs(q_now - prev_q) < tol * max(1.0, abs(prev_q)):
            res.converged = True
            res.n_iter = it
            break
        prev_q = q_now
    else:
        res.n_iter = max_iter

    res.rho = rho
    res.innovation_density = p_eps
    return res


# -----------------------------------------------------------------------------

def density_l1_distance(p: np.ndarray, q: np.ndarray, grid: Grid) -> float:
    """L1 distance between two tabulated densities; 0 identical, 2 disjoint."""
    return float(np.sum(np.abs(p - q) * grid.weights))
