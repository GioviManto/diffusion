"""
Belief propagation for diffusion scores in Markov chains:
exact grid BP, Gaussian grid validation, and Gaussian-message approximation error.

Main question
-------------
For a clean Markov chain a_1,...,a_n observed through coordinatewise OU noise,

    x_i = alpha_t a_i + sqrt(Delta_t) z_i,
    alpha_t = exp(-t), Delta_t = 1 - exp(-2t),

we have the exact identity

    score_i(x,t) = d/dx_i log p_t(x) = -(x_i - alpha_t E[a_i | x]) / Delta_t.

On a chain, BP computes the posterior marginal p(a_i | x) exactly at the level of
continuous message functions. In practice, however, non-Gaussian continuous messages
must be represented approximately. This script studies two numerical questions:

1. How accurate is grid BP?  We validate it in the Gaussian AR(1) case, where the
   exact score is available by linear algebra: s(x,t) = -Sigma_t^{-1} x.
2. In a non-Gaussian Laplace-innovation AR(1) chain, how large is the error caused
   by forcing BP messages to be Gaussian?

Gaussian message closure
------------------------
Gaussian closure is implemented in ``gaussian_closure_bp`` as a pure information-form
(lambda, h) recursion with no grid at all.  This matters, and it is the correct way to
do it:  a Gaussian is fully characterised by two scalars, so introducing a grid can only
add error.

Two consequences are worth stating explicitly.

* In the Gaussian AR(1) case the recursion is exact and reproduces the precision-matrix
  score to machine precision at every diffusion time.
* For *any* additive innovation a_{i+1} = rho a_i + eps with E[eps]=0, Var(eps)=q, moment
  matching only ever uses the first two moments of eps.  A Gaussian message convolved
  with the true kernel has mean rho*mu and variance q + rho^2*v regardless of the shape
  of eps.  Gaussian closure is therefore *blind to the non-Gaussianity of the
  innovations*: it collapses exactly onto the linear (LMMSE) denoiser
  m = alpha Sigma_0 Sigma_t^{-1} x.  What this script measures as "Gaussian message
  error" is thus precisely the excess error of the best linear denoiser over the exact
  one -- i.e. the nonlinearity that a learned score has to supply.

``gaussian_projected_bp_grid`` retains the earlier grid-based projection as an explicit
ablation.  It is *wrong*, on purpose, and quantifying how wrong is one of the outputs:
see its docstring for the failure mechanism.

Grid diagnostics
----------------
Grid BP truncates the state space to [grid_min, grid_max].  Beliefs and forward messages
are genuine probability densities, so a non-negligible probability mass at the grid edges
means the interval is too narrow and the results cannot be trusted.  ``grid_bp`` accumulates
this boundary mass and ``run_all`` reports the worst value seen; exceeding
``boundary_mass_tol`` raises.  (Backward messages R_i are likelihoods, not densities, and
are legitimately near-flat at large t, so they are excluded from the check.)

Outputs
-------
The script writes CSV files and PNG figures to the output directory:

    gaussian_grid_validation.csv
    gaussian_grid_validation_rel_error.png
    gaussian_closure_sanity_rel_error.png
    laplace_grid_convergence.csv
    laplace_grid_convergence_rel_error.png
    laplace_gaussian_message_error.csv
    laplace_posterior_mean_mse.png
    laplace_relative_score_error.png
    laplace_score_cosine_similarity.png
    example_laplace_belief_comparison.png
    example_gaussian_grid_vs_matrix_score.png

Run examples
------------
Quick run:
    python bp_general_markov_diffusion_experiments.py --mode quick

Report-quality run:
    python bp_general_markov_diffusion_experiments.py --mode report

Custom run:
    python bp_general_markov_diffusion_experiments.py --n 50 --rho 0.85 --n-trials 8 --ref-grid-size 401
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Grid sizes below are chosen so that the spacing is 0.16 / 0.08 / 0.04 on the
    # [-12,12] interval. The interval is wider than the [-8,8] used in the first version
    # of the report: at [-8,8] the boundary-mass diagnostic reached 7.4e-7, which is only
    # marginally acceptable, whereas [-12,12] brings it to 3e-11 at the same resolution.
    n: int = 50
    rho: float = 0.85
    grid_min: float = -12.0
    grid_max: float = 12.0
    validation_grid_sizes: tuple[int, ...] = (151, 301, 601)
    convergence_coarse_grid_size: int = 301
    ref_grid_size: int = 601
    n_trials: int = 50
    seed: int = 11
    output_dir: str = "bp_markov_diffusion_outputs"
    t_values: tuple[float, ...] = (0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30, 1.80, 2.40)
    # Fraction of a density allowed to sit in the outermost grid cells before the
    # truncation is considered to be corrupting the result.
    boundary_mass_tol: float = 1e-6
    boundary_edge_points: int = 3


# -----------------------------------------------------------------------------
# Basic densities and numerical helpers
# -----------------------------------------------------------------------------

def normal_pdf(x: np.ndarray, mean: np.ndarray | float, var: float) -> np.ndarray:
    """Evaluate a univariate Gaussian density N(mean,var)."""
    var = max(float(var), 1e-14)
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def laplace_pdf(x: np.ndarray, loc: np.ndarray | float, b: float) -> np.ndarray:
    """Evaluate a Laplace density with location loc and scale b."""
    b = max(float(b), 1e-14)
    return np.exp(-np.abs(x - loc) / b) / (2.0 * b)


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Composite trapezoidal integration weights for an equally spaced grid."""
    dx = float(grid[1] - grid[0])
    w = np.full_like(grid, dx, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def normalize_density(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative grid values as a density with respect to quadrature weights."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    mass = float(np.sum(values * weights))
    if not np.isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("Cannot normalize density: non-positive or non-finite mass.")
    return values / mass


def density_moments(values: np.ndarray, grid: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return mean and variance of a grid density after normalization."""
    p = normalize_density(values, weights)
    mean = float(np.sum(grid * p * weights))
    var = float(np.sum((grid - mean) ** 2 * p * weights))
    return mean, max(var, 1e-10)


def gaussian_message_values(grid: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Evaluate a Gaussian message. Infinite variance represents a flat message."""
    if np.isinf(var):
        return np.ones_like(grid)
    return normal_pdf(grid, mean, var)


def boundary_mass(values: np.ndarray, weights: np.ndarray, n_edge: int) -> float:
    """Fraction of a density's mass sitting in the outermost ``n_edge`` cells per side.

    A converged BP belief on a wide enough interval has essentially no mass at the grid
    edges.  A large value means the truncation [grid_min, grid_max] is biting and the
    interval must be widened.
    """
    p = normalize_density(values, weights)
    mass = p * weights
    return float(np.sum(mass[:n_edge]) + np.sum(mass[-n_edge:]))


class GridDiagnostics:
    """Accumulate the worst boundary mass seen across a set of BP runs."""

    def __init__(self, tol: float, n_edge: int) -> None:
        self.tol = tol
        self.n_edge = n_edge
        self.worst = 0.0
        self.worst_context = ""

    def check(self, values: np.ndarray, weights: np.ndarray, context: str) -> None:
        b = boundary_mass(values, weights, self.n_edge)
        if b > self.worst:
            self.worst = b
            self.worst_context = context
        if b > self.tol:
            raise FloatingPointError(
                f"Grid truncation error: {b:.3e} of the probability mass lies in the "
                f"outermost {self.n_edge} grid cells ({context}). "
                f"Tolerance is {self.tol:.1e}. Widen [grid_min, grid_max]."
            )

    def report(self) -> str:
        if not self.worst_context:
            return "no densities checked"
        return f"worst boundary mass {self.worst:.3e} at {self.worst_context} (tol {self.tol:.1e})"


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(u, v) / denom)


def score_from_mean(x: np.ndarray, mean: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Convert a posterior mean into the OU diffusion score."""
    return -(x - alpha * mean) / delta


def ar1_covariance(n: int, rho: float) -> np.ndarray:
    """Stationary AR(1) covariance Sigma_0(i,j)=rho^|i-j|."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def gaussian_precision_score(x: np.ndarray, rho: float, alpha: float, delta: float) -> np.ndarray:
    """Exact Gaussian AR(1) score s=-Sigma_t^{-1}x."""
    n = len(x)
    sigma0 = ar1_covariance(n, rho)
    sigmat = alpha ** 2 * sigma0 + delta * np.eye(n)
    return -np.linalg.solve(sigmat, x)


def gaussian_posterior_mean_exact(x: np.ndarray, rho: float, alpha: float, delta: float) -> np.ndarray:
    """Exact posterior mean E[a|x] in the Gaussian AR(1) case."""
    n = len(x)
    sigma0 = ar1_covariance(n, rho)
    sigmat = alpha ** 2 * sigma0 + delta * np.eye(n)
    return alpha * sigma0 @ np.linalg.solve(sigmat, x)


# -----------------------------------------------------------------------------
# Sampling clean chains and observations
# -----------------------------------------------------------------------------

def sample_clean_chain(rng: np.random.Generator, n: int, rho: float, prior: str) -> np.ndarray:
    """Sample a clean AR(1)-type chain with Gaussian or Laplace innovations."""
    if prior not in {"gaussian", "laplace"}:
        raise ValueError("prior must be either 'gaussian' or 'laplace'.")

    q = 1.0 - rho ** 2
    a = np.empty(n)
    a[0] = rng.normal(0.0, 1.0)

    if prior == "gaussian":
        for i in range(1, n):
            a[i] = rho * a[i - 1] + np.sqrt(q) * rng.normal()
    else:
        b = np.sqrt(q / 2.0)  # Var Laplace(0,b)=2b^2=q.
        for i in range(1, n):
            a[i] = rho * a[i - 1] + rng.laplace(0.0, b)
    return a


def sample_noisy_observation(rng: np.random.Generator, a: np.ndarray, t: float) -> tuple[np.ndarray, float, float]:
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    x = alpha * a + np.sqrt(delta) * rng.normal(size=a.shape)
    return x, alpha, delta


# -----------------------------------------------------------------------------
# Transition and likelihood matrices on a grid
# -----------------------------------------------------------------------------

def make_grid(config: ExperimentConfig, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(config.grid_min, config.grid_max, grid_size)
    weights = trapezoid_weights(grid)
    return grid, weights


def build_transition_matrix(grid: np.ndarray, rho: float, prior: str) -> np.ndarray:
    """Return K[out,in] = p(a_out | a_in) evaluated on the grid."""
    q = 1.0 - rho ** 2
    out = grid[:, None]
    inn = grid[None, :]
    if prior == "gaussian":
        return normal_pdf(out, rho * inn, q)
    if prior == "laplace":
        return laplace_pdf(out, rho * inn, np.sqrt(q / 2.0))
    raise ValueError("prior must be either 'gaussian' or 'laplace'.")


def likelihood_matrix(grid: np.ndarray, x: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """ell[i,k] = p(x_i | a_i=grid[k])."""
    return normal_pdf(x[:, None], alpha * grid[None, :], delta)


# -----------------------------------------------------------------------------
# Algorithm 1: reference grid BP
# -----------------------------------------------------------------------------

def grid_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray,
    ell: np.ndarray,
    diagnostics: GridDiagnostics | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sum-product BP on a chain by representing messages on a grid.

    L_i(a_i) is the left message entering node i, excluding the local likelihood ell_i.
    R_i(a_i) is the right message entering node i, excluding the local likelihood ell_i.

    The recursions are quadrature approximations to

        L_{i+1}(a') = int K(a'|a) L_i(a) ell_i(a) da,
        R_{i-1}(a)  = int K(a'|a) R_i(a') ell_i(a') da'.

    Returns posterior means, normalized beliefs, and posterior variances.

    If ``diagnostics`` is supplied, the boundary mass of every forward message and every
    belief is checked.  Both are genuine probability densities -- L_i is the filtered
    distribution p(a_i | x_1..x_{i-1}) and b_i is the posterior marginal -- so mass at the
    grid edges is a truncation artefact.  Backward messages are likelihoods rather than
    densities and are deliberately not checked.
    """
    n, m = ell.shape
    L = np.empty((n, m))
    R = np.empty((n, m))

    # Initial prior mu(a_1)=N(0,1).
    L[0] = normalize_density(normal_pdf(grid, 0.0, 1.0), weights)

    # Forward pass.
    for i in range(n - 1):
        incoming = L[i] * ell[i] * weights
        L[i + 1] = K @ incoming
        L[i + 1] = normalize_density(L[i + 1], weights)
        if diagnostics is not None:
            diagnostics.check(L[i + 1], weights, f"forward message L[{i + 1}]")

    # Backward pass. A flat terminal message is represented by ones.
    R[-1] = np.ones(m)
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[i] * weights
        R[i - 1] = K.T @ incoming
        R[i - 1] = normalize_density(R[i - 1], weights)

    beliefs = np.empty((n, m))
    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        b = normalize_density(L[i] * ell[i] * R[i], weights)
        if diagnostics is not None:
            diagnostics.check(b, weights, f"belief b[{i}]")
        beliefs[i] = b
        means[i] = float(np.sum(grid * b * weights))
        variances[i] = float(np.sum((grid - means[i]) ** 2 * b * weights))

    return means, beliefs, variances


# -----------------------------------------------------------------------------
# Algorithm 2: Gaussian message closure (information form, grid free)
# -----------------------------------------------------------------------------

def gaussian_closure_bp(
    x: np.ndarray,
    rho: float,
    q: float,
    alpha: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Gaussian message closure as a pure (lambda, h) recursion. No grid.

    Every message is written in information form

        m(a) propto exp(-lambda a^2 / 2 + h a),      mean = h / lambda,  var = 1 / lambda,

    which is the finite-dimensional closure of section 7.2 of the report.  Because a
    Gaussian is exactly characterised by two scalars, no discretisation is involved and
    no discretisation error can enter.

    Forward.  With L_i * ell_i propto N(mu, v), propagating through
    a_{i+1} = rho a_i + eps gives mean rho*mu and variance q + rho^2*v.

    Backward.  R_{i-1}(a) = int K(a'|a) R_i(a') ell_i(a') da' propto
    N(a; mu/rho, (q + v)/rho^2), written below directly in information form so that
    rho -> 0 stays well behaved.

    Only the first two moments of the innovation are ever used, so this routine is
    correct for a Gaussian chain and is *the* Gaussian closure for a non-Gaussian one:
    applied to a Laplace, Student or mixture innovation with the same variance q it
    returns the identical answer, namely the linear (LMMSE) denoiser.

    Returns posterior means, posterior variances, and the message parameters.
    """
    n = len(x)
    lam_obs = alpha ** 2 / delta          # precision contributed by one observation
    h_obs = alpha * x / delta             # information contributed by one observation

    lam_L = np.zeros(n)
    h_L = np.zeros(n)
    lam_L[0], h_L[0] = 1.0, 0.0           # prior mu(a_1) = N(0,1)

    for i in range(n - 1):
        lam_t = lam_L[i] + lam_obs        # multiply by the local likelihood
        h_t = h_L[i] + h_obs[i]
        mu_t, v_t = h_t / lam_t, 1.0 / lam_t
        v_out = q + rho ** 2 * v_t        # push through the transition
        lam_L[i + 1] = 1.0 / v_out
        h_L[i + 1] = rho * mu_t / v_out

    lam_R = np.zeros(n)
    h_R = np.zeros(n)
    lam_R[-1], h_R[-1] = 0.0, 0.0         # flat terminal message: zero precision

    for i in range(n - 1, 0, -1):
        lam_t = lam_R[i] + lam_obs
        h_t = h_R[i] + h_obs[i]
        mu_t, v_t = h_t / lam_t, 1.0 / lam_t
        denom = q + v_t
        lam_R[i - 1] = rho ** 2 / denom
        h_R[i - 1] = rho * mu_t / denom

    lam = lam_L + lam_R + lam_obs
    h = h_L + h_R + h_obs
    means = h / lam
    variances = 1.0 / lam

    params = {
        "L_mean": np.divide(h_L, lam_L, out=np.zeros(n), where=lam_L > 0),
        "L_var": np.divide(1.0, lam_L, out=np.full(n, np.inf), where=lam_L > 0),
        "R_mean": np.divide(h_R, lam_R, out=np.zeros(n), where=lam_R > 0),
        "R_var": np.divide(1.0, lam_R, out=np.full(n, np.inf), where=lam_R > 0),
        "belief_lambda": lam,
        "belief_h": h,
    }
    return means, variances, params


def gaussian_projected_bp_grid(
    grid: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray,
    ell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Grid-based Gaussian projection. WRONG ON PURPOSE, retained as an ablation.

    Each message is stored as N(mean,var) but is evaluated on the grid, updated there,
    and moment-matched back.  This is the algorithm used in the first version of the
    report and it fails badly.  The mechanism is worth recording:

    The terminal backward message is flat, stored as ``R_var = inf`` and evaluated as a
    vector of ones.  A constant function restricted to [-A, A] and then moment-matched
    yields variance (2A)^2/12 -- about 21 for A = 8 -- rather than infinity.  The
    projection therefore injects a spurious precision of roughly 1/21 into the backward
    pass.  The damage grows with diffusion time because alpha_t -> 0 flattens every
    message, so progressively more of them are effectively fitted to the truncation
    width rather than to their own shape.

    In the Gaussian AR(1) case, where the answer should be exact, this reaches a relative
    score error of order 10 at t = 1.8.  Compare ``gaussian_closure_bp``, which is exact
    to machine precision at every t.
    """
    n, m = ell.shape
    L_mean = np.empty(n)
    L_var = np.empty(n)
    R_mean = np.empty(n)
    R_var = np.empty(n)

    # Initial prior is exactly Gaussian.
    L_mean[0] = 0.0
    L_var[0] = 1.0

    for i in range(n - 1):
        L_values = gaussian_message_values(grid, L_mean[i], L_var[i])
        incoming = L_values * ell[i] * weights
        outgoing = K @ incoming
        L_mean[i + 1], L_var[i + 1] = density_moments(outgoing, grid, weights)

    # Terminal right message is flat: infinite variance.
    R_mean[-1] = 0.0
    R_var[-1] = np.inf

    for i in range(n - 1, 0, -1):
        R_values = gaussian_message_values(grid, R_mean[i], R_var[i])
        incoming = R_values * ell[i] * weights
        outgoing = K.T @ incoming
        R_mean[i - 1], R_var[i - 1] = density_moments(outgoing, grid, weights)

    beliefs = np.empty((n, m))
    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        L_values = gaussian_message_values(grid, L_mean[i], L_var[i])
        R_values = gaussian_message_values(grid, R_mean[i], R_var[i])
        b = normalize_density(L_values * ell[i] * R_values, weights)
        beliefs[i] = b
        means[i] = float(np.sum(grid * b * weights))
        variances[i] = float(np.sum((grid - means[i]) ** 2 * b * weights))

    params = {"L_mean": L_mean, "L_var": L_var, "R_mean": R_mean, "R_var": R_var}
    return means, beliefs, variances, params


# -----------------------------------------------------------------------------
# Experiment A: Gaussian grid BP validation against exact precision score
# -----------------------------------------------------------------------------

def run_gaussian_grid_validation(
    config: ExperimentConfig,
    rng: np.random.Generator,
    diagnostics: GridDiagnostics | None = None,
) -> list[dict[str, float]]:
    """Validate both algorithms in the Gaussian case, where the exact score is known.

    Grid BP should be accurate to quadrature error.  Gaussian closure should be exact to
    machine precision, since the Gaussian family is closed under the BP updates.  The
    grid-projection ablation is run alongside to quantify what the discarded
    implementation was costing.
    """
    rows: list[dict[str, float]] = []
    q = 1.0 - config.rho ** 2

    for grid_size in config.validation_grid_sizes:
        grid, weights = make_grid(config, grid_size)
        K = build_transition_matrix(grid, config.rho, prior="gaussian")

        for t in config.t_values:
            rel_grid_scores = []
            rel_closure_scores = []
            rel_ablation_scores = []
            mean_grid_mses = []
            mean_closure_mses = []

            for _ in range(config.n_trials):
                a = sample_clean_chain(rng, config.n, config.rho, prior="gaussian")
                x, alpha, delta = sample_noisy_observation(rng, a, t)
                ell = likelihood_matrix(grid, x, alpha, delta)

                mean_grid, _, _ = grid_bp(grid, weights, K, ell, diagnostics)
                mean_closure, _, _ = gaussian_closure_bp(x, config.rho, q, alpha, delta)
                mean_ablation, _, _, _ = gaussian_projected_bp_grid(grid, weights, K, ell)

                score_grid = score_from_mean(x, mean_grid, alpha, delta)
                score_closure = score_from_mean(x, mean_closure, alpha, delta)
                score_ablation = score_from_mean(x, mean_ablation, alpha, delta)
                score_exact = gaussian_precision_score(x, config.rho, alpha, delta)
                mean_exact = gaussian_posterior_mean_exact(x, config.rho, alpha, delta)

                norm_exact = float(np.linalg.norm(score_exact))
                rel_grid_scores.append(float(np.linalg.norm(score_grid - score_exact) / norm_exact))
                rel_closure_scores.append(float(np.linalg.norm(score_closure - score_exact) / norm_exact))
                rel_ablation_scores.append(float(np.linalg.norm(score_ablation - score_exact) / norm_exact))
                mean_grid_mses.append(float(np.mean((mean_grid - mean_exact) ** 2)))
                mean_closure_mses.append(float(np.mean((mean_closure - mean_exact) ** 2)))

            rows.append({
                "prior": "gaussian",
                "grid_size": float(grid_size),
                "t": float(t),
                "rel_score_error_grid_mean": float(np.mean(rel_grid_scores)),
                "rel_score_error_grid_std": float(np.std(rel_grid_scores, ddof=1)) if len(rel_grid_scores) > 1 else 0.0,
                "rel_score_error_gauss_closure_mean": float(np.mean(rel_closure_scores)),
                "rel_score_error_gauss_closure_std": float(np.std(rel_closure_scores, ddof=1)) if len(rel_closure_scores) > 1 else 0.0,
                "rel_score_error_grid_projection_ablation_mean": float(np.mean(rel_ablation_scores)),
                "posterior_mean_mse_grid_mean": float(np.mean(mean_grid_mses)),
                "posterior_mean_mse_gauss_closure_mean": float(np.mean(mean_closure_mses)),
            })

    return rows


# -----------------------------------------------------------------------------
# Experiment B: Laplace grid convergence and Gaussian-message approximation error
# -----------------------------------------------------------------------------

def run_laplace_grid_convergence(
    config: ExperimentConfig,
    rng: np.random.Generator,
    diagnostics: GridDiagnostics | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    coarse_size = config.convergence_coarse_grid_size
    fine_size = 2 * coarse_size - 1

    coarse_grid, coarse_weights = make_grid(config, coarse_size)
    fine_grid, fine_weights = make_grid(config, fine_size)
    K_coarse = build_transition_matrix(coarse_grid, config.rho, prior="laplace")
    K_fine = build_transition_matrix(fine_grid, config.rho, prior="laplace")

    for t in config.t_values:
        rel_scores = []
        mean_mses = []
        cos_scores = []

        for _ in range(config.n_trials):
            a = sample_clean_chain(rng, config.n, config.rho, prior="laplace")
            x, alpha, delta = sample_noisy_observation(rng, a, t)

            ell_c = likelihood_matrix(coarse_grid, x, alpha, delta)
            ell_f = likelihood_matrix(fine_grid, x, alpha, delta)

            mean_c, _, _ = grid_bp(coarse_grid, coarse_weights, K_coarse, ell_c, diagnostics)
            mean_f, _, _ = grid_bp(fine_grid, fine_weights, K_fine, ell_f, diagnostics)

            score_c = score_from_mean(x, mean_c, alpha, delta)
            score_f = score_from_mean(x, mean_f, alpha, delta)

            rel_scores.append(float(np.linalg.norm(score_c - score_f) / np.linalg.norm(score_f)))
            mean_mses.append(float(np.mean((mean_c - mean_f) ** 2)))
            cos_scores.append(cosine_similarity(score_c, score_f))

        rows.append({
            "prior": "laplace",
            "coarse_grid_size": float(coarse_size),
            "fine_grid_size": float(fine_size),
            "t": float(t),
            "relative_score_error_mean": float(np.mean(rel_scores)),
            "relative_score_error_std": float(np.std(rel_scores, ddof=1)) if len(rel_scores) > 1 else 0.0,
            "posterior_mean_mse_mean": float(np.mean(mean_mses)),
            "posterior_mean_mse_std": float(np.std(mean_mses, ddof=1)) if len(mean_mses) > 1 else 0.0,
            "score_cosine_mean": float(np.mean(cos_scores)),
            "score_cosine_std": float(np.std(cos_scores, ddof=1)) if len(cos_scores) > 1 else 0.0,
        })

    return rows


def run_laplace_gaussian_message_error(
    config: ExperimentConfig,
    rng: np.random.Generator,
    diagnostics: GridDiagnostics | None = None,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Measure what Gaussian message closure loses on a non-Gaussian chain.

    Because closure is provably identical to the linear denoiser (see the module
    docstring), this quantity is the excess error of the best linear estimator over the
    exact one.  The column ``lmmse_gap`` verifies that identity numerically: it should be
    at machine precision.
    """
    rows: list[dict[str, float]] = []
    example: dict[str, object] | None = None
    q = 1.0 - config.rho ** 2

    grid, weights = make_grid(config, config.ref_grid_size)
    K = build_transition_matrix(grid, config.rho, prior="laplace")

    for t in config.t_values:
        mse_m_list = []
        mse_s_list = []
        rel_s_list = []
        cos_s_list = []
        identity_error_list = []
        lmmse_gap_list = []

        for trial in range(config.n_trials):
            a = sample_clean_chain(rng, config.n, config.rho, prior="laplace")
            x, alpha, delta = sample_noisy_observation(rng, a, t)
            ell = likelihood_matrix(grid, x, alpha, delta)

            mean_ref, beliefs_ref, _ = grid_bp(grid, weights, K, ell, diagnostics)
            mean_g, var_g, params_g = gaussian_closure_bp(x, config.rho, q, alpha, delta)

            score_ref = score_from_mean(x, mean_ref, alpha, delta)
            score_g = score_from_mean(x, mean_g, alpha, delta)

            mse_m = float(np.mean((mean_g - mean_ref) ** 2))
            mse_s = float(np.mean((score_g - score_ref) ** 2))
            rel_s = float(np.linalg.norm(score_g - score_ref) / np.linalg.norm(score_ref))
            cos_s = cosine_similarity(score_g, score_ref)

            # This should be numerically zero: score error equals alpha/delta times mean error.
            direct = score_g - score_ref
            predicted = (alpha / delta) * (mean_g - mean_ref)
            identity_error = float(np.linalg.norm(direct - predicted) / (np.linalg.norm(direct) + 1e-14))

            # Gaussian closure should coincide *exactly* with the linear denoiser built
            # from the AR(1) covariance, even though the chain is Laplace.
            mean_lmmse = gaussian_posterior_mean_exact(x, config.rho, alpha, delta)
            lmmse_gap = float(np.linalg.norm(mean_g - mean_lmmse) / (np.linalg.norm(mean_lmmse) + 1e-14))

            mse_m_list.append(mse_m)
            mse_s_list.append(mse_s)
            rel_s_list.append(rel_s)
            cos_s_list.append(cos_s)
            identity_error_list.append(identity_error)
            lmmse_gap_list.append(lmmse_gap)

            if example is None and abs(t - 0.40) < 1e-12 and trial == 0:
                site = config.n // 2
                belief_g = normalize_density(
                    normal_pdf(grid, mean_g[site], var_g[site]), weights
                )
                example = {
                    "t": float(t),
                    "site": site,
                    "grid": grid.copy(),
                    "belief_ref": beliefs_ref[site].copy(),
                    "belief_g": belief_g,
                    "x": x.copy(),
                    "a": a.copy(),
                    "mean_ref": mean_ref.copy(),
                    "mean_g": mean_g.copy(),
                    "score_ref": score_ref.copy(),
                    "score_g": score_g.copy(),
                    "params_g": params_g,
                }

        rows.append({
            "prior": "laplace",
            "grid_size": float(config.ref_grid_size),
            "t": float(t),
            "alpha": float(np.exp(-t)),
            "delta": float(1.0 - np.exp(-2.0 * t)),
            "posterior_mean_mse_mean": float(np.mean(mse_m_list)),
            "posterior_mean_mse_std": float(np.std(mse_m_list, ddof=1)) if len(mse_m_list) > 1 else 0.0,
            "score_mse_mean": float(np.mean(mse_s_list)),
            "score_mse_std": float(np.std(mse_s_list, ddof=1)) if len(mse_s_list) > 1 else 0.0,
            "relative_score_error_mean": float(np.mean(rel_s_list)),
            "relative_score_error_std": float(np.std(rel_s_list, ddof=1)) if len(rel_s_list) > 1 else 0.0,
            "score_cosine_mean": float(np.mean(cos_s_list)),
            "score_cosine_std": float(np.std(cos_s_list, ddof=1)) if len(cos_s_list) > 1 else 0.0,
            "score_error_identity_rel_residual_mean": float(np.mean(identity_error_list)),
            "closure_vs_lmmse_rel_gap_mean": float(np.mean(lmmse_gap_list)),
        })

    return rows, (example if example is not None else {})


# -----------------------------------------------------------------------------
# CSV and plotting utilities
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_to_arrays(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def plot_gaussian_validation(rows: list[dict[str, float]], output_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.6))
    sizes = sorted({int(r["grid_size"]) for r in rows})
    for size in sizes:
        sub = [r for r in rows if int(r["grid_size"]) == size]
        t = rows_to_arrays(sub, "t")
        y = rows_to_arrays(sub, "rel_score_error_grid_mean")
        e = rows_to_arrays(sub, "rel_score_error_grid_std")
        plt.errorbar(t, y, yerr=e, marker="o", capsize=3, label=f"grid M={size}")
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("relative score error vs exact matrix score")
    plt.title("Gaussian AR(1): grid BP validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gaussian_grid_validation_rel_error.png", dpi=220)
    plt.close()

    # Gaussian closure is grid free, so it has a single curve. The discarded grid
    # projection is overlaid to show what the discretisation was costing.
    floor = 1e-18
    plt.figure(figsize=(7.2, 4.6))
    sub0 = [r for r in rows if int(r["grid_size"]) == sizes[0]]
    t = rows_to_arrays(sub0, "t")
    y = np.maximum(rows_to_arrays(sub0, "rel_score_error_gauss_closure_mean"), floor)
    plt.plot(t, y, marker="o", color="C2", linewidth=2,
             label="Gaussian closure, information form (grid free)")
    for size in sizes:
        sub = [r for r in rows if int(r["grid_size"]) == size]
        ta = rows_to_arrays(sub, "t")
        ya = np.maximum(rows_to_arrays(sub, "rel_score_error_grid_projection_ablation_mean"), floor)
        plt.plot(ta, ya, marker="x", linestyle="--", alpha=0.75,
                 label=f"ablation: grid projection M={size}")
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("relative score error vs exact matrix score")
    plt.title("Gaussian AR(1): Gaussian closure is exact; grid projection is not")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "gaussian_closure_sanity_rel_error.png", dpi=220)
    plt.close()


def plot_laplace_convergence(rows: list[dict[str, float]], output_dir: Path) -> None:
    t = rows_to_arrays(rows, "t")
    rel = rows_to_arrays(rows, "relative_score_error_mean")
    rel_std = rows_to_arrays(rows, "relative_score_error_std")

    coarse = int(rows[0]["coarse_grid_size"])
    fine = int(rows[0]["fine_grid_size"])
    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(t, rel, yerr=rel_std, marker="o", capsize=3)
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("relative score difference")
    plt.title(f"Laplace chain: grid convergence M={coarse} vs M={fine}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_grid_convergence_rel_error.png", dpi=220)
    plt.close()


def plot_laplace_gaussian_message(rows: list[dict[str, float]], output_dir: Path) -> None:
    t = rows_to_arrays(rows, "t")
    mse_m = rows_to_arrays(rows, "posterior_mean_mse_mean")
    mse_m_std = rows_to_arrays(rows, "posterior_mean_mse_std")
    rel_s = rows_to_arrays(rows, "relative_score_error_mean")
    rel_s_std = rows_to_arrays(rows, "relative_score_error_std")
    cos_s = rows_to_arrays(rows, "score_cosine_mean")
    cos_s_std = rows_to_arrays(rows, "score_cosine_std")

    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(t, mse_m, yerr=mse_m_std, marker="o", capsize=3)
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("posterior mean MSE")
    plt.title("Laplace chain: Gaussian-message posterior-mean error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_posterior_mean_mse.png", dpi=220)
    plt.close()

    # The corrected closure spans several decades in t, so this needs a log axis.
    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(t, rel_s, yerr=rel_s_std, marker="o", capsize=3)
    plt.yscale("log")
    plt.xlabel("diffusion time t")
    plt.ylabel("relative score error")
    plt.title("Laplace chain: Gaussian-message (= linear denoiser) score error")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_relative_score_error.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(t, cos_s, yerr=cos_s_std, marker="o", capsize=3)
    plt.xlabel("diffusion time t")
    plt.ylabel("score cosine similarity")
    plt.title("Laplace chain: score direction, Gaussian messages vs grid BP")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "laplace_score_cosine_similarity.png", dpi=220)
    plt.close()


def plot_examples(config: ExperimentConfig, example: dict[str, object], output_dir: Path) -> None:
    if not example:
        return
    grid = example["grid"]
    site = int(example["site"])

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(grid, example["belief_ref"], label="reference grid BP (exact)")
    plt.plot(grid, example["belief_g"], linestyle="--", label="Gaussian closure (= LMMSE)")
    plt.axvline(float(example["mean_ref"][site]), linestyle=":", label="reference mean")
    plt.axvline(float(example["mean_g"][site]), linestyle=":", label="Gaussian-message mean")
    plt.xlabel(f"a_{site}")
    plt.ylabel("posterior belief density")
    plt.title(f"Laplace chain posterior belief at site {site}, t={example['t']}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "example_laplace_belief_comparison.png", dpi=220)
    plt.close()


def plot_gaussian_example(config: ExperimentConfig, rng: np.random.Generator, output_dir: Path) -> None:
    # Use largest validation grid for a visual sanity check.
    grid_size = max(config.validation_grid_sizes)
    grid, weights = make_grid(config, grid_size)
    K = build_transition_matrix(grid, config.rho, prior="gaussian")
    t = 0.40
    a = sample_clean_chain(rng, config.n, config.rho, prior="gaussian")
    x, alpha, delta = sample_noisy_observation(rng, a, t)
    ell = likelihood_matrix(grid, x, alpha, delta)
    mean_grid, _, _ = grid_bp(grid, weights, K, ell)
    score_grid = score_from_mean(x, mean_grid, alpha, delta)
    score_exact = gaussian_precision_score(x, config.rho, alpha, delta)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(score_exact, marker="o", markersize=3, label="exact matrix score")
    plt.plot(score_grid, marker="x", markersize=3, linestyle="--", label=f"grid BP score M={grid_size}")
    plt.xlabel("site i")
    plt.ylabel("score component")
    plt.title("Gaussian AR(1): exact score vs grid BP score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "example_gaussian_grid_vs_matrix_score.png", dpi=220)
    plt.close()


def write_summary(config: ExperimentConfig, output_dir: Path, diagnostics: GridDiagnostics | None = None) -> None:
    diag_line = f"\nGrid diagnostics: {diagnostics.report()}\n" if diagnostics is not None else ""
    summary = f"""# BP Markov diffusion experiment outputs

Configuration used for this run:

- n = {config.n}
- rho = {config.rho}
- grid interval = [{config.grid_min}, {config.grid_max}]
- validation grid sizes = {config.validation_grid_sizes}
- Laplace convergence coarse grid size = {config.convergence_coarse_grid_size}
- reference grid size = {config.ref_grid_size}
- n_trials = {config.n_trials}
- seed = {config.seed}
- diffusion times = {config.t_values}
- boundary mass tolerance = {config.boundary_mass_tol} over {config.boundary_edge_points} edge cells
{diag_line}
The Gaussian validation compares grid BP with the exact precision-matrix score, and
checks that Gaussian closure reproduces that score to machine precision. The retained
grid-projection ablation shows what the earlier discretised projection was costing.

The Laplace convergence compares grid BP at two resolutions.

The Laplace Gaussian-message experiment compares Gaussian closure with reference grid BP.
Because closure coincides exactly with the linear (LMMSE) denoiser, the reported error is
the excess error of the best linear estimator over the exact one. The column
`closure_vs_lmmse_rel_gap_mean` verifies that identity numerically.
"""
    (output_dir / "README_outputs.md").write_text(summary, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_all(config: ExperimentConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    diagnostics = GridDiagnostics(config.boundary_mass_tol, config.boundary_edge_points)

    print("Running Gaussian grid validation...")
    gaussian_rows = run_gaussian_grid_validation(config, rng, diagnostics)
    write_csv(output_dir / "gaussian_grid_validation.csv", gaussian_rows)
    plot_gaussian_validation(gaussian_rows, output_dir)

    print("Running Laplace grid convergence...")
    convergence_rows = run_laplace_grid_convergence(config, rng, diagnostics)
    write_csv(output_dir / "laplace_grid_convergence.csv", convergence_rows)
    plot_laplace_convergence(convergence_rows, output_dir)

    print("Running Laplace Gaussian-message approximation error...")
    laplace_rows, example = run_laplace_gaussian_message_error(config, rng, diagnostics)
    write_csv(output_dir / "laplace_gaussian_message_error.csv", laplace_rows)
    plot_laplace_gaussian_message(laplace_rows, output_dir)
    plot_examples(config, example, output_dir)

    print("Writing Gaussian visual example...")
    plot_gaussian_example(config, rng, output_dir)
    write_summary(config, output_dir, diagnostics)

    print(f"Grid diagnostics: {diagnostics.report()}")

    max_closure_err = max(r["rel_score_error_gauss_closure_mean"] for r in gaussian_rows)
    max_lmmse_gap = max(r["closure_vs_lmmse_rel_gap_mean"] for r in laplace_rows)
    print(f"Gaussian closure vs exact matrix score, worst over t: {max_closure_err:.3e} (expect ~1e-15)")
    print(f"Closure vs LMMSE on the Laplace chain, worst over t: {max_lmmse_gap:.3e} (expect ~1e-15)")

    print("Finished all experiments.")
    print(f"Outputs written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BP diffusion score experiments for Markov chains.")
    parser.add_argument("--mode", choices=["quick", "report", "custom"], default="report")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--ref-grid-size", type=int, default=None)
    parser.add_argument("--coarse-grid-size", type=int, default=None)
    parser.add_argument("--grid-min", type=float, default=None)
    parser.add_argument("--grid-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.mode == "quick":
        cfg = ExperimentConfig(
            n=25,
            validation_grid_sizes=(121, 241),
            convergence_coarse_grid_size=151,
            ref_grid_size=301,
            n_trials=2,
            output_dir="bp_markov_diffusion_outputs_quick",
            t_values=(0.12, 0.27, 0.60, 1.30),
        )
    elif args.mode == "report":
        cfg = ExperimentConfig(
            n=50,
            validation_grid_sizes=(151, 301, 601),
            convergence_coarse_grid_size=301,
            ref_grid_size=601,
            n_trials=50,
            output_dir="bp_markov_diffusion_outputs_report",
        )
    else:
        cfg = ExperimentConfig()

    updates = {
        "n": args.n,
        "rho": args.rho,
        "n_trials": args.n_trials,
        "ref_grid_size": args.ref_grid_size,
        "convergence_coarse_grid_size": args.coarse_grid_size,
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    for name, value in updates.items():
        if value is not None:
            setattr(cfg, name, value)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = config_from_args(args)
    run_all(config)
