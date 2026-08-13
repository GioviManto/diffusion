"""Belief propagation on the posterior chain: grid BP, Gaussian closure, and the exact
Gaussian reference.

Setting
-------
For a Markov prior observed through coordinatewise OU noise, the posterior

    p(a | x) propto mu(a_1) prod_i K(a_i | a_{i-1}) prod_i ell_i(a_i; x_i)

is itself a chain, so sum-product BP is *exact* -- the observations attach as unary
factors and create no loops.  The score follows from the posterior mean by

    s_i(x,t) = -(x_i - alpha_t E[a_i | x]) / Delta_t.

Messages
--------
The two messages play genuinely different probabilistic roles, and conflating them is
what broke the earlier Gaussian projection:

    L_i(a_i) propto p(a_i | x_1..x_{i-1})     -- a probability *density* in a_i
    R_i(a_i)         = p(x_{i+1}..x_n | a_i)  -- a *likelihood*, not a density in a_i

``R_i`` need not integrate to one over ``a_i``, and at large diffusion time it is
legitimately almost flat, because ``alpha_t -> 0`` means future observations say nothing
about ``a_i``.  The terminal value ``R_n = 1`` encodes "no evidence from the right", not
"a uniform distribution over a_n".  Messages may be rescaled freely for numerical
stability -- the belief is normalised -- but a normalised ``R_i`` must never be treated
as a probability law and have moments taken from it.

Pairwise marginals
------------------
EM needs more than the single-site beliefs.  The M-step requires the pairwise posterior

    xi_i(a_i, a_{i+1}) propto [L_i ell_i](a_i) K(a_{i+1} | a_i) [R_{i+1} ell_{i+1}](a_{i+1}),

from which the expected sufficient statistics E[a_i^2], E[a_i a_{i+1}] and the expected
innovation distribution are read off.  Both the grid and the Gaussian routines return
these.

Complexity
----------
    Gaussian closure   O(n) time, O(n) space   -- two scalars per message
    grid BP            O(n M^2) time           -- an M x M kernel applied per site
    dense matrix solve O(n^3) time, O(n^2) space
    tridiagonal solve  O(n) time, O(n) space

Gaussian BP is not merely *like* the tridiagonal solve; it is the same elimination,
organised as message passing.  ``analytics.py`` verifies this numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chain_models import ChainConfig, innovation_pdf, ou_coefficients


# -----------------------------------------------------------------------------
# Grid utilities
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """An equally spaced quadrature grid with composite trapezoidal weights."""

    points: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(cls, lo: float, hi: float, size: int) -> "Grid":
        pts = np.linspace(lo, hi, size)
        dx = float(pts[1] - pts[0])
        w = np.full(size, dx)
        w[0] *= 0.5
        w[-1] *= 0.5
        return cls(pts, w)

    @property
    def size(self) -> int:
        return self.points.size

    @property
    def spacing(self) -> float:
        return float(self.points[1] - self.points[0])


def normalize_density(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalise non-negative grid values to unit mass under the quadrature weights."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    mass = float(np.sum(values * weights))
    if not np.isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("cannot normalise: non-positive or non-finite mass")
    return values / mass


def boundary_mass(values: np.ndarray, weights: np.ndarray, n_edge: int = 3) -> float:
    """Fraction of a density's mass in the outermost ``n_edge`` cells on each side.

    Grid BP silently redistributes any mass that should have escaped ``[lo, hi]``, because
    messages are renormalised at every step.  Monitoring this quantity is the only way to
    detect that the interval is too narrow.  Apply it to densities (forward messages and
    beliefs) but never to the backward likelihood messages.
    """
    p = normalize_density(values, weights)
    m = p * weights
    return float(np.sum(m[:n_edge]) + np.sum(m[-n_edge:]))


def build_transition_matrix(grid: Grid, rho: float, config: ChainConfig) -> np.ndarray:
    """``K[out, in] = p(a_out | a_in)`` for the additive kernel a' = rho a + eps."""
    diff = grid.points[:, None] - rho * grid.points[None, :]
    return innovation_pdf(diff.ravel(), config).reshape(diff.shape)


def transition_matrix_from_innovation(
    grid: Grid, rho: float, innovation_density: np.ndarray
) -> np.ndarray:
    """Build ``K`` from an arbitrary tabulated innovation density.

    This is what nonparametric EM needs: the innovation density is itself the learned
    object, so the kernel must be reconstructible from a grid of values rather than from
    a closed-form family.  ``innovation_density`` is interpolated on the offset grid.
    """
    diff = grid.points[:, None] - rho * grid.points[None, :]
    return np.interp(diff.ravel(), grid.points, innovation_density, left=0.0, right=0.0).reshape(
        diff.shape
    )


def likelihood_matrix(grid: Grid, x: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """``ell[i, k] = p(x_i | a_i = grid_k)`` -- the diffusion noise enters here, and only here."""
    resid = x[:, None] - alpha * grid.points[None, :]
    return np.exp(-0.5 * resid ** 2 / delta) / np.sqrt(2.0 * np.pi * delta)


# -----------------------------------------------------------------------------
# Grid BP (reference / exact up to quadrature)
# -----------------------------------------------------------------------------

@dataclass
class GridBPResult:
    means: np.ndarray            # (n,) posterior means E[a_i | x]
    variances: np.ndarray        # (n,)
    beliefs: np.ndarray          # (n, M) normalised posterior marginals
    worst_boundary_mass: float


def grid_bp(
    grid: Grid,
    K: np.ndarray,
    ell: np.ndarray,
    prior: np.ndarray,
    *,
    want_pairwise: bool = False,
) -> tuple[GridBPResult, np.ndarray | None]:
    """Exact sum-product BP with messages tabulated on ``grid``.

    Parameters
    ----------
    K:
        Transition matrix, ``K[out, in]``.
    ell:
        Likelihood matrix, shape ``(n, M)``.
    prior:
        Density of ``a_1`` on the grid.
    want_pairwise:
        If true, also return ``xi`` of shape ``(n-1, M, M)`` with
        ``xi[i, l, k] = P(a_i = u_l, a_{i+1} = u_k | x)``, already summing to one.
        This is the object the EM M-step consumes; it costs ``O(n M^2)`` memory, so it is
        opt-in.

    Returns
    -------
    ``(result, xi)`` where ``xi`` is ``None`` unless ``want_pairwise``.
    """
    n, m = ell.shape
    w = grid.weights
    worst = 0.0

    L = np.empty((n, m))
    R = np.empty((n, m))

    L[0] = normalize_density(prior, w)
    for i in range(n - 1):
        L[i + 1] = normalize_density(K @ (L[i] * ell[i] * w), w)
        worst = max(worst, boundary_mass(L[i + 1], w))

    # Terminal likelihood message: no evidence from the right. Represented as a constant;
    # this is NOT a uniform density and no moments are ever taken from it.
    R[-1] = np.ones(m)
    for i in range(n - 1, 0, -1):
        R[i - 1] = normalize_density(K.T @ (R[i] * ell[i] * w), w)

    beliefs = np.empty((n, m))
    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        b = normalize_density(L[i] * ell[i] * R[i], w)
        worst = max(worst, boundary_mass(b, w))
        beliefs[i] = b
        means[i] = float(np.sum(grid.points * b * w))
        variances[i] = float(np.sum((grid.points - means[i]) ** 2 * b * w))

    result = GridBPResult(means, variances, beliefs, worst)

    if not want_pairwise:
        return result, None

    xi = np.empty((n - 1, m, m))
    for i in range(n - 1):
        left = L[i] * ell[i] * w                 # (M,) indexed by a_i
        right = R[i + 1] * ell[i + 1] * w        # (M,) indexed by a_{i+1}
        # joint[l, k] propto left[l] * K[k, l] * right[k]
        joint = left[:, None] * K.T * right[None, :]
        total = float(joint.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError(f"degenerate pairwise marginal at site {i}")
        xi[i] = joint / total

    return result, xi


# -----------------------------------------------------------------------------
# Gaussian message closure (information form, grid free)
# -----------------------------------------------------------------------------

@dataclass
class GaussianBPResult:
    means: np.ndarray        # (..., n)
    variances: np.ndarray    # (..., n)
    lam_left: np.ndarray
    h_left: np.ndarray
    lam_right: np.ndarray
    h_right: np.ndarray


def gaussian_closure_bp(
    x: np.ndarray,
    rho: float,
    q: float,
    alpha: float,
    delta: float,
    prior_var: float | None = None,
) -> GaussianBPResult:
    """Gaussian message closure as a pure ``(lambda, h)`` recursion. No grid.

    Messages are written ``m(a) propto exp(-lambda a^2 / 2 + h a)``, so ``mean = h/lambda``
    and ``var = 1/lambda``.  Multiplying messages adds their parameters; the terminal flat
    message is exactly ``lambda = h = 0``.

    ``x`` may be ``(n,)`` or ``(M, n)``; the recursion is vectorised over the leading axis.

    Only the first two moments of the innovation are used, so for *any* additive
    innovation with mean zero and variance ``q`` this returns the same answer -- namely
    the LMMSE denoiser ``alpha Sigma_0 Sigma_t^{-1} x``.  That is the content of the
    closure/LMMSE proposition, and it is why this routine takes ``q`` rather than an
    innovation family.
    """
    x = np.atleast_2d(x)
    batch, n = x.shape
    if prior_var is None:
        prior_var = q / (1.0 - rho ** 2)

    lam_obs = alpha ** 2 / delta
    h_obs = alpha * x / delta

    lam_L = np.zeros((batch, n))
    h_L = np.zeros((batch, n))
    lam_L[:, 0] = 1.0 / prior_var

    for i in range(n - 1):
        lam_t = lam_L[:, i] + lam_obs
        h_t = h_L[:, i] + h_obs[:, i]
        v_out = q + rho ** 2 / lam_t
        lam_L[:, i + 1] = 1.0 / v_out
        h_L[:, i + 1] = rho * (h_t / lam_t) / v_out

    lam_R = np.zeros((batch, n))
    h_R = np.zeros((batch, n))
    # lam_R[:, -1] = h_R[:, -1] = 0 : no evidence from the right.

    for i in range(n - 1, 0, -1):
        lam_t = lam_R[:, i] + lam_obs
        h_t = h_R[:, i] + h_obs[:, i]
        denom = q + 1.0 / lam_t
        lam_R[:, i - 1] = rho ** 2 / denom
        h_R[:, i - 1] = rho * (h_t / lam_t) / denom

    lam = lam_L + lam_R + lam_obs
    h = h_L + h_R + h_obs
    return GaussianBPResult(h / lam, 1.0 / lam, lam_L, h_L, lam_R, h_R)


def gaussian_pairwise_moments(
    res: GaussianBPResult,
    x: np.ndarray,
    rho: float,
    q: float,
    alpha: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected sufficient statistics from the Gaussian pairwise posteriors.

    Returns ``(E[a_i^2], E[a_{i+1}^2], E[a_i a_{i+1}])``, each of shape ``(batch, n-1)``.

    The pair ``(a_i, a_{i+1})`` has joint precision

        J = [[A + rho^2/q,  -rho/q],
             [-rho/q,        B + 1/q]]

    with ``A`` the left-plus-local information at site ``i`` and ``B`` the right-plus-local
    information at site ``i+1``; the linear term is ``(hA, hB)``.  Inverting this 2x2
    system gives the moments in closed form.
    """
    x = np.atleast_2d(x)
    lam_obs = alpha ** 2 / delta
    h_obs = alpha * x / delta

    A = res.lam_left[:, :-1] + lam_obs
    hA = res.h_left[:, :-1] + h_obs[:, :-1]
    B = res.lam_right[:, 1:] + lam_obs
    hB = res.h_right[:, 1:] + h_obs[:, 1:]

    J00 = A + rho ** 2 / q
    J11 = B + 1.0 / q
    J01 = -rho / q

    det = J00 * J11 - J01 ** 2
    if np.any(det <= 0.0):
        raise FloatingPointError("non-positive-definite pairwise precision")

    C00, C11, C01 = J11 / det, J00 / det, -J01 / det
    m0 = (J11 * hA - J01 * hB) / det
    m1 = (-J01 * hA + J00 * hB) / det

    return C00 + m0 * m0, C11 + m1 * m1, C01 + m0 * m1


# -----------------------------------------------------------------------------
# Exact Gaussian reference
# -----------------------------------------------------------------------------

def exact_gaussian_posterior_mean(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """``alpha Sigma_0 Sigma_t^{-1} x`` -- exact for a Gaussian chain, LMMSE otherwise."""
    x = np.atleast_2d(x)
    sigma_t = alpha ** 2 * sigma0 + delta * np.eye(sigma0.shape[0])
    return alpha * (sigma0 @ np.linalg.solve(sigma_t, x.T)).T


def exact_gaussian_score(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """``-Sigma_t^{-1} x``."""
    x = np.atleast_2d(x)
    sigma_t = alpha ** 2 * sigma0 + delta * np.eye(sigma0.shape[0])
    return -np.linalg.solve(sigma_t, x.T).T


def score_from_mean(
    x: np.ndarray, mean: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Convert a posterior mean into the diffusion score."""
    return -(np.atleast_2d(x) - alpha * mean) / delta


def marginal_loglik_gaussian(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> float:
    """Mean marginal log-likelihood of noisy chains, via the dense covariance.

    ``O(n^3)`` time and ``O(n^2)`` memory because it forms and factorises ``Sigma_t``.
    Kept as an independent reference for :func:`marginal_loglik_ar1`, which is the routine
    that should actually be called: it is ``O(n)`` and exact for the AR(1) structure.
    """
    x = np.atleast_2d(x)
    n = sigma0.shape[0]
    sigma_t = alpha ** 2 * sigma0 + delta * np.eye(n)
    _, logdet = np.linalg.slogdet(sigma_t)
    quad = np.einsum("ij,ji->i", x, np.linalg.solve(sigma_t, x.T))
    return float(np.mean(-0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)))


def marginal_loglik_ar1(
    x: np.ndarray,
    rho: float,
    q: float,
    alpha: float,
    delta: float,
    prior_var: float | None = None,
) -> float:
    """Mean marginal log-likelihood of noisy AR(1) chains in ``O(n)`` per chain.

    This is the prediction-error decomposition, which is what the forward BP pass already
    computes and then throws away.  Writing the joint density as a product of one-step
    predictive densities,

        log p(x) = sum_i log p(x_i | x_1..x_{i-1}),

    each factor is Gaussian because the predictive law of ``a_i`` given the past is the
    forward message ``N(mu_i, v_i)`` and the observation adds independent noise:

        x_i | x_{1..i-1}  ~  N(alpha mu_i,  alpha^2 v_i + Delta).

    So the whole likelihood falls out of the same ``(lambda, h)`` recursion used for the
    posterior mean, at no extra asymptotic cost.

    Why this matters here rather than being a micro-optimisation: the dense form
    :func:`marginal_loglik_gaussian` is ``O(n^3)``, and it sits inside the EM convergence
    test *and* inside every point of the profile-likelihood scan.  At ``n = 80`` with a
    61-point scan that dominated the entire sweep and caused the first identifiability run
    to exceed its walltime.  The two agree to machine precision; see
    ``test_bp_core.test_loglik_agreement``.
    """
    x = np.atleast_2d(x)
    batch, n = x.shape
    if prior_var is None:
        prior_var = q / (1.0 - rho ** 2)

    lam = np.full(batch, 1.0 / prior_var)
    h = np.zeros(batch)
    total = np.zeros(batch)

    for i in range(n):
        v = 1.0 / lam                       # predictive variance of a_i given the past
        mu = h / lam                        # predictive mean
        s = alpha ** 2 * v + delta          # predictive variance of x_i
        resid = x[:, i] - alpha * mu
        total += -0.5 * (np.log(2.0 * np.pi * s) + resid ** 2 / s)

        if i < n - 1:
            lam_t = lam + alpha ** 2 / delta          # condition on x_i
            h_t = h + alpha * x[:, i] / delta
            v_out = q + rho ** 2 / lam_t              # propagate to a_{i+1}
            lam = 1.0 / v_out
            h = rho * (h_t / lam_t) / v_out

    return float(np.mean(total))
