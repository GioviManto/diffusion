"""Exact EM for chain-prior parameters, with belief propagation as the E-step.

Problem
-------
The clean data is a Markov chain whose transition kernel is *unknown*:

    p_theta(a) = mu(a_1) prod_{i=2}^n K_theta(a_i | a_{i-1}),

observed only through the OU channel  x = alpha_t a + sqrt(Delta_t) z. We are
given a sample {(x^(d), t_d)} and want theta. This is a latent-variable problem
(the clean chain a is never seen), so the natural estimator is maximum marginal
likelihood, computed by EM.

Why the E-step is exact here
---------------------------
The posterior p_theta(a | x) is again a chain: the likelihood factorizes
sitewise, so it only reweights the node potentials and leaves the edge
structure untouched. A chain is a tree, hence sum-product BP returns the
*exact* single-site and pairwise posterior marginals. There is no loopy-BP
approximation anywhere; the only error is the grid representation of the
continuous messages, which the Layer-2 experiments pin at the 1e-15 floor for
M = 401, A = 8.

The one matrix that is the whole E-step
---------------------------------------
Write the grid as u_1..u_M with quadrature weights w, and let

    f_i = w * L_i * ell_i        ("forward message including its local factor")
    g_i = w * R_i * ell_i        ("backward message including its local factor")

for one chain. The pairwise posterior on edge (i, i+1) is

    b_{i,i+1}(u_j, u_k)  =  f_i(j) K[k, j] g_{i+1}(k) / Z_i,
    Z_i = g_{i+1}^T K f_i.

Every quantity EM needs is an expectation under these pairwise beliefs summed
over edges and over the dataset, so the entire E-step compresses into a single
M x M matrix

    C = sum_{d, i} g^{(d)}_{i+1} (f^{(d)}_i)^T / Z^{(d)}_i,      Xi = C * K,

with `*` elementwise. Xi[k, j] is the expected posterior mass placed on the
transition u_j -> u_k, summed over the whole dataset: exactly the continuous
analogue of the expected transition-count matrix of Baum-Welch. It is a
*sufficient statistic of the E-step*: it does not depend on how K_theta is
parameterized, and sum(Xi) equals the number of edges in the dataset.

Consequences used throughout this module:

1. The expected complete-data log-likelihood is a finite bilinear form,

       Q(theta | theta') = <Xi(theta'), log K_theta>  +  site-1 term  +  const,

   so the M-step never touches the data again -- only the M x M matrix Xi.

2. Fisher's identity gives the gradient of the *exact* marginal log-likelihood
   with no differentiation through the forward-backward recursion:

       grad_theta L(theta) = <Xi(theta), grad_theta log K_theta>.

   This is the key computational point: BP is not something we backpropagate
   through, it is the thing that supplies the expectation. No autodiff needed.

3. Cost per EM iteration is O(N n M^2), independent of the number of
   parameters, and the M-step is O(P M^2) with P = dim(theta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpectedStatistics:
    """Sufficient statistics of one E-step over a whole dataset.

    xi           : (M, M) expected pairwise transition mass, xi[k, j] is the mass
                   on u_j -> u_k summed over all edges of all chains. Sums to
                   the total number of edges.
    site1        : (M,) expected posterior mass at the first site, summed over
                   chains (sums to the number of chains). Only needed if the
                   initial density mu is also learned.
    log_evidence : sum_d log p_{t_d, theta}(x^(d)), the exact marginal
                   log-likelihood at the theta used in the E-step. Monotone
                   increase of this number across EM iterations is the
                   correctness check for the whole algorithm.
    n_edges      : total number of edges (chains x (n - 1)).
    n_chains     : total number of chains.
    """

    xi: np.ndarray
    site1: np.ndarray
    log_evidence: float
    n_edges: int
    n_chains: int

    def __add__(self, other: "ExpectedStatistics") -> "ExpectedStatistics":
        return ExpectedStatistics(
            xi=self.xi + other.xi,
            site1=self.site1 + other.site1,
            log_evidence=self.log_evidence + other.log_evidence,
            n_edges=self.n_edges + other.n_edges,
            n_chains=self.n_chains + other.n_chains,
        )


def e_step(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    chunk: int = 256,
) -> ExpectedStatistics:
    """Exact BP E-step over a batch of noisy chains observed at one noise level.

    Parameters
    ----------
    log_K : (M, M) log K[out, in] = log K_theta(u_out | u_in) at the current theta.
    X     : (B, n) noisy observations, all at the same (alpha, delta).
    log_mu: (M,) log initial density; defaults to the standard normal used by
            `priors` when it draws a_1.
    chunk : batch rows processed at once (memory knob; results are exact and
            chunk-independent).

    Returns the accumulated `ExpectedStatistics`. The forward-backward pass is
    the same recursion as `bp_grid.grid_bp_batch`; the extra work is the rank-one
    accumulation of the pairwise statistics and the evidence bookkeeping.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = len(grid)
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    K = np.exp(log_K)

    total = None
    for start in range(0, X.shape[0], chunk):
        part = _e_step_chunk(
            grid, weights, K, X[start : start + chunk], alpha, delta, log_mu, m
        )
        total = part if total is None else total + part
    if total is None:
        raise ValueError("Empty observation batch.")
    return total


def _e_step_chunk(
    grid: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray,
    m: int,
) -> ExpectedStatistics:
    b_size, n = X.shape
    if n < 2:
        raise ValueError("EM on a chain needs at least two sites.")

    # Row-shifted likelihood tables, one per (chain, site); the removed shifts
    # are restored in the evidence at the end.
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    row_shift = log_ell.max(axis=2)  # (B, n)
    log_ell = log_ell - row_shift[:, :, None]
    ell = np.exp(log_ell)  # (B, n, M)

    L = np.empty((n, b_size, m))
    R = np.empty((n, b_size, m))
    L[0] = np.exp(log_mu)[None, :]
    log_z_fwd = np.zeros(b_size)

    for i in range(n - 1):
        incoming = L[i] * ell[:, i, :] * weights[None, :]  # (B, M)
        out = incoming @ K.T  # (B, M);  out[b, k] = sum_j K[k, j] incoming[b, j]
        mass = out @ weights
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise FloatingPointError(f"Forward message {i + 1} lost all mass.")
        L[i + 1] = out / mass[:, None]
        log_z_fwd += np.log(mass)

    R[-1] = 1.0
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[:, i, :] * weights[None, :]
        out = incoming @ K
        mass = out @ weights
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise FloatingPointError(f"Backward message {i - 1} lost all mass.")
        R[i - 1] = out / mass[:, None]

    # Pairwise accumulation. F[i] and G[i] carry the local factors so that the
    # unnormalized pairwise belief on edge (i, i+1) is F[i](j) K[k, j] G[i+1](k).
    F = L * ell.transpose(1, 0, 2) * weights[None, None, :]  # (n, B, M)
    G = R * ell.transpose(1, 0, 2) * weights[None, None, :]  # (n, B, M)

    f_all = F[:-1].reshape(-1, m)  # (B(n-1), M)
    g_all = G[1:].reshape(-1, m)  # (B(n-1), M)
    # Z_e = g_e^T K f_e for every edge e, as two BLAS calls rather than a
    # three-operand einsum (which would contract at O(E M^2) outside BLAS).
    partition = np.einsum("ek,ek->e", g_all, f_all @ K.T)
    if not np.all(np.isfinite(partition)) or np.any(partition <= 0.0):
        raise FloatingPointError("Pairwise belief lost all mass.")
    c_mat = (g_all / partition[:, None]).T @ f_all  # (M, M), C[k, j]
    xi = c_mat * K

    # Site-1 posterior mass (for a learnable initial density).
    raw1 = L[0] * ell[:, 0, :] * R[0]
    mass1 = raw1 @ weights
    site1 = ((raw1 / mass1[:, None]) * weights[None, :]).sum(axis=0)

    # Exact evidence: forward normalizers x tail integral x restored constants.
    tail = (L[-1] * ell[:, -1, :]) @ weights
    log_const = row_shift.sum(axis=1) + n * (-0.5 * np.log(2.0 * np.pi * delta))
    log_evidence = float(np.sum(log_z_fwd + np.log(tail) + log_const))

    return ExpectedStatistics(
        xi=xi,
        site1=site1,
        log_evidence=log_evidence,
        n_edges=b_size * (n - 1),
        n_chains=b_size,
    )


def e_step_multi(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    groups,
    log_mu: np.ndarray | None = None,
    chunk: int = 256,
) -> ExpectedStatistics:
    """E-step over several noise levels at once.

    `groups` is an iterable of (X, alpha, delta). Because the noise level enters
    only through the local likelihood factors, the statistics of different levels
    are simply added: one Xi matrix summarizes a dataset spread over the whole
    diffusion schedule.
    """
    total = None
    for X, alpha, delta in groups:
        part = e_step(grid, weights, log_K, X, alpha, delta, log_mu, chunk)
        total = part if total is None else total + part
    if total is None:
        raise ValueError("No observation groups supplied.")
    return total


def clean_statistics(
    grid: np.ndarray, A: np.ndarray
) -> ExpectedStatistics:
    """Statistics for *clean* chains, i.e. the t -> 0 limit of the E-step.

    With no noise the posterior collapses onto the observation, the pairwise
    belief becomes a product of deltas, and Xi degenerates to a histogram of the
    observed transitions. Binning those deltas onto the same grid (linear
    interpolation, which preserves the first moment exactly) puts the clean-data
    MLE and the noisy-data EM on a common footing: identical M-step code, only
    the way Xi is built differs. This is the "no noising required" baseline.

    No quadrature weights appear here, and that is not an oversight: the
    clean-data Xi is a *counting* measure over observed transitions, whereas the
    BP Xi is belief mass that already absorbed the weights. Both sum to the edge
    count, which is exactly the property the M-steps rely on, so the two are
    interchangeable inputs despite being built from different objects.

    A : (B, n) clean chains.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m = len(grid)
    src = _interp_weights(grid, A[:, :-1].ravel())
    dst = _interp_weights(grid, A[:, 1:].ravel())
    xi = _bilinear_accumulate(m, src, dst)
    site1 = np.zeros(m)
    s0 = _interp_weights(grid, A[:, 0])
    np.add.at(site1, s0[0], 1.0 - s0[2])
    np.add.at(site1, s0[1], s0[2])
    return ExpectedStatistics(
        xi=xi,
        site1=site1,
        log_evidence=float("nan"),  # complete-data case: use `complete_log_lik`
        n_edges=A.shape[0] * (A.shape[1] - 1),
        n_chains=A.shape[0],
    )


def _interp_weights(grid: np.ndarray, values: np.ndarray):
    """Linear-interpolation indices/weights placing `values` on `grid`."""
    lo = grid[0]
    dx = float(grid[1] - grid[0])
    pos = np.clip((values - lo) / dx, 0.0, len(grid) - 1 - 1e-12)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, len(grid) - 1)
    return i0, i1, frac


def _bilinear_accumulate(m: int, src, dst) -> np.ndarray:
    """Scatter unit mass at each (source, destination) pair bilinearly."""
    j0, j1, fj = src
    k0, k1, fk = dst
    xi = np.zeros((m, m))
    for kk, wk in ((k0, 1.0 - fk), (k1, fk)):
        for jj, wj in ((j0, 1.0 - fj), (j1, fj)):
            np.add.at(xi, (kk, jj), wk * wj)
    return xi


@dataclass
class EMTrace:
    """Per-iteration record of an EM run.

    log_evidence is the exact marginal log-likelihood *at the parameters used
    for that E-step*, so entry k is L(theta^(k)). A non-monotone sequence means
    a bug in the M-step (or a learning rate too large in the generalized case),
    which is why it is recorded rather than discarded.
    """

    log_evidence: list[float]
    theta: list[np.ndarray]
    seconds: list[float]

    @property
    def monotone_violation(self) -> float:
        """Largest decrease of the log-likelihood across iterations (0 if none)."""
        d = np.diff(np.asarray(self.log_evidence, dtype=float))
        return float(max(0.0, -d.min())) if d.size else 0.0


def fit_em(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    groups,
    n_iters: int = 50,
    log_mu: np.ndarray | None = None,
    tol: float = 1e-9,
    chunk: int = 256,
):
    """Run EM on noisy observations. Returns (fitted_kernel, EMTrace).

    `groups` is a list of (X, alpha, delta): one entry per noise level, each
    with its own batch of noisy chains. Every iteration is one exact BP E-step
    over the whole dataset followed by the kernel's own M-step.

    Stops early when the relative increase of the marginal log-likelihood falls
    below `tol`.
    """
    import time

    trace = EMTrace(log_evidence=[], theta=[], seconds=[])
    current = kernel
    prev = -np.inf
    for _ in range(n_iters):
        t0 = time.perf_counter()
        log_k = current.log_transition_matrix(grid)
        stats = e_step_multi(grid, weights, log_k, groups, log_mu, chunk)
        trace.log_evidence.append(stats.log_evidence)
        trace.theta.append(np.asarray(current.theta, dtype=float).copy())
        current = current.m_step(stats, grid)
        trace.seconds.append(time.perf_counter() - t0)
        if np.isfinite(prev) and abs(stats.log_evidence - prev) <= tol * abs(prev):
            break
        prev = stats.log_evidence
    return current, trace


def fit_clean(kernel, grid: np.ndarray, A: np.ndarray, n_iters: int = 1):
    """Maximum likelihood from *clean* chains -- the t = 0 degenerate case.

    There are no latent variables here, so this is plain MLE and one M-step
    already maximizes the likelihood for the closed-form kernels. Kernels with
    their own inner latent variable (a mixture label, a network) still need
    iterations, but none of them re-touches the data: Xi is built once.

    This is the baseline Marc's remark points at -- if the clean chain is
    available, no noising is needed to identify the prior, and the denoiser at
    every noise level follows from BP for free.
    """
    stats = clean_statistics(grid, A)
    current = kernel
    for _ in range(n_iters):
        current = current.m_step(stats, grid)
    return current, stats


def q_value(stats: ExpectedStatistics, log_K: np.ndarray) -> float:
    """Q(theta | theta') = <Xi(theta'), log K_theta>, the edge part of the ELBO.

    Constants that do not depend on theta (the likelihood terms and the
    posterior entropy) are dropped, so only *differences* of `q_value` across
    theta at fixed Xi are meaningful -- which is all the M-step needs.
    """
    return float(np.sum(stats.xi * log_K))


def q_gradient(stats: ExpectedStatistics, grad_log_K: np.ndarray) -> np.ndarray:
    """grad_theta Q = <Xi, grad_theta log K_theta>, given grad_log_K of shape (P, M, M).

    By Fisher's identity this is also the exact gradient of the marginal
    log-likelihood L(theta) whenever Xi was computed at the same theta.
    """
    return np.einsum("pkj,kj->p", grad_log_K, stats.xi)
