"""Correctness tests for the EM layer.

The claims being guarded are the ones Section 2 of `report/em_bp_learning.tex`
rests on: the E-step is exact, Xi is a sufficient statistic that conserves mass,
Fisher's identity gives the marginal-likelihood gradient without differentiating
through BP, the M-steps really maximize Q, and EM ascends the exact evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import (
    clean_statistics,
    e_step,
    e_step_multi,
    fit_clean,
    fit_em,
    q_gradient,
    q_value,
)
from src.exact_scores import gaussian_log_evidence
from src.kernels import (
    GaussianAR1Kernel,
    LaplaceAR1Kernel,
    MDNKernel,
    MixtureInnovationKernel,
)
from src.noising import alpha_delta
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

RHO = 0.8
N_SITES = 10


def _sample(prior, n_chains, t, rng, n_sites=N_SITES):
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])
    alpha, delta = alpha_delta(t)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    return A, X, alpha, delta


def test_evidence_matches_closed_form_gaussian():
    """The E-step's log-evidence is the exact chain marginal likelihood."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-evidence")
    _, X, alpha, delta = _sample(prior, 8, 0.4, rng)

    stats = e_step(grid, weights, prior.log_transition_matrix(grid), X, alpha, delta)
    sigma0 = prior.covariance(N_SITES)
    closed = sum(gaussian_log_evidence(x, sigma0, alpha, delta) for x in X)
    assert abs(stats.log_evidence - closed) / abs(closed) < 1e-12


def test_xi_conserves_mass():
    """Xi sums to the edge count and site1 to the chain count, exactly."""
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mass")
    _, X, alpha, delta = _sample(prior, 5, 0.3, rng)

    stats = e_step(grid, weights, prior.log_transition_matrix(grid), X, alpha, delta)
    assert stats.n_edges == 5 * (N_SITES - 1)
    assert np.isclose(stats.xi.sum(), stats.n_edges, rtol=1e-10)
    assert np.isclose(stats.site1.sum(), stats.n_chains, rtol=1e-10)
    assert np.all(stats.xi >= 0.0)


def test_chunking_does_not_change_statistics():
    """Batching is a memory knob only."""
    grid, weights = make_grid(8.0, 161)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-chunk")
    _, X, alpha, delta = _sample(prior, 7, 0.5, rng)
    log_k = prior.log_transition_matrix(grid)

    big = e_step(grid, weights, log_k, X, alpha, delta, chunk=64)
    small = e_step(grid, weights, log_k, X, alpha, delta, chunk=2)
    assert np.allclose(big.xi, small.xi, rtol=1e-12, atol=1e-14)
    assert np.isclose(big.log_evidence, small.log_evidence, rtol=1e-12)


def test_multi_level_statistics_are_additive():
    """Observations at different noise levels contribute additively to Xi."""
    grid, weights = make_grid(8.0, 161)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-multi")
    _, X1, a1, d1 = _sample(prior, 4, 0.2, rng)
    _, X2, a2, d2 = _sample(prior, 4, 0.9, rng)
    log_k = prior.log_transition_matrix(grid)

    joint = e_step_multi(grid, weights, log_k, [(X1, a1, d1), (X2, a2, d2)])
    s1 = e_step(grid, weights, log_k, X1, a1, d1)
    s2 = e_step(grid, weights, log_k, X2, a2, d2)
    assert np.allclose(joint.xi, s1.xi + s2.xi, rtol=1e-12)
    assert np.isclose(joint.log_evidence, s1.log_evidence + s2.log_evidence, rtol=1e-12)


def test_fisher_identity_matches_finite_differences():
    """grad of the exact evidence == <Xi, grad log K>, for a smooth kernel.

    Uses the Gaussian kernel: the Laplace kernel's rho-derivative carries a sign
    discontinuity whose quadrature error is documented in kernels.py and tested
    separately below.
    """
    grid, weights = make_grid(8.0, 401)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-fisher")
    _, X, alpha, delta = _sample(prior, 6, 0.4, rng)

    kernel = GaussianAR1Kernel(rho=0.72, q=0.5)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha, delta)
    analytic = q_gradient(stats, kernel.grad_log_transition_matrix(grid))

    eps = 1e-5
    builders = (
        lambda v: GaussianAR1Kernel(v, kernel.q),
        lambda v: GaussianAR1Kernel(kernel.rho, v),
    )
    for p, (base, build) in enumerate(zip((kernel.rho, kernel.q), builders)):
        hi = e_step(
            grid, weights, build(base + eps).log_transition_matrix(grid),
            X, alpha, delta,
        ).log_evidence
        lo = e_step(
            grid, weights, build(base - eps).log_transition_matrix(grid),
            X, alpha, delta,
        ).log_evidence
        fd = (hi - lo) / (2 * eps)
        assert abs(analytic[p] - fd) / abs(fd) < 1e-6


def test_laplace_scale_gradient_is_exact_but_rho_gradient_is_not():
    """Documents the quadrature caveat of Remark 'A discretization caveat'.

    d/db is smooth and matches a finite difference to high accuracy; d/drho
    carries sign(e) and does not. The test pins both behaviours so that a future
    change to the quadrature is noticed.
    """
    grid, weights = make_grid(8.0, 401)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-laplace-grad")
    _, X, alpha, delta = _sample(prior, 6, 0.4, rng)

    kernel = LaplaceAR1Kernel(rho=0.75, b=0.4)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha, delta)
    analytic = q_gradient(stats, kernel.grad_log_transition_matrix(grid))

    eps = 1e-5

    def fd(build, base):
        hi = e_step(grid, weights, build(base + eps).log_transition_matrix(grid),
                    X, alpha, delta).log_evidence
        lo = e_step(grid, weights, build(base - eps).log_transition_matrix(grid),
                    X, alpha, delta).log_evidence
        return (hi - lo) / (2 * eps)

    fd_b = fd(lambda v: LaplaceAR1Kernel(kernel.rho, v), kernel.b)
    assert abs(analytic[1] - fd_b) / abs(fd_b) < 1e-6

    fd_rho = fd(lambda v: LaplaceAR1Kernel(v, kernel.b), kernel.rho)
    rel = abs(analytic[0] - fd_rho) / abs(fd_rho)
    assert rel > 1e-3, "sign discontinuity unexpectedly integrated exactly"
    assert rel < 0.5, "rho gradient is far worse than the documented quadrature error"


def test_gaussian_m_step_maximizes_q():
    """The closed-form Gaussian M-step beats nearby parameters on Q."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-mstep")
    _, X, alpha, delta = _sample(prior, 12, 0.3, rng)

    start = GaussianAR1Kernel(0.4, 0.7)
    stats = e_step(grid, weights, start.log_transition_matrix(grid), X, alpha, delta)
    best = start.m_step(stats, grid)
    q_best = q_value(stats, best.log_transition_matrix(grid))

    perturb = rng_for("test-em-mstep-perturb")
    for _ in range(20):
        cand = GaussianAR1Kernel(
            best.rho + float(perturb.normal(0, 0.05)),
            max(best.q + float(perturb.normal(0, 0.05)), 1e-3),
        )
        assert q_value(stats, cand.log_transition_matrix(grid)) <= q_best + 1e-9


def test_laplace_m_step_maximizes_q():
    """Same for the weighted-median / mean-absolute-residual M-step."""
    grid, weights = make_grid(8.0, 301)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mstep-lap")
    _, X, alpha, delta = _sample(prior, 12, 0.3, rng)

    start = LaplaceAR1Kernel(0.4, 0.7)
    stats = e_step(grid, weights, start.log_transition_matrix(grid), X, alpha, delta)
    best = start.m_step(stats, grid)
    q_best = q_value(stats, best.log_transition_matrix(grid))

    perturb = rng_for("test-em-mstep-lap-perturb")
    for _ in range(20):
        cand = LaplaceAR1Kernel(
            best.rho + float(perturb.normal(0, 0.05)),
            max(best.b + float(perturb.normal(0, 0.05)), 1e-3),
        )
        assert q_value(stats, cand.log_transition_matrix(grid)) <= q_best + 1e-9


@pytest.mark.parametrize("kernel_name", ["gaussian", "laplace", "mixture"])
def test_em_is_monotone(kernel_name):
    """The exact marginal log-likelihood never decreases across EM iterations."""
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-monotone", kernel_name)
    groups = []
    for t in (0.2, 0.6):
        _, X, alpha, delta = _sample(prior, 24, t, rng)
        groups.append((X, alpha, delta))

    start = {
        "gaussian": GaussianAR1Kernel(0.2, 0.9),
        "laplace": LaplaceAR1Kernel(0.2, 0.9),
        "mixture": MixtureInnovationKernel.init(
            3, rho=0.2, var=0.9, rng=rng_for("test-em-monotone-init")
        ),
    }[kernel_name]

    _, trace = fit_em(start, grid, weights, groups, n_iters=25)
    assert trace.monotone_violation < 1e-8
    assert trace.log_evidence[-1] > trace.log_evidence[0]


def test_em_recovers_gaussian_parameters():
    """A well-specified fit lands on the truth from a bad initialization."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-recover")
    groups = []
    for t in (0.1, 0.3, 0.6):
        _, X, alpha, delta = _sample(prior, 200, t, rng, n_sites=24)
        groups.append((X, alpha, delta))

    fitted, _ = fit_em(GaussianAR1Kernel(0.1, 1.2), grid, weights, groups, n_iters=100)
    assert abs(fitted.rho - RHO) < 0.02
    assert abs(fitted.q - (1.0 - RHO**2)) < 0.02


def test_clean_statistics_recover_the_least_squares_estimate():
    """The t -> 0 limit reduces to ordinary AR(1) regression."""
    grid, weights = make_grid(8.0, 601)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-clean")
    A = np.stack([prior.sample(rng, 40) for _ in range(400)])

    fitted, _ = fit_clean(GaussianAR1Kernel(0.1, 1.0), grid, A)
    src, dst = A[:, :-1].ravel(), A[:, 1:].ravel()
    rho_ols = float(src @ dst / (src @ src))
    q_ols = float(np.mean((dst - rho_ols * src) ** 2))

    assert abs(fitted.rho - rho_ols) < 5e-3
    assert abs(fitted.q - q_ols) < 5e-3


def test_small_noise_em_approaches_clean_mle():
    """EM at a small noise level agrees with the clean-data MLE.

    The continuity claimed in Section 3 ("the two limits"): as t -> 0 the
    likelihood factors sharpen, Xi degenerates to a transition histogram, and
    the estimator tends to the complete-data one.
    """
    grid, weights = make_grid(8.0, 801)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-smallt")
    A = np.stack([prior.sample(rng, 24) for _ in range(300)])
    clean, _ = fit_clean(GaussianAR1Kernel(0.1, 1.0), grid, A)

    alpha, delta = alpha_delta(0.05)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    noisy, _ = fit_em(
        GaussianAR1Kernel(0.1, 1.0), grid, weights, [(X, alpha, delta)], n_iters=80
    )
    assert abs(noisy.rho - clean.rho) < 0.03
    assert abs(noisy.q - clean.q) < 0.03


def test_mixture_kernel_recovers_a_heavy_tail():
    """A mixture that never heard of Laplace reproduces its excess kurtosis."""
    grid, weights = make_grid(8.0, 301)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mixture-tail")
    A = np.stack([prior.sample(rng, 48) for _ in range(600)])

    fitted, _ = fit_clean(
        MixtureInnovationKernel.init(
            6, rho=0.3, var=0.8, rng=rng_for("test-em-mixture-init")
        ),
        grid, A, n_iters=60,
    )
    mom = fitted.innovation_moments
    assert abs(fitted.rho - RHO) < 0.03
    assert abs(mom["innovation_var"] - (1.0 - RHO**2)) < 0.03
    assert mom["innovation_excess_kurtosis"] > 2.0  # true value 3.0


def test_mdn_gradient_matches_finite_differences():
    """The fused Fisher-identity gradient through the network is correct."""
    grid, weights = make_grid(6.0, 101)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mdn")
    _, X, alpha, delta = _sample(prior, 4, 0.4, rng)

    from src.kernels import _logsumexp_axis0, _quad_weights

    mdn = MDNKernel.init(2, 12, rng_for("test-em-mdn-init"))
    stats = e_step(grid, weights, mdn.log_transition_matrix(grid), X, alpha, delta)

    comp = mdn._component_logs(grid)
    resp = np.exp(comp - _logsumexp_axis0(comp)[None, :, :])
    _, m, log_s2, cache = mdn._heads(grid)
    log_k = _logsumexp_axis0(comp)
    log_w = np.log(_quad_weights(grid))
    p_col = np.exp(
        log_k + log_w[:, None] - _logsumexp_axis0(log_k + log_w[:, None])[None, :]
    )
    xi = stats.xi - stats.xi.sum(axis=0)[None, :] * p_col
    d = grid[None, :, None] - m[:, None, :]
    inv_s2 = np.exp(-log_s2)[:, None, :]
    grad_out = np.concatenate([
        np.einsum("ckj,kj->cj", resp, xi).T,
        np.einsum("ckj,ckj,kj->cj", resp, d * inv_s2, xi).T,
        0.5 * np.einsum("ckj,ckj,kj->cj", resp, d**2 * inv_s2 - 1.0, xi).T,
    ], axis=1)
    grads = mdn.net.backward(cache, grad_out)

    eps = 1e-6
    probe = rng_for("test-em-mdn-probe")
    for layer in (0, 2):
        P = mdn.net.params[layer]
        for _ in range(3):
            i = int(probe.integers(0, P.shape[0]))
            j = int(probe.integers(0, P.shape[1]))
            old = P[i, j]
            P[i, j] = old + eps
            hi = e_step(grid, weights, mdn.log_transition_matrix(grid),
                        X, alpha, delta).log_evidence
            P[i, j] = old - eps
            lo = e_step(grid, weights, mdn.log_transition_matrix(grid),
                        X, alpha, delta).log_evidence
            P[i, j] = old
            fd = (hi - lo) / (2 * eps)
            if abs(fd) > 1e-6:
                assert abs(grads[layer][i, j] - fd) / abs(fd) < 1e-4
