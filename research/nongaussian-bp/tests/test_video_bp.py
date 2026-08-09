"""BP on a temporal chain of spatial trees, against a dense Gaussian solve.

The reference is built from the generative recursion of the *whole* caterpillar
-- temporal backbone and per-frame quadtrees together, by forward substitution --
so nothing about the composition (upward pass, chain, downward pass) is assumed
by the thing it is checked against. If the chain context were fed back at the
wrong point, or the root prior double-counted, this test fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.hierarchy import TreeIndex
from src.noising import alpha_delta
from src.utils import rng_for
from src.video_bp import caterpillar_bp, chain_bp_potentials
from src.wavelet_bp import node_delta

DEPTH = 2
BRANCHING = 4
FRAMES = 5
RHO_SPACE = [0.7, 0.5]
RHO_TIME = 0.85
_LOG_2PI = float(np.log(2.0 * np.pi))


def _gaussian_kernel(grid, rho):
    var = 1.0 - rho**2
    resid = grid[:, None] - rho * grid[None, :]
    return -0.5 * (_LOG_2PI + np.log(var)) - 0.5 * resid**2 / var


def _caterpillar_linear_map(ti: TreeIndex, frames: int, rho_space, rho_time):
    """`a = M eps` for the whole video, frame-major node ordering.

    Built by forward substitution over the generative recursion: the roots form
    a unit-variance AR(1) across frames, and each frame's quadtree hangs off its
    own root with per-level rho. Every node has unit marginal variance.
    """
    n = ti.n_nodes
    total = frames * n
    mat = np.zeros((total, total))
    for f in range(frames):
        base = f * n
        if f == 0:
            mat[base, base] = 1.0
        else:
            mat[base] = rho_time * mat[base - n]
            mat[base, base] += np.sqrt(1.0 - rho_time**2)
        for d in range(ti.depth):
            rho = rho_space[d]
            for node in ti.nodes_at(d):
                for child in ti.children(int(node), d):
                    mat[base + child] = rho * mat[base + int(node)]
                    mat[base + child, base + child] += np.sqrt(1.0 - rho**2)
    return mat


def _dense_posterior(sigma, x, alpha, deltas):
    obs_cov = alpha**2 * sigma + np.diag(deltas)
    mean = alpha * sigma @ np.linalg.solve(obs_cov, x.T)
    sign, logdet = np.linalg.slogdet(obs_cov)
    assert sign > 0
    quad = np.einsum("bi,ib->b", x, np.linalg.solve(obs_cov, x.T))
    log_ev = float(np.sum(-0.5 * (quad + logdet + len(deltas) * _LOG_2PI)))
    return mean.T, log_ev


def _setup(deltas_by_depth, n_videos=4, size=901, half_width=8.0):
    ti = TreeIndex(DEPTH, BRANCHING)
    grid, weights = make_grid(half_width, size)
    log_k_space = [_gaussian_kernel(grid, r) for r in RHO_SPACE]
    k_time = np.exp(_gaussian_kernel(grid, RHO_TIME))
    log_mu = -0.5 * grid**2 - 0.5 * _LOG_2PI

    mat = _caterpillar_linear_map(ti, FRAMES, RHO_SPACE, RHO_TIME)
    sigma = mat @ mat.T
    rng = rng_for("video-bp-test")
    total = FRAMES * ti.n_nodes
    a = rng.standard_normal((n_videos, total)) @ mat.T
    alpha, _ = alpha_delta(0.6)
    nd_frame = node_delta(ti, deltas_by_depth)
    nd = np.tile(nd_frame, FRAMES)
    x = alpha * a + rng.standard_normal(a.shape) * np.sqrt(nd)
    return ti, grid, weights, log_k_space, k_time, log_mu, sigma, x, alpha, nd


def test_caterpillar_matches_dense_gaussian_solve():
    deltas = [0.4, 0.6, 0.9]
    ti, grid, w, log_ks, k_time, log_mu, sigma, x, alpha, nd = _setup(deltas)
    xr = x.reshape(x.shape[0], FRAMES, ti.n_nodes)

    got = caterpillar_bp(
        grid, w, log_ks, k_time, log_mu, xr, alpha, deltas, BRANCHING, DEPTH
    )
    want, _ = _dense_posterior(sigma, x, alpha, nd)
    want = want.reshape(x.shape[0], FRAMES, ti.n_nodes)
    assert np.max(np.abs(got.posterior_mean - want)) < 1e-8


def test_caterpillar_log_evidence_matches_closed_form():
    deltas = [0.4, 0.6, 0.9]
    ti, grid, w, log_ks, k_time, log_mu, sigma, x, alpha, nd = _setup(deltas)
    xr = x.reshape(x.shape[0], FRAMES, ti.n_nodes)

    got = caterpillar_bp(
        grid, w, log_ks, k_time, log_mu, xr, alpha, deltas, BRANCHING, DEPTH
    )
    _, want = _dense_posterior(sigma, x, alpha, nd)
    assert abs(got.log_evidence - want) / abs(want) < 1e-7


def test_temporal_coupling_actually_does_something():
    """A guard against a caterpillar that is secretly F independent trees.

    With rho_time = 0 the frames are independent and the posterior must equal
    the per-frame one; with rho_time = 0.85 it must not. Without this, an
    implementation that dropped the chain context entirely would still pass the
    dense-solve test only if that test were also wrong.
    """
    deltas = [0.4, 0.6, 0.9]
    ti, grid, w, log_ks, k_time, log_mu, _, x, alpha, _ = _setup(deltas)
    xr = x.reshape(x.shape[0], FRAMES, ti.n_nodes)

    coupled = caterpillar_bp(
        grid, w, log_ks, k_time, log_mu, xr, alpha, deltas, BRANCHING, DEPTH
    ).posterior_mean
    k_indep = np.exp(_gaussian_kernel(grid, 1e-9))
    indep = caterpillar_bp(
        grid, w, log_ks, k_indep, log_mu, xr, alpha, deltas, BRANCHING, DEPTH
    ).posterior_mean
    assert np.max(np.abs(coupled - indep)) > 0.05


def test_chain_bp_potentials_reproduces_a_gaussian_chain():
    """The one genuinely new routine, checked on its own against a dense solve."""
    grid, w = make_grid(8.0, 901)
    rho = 0.8
    k = np.exp(_gaussian_kernel(grid, rho))
    mu = np.exp(-0.5 * grid**2 - 0.5 * _LOG_2PI)
    f_len = 6
    rng = rng_for("video-chain")

    sigma = rho ** np.abs(np.subtract.outer(np.arange(f_len), np.arange(f_len)))
    a = rng.standard_normal((3, f_len)) @ np.linalg.cholesky(sigma).T
    alpha, delta = alpha_delta(0.5)
    obs = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    pot = np.exp(-0.5 * (obs[:, :, None] - alpha * grid[None, None, :]) ** 2 / delta)
    res = chain_bp_potentials(pot, k, w, mu)
    belief = pot * res.context
    belief = belief / (belief * w[None, None, :]).sum(axis=2, keepdims=True)
    got = (belief * w[None, None, :]) @ grid

    want, want_ev = _dense_posterior(sigma, obs, alpha, np.full(f_len, delta))
    assert np.max(np.abs(got - want)) < 1e-9

    # The potentials above drop the Gaussian normaliser, so restore it before
    # comparing evidences rather than comparing an unnormalised quantity.
    const = -0.5 * f_len * np.log(2 * np.pi * delta)
    assert abs(float(np.sum(res.log_z + const)) - want_ev) / abs(want_ev) < 1e-9


def test_statistics_carry_the_right_mass():
    """Xi must sum to the edge count on both axes, spatial and temporal."""
    deltas = [0.4, 0.6, 0.9]
    n_videos = 4
    ti, grid, w, log_ks, k_time, log_mu, _, x, alpha, _ = _setup(deltas, n_videos)
    xr = x.reshape(n_videos, FRAMES, ti.n_nodes)

    res = caterpillar_bp(
        grid, w, log_ks, k_time, log_mu, xr, alpha, deltas,
        BRANCHING, DEPTH, want_stats=True,
    )
    assert res.xi_time is not None and res.xi_space is not None

    want_time = n_videos * (FRAMES - 1)
    assert abs(float(res.xi_time.sum()) - want_time) < 1e-6 * want_time
    for d, xi in enumerate(res.xi_space):
        want = n_videos * FRAMES * BRANCHING ** (d + 1)
        assert abs(float(xi.sum()) - want) < 1e-6 * want


def test_temporal_m_step_recovers_rho_time():
    """The statistic has to be *correct*, not merely correctly normalised.

    Feeding the temporal Xi to the existing Gaussian M-step must return the rho
    that generated the data. A Xi that summed to the right mass but paired the
    wrong sites would pass the mass check and fail this one.
    """
    from src.kernels import GaussianAR1Kernel

    deltas = [0.25, 0.25, 0.25]
    n_videos = 200
    ti, grid, w, log_ks, k_time, log_mu, _, x, alpha, _ = _setup(
        deltas, n_videos, size=401, half_width=8.0
    )
    xr = x.reshape(n_videos, FRAMES, ti.n_nodes)

    res = caterpillar_bp(
        grid, w, log_ks, k_time, log_mu, xr, alpha, deltas,
        BRANCHING, DEPTH, want_stats=True,
    )
    from src.em import ExpectedStatistics

    stats = ExpectedStatistics(
        xi=res.xi_time, site1=np.zeros(grid.size), log_evidence=0.0,
        n_edges=n_videos * (FRAMES - 1), n_chains=n_videos,
    )
    fitted = GaussianAR1Kernel(rho=0.2, q=0.5).m_step(stats, grid)
    assert abs(fitted.rho - RHO_TIME) < 0.05, f"got {fitted.rho}, want {RHO_TIME}"
    assert abs(fitted.q - (1.0 - RHO_TIME**2)) < 0.05


def test_rejects_wrong_kernel_shape():
    grid, w = make_grid(8.0, 101)
    pot = np.ones((2, 3, 101))
    with pytest.raises(ValueError, match="k_time"):
        chain_bp_potentials(pot, np.ones((50, 50)), w, np.ones(101))
