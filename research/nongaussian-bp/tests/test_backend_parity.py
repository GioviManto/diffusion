"""CPU/GPU parity for the BP recursion.

This is the gate on every GPU number the project produces. The exactness claims -- grid BP
agreeing with the closed-form Gaussian score to 9.2e-15 and with brute-force enumeration to
1.0e-14 -- were all established on CPU. Running the same recursion on a different device with
different BLAS kernels and a different reduction order does not automatically preserve them.

So: no result computed on a GPU is trusted until these pass on that GPU.

The tests skip cleanly when no device is present, which is the normal state on a login node
or a laptop. A skip is not a pass, and the batch script runs them on the compute node before
the sweep rather than assuming a green laptop run means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.backend import device_name, get_xp, gpu_available, to_host
from src.bp_grid import grid_bp_batch, make_grid
from src.exact_scores import exact_gaussian_posterior_mean, sigma_t
from src.priors import GaussianAR1, LaplaceAR1

pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="no usable CUDA device on this machine"
)


def _setup(prior, n_sites, n_chains, m_grid, t, seed=0):
    grid, weights = make_grid(8.0, m_grid)
    log_K = prior.log_transition_matrix(grid)
    rng = np.random.default_rng(seed)
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    return grid, weights, log_K, X, alpha, delta, A


def test_gpu_is_actually_selected():
    """Guard against the silent numpy fallback masquerading as a GPU run."""
    xp = get_xp("gpu")
    assert xp is not np, "get_xp('gpu') returned numpy -- the fallback fired"
    assert device_name(xp).startswith("gpu:")
    print(f"\n    device: {device_name(xp)}")


@pytest.mark.parametrize("family", ["gaussian", "laplace"])
@pytest.mark.parametrize("t", [0.05, 0.4, 1.6])
def test_cpu_and_gpu_posterior_means_agree(family, t):
    """The load-bearing test: identical inputs, both devices, machine precision.

    Tolerance is 1e-12 relative rather than exact equality. Floating-point addition is not
    associative and the two devices reduce in different orders, so bitwise identity is not a
    reasonable demand; 1e-12 is still three orders of magnitude tighter than the 9.2e-15
    discretisation error is loose, and far below any learning error in the project.
    """
    prior = GaussianAR1(0.85) if family == "gaussian" else LaplaceAR1(0.85)
    grid, weights, log_K, X, alpha, delta, _ = _setup(prior, 24, 64, 401, t)

    m_cpu, v_cpu = grid_bp_batch(grid, weights, log_K, X, alpha, delta, xp=np)
    m_gpu, v_gpu = grid_bp_batch(grid, weights, log_K, X, alpha, delta, xp=get_xp("gpu"))
    m_gpu, v_gpu = to_host(m_gpu), to_host(v_gpu)

    rel_m = np.linalg.norm(m_gpu - m_cpu) / np.linalg.norm(m_cpu)
    rel_v = np.linalg.norm(v_gpu - v_cpu) / np.linalg.norm(v_cpu)
    print(f"\n    {family} t={t}: rel mean diff {rel_m:.3e}, rel var diff {rel_v:.3e}")
    assert rel_m < 1e-12, f"posterior means diverge across devices: {rel_m:.3e}"
    assert rel_v < 1e-12, f"posterior variances diverge across devices: {rel_v:.3e}"


@pytest.mark.parametrize("t", [0.05, 0.4, 1.6])
def test_gpu_reproduces_the_closed_form_gaussian_score(t):
    """The GPU path must inherit the exactness claim, not merely match the CPU path.

    Checked against the analytic Gaussian posterior mean, which is the same reference the
    original 9.2e-15 audit used -- so this asserts the property the project actually relies
    on, independently of whether the CPU code drifted too.
    """
    prior = GaussianAR1(0.85)
    grid, weights, log_K, X, alpha, delta, _ = _setup(prior, 24, 64, 401, t)
    sigma0 = prior.rho ** np.abs(np.subtract.outer(np.arange(24), np.arange(24)))

    m_gpu = to_host(grid_bp_batch(grid, weights, log_K, X, alpha, delta, xp=get_xp("gpu"))[0])
    m_exact = exact_gaussian_posterior_mean(X.T, sigma0, alpha, delta).T

    rel = np.linalg.norm(m_gpu - m_exact) / np.linalg.norm(m_exact)
    print(f"\n    t={t}: GPU vs closed form {rel:.3e}")
    assert rel < 1e-10, f"GPU BP does not reproduce the exact Gaussian score: {rel:.3e}"


def test_dtype_is_preserved_as_float64():
    """fp32 would silently destroy the precision every exactness claim depends on."""
    prior = LaplaceAR1(0.85)
    grid, weights, log_K, X, alpha, delta, _ = _setup(prior, 16, 32, 201, 0.4)
    m = grid_bp_batch(grid, weights, log_K, X, alpha, delta, xp=get_xp("gpu"))[0]
    assert to_host(m).dtype == np.float64


def test_large_batch_matches_small_batch():
    """Batching must not change a per-chain answer.

    The GPU path exists to run very wide batches; if the result depended on batch width the
    speedup would be buying wrong numbers.
    """
    prior = LaplaceAR1(0.85)
    grid, weights, log_K, X, alpha, delta, _ = _setup(prior, 16, 256, 401, 0.4)
    xp = get_xp("gpu")

    m_all = to_host(grid_bp_batch(grid, weights, log_K, X, alpha, delta, xp=xp)[0])
    m_split = np.concatenate([
        to_host(grid_bp_batch(grid, weights, log_K, X[s:s + 32], alpha, delta, xp=xp)[0])
        for s in range(0, 256, 32)
    ])
    rel = np.linalg.norm(m_split - m_all) / np.linalg.norm(m_all)
    assert rel < 1e-12, f"result depends on batch width: {rel:.3e}"
