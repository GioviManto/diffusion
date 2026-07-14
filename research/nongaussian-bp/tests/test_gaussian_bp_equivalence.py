"""Equivalence proposition and Markov-approximation identities.

1. Single-Gaussian moment-matched (ADF) BP on a *linear* non-Gaussian chain
   equals analytic Gaussian BP on the covariance-matched Gaussian model.
   The legacy grid projection realizes the same map up to grid numerics, so at a
   moderate t (no boundary collapse) the two must agree closely.
2. The Chow-Liu chain approximation of an AR(1) covariance is the identity.
3. The Woodbury rank-one correction reproduces the exact residual score of the
   AR(1)+global model against its naive chain part.
"""

import numpy as np

from src.bp_gaussian import gaussian_chain_bp, grid_projected_gaussian_bp
from src.bp_grid import make_grid
from src.exact_scores import precision_matrix_score, sigma_t
from src.markov_approx import chow_liu_chain_covariance, rank_one_global_correction
from src.noising import log_likelihood_matrix, ou_noise_sample
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

N, RHO, T = 30, 0.85, 0.4


def test_adf_equals_gaussian_model_bp_on_laplace_chain():
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-equiv", 5)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)

    grid, w = make_grid(8.0, 801)
    m_proj, _ = grid_projected_gaussian_bp(
        grid, w, prior.log_transition_matrix(grid),
        log_likelihood_matrix(grid, x, alpha, delta),
    )
    res = gaussian_chain_bp(x, RHO, prior.q, alpha, delta)
    # Same projection map, one computed on the grid, one analytically:
    # they agree up to grid quadrature/truncation error.
    assert np.max(np.abs(m_proj - res.means)) < 5e-3


def test_chow_liu_is_identity_on_markov_covariance():
    sigma = GaussianAR1(0.7).covariance(20)
    sigma_cl = chow_liu_chain_covariance(sigma)
    assert np.allclose(sigma_cl, sigma, atol=1e-12)


def test_woodbury_rank_one_residual():
    n, beta = 25, 0.5
    rng = rng_for("test-woodbury")
    x = rng.standard_normal(n)
    alpha, delta = np.exp(-0.5), 1 - np.exp(-1.0)
    sigma_chain = GaussianAR1(RHO).covariance(n)
    c = beta**2
    sigma_true = sigma_chain + c * np.ones((n, n))

    s_true = precision_matrix_score(x, sigma_true, alpha, delta)
    s_chain = precision_matrix_score(x, sigma_chain, alpha, delta)
    correction = rank_one_global_correction(
        x, sigma_t(sigma_chain, alpha, delta), alpha**2 * c
    )
    assert np.allclose(s_true, s_chain + correction, atol=1e-10)
