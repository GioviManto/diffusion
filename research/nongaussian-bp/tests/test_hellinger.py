"""The density-level recovery metric.

Guards the properties a distance has to have before a recovery claim can rest on
it: zero exactly at identity, strictly positive off it, monotone in a parameter
that moves the density, and invariant to the normalisation convention the two
kernels happen to be written in.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel, MixtureInnovationKernel
from src.metrics import transition_hellinger
from src.utils import rng_for

RHO = 0.85


def _h(a, b, grid, weights, key="hellinger_median_interior"):
    return transition_hellinger(
        a.log_transition_matrix(grid), b.log_transition_matrix(grid), grid, weights
    )[key]


def test_identical_kernels_sit_at_the_resolution_floor():
    """A kernel against itself gives ~4e-8, not 0, and that is the metric's floor.

    H = sqrt(1 - BC), and BC carries ~1e-15 of arithmetic error, so the square
    root amplifies it to ~1e-8. Pinning the floor rather than asserting an exact
    zero is the honest version: it records what the metric can actually resolve,
    so a reported 1e-8 is read as "identical to arithmetic" and not as a real
    distance. Score-level metrics do not have this floor.
    """
    grid, weights = make_grid(8.0, 401)
    k = GaussianAR1Kernel(RHO, 1 - RHO**2)
    out = transition_hellinger(
        k.log_transition_matrix(grid), k.log_transition_matrix(grid), grid, weights
    )
    for name, v in out.items():
        assert v < 1e-7, f"{name} = {v:.3e} for a kernel against itself"
    assert out["hellinger_max"] > 0.0, (
        "an exact zero would mean the floor moved; if this fires, the metric "
        "changed and the ~4e-8 documented in transition_hellinger is stale")


def test_distance_grows_with_the_gap_in_rho():
    """Monotone in a parameter that moves the density, which is the whole point.

    A metric that merely separates identical from different does not support
    "the fitted transition approaches the truth"; it has to order the near
    misses too.
    """
    grid, weights = make_grid(8.0, 401)
    truth = GaussianAR1Kernel(RHO, 1 - RHO**2)
    ds = [_h(GaussianAR1Kernel(RHO - d, 1 - RHO**2), truth, grid, weights)
          for d in (0.01, 0.03, 0.10, 0.30)]
    assert all(a < b for a, b in zip(ds, ds[1:])), f"not monotone: {ds}"
    assert ds[0] > 0.0


def test_families_differing_beyond_second_moments_are_separated():
    """Variance-matched Gaussian and Laplace: identical covariance, different law.

    This is the case the score-level metrics are weakest on -- at moderate noise
    the induced scores nearly agree -- so it is exactly where a density-level
    metric has to earn its place.
    """
    grid, weights = make_grid(8.0, 401)
    q = 1 - RHO**2
    gauss = GaussianAR1Kernel(RHO, q)
    lap = LaplaceAR1Kernel(RHO, np.sqrt(q / 2.0))   # Laplace variance is 2b^2
    d = _h(lap, gauss, grid, weights)
    assert d > 0.05, f"variance-matched Gaussian and Laplace only {d:.4f} apart"


def test_invariant_to_the_normalisation_convention():
    """An unnormalised kernel must score the same as its normalised twin.

    The metric column-normalises under the grid quadrature before comparing,
    because otherwise it measures which convention each kernel was written in
    rather than how different the densities are -- a real hazard here, where the
    analytic kernels are normalised on the continuum and the MDN kernel is
    normalised on the grid.
    """
    grid, weights = make_grid(8.0, 401)
    a = GaussianAR1Kernel(RHO, 1 - RHO**2)
    b = LaplaceAR1Kernel(RHO, 0.3)
    ref = transition_hellinger(a.log_transition_matrix(grid),
                               b.log_transition_matrix(grid), grid, weights)
    # Same densities, scaled per column by an arbitrary positive factor.
    rng = rng_for("hellinger-scale")
    bump = rng.uniform(0.5, 2.0, size=len(grid))
    shifted = transition_hellinger(
        a.log_transition_matrix(grid) + np.log(bump)[None, :],
        b.log_transition_matrix(grid), grid, weights)
    for k in ref:
        assert shifted[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12)


def test_the_distance_is_bounded_in_the_unit_interval():
    """H in [0, 1] even for kernels with nearly disjoint support.

    Worth pinning because the bound is what makes the number interpretable
    without a scale: 0.3 means the same thing at every grid and every noise
    level, which is not true of the relative score error.
    """
    grid, weights = make_grid(8.0, 401)
    # Far apart: opposite autoregression and very different spread.
    a = GaussianAR1Kernel(0.95, 0.01)
    b = GaussianAR1Kernel(-0.95, 2.0)
    out = transition_hellinger(a.log_transition_matrix(grid),
                               b.log_transition_matrix(grid), grid, weights)
    for name, v in out.items():
        assert 0.0 <= v <= 1.0, f"{name} = {v} outside [0, 1]"
    assert out["hellinger_median_interior"] > 0.5, (
        f"kernels this different should be far apart, got "
        f"{out['hellinger_median_interior']:.3f}")


def test_a_fitted_mixture_approaches_the_truth_it_was_fitted_to():
    """End to end: closer parameters give a smaller density distance."""
    grid, weights = make_grid(8.0, 401)
    truth = LaplaceAR1Kernel(RHO, np.sqrt((1 - RHO**2) / 2.0))
    near = MixtureInnovationKernel.init(
        8, rho=RHO, var=1 - RHO**2, rng=rng_for("hellinger-near"))
    far = MixtureInnovationKernel.init(
        8, rho=0.2, var=3.0, rng=rng_for("hellinger-far"))
    assert _h(near, truth, grid, weights) < _h(far, truth, grid, weights)
