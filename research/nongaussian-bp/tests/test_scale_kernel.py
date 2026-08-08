"""The scale-mixture kernel: does it learn what the linear-AR family cannot?

The decisive test is `test_recovers_magnitude_dependence`, which fits both
families to data whose child *spread* depends on the parent magnitude and checks
that this kernel recovers it while `MixtureInnovationKernel` returns a flat
conditional standard deviation. That is not a criticism of the mixture kernel --
a location-shift family cannot represent a scale effect, by construction -- it is
the measurement that justifies adding a second family at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import clean_statistics
from src.kernels import MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel, linear_ar_magnitude_ratio
from src.utils import rng_for

_LOG_2PI = float(np.log(2.0 * np.pi))


def _q_value(kernel, xi, grid):
    """Q(theta) = sum_{kj} Xi[k,j] log K(u_k | u_j), the ECM objective."""
    return float((xi * kernel.log_transition_matrix(grid)).sum())


def _pairs_to_xi(grid, parent, child):
    return clean_statistics(grid, np.stack([parent, child], axis=1)).xi


def _empirical_ratio(parent, child):
    q = np.quantile(np.abs(parent), [0.25, 0.75])
    lo = child[np.abs(parent) <= q[0]].std()
    hi = child[np.abs(parent) >= q[1]].std()
    return float(hi / lo)


# ----------------------------------------------------------------------------
# The density
# ----------------------------------------------------------------------------

def test_column_mass_deficit_is_truncation_and_responds_to_the_domain():
    """Columns integrate to 1 up to *truncation*, not up to quadrature error.

    The residual is measured the way the project measures it elsewhere
    (advisor document question A, Fig. 2): truncation responds to the domain A
    and not to the grid size N_g, while quadrature responds to the spacing. A
    scale mixture makes this visible because its widest component is wide -- at
    var = 0.6 the broadest component has standard deviation 1.7, so on a +-9
    domain a parent at u = 4 still has ~2e-5 of its child mass off-grid.
    """
    rng = rng_for("scale-kernel-norm")
    k = ScaleMixtureKernel.init(4, rho=0.5, var=0.6, rng=rng)

    def deficit(half_width, size):
        grid, weights = make_grid(half_width, size)
        mass = (np.exp(k.log_transition_matrix(grid)) * weights[:, None]).sum(axis=0)
        return float(np.max(np.abs(mass[np.abs(grid) <= 4.0] - 1.0)))

    coarse, fine = deficit(9.0, 801), deficit(9.0, 1601)
    # Doubling N_g at fixed domain moves the residual by ~2%: it is not
    # quadrature error. Enlarging the domain kills it by nine orders of
    # magnitude: it is truncation.
    assert abs(coarse - fine) < 0.03 * coarse
    assert coarse < 1e-4
    assert deficit(12.0, 1201) < 1e-8
    assert deficit(14.0, 1201) < 1e-11


def test_single_component_is_exactly_gaussian_ar1():
    grid, _ = make_grid(8.0, 401)
    k = ScaleMixtureKernel(
        rho=np.array([0.7]), s2=np.array([0.4]),
        beta=np.zeros(1), gamma=np.zeros(1),
    )
    resid = grid[:, None] - 0.7 * grid[None, :]
    want = -0.5 * (_LOG_2PI + np.log(0.4)) - 0.5 * resid**2 / 0.4
    assert np.max(np.abs(k.log_transition_matrix(grid) - want)) < 1e-12


def test_gate_equals_a_gaussian_scale_mixture_responsibility():
    """The quadratic logit is not an arbitrary parametrisation: it is exactly the
    responsibility of a zero-mean Gaussian scale mixture, which is what the
    docstring claims and what makes the fitted gate interpretable."""
    grid, _ = make_grid(6.0, 301)
    pi = np.array([0.6, 0.4])
    tau2 = np.array([0.5, 3.0])
    beta = np.log(pi) - 0.5 * np.log(tau2)
    gamma = -0.5 / tau2
    k = ScaleMixtureKernel(
        rho=np.zeros(2), s2=np.ones(2),
        beta=beta - beta[0], gamma=gamma - gamma[0],
    )
    log_r = (np.log(pi)[None, :] - 0.5 * np.log(2 * np.pi * tau2)[None, :]
             - 0.5 * (grid**2)[:, None] / tau2[None, :])
    want = np.exp(log_r - log_r.max(1, keepdims=True))
    want = want / want.sum(1, keepdims=True)
    assert np.max(np.abs(k.gate(grid) - want)) < 1e-12


# ----------------------------------------------------------------------------
# The M-step
# ----------------------------------------------------------------------------

def test_m_step_increases_q_every_iteration():
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-monotone")
    p = rng.standard_normal(40000)
    c = 0.5 * p + (0.4 + 0.8 * np.abs(p)) * rng.standard_normal(40000)
    xi = _pairs_to_xi(grid, p, c)

    k = ScaleMixtureKernel.init(3, rho=0.2, var=1.0, rng=rng)
    q_prev = _q_value(k, xi, grid)
    for _ in range(8):
        k = k.m_step(xi_stats(xi), grid)
        q_now = _q_value(k, xi, grid)
        assert q_now >= q_prev - 1e-6 * abs(q_prev)
        q_prev = q_now


def xi_stats(xi):
    from src.em import ExpectedStatistics

    return ExpectedStatistics(
        xi=xi, site1=np.zeros(xi.shape[0]), log_evidence=float("nan"),
        n_edges=int(xi.sum()), n_chains=1,
    )


def test_recovers_magnitude_dependence_where_the_mixture_cannot():
    """The reason this module exists.

    Data with a parent-magnitude-dependent child spread. The scale-mixture kernel
    must recover a conditional standard deviation that grows with |parent|; the
    linear-AR mixture must return a flat one whatever its component count.
    """
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-recover")
    n = 200000
    p = rng.standard_normal(n)
    c = 0.35 * p + (0.35 + 0.85 * np.abs(p)) * rng.standard_normal(n)
    target = _empirical_ratio(p, c)
    assert target > 2.0, "test data must actually have magnitude dependence"

    xi = _pairs_to_xi(grid, p, c)

    scale_k = ScaleMixtureKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        scale_k = scale_k.m_step(xi_stats(xi), grid)

    mix_k = MixtureInnovationKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        mix_k = mix_k.m_step(xi_stats(xi), grid)

    got = scale_k.magnitude_ratio(grid)
    null = linear_ar_magnitude_ratio(0.35)
    assert got > null + 1.0, (
        f"scale mixture recovered {got:.3f}, barely above the linear-AR null "
        f"{null:.3f}; target (empirical) is {target:.3f}"
    )

    # The mixture kernel's conditional variance is independent of the parent by
    # construction; confirm numerically rather than by assertion.
    logk = mix_k.log_transition_matrix(grid)
    k_mat = np.exp(logk)
    k_mat = k_mat / np.maximum(k_mat.sum(axis=0, keepdims=True), 1e-300)
    mean = (k_mat * grid[:, None]).sum(axis=0)
    var = (k_mat * (grid[:, None] - mean[None, :]) ** 2).sum(axis=0)
    near = np.abs(grid) <= 2.0
    flat = float(np.sqrt(var[near]).max() / np.sqrt(var[near]).min())
    assert flat < 1.15, f"mixture kernel unexpectedly varied its scale: {flat}"

    # And the scale kernel must beat it on the objective it is fitted to.
    assert _q_value(scale_k, xi, grid) > _q_value(mix_k, xi, grid)


def test_linear_ar_null_matches_simulation():
    """The null has to be right, because the headline claim is measured against it."""
    rng = rng_for("scale-kernel-null")
    n = 2000000
    for rho in (0.0, 0.148, 0.452, 0.6):
        p = rng.standard_normal(n)
        c = rho * p + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        assert abs(_empirical_ratio(p, c) - linear_ar_magnitude_ratio(rho)) < 0.01


def test_gaussian_control_reproduces_the_linear_null_and_no_more():
    """The negative control the project uses everywhere.

    On genuinely Gaussian AR data the extra capacity must not manufacture a
    magnitude effect. Note the target is *not* 1: at rho = 0.6 a homoscedastic
    AR(1) already produces a quartile ratio of 1.61, because conditioning on a
    set of parent values picks up the spread of the conditional mean. The
    control is that the fitted kernel reproduces that null and nothing beyond it.
    """
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-gauss-control")
    n = 200000
    p = rng.standard_normal(n)
    c = 0.6 * p + np.sqrt(1 - 0.36) * rng.standard_normal(n)
    xi = _pairs_to_xi(grid, p, c)

    k = ScaleMixtureKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        k = k.m_step(xi_stats(xi), grid)

    null = linear_ar_magnitude_ratio(0.6)
    assert abs(k.magnitude_ratio(grid) - null) < 0.08, (
        f"fitted {k.magnitude_ratio(grid):.3f} vs linear-AR null {null:.3f}"
    )
    eff_rho = float((k.gate(grid) * k.rho[None, :]).sum(1).mean())
    assert abs(eff_rho - 0.6) < 0.08


def test_gradient_interface_is_refused_explicitly():
    grid, _ = make_grid(8.0, 201)
    k = ScaleMixtureKernel.init(2, rho=0.4, var=1.0, rng=rng_for("scale-kernel-grad"))
    with pytest.raises(NotImplementedError, match="ECM"):
        k.grad_log_transition_matrix(grid)
