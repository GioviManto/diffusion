"""Validation of chain_models.py and bp_core.py.

These checks are the foundation for everything downstream: if BP or the pairwise marginals
are wrong, every EM result is wrong.  Each test states the identity it verifies and the
tolerance it demands.

Run:  python test_bp_core.py
"""

from __future__ import annotations

import numpy as np

from chain_models import (
    INNOVATION_FAMILIES,
    ChainConfig,
    ar1_covariance,
    innovation_pdf,
    noise_chains,
    ou_coefficients,
    sample_chains,
    sample_innovations,
)
from bp_core import (
    Grid,
    build_transition_matrix,
    exact_gaussian_posterior_mean,
    exact_gaussian_score,
    gaussian_closure_bp,
    gaussian_pairwise_moments,
    grid_bp,
    likelihood_matrix,
    normalize_density,
    score_from_mean,
)

FAILURES: list[str] = []


def check(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    status = "PASS" if ok else "FAIL"
    suffix = f"  {note}" if note else ""
    print(f"  [{status}] {name:<58s} {value:.3e}  (tol {tol:.1e}){suffix}")
    if not ok:
        FAILURES.append(name)


# -----------------------------------------------------------------------------

#: Quadrature tolerance for ``innovation_pdf`` on the -40..40 grid with h = 0.004, by family.
#:
#: These differ by orders of magnitude, and the reason is structural rather than a defect:
#: composite trapezoidal quadrature converges at a rate set by the *smoothness* of the
#: integrand. See ``test_quadrature_rates`` for the measured rates.
#:
#:   gaussian, mixture : smooth everywhere            -> machine precision
#:   laplace           : cusp at 0                    -> O(h^2), here ~1e-5
#:   uniform           : jump discontinuities at +-h  -> O(h),   here ~2e-3
#:   student           : heavy tail truncated at 40   -> h-independent floor ~2e-9
PDF_MASS_TOL = {
    "gaussian": 1e-10,
    "mixture": 1e-10,
    "student": 1e-7,
    "laplace": 1e-4,
    "uniform": 5e-3,
}
PDF_VAR_TOL = {
    "gaussian": 1e-9,
    "mixture": 1e-9,
    "student": 1e-3,   # tail truncation bites harder on the second moment
    "laplace": 1e-4,
    "uniform": 1e-2,
}


def test_innovation_moments() -> None:
    """Every family must have mean 0 and variance exactly q."""
    print("\n1. Innovation families: mean 0, variance q, and pdf integrates to 1")
    rng = np.random.default_rng(0)
    grid = Grid.build(-40.0, 40.0, 20001)
    for fam in INNOVATION_FAMILIES:
        cfg = ChainConfig(rho=0.85, innovation=fam)
        q = cfg.innovation_variance
        s = sample_innovations(rng, (400_000,), cfg)
        # Monte Carlo error on the variance is ~ sqrt(2/N) relative for light tails; the
        # Student-t case has a heavier tail so we allow a looser tolerance.
        tol = 0.08 if fam == "student" else 0.02
        check(f"{fam}: |sample mean| ", abs(float(s.mean())), 0.02)
        check(f"{fam}: |sample var/q - 1| ", abs(float(s.var()) / q - 1.0), tol)

        pdf = innovation_pdf(grid.points, cfg)
        mass = float(np.sum(pdf * grid.weights))
        var = float(np.sum(grid.points ** 2 * pdf * grid.weights))
        check(f"{fam}: |pdf mass - 1| ", abs(mass - 1.0), PDF_MASS_TOL[fam])
        check(f"{fam}: |pdf var/q - 1| ", abs(var / q - 1.0), PDF_VAR_TOL[fam])


def test_quadrature_rates() -> None:
    """Measure the quadrature convergence rate of each family.

    This is a property of the method that matters downstream: grid BP inherits it, so a
    bounded-support innovation cannot be integrated to the same accuracy as a smooth one
    at fixed grid resolution.  We assert the *rate*, which is the meaningful invariant,
    rather than an absolute error.
    """
    print("\n1b. Quadrature convergence rate by innovation smoothness")
    sizes = (5001, 10001, 20001, 40001)
    # Expected asymptotic rate, and the tolerance on the measured exponent.
    expected = {"laplace": (2.0, 0.3), "uniform": (1.0, 0.4)}
    for fam in INNOVATION_FAMILIES:
        cfg = ChainConfig(rho=0.85, innovation=fam)
        errs = []
        for N in sizes:
            g = Grid.build(-40.0, 40.0, N)
            errs.append(abs(float(np.sum(innovation_pdf(g.points, cfg) * g.weights)) - 1.0))
        if fam not in expected:
            # Smooth or tail-limited: error must simply stay negligible, no rate to assert.
            print(f"  [INFO] {fam:<10s} errors: " + " ".join(f"{e:.1e}" for e in errs)
                  + "   (at floating-point / tail floor)")
            continue
        rate = float(np.log2(errs[0] / errs[-1]) / (len(sizes) - 1))
        want, tol = expected[fam]
        ok = abs(rate - want) <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {fam:<10s} measured O(h^{rate:.2f}), "
              f"expected O(h^{want:.0f})  " + " ".join(f"{e:.1e}" for e in errs))
        if not ok:
            FAILURES.append(f"{fam} quadrature rate {rate:.2f} != {want}")


def test_chain_covariance() -> None:
    """Sampled chains must reproduce Sigma_0 = stationary_var * rho^{|i-j|} for all families."""
    print("\n2. Chain covariance is rho^{|i-j|} regardless of innovation family")
    rng = np.random.default_rng(1)
    for fam in INNOVATION_FAMILIES:
        cfg = ChainConfig(n=12, rho=0.8, innovation=fam)
        a = sample_chains(rng, 400_000, cfg)
        emp = np.cov(a, rowvar=False)
        target = ar1_covariance(cfg)
        rel = float(np.abs(emp - target).max() / np.abs(target).max())
        check(f"{fam}: max |emp - Sigma_0| / |Sigma_0|", rel, 0.03)


def test_grid_bp_vs_exact_gaussian() -> None:
    """Grid BP must reproduce the exact Gaussian score."""
    print("\n3. Grid BP vs exact precision-matrix score (Gaussian chain)")
    rng = np.random.default_rng(2)
    cfg = ChainConfig(n=30, rho=0.85, innovation="gaussian")
    grid = Grid.build(-12.0, 12.0, 601)
    K = build_transition_matrix(grid, cfg.rho, cfg)
    sigma0 = ar1_covariance(cfg)
    prior = np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance)

    worst = 0.0
    for t in (0.1, 0.4, 0.9, 1.8, 2.4):
        alpha, delta = ou_coefficients(t)
        a = sample_chains(rng, 1, cfg)
        x, _, _ = noise_chains(rng, a, t)
        ell = likelihood_matrix(grid, x[0], alpha, delta)
        res, _ = grid_bp(grid, K, ell, prior)
        s_grid = score_from_mean(x[0], res.means, alpha, delta)
        s_exact = exact_gaussian_score(x[0], sigma0, alpha, delta)
        rel = float(np.linalg.norm(s_grid - s_exact) / np.linalg.norm(s_exact))
        check(f"t={t}: relative score error", rel, 1e-8)
        worst = max(worst, res.worst_boundary_mass)
    check("worst boundary mass over sweep", worst, 1e-6, "(grid wide enough)")


def test_gaussian_closure_is_exact_gaussian() -> None:
    """Closure must equal the exact posterior mean on a Gaussian chain, to machine precision."""
    print("\n4. Gaussian closure vs exact posterior mean (Gaussian chain)")
    rng = np.random.default_rng(3)
    cfg = ChainConfig(n=40, rho=0.85, innovation="gaussian")
    sigma0 = ar1_covariance(cfg)
    for t in (0.1, 0.4, 0.9, 1.8, 2.4, 3.5):
        alpha, delta = ou_coefficients(t)
        a = sample_chains(rng, 8, cfg)
        x, _, _ = noise_chains(rng, a, t)
        res = gaussian_closure_bp(x, cfg.rho, cfg.innovation_variance, alpha, delta)
        exact = exact_gaussian_posterior_mean(x, sigma0, alpha, delta)
        rel = float(np.linalg.norm(res.means - exact) / np.linalg.norm(exact))
        check(f"t={t}: relative posterior-mean error", rel, 1e-12)


def test_closure_equals_lmmse_on_nongaussian() -> None:
    """The closure/LMMSE proposition: closure ignores innovation shape entirely."""
    print("\n5. Closure = LMMSE on NON-Gaussian chains (the proposition)")
    rng = np.random.default_rng(4)
    for fam in INNOVATION_FAMILIES:
        cfg = ChainConfig(n=40, rho=0.85, innovation=fam)
        sigma0 = ar1_covariance(cfg)
        worst = 0.0
        for t in (0.1, 0.9, 2.4):
            alpha, delta = ou_coefficients(t)
            a = sample_chains(rng, 8, cfg)
            x, _, _ = noise_chains(rng, a, t)
            res = gaussian_closure_bp(x, cfg.rho, cfg.innovation_variance, alpha, delta)
            exact = exact_gaussian_posterior_mean(x, sigma0, alpha, delta)
            worst = max(worst, float(np.linalg.norm(res.means - exact) / np.linalg.norm(exact)))
        check(f"{fam}: worst |closure - LMMSE| / |LMMSE|", worst, 1e-12)


def test_pairwise_marginals_consistency() -> None:
    """Pairwise marginals must marginalise back to the single-site beliefs."""
    print("\n6. Grid pairwise marginals are consistent with the beliefs")
    rng = np.random.default_rng(5)
    cfg = ChainConfig(n=14, rho=0.8, innovation="laplace")
    grid = Grid.build(-12.0, 12.0, 401)
    K = build_transition_matrix(grid, cfg.rho, cfg)
    prior = np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance)
    alpha, delta = ou_coefficients(0.5)
    a = sample_chains(rng, 1, cfg)
    x, _, _ = noise_chains(rng, a, 0.5)
    ell = likelihood_matrix(grid, x[0], alpha, delta)
    res, xi = grid_bp(grid, K, ell, prior, want_pairwise=True)

    check("sum of each pairwise marginal - 1", float(np.abs(xi.sum(axis=(1, 2)) - 1.0).max()), 1e-10)

    # Marginalising xi_i over a_{i+1} must give belief_i (as a probability vector).
    worst_l, worst_r = 0.0, 0.0
    for i in range(cfg.n - 1):
        b_i = res.beliefs[i] * grid.weights
        b_ip1 = res.beliefs[i + 1] * grid.weights
        worst_l = max(worst_l, float(np.abs(xi[i].sum(axis=1) - b_i).max()))
        worst_r = max(worst_r, float(np.abs(xi[i].sum(axis=0) - b_ip1).max()))
    check("max |sum_k xi[i,l,k] - belief_i[l]|", worst_l, 1e-9)
    check("max |sum_l xi[i,l,k] - belief_{i+1}[k]|", worst_r, 1e-9)


def test_gaussian_pairwise_vs_grid() -> None:
    """Analytic Gaussian pairwise moments must match the grid ones on a Gaussian chain."""
    print("\n7. Gaussian pairwise moments vs grid pairwise moments (Gaussian chain)")
    rng = np.random.default_rng(6)
    cfg = ChainConfig(n=14, rho=0.8, innovation="gaussian")
    q = cfg.innovation_variance
    grid = Grid.build(-14.0, 14.0, 801)
    K = build_transition_matrix(grid, cfg.rho, cfg)
    prior = np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance)
    alpha, delta = ou_coefficients(0.6)
    a = sample_chains(rng, 1, cfg)
    x, _, _ = noise_chains(rng, a, 0.6)
    ell = likelihood_matrix(grid, x[0], alpha, delta)
    _, xi = grid_bp(grid, K, ell, prior, want_pairwise=True)

    u = grid.points
    exx_grid = np.array([float(xi[i].sum(axis=1) @ (u ** 2)) for i in range(cfg.n - 1)])
    eyy_grid = np.array([float(xi[i].sum(axis=0) @ (u ** 2)) for i in range(cfg.n - 1)])
    exy_grid = np.array([float(u @ xi[i] @ u) for i in range(cfg.n - 1)])

    res = gaussian_closure_bp(x, cfg.rho, q, alpha, delta)
    exx, eyy, exy = gaussian_pairwise_moments(res, x, cfg.rho, q, alpha, delta)

    check("max rel |E[a_i^2] grid vs analytic|",
          float(np.abs(exx[0] - exx_grid).max() / np.abs(exx_grid).max()), 1e-6)
    check("max rel |E[a_{i+1}^2] grid vs analytic|",
          float(np.abs(eyy[0] - eyy_grid).max() / np.abs(eyy_grid).max()), 1e-6)
    check("max rel |E[a_i a_{i+1}] grid vs analytic|",
          float(np.abs(exy[0] - exy_grid).max() / np.abs(exy_grid).max()), 1e-6)


def test_pairwise_recovers_covariance() -> None:
    """With no observations the posterior is the prior, so E[a_i a_{i+1}] = rho * Var."""
    print("\n8. Sanity: at very large t the pairwise moments return the prior covariance")
    rng = np.random.default_rng(7)
    cfg = ChainConfig(n=20, rho=0.75, innovation="gaussian")
    q = cfg.innovation_variance
    t = 12.0  # alpha_t ~ 6e-6: observations carry essentially no information
    alpha, delta = ou_coefficients(t)
    a = sample_chains(rng, 64, cfg)
    x, _, _ = noise_chains(rng, a, t)
    res = gaussian_closure_bp(x, cfg.rho, q, alpha, delta)
    exx, _, exy = gaussian_pairwise_moments(res, x, cfg.rho, q, alpha, delta)
    lag1 = float(np.mean(exy / exx))
    check("|E[a_i a_{i+1}]/E[a_i^2] - rho|", abs(lag1 - cfg.rho), 1e-3)
    check("|E[a_i^2] - stationary var| / var",
          abs(float(np.mean(exx)) - cfg.stationary_variance) / cfg.stationary_variance, 1e-2)


def test_grid_bp_matches_closure_on_gaussian() -> None:
    """Grid BP and the closure agree on a Gaussian chain; they must NOT on a Laplace one."""
    print("\n9. Grid BP vs closure: agree for Gaussian, differ for non-Gaussian")
    rng = np.random.default_rng(8)
    grid = Grid.build(-12.0, 12.0, 601)
    alpha, delta = ou_coefficients(0.3)
    for fam, expect_close in (("gaussian", True), ("laplace", False)):
        cfg = ChainConfig(n=30, rho=0.85, innovation=fam)
        K = build_transition_matrix(grid, cfg.rho, cfg)
        prior = np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance)
        a = sample_chains(rng, 1, cfg)
        x, _, _ = noise_chains(rng, a, 0.3)
        ell = likelihood_matrix(grid, x[0], alpha, delta)
        res_g, _ = grid_bp(grid, K, ell, prior)
        res_c = gaussian_closure_bp(x, cfg.rho, cfg.innovation_variance, alpha, delta)
        rel = float(np.linalg.norm(res_c.means[0] - res_g.means) / np.linalg.norm(res_g.means))
        if expect_close:
            check(f"{fam}: |closure - gridBP| (should vanish)", rel, 1e-6)
        else:
            # Report as a "failure to be small": we want this to be clearly non-zero.
            ok = rel > 1e-3
            print(f"  [{'PASS' if ok else 'FAIL'}] "
                  f"{fam}: |closure - gridBP| (should be LARGE)      {rel:.3e}  (want > 1e-3)")
            if not ok:
                FAILURES.append(f"{fam} closure/grid gap too small")


if __name__ == "__main__":
    print("=" * 88)
    print("Validation of chain_models.py and bp_core.py")
    print("=" * 88)
    test_innovation_moments()
    test_quadrature_rates()
    test_chain_covariance()
    test_grid_bp_vs_exact_gaussian()
    test_gaussian_closure_is_exact_gaussian()
    test_closure_equals_lmmse_on_nongaussian()
    test_pairwise_marginals_consistency()
    test_gaussian_pairwise_vs_grid()
    test_pairwise_recovers_covariance()
    test_grid_bp_matches_closure_on_gaussian()
    print("\n" + "=" * 88)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")
    print("=" * 88)
