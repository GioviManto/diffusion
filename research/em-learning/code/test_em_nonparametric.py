"""Validation of em_nonparametric.py.

The claim under test is specific: nonparametric EM recovers the *shape* of the innovation
law from noisy chains, which rung 1 provably cannot. The decisive comparison is therefore
not "EM is accurate" but "EM distinguishes two families that share a covariance".

Run:  python test_em_nonparametric.py
"""

from __future__ import annotations

import numpy as np

from chain_models import ChainConfig, innovation_pdf, noise_chains, sample_chains
from bp_core import Grid, normalize_density
from em_nonparametric import (
    density_l1_distance,
    em_nonparametric_fit,
    stationary_density,
)
from bp_core import build_transition_matrix

FAILURES: list[str] = []


def check(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<54s} {value:.4e}  (tol {tol:.1e}) {note}")
    if not ok:
        FAILURES.append(name)


def test_stationary_density() -> None:
    """Power iteration must reproduce the analytic stationary law for a Gaussian chain."""
    print("\n1. Stationary density by power iteration (Gaussian chain, known answer)")
    cfg = ChainConfig(n=10, rho=0.8, innovation="gaussian")
    grid = Grid.build(-12.0, 12.0, 401)
    K = build_transition_matrix(grid, cfg.rho, cfg)
    pi = stationary_density(K, grid.weights)
    target = normalize_density(
        np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance), grid.weights
    )
    check("L1(pi_power_iteration, pi_analytic)", density_l1_distance(pi, target, grid), 2e-3)


def test_recovers_gaussian_shape() -> None:
    """Sanity: on a Gaussian chain the learned innovation must look Gaussian."""
    print("\n2. Gaussian chain: learned innovation matches the true Gaussian")
    rng = np.random.default_rng(0)
    cfg = ChainConfig(n=24, rho=0.80, innovation="gaussian")
    grid = Grid.build(-6.0, 6.0, 241)
    a = sample_chains(rng, 300, cfg)
    x, al, de = noise_chains(rng, a, 0.15)
    fit = em_nonparametric_fit(x, al, de, grid, rho_init=0.5, max_iter=60)
    truth = normalize_density(innovation_pdf(grid.points, cfg), grid.weights)
    check("|rho_hat - rho|", abs(fit.rho - cfg.rho), 0.08, f"rho_hat={fit.rho:.3f}")
    check("L1(p_learned, p_true)", density_l1_distance(fit.innovation_density, truth, grid),
          0.30, f"(converged={fit.converged}, {fit.n_iter} iters)")


def test_distinguishes_families_sharing_covariance() -> None:
    """The decisive test.

    Laplace and Gaussian innovations with the same variance produce chains with *identical*
    covariance rho^{|i-j|}. Any second-moment method -- including the entire rung-1 EM and
    the Gaussian closure -- is blind to the difference. Nonparametric EM must not be: the
    learned density has to sit closer to its own family than to the other one.
    """
    print("\n3. Distinguishing Laplace from Gaussian at matched covariance (the key claim)")
    rng = np.random.default_rng(1)
    grid = Grid.build(-6.0, 6.0, 241)
    learned = {}
    for fam in ("gaussian", "laplace"):
        cfg = ChainConfig(n=24, rho=0.80, innovation=fam)
        a = sample_chains(rng, 400, cfg)
        x, al, de = noise_chains(rng, a, 0.15)
        fit = em_nonparametric_fit(x, al, de, grid, rho_init=0.5, max_iter=60)
        learned[fam] = fit.innovation_density
        print(f"    {fam:9s}: rho_hat={fit.rho:.3f}  var_hat={fit.innovation_variance:.4f}"
              f"  (true q={cfg.innovation_variance:.4f})")

    truths = {
        f: normalize_density(
            innovation_pdf(grid.points, ChainConfig(n=24, rho=0.80, innovation=f)),
            grid.weights,
        )
        for f in ("gaussian", "laplace")
    }
    for fam in ("gaussian", "laplace"):
        other = "laplace" if fam == "gaussian" else "gaussian"
        d_own = density_l1_distance(learned[fam], truths[fam], grid)
        d_other = density_l1_distance(learned[fam], truths[other], grid)
        print(f"    learned[{fam}]  L1 to own={d_own:.4f}  L1 to {other}={d_other:.4f}")
        ok = d_own < d_other
        print(f"  [{'PASS' if ok else 'FAIL'}] learned {fam} is closer to {fam} than to {other}")
        if not ok:
            FAILURES.append(f"{fam} not distinguished")


def test_peak_height_separates() -> None:
    """A Laplace innovation has a sharp peak at 0; a Gaussian of equal variance does not.

    This is a single interpretable statistic rather than a whole-density distance, and it is
    what we would quote in the paper.
    """
    print("\n4. Peak height at 0 separates the families (interpretable statistic)")
    rng = np.random.default_rng(2)
    grid = Grid.build(-6.0, 6.0, 241)
    mid = grid.size // 2
    peaks = {}
    for fam in ("gaussian", "laplace"):
        cfg = ChainConfig(n=24, rho=0.80, innovation=fam)
        a = sample_chains(rng, 400, cfg)
        x, al, de = noise_chains(rng, a, 0.15)
        fit = em_nonparametric_fit(x, al, de, grid, rho_init=0.5, max_iter=60)
        peaks[fam] = float(fit.innovation_density[mid])
        true_peak = float(
            normalize_density(innovation_pdf(grid.points, cfg), grid.weights)[mid]
        )
        print(f"    {fam:9s}: learned peak={peaks[fam]:.4f}   true peak={true_peak:.4f}")
    ok = peaks["laplace"] > peaks["gaussian"]
    print(f"  [{'PASS' if ok else 'FAIL'}] learned Laplace peak exceeds learned Gaussian peak")
    if not ok:
        FAILURES.append("peak height does not separate families")


def test_monotone_objective() -> None:
    """Generalised EM must not decrease the expected complete-data log-likelihood."""
    print("\n5. The EM objective Q is non-decreasing across iterations")
    rng = np.random.default_rng(3)
    cfg = ChainConfig(n=20, rho=0.75, innovation="laplace")
    grid = Grid.build(-6.0, 6.0, 201)
    a = sample_chains(rng, 200, cfg)
    x, al, de = noise_chains(rng, a, 0.2)
    fit = em_nonparametric_fit(x, al, de, grid, rho_init=0.4, max_iter=20)
    q = np.array(fit.q_trace)
    # Allow a small tolerance: smoothing and the zero-mean reprojection are not part of the
    # M-step maximisation, so tiny decreases are possible and must be reported honestly.
    worst_drop = float(np.min(np.diff(q))) if q.size > 1 else 0.0
    rel = abs(min(worst_drop, 0.0)) / max(1.0, abs(float(q[-1])))
    print(f"    Q: {q[0]:.4f} -> {q[-1]:.4f} over {q.size} iters, worst step {worst_drop:+.3e}")
    check("relative size of worst Q decrease", rel, 1e-3)


if __name__ == "__main__":
    print("=" * 92)
    print("Validation of em_nonparametric.py")
    print("=" * 92)
    test_stationary_density()
    test_recovers_gaussian_shape()
    test_distinguishes_families_sharing_covariance()
    test_peak_height_separates()
    test_monotone_objective()
    print("\n" + "=" * 92)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")
    print("=" * 92)
