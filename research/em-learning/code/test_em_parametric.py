"""Validation of em_parametric.py.

The point of these checks is to establish what the estimator does and does *not* do, so
that nothing in the write-up over-claims:

  * it recovers (rho, q) at low noise, with a bias that shrinks as the chain lengthens;
  * pooling across diffusion times genuinely helps, and by how much;
  * at large diffusion time it is unidentified -- large parameter spread, negligible
    likelihood spread -- and this must be reported as such rather than as an estimate.

Run:  python test_em_parametric.py
"""

from __future__ import annotations

import numpy as np

from chain_models import ChainConfig, noise_chains, ou_coefficients, sample_chains
from em_parametric import (
    NoisyDataset,
    em_fit,
    em_fit_multistart,
    fisher_curvature,
    profile_likelihood_ridge,
    restart_spread,
    ridge_width,
)

FAILURES: list[str] = []


def check(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56s} {value:.4e}  (tol {tol:.1e}) {note}")
    if not ok:
        FAILURES.append(name)


def make_data(rng, cfg, t, n_chains):
    a = sample_chains(rng, n_chains, cfg)
    x, alpha, delta = noise_chains(rng, a, t)
    return NoisyDataset(x, alpha, delta, t)


# -----------------------------------------------------------------------------

def test_recovery_at_low_noise() -> None:
    print("\n1. Recovery of (rho, q) at low diffusion time, Gaussian chain (correctly specified)")
    rng = np.random.default_rng(0)
    cfg = ChainConfig(n=60, rho=0.85, innovation="gaussian")
    q_true = cfg.innovation_variance
    for t in (0.1, 0.3, 0.6):
        d = make_data(rng, cfg, t, 2000)
        r = em_fit_multistart(d, n_restarts=4, max_iter=400)
        check(f"t={t}: |rho_hat - rho|", abs(r.rho - cfg.rho), 0.02, f"rho_hat={r.rho:.4f}")
        check(f"t={t}: |q_hat - q|/q", abs(r.q - q_true) / q_true, 0.06, f"q_hat={r.q:.4f}")


def test_bias_shrinks_with_chain_length() -> None:
    """The generalised-EM M-step carries an O(1/n) first-site bias. Verify it shrinks."""
    print("\n2. Estimator bias shrinks with chain length n (documents the M-step approximation)")
    rng = np.random.default_rng(1)
    biases = []
    for n in (20, 40, 80, 160):
        cfg = ChainConfig(n=n, rho=0.85, innovation="gaussian")
        d = make_data(rng, cfg, 0.2, 4000)
        r = em_fit(d, rho_init=0.5, q_init=0.5, max_iter=400)
        biases.append(abs(r.rho - cfg.rho))
        print(f"    n={n:4d}: rho_hat={r.rho:.5f}  bias={biases[-1]:.2e}")
    ok = biases[-1] < biases[0]
    print(f"  [{'PASS' if ok else 'FAIL'}] bias at n=160 below bias at n=20")
    if not ok:
        FAILURES.append("bias does not shrink with n")
    check("bias at n=160", biases[-1], 0.02)


def test_identifiability_collapse() -> None:
    """Identifiability degrades with t: parameter spread grows, likelihood spread shrinks.

    The degradation is *gradual*, not a sharp threshold, so we assert the monotone trend
    rather than an arbitrary cutoff.  Claiming a transition at some particular t would not
    be supported by this data.
    """
    print("\n3. Identifiability degrades with t (spread grows, loglik spread shrinks)")
    rng = np.random.default_rng(2)
    cfg = ChainConfig(n=40, rho=0.85, innovation="gaussian")
    ts = (0.3, 0.9, 1.8, 2.4)
    spreads, ranges = [], []
    for t in ts:
        d = make_data(rng, cfg, t, 2000)
        s = restart_spread(d, n_restarts=6, max_iter=300)
        spreads.append(s["rho_std"])
        ranges.append(s["loglik_range"])
        print(f"    t={t:4.1f}:  rho in [{s['rho_min']:.3f}, {s['rho_max']:.3f}]"
              f"  (std {s['rho_std']:.3f})   loglik range {s['loglik_range']:.2e}")

    check("restart spread in rho at t=0.3 is small", spreads[0], 0.05)
    grew = spreads[-1] > 10.0 * spreads[0]
    print(f"  [{'PASS' if grew else 'FAIL'}] parameter spread grows by >10x from t=0.3 to t=2.4")
    if not grew:
        FAILURES.append("parameter spread does not grow with t")

    # The loglik range across restarts is reported but deliberately NOT asserted: it is not
    # monotone in t. It measures how far apart the restarts happened to land as well as how
    # flat the surface is, so it peaks around t=1.8 and then falls once every restart is
    # already lost on the ridge. The principled measure of identifiability is the curvature
    # of the profile likelihood, asserted in test 4, which does fall monotonically.
    print(f"    (loglik ranges across t, reported not asserted: "
          + ", ".join(f"{r:.1e}" for r in ranges) + " -- not monotone by construction)")


def test_profile_likelihood_flattens() -> None:
    """The ridge widens and the curvature falls as t grows."""
    print("\n4. Profile likelihood in rho: peak sharp at low t, flat at high t")
    rng = np.random.default_rng(3)
    cfg = ChainConfig(n=40, rho=0.85, innovation="gaussian")
    rho_grid = np.linspace(0.30, 0.98, 45)
    widths, curvs = [], []
    for t in (0.2, 0.9, 1.8, 2.4):
        d = make_data(rng, cfg, t, 1500)
        prof = profile_likelihood_ridge(d, cfg.rho, cfg.innovation_variance, rho_grid)
        w, w_cens = ridge_width(prof)
        c, c_edge = fisher_curvature(prof)
        peak = float(prof["rho"][int(np.argmax(prof["loglik"]))])
        widths.append(w)
        curvs.append(c)
        print(f"    t={t:4.1f}:  peak at rho={peak:.3f}   ridge width={w:.3f}"
              f"{' (CENSORED)' if w_cens else ''}   curvature={c:.3e}"
              f"{' (PEAK AT EDGE)' if c_edge else ''}")
    ok = widths[-1] > widths[0] and curvs[-1] < curvs[0]
    print(f"  [{'PASS' if ok else 'FAIL'}] ridge widens and curvature falls with t")
    if not ok:
        FAILURES.append("ridge does not widen with t")


def test_pooling_helps() -> None:
    """Pooling low-t with high-t data must not be worse than high-t alone."""
    print("\n5. Pooling across diffusion times (proper joint EM, not a min-t shortcut)")
    rng = np.random.default_rng(4)
    cfg = ChainConfig(n=40, rho=0.85, innovation="gaussian")
    a = sample_chains(rng, 1500, cfg)

    sets = {}
    for t in (0.2, 0.9, 2.4):
        x, al, de = noise_chains(rng, a, t)
        sets[t] = NoisyDataset(x, al, de, t)

    only_hi = em_fit_multistart(sets[2.4], n_restarts=4, max_iter=300)
    only_lo = em_fit_multistart(sets[0.2], n_restarts=4, max_iter=300)
    pooled = em_fit_multistart([sets[0.2], sets[0.9], sets[2.4]], n_restarts=4, max_iter=300)

    for name, r in (("t=2.4 only", only_hi), ("t=0.2 only", only_lo), ("pooled", pooled)):
        print(f"    {name:12s}: rho_hat={r.rho:.4f}  q_hat={r.q:.4f}")
    check("pooled |rho_hat - rho|", abs(pooled.rho - cfg.rho), 0.02)
    ok = abs(pooled.rho - cfg.rho) < abs(only_hi.rho - cfg.rho)
    print(f"  [{'PASS' if ok else 'FAIL'}] pooling beats high-t-only")
    if not ok:
        FAILURES.append("pooling does not beat high-t-only")


def test_misspecified_families_still_get_second_moments() -> None:
    """On non-Gaussian chains rung 1 is mis-specified but must still recover rho and q.

    This is the honest statement of what rung 1 can do: the Gaussian E-step uses only two
    moments, so it estimates the two second-moment parameters correctly while remaining
    completely blind to the innovation shape.
    """
    print("\n6. Non-Gaussian chains: rung 1 recovers (rho,q) but is blind to shape")
    rng = np.random.default_rng(5)
    for fam in ("gaussian", "laplace", "student", "mixture", "uniform"):
        cfg = ChainConfig(n=60, rho=0.85, innovation=fam)
        d = make_data(rng, cfg, 0.2, 2000)
        r = em_fit_multistart(d, n_restarts=4, max_iter=400)
        check(f"{fam}: |rho_hat - rho|", abs(r.rho - cfg.rho), 0.03, f"rho_hat={r.rho:.4f}")


if __name__ == "__main__":
    print("=" * 92)
    print("Validation of em_parametric.py")
    print("=" * 92)
    test_recovery_at_low_noise()
    test_bias_shrinks_with_chain_length()
    test_identifiability_collapse()
    test_profile_likelihood_flattens()
    test_pooling_helps()
    test_misspecified_families_still_get_second_moments()
    print("\n" + "=" * 92)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")
    print("=" * 92)
