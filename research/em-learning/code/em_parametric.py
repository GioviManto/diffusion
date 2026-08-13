"""Rung 1 of the EM ladder: learn the AR(1) parameters (rho, q) from *noisy* chains.

The setting is the one Marc proposed.  We are told the data is Markov plus known OU noise,
but we are *not* given the transition kernel.  Starting from a random initialisation we
maximise the marginal likelihood of the observed noisy chains with EM, using BP as the
E-step.  Once (rho, q) are known, BP returns the exact score at *every* diffusion time, at
O(n) cost, with no network involved.

Algorithm
---------
E-step
    Run the Gaussian information-form BP on the current (rho, q) and read off the expected
    sufficient statistics from the pairwise posteriors:
        S_xx = sum_i E[a_i^2],  S_yy = sum_i E[a_{i+1}^2],  S_xy = sum_i E[a_i a_{i+1}].

M-step
    rho <- S_xy / S_xx
    q   <- (S_yy - 2 rho S_xy + rho^2 S_xx) / (#pairs)

Scope and honest limitations
----------------------------
* The E-step here is the *Gaussian* BP.  For a non-Gaussian chain this is the Gaussian
  closure, which by the closure/LMMSE proposition uses only the first two moments of the
  innovation.  Consequently this rung can only ever learn (rho, q) -- it is structurally
  blind to the innovation *shape*.  That is not a defect of the implementation but the
  reason rungs 2 and 3 exist.  It also means rung 1 is *correctly specified* for a Gaussian
  chain and *deliberately mis-specified* for the others, which is worth reporting as such.

* The M-step is the conditional (Baum-Welch) update, which treats the stationary initial
  distribution as fixed within each iteration while the E-step uses the current (rho, q)
  for it.  This is a generalised EM step rather than an exact one.  The resulting estimator
  carries an O(1/n) bias from the first site; ``test_em_parametric.py`` measures it and
  confirms it shrinks with chain length.  We do not claim exact unbiasedness.

* At large diffusion time the likelihood develops a nearly flat ridge along which (rho, q)
  trade off while leaving the observable covariance almost unchanged, and EM lands at an
  essentially arbitrary point on it.

  It is tempting -- and wrong -- to call this non-identifiability.  Since alpha_t > 0 for
  every finite t and both alpha_t and Delta_t are *known*, the map
  Sigma_0 = (Sigma_t - Delta_t I) / alpha_t^2 is invertible, so the law of x determines
  Sigma_0 and hence (rho, q) exactly at every finite t.  We verified this numerically to
  machine precision out to t = 5 (max |Sigma_0 recovered - Sigma_0| = 1.3e-13).

  What actually decays is the *information*.  The Fisher information for rho per chain,
  computed as 0.5 tr(S^-1 dS S^-1 dS), falls off asymptotically like alpha_t^4 = e^{-4t};
  the measured local decay rate rises through 3.16, 3.63, 3.86 as t goes from 1.5 to 3.0,
  approaching 4.  So the sample size required for fixed accuracy grows exponentially in t.

  The practical consequence is the same -- large-t fits at a fixed sample size are poorly
  determined and should be reported with their restart spread and profile curvature rather
  than as point estimates -- but the *reason* is exponential ill-conditioning, not a
  degenerate model.  The distinction matters: a reviewer can refute "unidentified" with two
  lines of algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bp_core import (
    gaussian_closure_bp,
    gaussian_pairwise_moments,
    marginal_loglik_gaussian,
)


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NoisyDataset:
    """Noisy chains observed at one diffusion time.

    ``x`` has shape ``(n_chains, n)``.  ``alpha`` and ``delta`` are the OU coefficients,
    which are *known* -- the diffusion is our own construction.
    """

    x: np.ndarray
    alpha: float
    delta: float
    t: float

    @property
    def n_chains(self) -> int:
        return int(self.x.shape[0])

    @property
    def n(self) -> int:
        return int(self.x.shape[1])


@dataclass
class EMResult:
    rho: float
    q: float
    loglik: float
    n_iter: int
    converged: bool
    rho_trace: list[float] = field(default_factory=list)
    q_trace: list[float] = field(default_factory=list)
    loglik_trace: list[float] = field(default_factory=list)
    n_restarts: int = 1

    @property
    def stationary_variance(self) -> float:
        return self.q / (1.0 - self.rho ** 2)


# -----------------------------------------------------------------------------
# Sufficient statistics and the EM iteration
# -----------------------------------------------------------------------------

def _expected_sufficient_statistics(
    data: NoisyDataset, rho: float, q: float
) -> tuple[float, float, float, int]:
    """E-step: expected pair statistics summed over chains and sites."""
    res = gaussian_closure_bp(data.x, rho, q, data.alpha, data.delta)
    exx, eyy, exy = gaussian_pairwise_moments(res, data.x, rho, q, data.alpha, data.delta)
    return float(exx.sum()), float(eyy.sum()), float(exy.sum()), int(exx.size)


def _total_loglik(datasets: list[NoisyDataset], rho: float, q: float) -> float:
    """Marginal log-likelihood per chain, summed over datasets (weighted by chain count)."""
    total = 0.0
    weight = 0
    for d in datasets:
        idx = np.arange(d.n)
        sigma0 = (q / (1.0 - rho ** 2)) * rho ** np.abs(idx[:, None] - idx[None, :])
        total += marginal_loglik_gaussian(d.x, sigma0, d.alpha, d.delta) * d.n_chains
        weight += d.n_chains
    return total / weight


def moment_initialiser(data: NoisyDataset) -> tuple[float, float]:
    """Consistent method-of-moments estimate of ``(rho, q)``, used to seed EM.

    Why this exists.  At large diffusion time EM has a *spurious local optimum* and random
    starts fall into it.  Measured at ``t = 1.8`` with 100000 chains: starting from
    ``rho = 0.30`` EM converges to ``rho = 0.65`` with a log-likelihood **735.6 nats worse
    in total** than the solution found by starting at the truth, and only 2 of 24 random
    restarts reached the right basin.  The problem is therefore optimisation, not
    information -- the likelihood does prefer the truth, and by a wide margin -- so the fix
    is a good starting point rather than more data or more restarts.

    The estimator inverts the noising directly.  Since ``alpha_t`` and ``Delta_t`` are known
    and ``Sigma_t = alpha^2 Sigma_0 + Delta I``,

        Sigma_0 = (Sigma_t - Delta I) / alpha^2,

    and ``Sigma_t`` is estimated by the empirical covariance of ``x``.  For an AR(1) chain
    ``rho`` is then the ratio of the lag-one to the lag-zero diagonal average, and
    ``q = var (1 - rho^2)``.  This is ``O(n_chains * n)`` and adds nothing meaningful to the
    cost of a fit.

    Returns values clipped to a valid range; degenerate inputs fall back to ``(0.5, 0.5)``.
    """
    x = data.x
    n = data.n
    alpha2 = max(data.alpha ** 2, 1e-300)

    # Average lag-0 and lag-1 second moments of the noisy data.
    lag0_x = float(np.mean(x ** 2))
    lag1_x = float(np.mean(x[:, :-1] * x[:, 1:])) if n > 1 else 0.0

    # Remove the known observation noise. It contributes Delta to lag 0 and nothing to lag 1,
    # because the diffusion noise is independent across sites.
    lag0 = (lag0_x - data.delta) / alpha2
    lag1 = lag1_x / alpha2

    # Reliability of the noise subtraction. lag0_x is about Delta + alpha^2 Var(a), so the
    # signal being extracted is alpha^2 Var(a) while the sampling error on lag0_x is roughly
    # lag0_x * sqrt(2 / n_samples). Once the latter exceeds the former the estimate is noise.
    # Measured at t=3.5 with 4000 chains of length 40: alpha^2 = 9.1e-4 against a standard
    # error of 3.5e-3, and the estimator returned rho = -0.436 for a true rho of 0.85.
    n_samples = x.size
    signal = abs(lag0_x - data.delta)
    noise = lag0_x * np.sqrt(2.0 / max(n_samples, 1))
    if not np.isfinite(lag0) or lag0 <= 0.0 or signal < 2.0 * noise:
        import warnings

        warnings.warn(
            f"moment initialiser is unreliable here: the signal left after removing the "
            f"known observation noise ({signal:.3g}) is not large compared with its own "
            f"sampling error ({noise:.3g}). alpha_t^2 = {alpha2:.3g}, {n_samples} samples. "
            "At this diffusion time the data carries too little information about "
            "(rho, q); treat any fit as poorly determined rather than as an estimate.",
            RuntimeWarning,
            stacklevel=2,
        )
        if not np.isfinite(lag0) or lag0 <= 0.0:
            return 0.5, 0.5

    rho = float(np.clip(lag1 / lag0, -0.99, 0.99))
    q = float(max(lag0 * (1.0 - rho ** 2), 1e-6))
    return rho, q


def em_fit(
    datasets: NoisyDataset | list[NoisyDataset],
    *,
    rho_init: float | None = None,
    q_init: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-9,
    rho_max: float = 0.999,
    q_min: float = 1e-6,
    seed: int = 0,
    track: bool = False,
) -> EMResult:
    """Fit ``(rho, q)`` by EM.

    Passing several :class:`NoisyDataset` objects pools them: the E-step runs separately at
    each diffusion time (each has its own ``alpha, delta``) and the M-step sums the
    sufficient statistics.  This is the correct way to use observations at several noise
    levels jointly.
    """
    if isinstance(datasets, NoisyDataset):
        datasets = [datasets]
    if not datasets:
        raise ValueError("need at least one dataset")

    rng = np.random.default_rng(seed)
    rho = float(rng.uniform(0.05, 0.95)) if rho_init is None else float(rho_init)
    q = float(rng.uniform(0.1, 1.5)) if q_init is None else float(q_init)

    res = EMResult(rho, q, -np.inf, 0, False)
    prev_ll = -np.inf

    for it in range(1, max_iter + 1):
        Sxx = Syy = Sxy = 0.0
        npairs = 0
        for d in datasets:
            a, b, c, k = _expected_sufficient_statistics(d, rho, q)
            Sxx += a
            Syy += b
            Sxy += c
            npairs += k

        rho = float(np.clip(Sxy / Sxx, -rho_max, rho_max))
        q = float(max((Syy - 2.0 * rho * Sxy + rho ** 2 * Sxx) / npairs, q_min))

        ll = _total_loglik(datasets, rho, q)
        if track:
            res.rho_trace.append(rho)
            res.q_trace.append(q)
            res.loglik_trace.append(ll)

        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            res.converged = True
            res.n_iter = it
            break
        prev_ll = ll
    else:
        res.n_iter = max_iter

    res.rho, res.q, res.loglik = rho, q, _total_loglik(datasets, rho, q)
    return res


def em_fit_multistart(
    datasets: NoisyDataset | list[NoisyDataset],
    *,
    n_restarts: int = 8,
    seed: int = 0,
    **kwargs,
) -> EMResult:
    """Run EM from several random initialisations and keep the highest-likelihood fit.

    Restarts matter here for a specific reason.  At large diffusion time the likelihood is
    nearly flat in a whole direction of parameter space, so different initialisations land
    at very different (rho, q) with almost identical likelihood.  Multistart does *not*
    repair that -- nothing can, from that data alone -- but the spread across restarts is a
    usable, cheap diagnostic of the degeneracy, and it is reported by
    :func:`restart_spread`.
    """
    best: EMResult | None = None
    for r in range(n_restarts):
        out = em_fit(datasets, seed=seed * 1000 + r, **kwargs)
        if best is None or out.loglik > best.loglik:
            best = out
    assert best is not None
    best.n_restarts = n_restarts
    return best


def restart_spread(
    datasets: NoisyDataset | list[NoisyDataset],
    *,
    n_restarts: int = 8,
    seed: int = 0,
    **kwargs,
) -> dict[str, float]:
    """Spread of EM solutions across random restarts, plus their likelihood spread.

    A large parameter spread combined with a *tiny* likelihood spread is the signature of
    an unidentified problem: many parameter values explain the data equally well.
    """
    rhos, qs, lls = [], [], []
    for r in range(n_restarts):
        out = em_fit(datasets, seed=seed * 1000 + r, **kwargs)
        rhos.append(out.rho)
        qs.append(out.q)
        lls.append(out.loglik)
    return {
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "rho_min": float(np.min(rhos)),
        "rho_max": float(np.max(rhos)),
        "q_mean": float(np.mean(qs)),
        "q_std": float(np.std(qs)),
        "loglik_mean": float(np.mean(lls)),
        "loglik_std": float(np.std(lls)),
        "loglik_range": float(np.max(lls) - np.min(lls)),
    }


# -----------------------------------------------------------------------------
# Identifiability diagnostics
# -----------------------------------------------------------------------------

def profile_likelihood_ridge(
    data: NoisyDataset,
    rho_true: float,
    q_true: float,
    rho_values: np.ndarray,
    *,
    q_start: float | None = None,
    max_inner: int = 60,
    inner_tol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Profile the log-likelihood along rho, maximising over q at each point.

    At small diffusion time this has a sharp peak near the true ``rho``.  As ``t`` grows it
    flattens into a ridge.  The width of that ridge, and its curvature at the peak, are the
    honest way to express "how well can rho be determined from this much data".

    ``rho_true`` is accepted only so callers can pass the generating value for bookkeeping;
    it is deliberately **not** used in the computation, which must not have oracle access to
    the answer.  ``q_start`` seeds the profile and defaults to ``q_true`` purely as a
    starting guess -- it is refined away by the inner iteration at every point.

    Performance note.  The inner loop profiles out ``q`` by iterating its M-step at fixed
    ``rho``.  Restarting that from scratch at every ``rho`` was the single largest cost in
    the sweep: profiling one cell (n=80, 4000 chains) showed this function taking 89 s of a
    137 s total, dominated by 3660 E-step evaluations.  Because adjacent ``rho`` values have
    almost the same profiled ``q``, we now sweep ``rho`` in order and warm-start each point
    from the previous solution, which cuts the inner iteration count by roughly an order of
    magnitude for the same answer.
    """
    order = np.argsort(rho_values)
    ll = np.empty(rho_values.size)
    q_hat = np.empty(rho_values.size)
    q = float(q_true if q_start is None else q_start)

    for j in order:
        r = float(rho_values[j])
        for _ in range(max_inner):
            Sxx, Syy, Sxy, npairs = _expected_sufficient_statistics(data, r, q)
            q_new = max((Syy - 2.0 * r * Sxy + r ** 2 * Sxx) / npairs, 1e-6)
            converged = abs(q_new - q) < inner_tol * max(1.0, q)
            q = q_new
            if converged:
                break
        q_hat[j] = q
        ll[j] = _total_loglik([data], r, q)

    return {"rho": rho_values, "loglik": ll, "q_hat": q_hat}


def em_fit_with_spread(
    datasets: NoisyDataset | list[NoisyDataset],
    *,
    n_restarts: int = 8,
    seed: int = 0,
    **kwargs,
) -> tuple[EMResult, dict[str, float]]:
    """Run the restarts *once* and return both the best fit and the spread across them.

    :func:`em_fit_multistart` and :func:`restart_spread` each ran the same set of restarts
    independently, so calling both -- which the sweep did -- doubled the EM cost for no extra
    information.  This runs them once.  The two original functions are kept for callers that
    genuinely want only one of the two.
    """
    ds = [datasets] if isinstance(datasets, NoisyDataset) else list(datasets)

    # Always include a moment-initialised run alongside the random restarts. At large
    # diffusion time the random starts land in a spurious basin ~92% of the time, so relying
    # on them alone gives a ~40% chance of finding the right optimum with six restarts.
    # The moment start costs O(n_chains * n) and is seeded from the dataset with the most
    # signal, i.e. the largest alpha.
    seed_data = max(ds, key=lambda d: d.alpha)
    r0, q0 = moment_initialiser(seed_data)
    results = [em_fit(datasets, rho_init=r0, q_init=q0, **kwargs)]
    results += [em_fit(datasets, seed=seed * 1000 + r, **kwargs) for r in range(n_restarts)]

    best = max(results, key=lambda r: r.loglik)
    best.n_restarts = n_restarts + 1

    rhos = [r.rho for r in results]
    qs = [r.q for r in results]
    lls = [r.loglik for r in results]
    spread = {
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "rho_min": float(np.min(rhos)),
        "rho_max": float(np.max(rhos)),
        "q_mean": float(np.mean(qs)),
        "q_std": float(np.std(qs)),
        "loglik_mean": float(np.mean(lls)),
        "loglik_std": float(np.std(lls)),
        "loglik_range": float(np.max(lls) - np.min(lls)),
    }
    return best, spread


def ridge_width(profile: dict[str, np.ndarray], drop: float = 0.5) -> tuple[float, bool]:
    """Width in ``rho`` within ``drop`` log-likelihood units of the peak, and a censoring flag.

    ``drop = 0.5`` corresponds to the usual one-unit-of-2*loglik interval for a single
    parameter.

    The flag matters.  The width can never exceed the scanned range, so once the region
    reaches either end of the scan the returned number is a *lower bound*, not a
    measurement.  Because the scan bracket in the sweep is built relative to the true
    ``rho``, the ceiling differs from cell to cell, and censored widths are therefore not
    comparable across ``rho``.  Callers must propagate the flag and exclude censored cells
    from any trend, rather than averaging them in.

    Returns ``(width, censored)``.
    """
    ll = profile["loglik"]
    rho = profile["rho"]
    inside = rho[ll >= ll.max() - drop]
    if inside.size == 0:
        return 0.0, False
    width = float(inside.max() - inside.min())
    censored = bool(inside.min() <= rho[0] or inside.max() >= rho[-1])
    return width, censored


def fisher_curvature(profile: dict[str, np.ndarray]) -> tuple[float, bool]:
    """Curvature of the profile log-likelihood at its maximum, and an edge flag.

    Larger curvature means a sharper peak and a better-determined ``rho``.

    When the peak sits at the edge of the scanned range the curvature is not defined by a
    central difference, and we return ``nan`` rather than ``0.0``.  Returning zero would be
    actively misleading: zero curvature reads as "maximally flat, nothing is determined",
    whereas an edge peak means the scan bracket was simply too narrow to contain the
    maximum.  Conflating the two would manufacture a spurious "curvature falls to zero with
    diffusion time" trend out of a bracketing failure.

    Returns ``(curvature, peak_at_edge)``.
    """
    ll, rho = profile["loglik"], profile["rho"]
    k = int(np.argmax(ll))
    if k == 0 or k == ll.size - 1:
        return float("nan"), True
    h = float(rho[k + 1] - rho[k])
    return float(-(ll[k + 1] - 2.0 * ll[k] + ll[k - 1]) / h ** 2), False
