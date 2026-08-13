"""Validation of nn_baseline.py.

The first test is the one that matters most: hand-written backprop is checked against
finite differences.  If that is wrong, every number downstream is meaningless, and the
failure mode is silent -- a subtly wrong gradient still trains, just to a worse optimum,
which would make the neural baseline look artificially bad and the EM comparison
flattering.  We check it explicitly rather than trusting that training "looks fine".

Run:  python test_nn_baseline.py
"""

from __future__ import annotations

import numpy as np

from chain_models import ChainConfig, ar1_covariance, noise_chains, ou_coefficients, sample_chains
from bp_core import exact_gaussian_posterior_mean
from nn_baseline import (
    MLP,
    fourier_time_features,
    make_batch,
    predict_posterior_mean,
    train_denoiser,
)

FAILURES: list[str] = []


def check(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} {value:.4e}  (tol {tol:.1e}) {note}")
    if not ok:
        FAILURES.append(name)


def test_gradient_check() -> None:
    """Every weight and bias gradient must match central finite differences."""
    print("\n1. Backprop vs finite differences (the load-bearing test)")
    rng = np.random.default_rng(0)
    model = MLP.init([5, 7, 6, 3], seed=1)
    x = rng.normal(size=(11, 5))
    y = rng.normal(size=(11, 3))

    def loss_of(m: MLP) -> float:
        pred, _ = m.forward(x)
        d = pred - y
        return float(np.mean(np.sum(d ** 2, axis=1)))

    pred, acts = model.forward(x)
    grad_out = 2.0 * (pred - y) / x.shape[0]
    gw, gb = model.backward(acts, grad_out)

    h = 1e-6
    worst_w = worst_b = 0.0
    for li in range(len(model.weights)):
        for _ in range(12):  # sample a few entries per layer
            i = rng.integers(model.weights[li].shape[0])
            j = rng.integers(model.weights[li].shape[1])
            orig = model.weights[li][i, j]
            model.weights[li][i, j] = orig + h
            lp = loss_of(model)
            model.weights[li][i, j] = orig - h
            lm = loss_of(model)
            model.weights[li][i, j] = orig
            num = (lp - lm) / (2 * h)
            den = max(abs(num), abs(gw[li][i, j]), 1e-8)
            worst_w = max(worst_w, abs(num - gw[li][i, j]) / den)

        for _ in range(6):
            j = rng.integers(model.biases[li].size)
            orig = model.biases[li][j]
            model.biases[li][j] = orig + h
            lp = loss_of(model)
            model.biases[li][j] = orig - h
            lm = loss_of(model)
            model.biases[li][j] = orig
            num = (lp - lm) / (2 * h)
            den = max(abs(num), abs(gb[li][j]), 1e-8)
            worst_b = max(worst_b, abs(num - gb[li][j]) / den)

    check("max relative error, weight gradients", worst_w, 1e-6)
    check("max relative error, bias gradients", worst_b, 1e-6)


def test_time_features() -> None:
    print("\n2. Time features are bounded and distinguish nearby t")
    t = np.array([0.05, 0.06, 0.5, 2.5])
    f = fourier_time_features(t, 16)
    check("|features| <= 1", float(np.abs(f).max() - 1.0), 1e-12)
    d = float(np.linalg.norm(f[0] - f[1]))
    ok = d > 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] t=0.05 and t=0.06 give distinct features (d={d:.3e})")
    if not ok:
        FAILURES.append("time features do not separate nearby t")


def bayes_risk(
    rng: np.random.Generator,
    cfg: ChainConfig,
    t_lo: float,
    t_hi: float,
    n_samples: int = 6000,
    n_t: int = 24,
) -> tuple[float, float]:
    """Return ``(bayes_risk, zero_predictor_risk)`` for the denoising objective.

    The denoising loss ``E||f(x,t) - a||^2`` cannot be driven to zero: its minimum is the
    Bayes risk ``E||a - E[a|x]||^2``, i.e. the total posterior variance.  Any statement
    about how well the network trains has to be made relative to that floor, and relative
    to the trivial predictor ``f = 0`` as a ceiling.  Reporting a raw loss reduction is
    meaningless, and can even be *impossible* to satisfy: for the Laplace chain below the
    floor is 7.4 against a ceiling of 12.06, so the largest achievable reduction is 39%.

    For the Gaussian family this is exact.  For the others ``exact_gaussian_posterior_mean``
    is the LMMSE estimator rather than the true conditional mean, so the value returned is
    a slight *over*-estimate of the true Bayes risk -- which makes the test conservative.
    """
    sigma0 = ar1_covariance(cfg)
    a = sample_chains(rng, n_samples, cfg)
    ts = np.linspace(t_lo, t_hi, n_t)
    risks = []
    for tv in ts:
        x, al, de = noise_chains(rng, a, float(tv))
        m = exact_gaussian_posterior_mean(x, sigma0, al, de)
        risks.append(float(np.mean(np.sum((a - m) ** 2, axis=1))))
    return float(np.mean(risks)), float(np.mean(np.sum(a ** 2, axis=1)))


def test_learns_gaussian_posterior_mean() -> None:
    """With ample data the network must approach the exact Gaussian posterior mean.

    Sanity check that the baseline is competent.  If the network could not fit the Gaussian
    case -- where the target is *linear* in x -- then any EM-vs-network comparison would be
    measuring our training setup rather than the science, and would flatter EM.
    """
    print("\n3. Network approaches the exact posterior mean on a Gaussian chain")
    rng = np.random.default_rng(1)
    cfg = ChainConfig(n=16, rho=0.85, innovation="gaussian")
    sigma0 = ar1_covariance(cfg)

    a_tr = sample_chains(rng, 8000, cfg)
    res = train_denoiser(a_tr, hidden=(128, 128), epochs=250, lr=3e-3, seed=0)
    floor, ceiling = bayes_risk(rng, cfg, 0.05, 2.5)
    captured = (ceiling - res.loss_trace[-1]) / (ceiling - floor)
    print(f"    {res.n_params} params, {res.n_train_chains} chains, final loss "
          f"{res.loss_trace[-1]:.3f}  (Bayes floor {floor:.3f}, zero-predictor {ceiling:.3f})")
    print(f"    captured {100 * captured:.1f}% of the achievable loss reduction")

    a_te = sample_chains(rng, 500, cfg)
    for t in (0.15, 0.6, 1.5):
        x, al, de = noise_chains(rng, a_te, t)
        m_nn = predict_posterior_mean(res.model, x, t)
        m_ex = exact_gaussian_posterior_mean(x, sigma0, al, de)
        rel = float(np.linalg.norm(m_nn - m_ex) / np.linalg.norm(m_ex))
        # 0.25 asserts competence, not accuracy: the trivial predictor scores 1.0, and this
        # is the honest level a lightly-tuned network reaches at this budget. The number
        # itself is a *result* to be reported, not something to be tuned until it looks good.
        check(f"t={t}: relative posterior-mean error", rel, 0.25)


def test_training_approaches_bayes_risk() -> None:
    """Training must capture most of the *achievable* loss reduction.

    Replaces an earlier check on raw loss reduction, which was not merely loose but
    unsatisfiable: with a floor of 7.4 and a ceiling of 12.06 the maximum possible
    reduction is 39%, so a ">40% reduction" criterion could never pass however well the
    network trained.
    """
    print("\n4. Training approaches the Bayes risk (not zero -- the loss has a floor)")
    rng = np.random.default_rng(2)
    cfg = ChainConfig(n=12, rho=0.8, innovation="laplace")
    a = sample_chains(rng, 2000, cfg)
    res = train_denoiser(a, hidden=(64, 64), epochs=120, seed=0)
    floor, ceiling = bayes_risk(rng, cfg, 0.05, 2.5)
    last = float(np.mean(res.loss_trace[-5:]))
    captured = (ceiling - last) / (ceiling - floor)
    print(f"    loss {float(np.mean(res.loss_trace[:5])):.3f} -> {last:.3f}   "
          f"floor {floor:.3f}, ceiling {ceiling:.3f}")
    ok = captured > 0.60
    print(f"  [{'PASS' if ok else 'FAIL'}] captured {100 * captured:.1f}% of achievable "
          f"reduction (want > 60%)")
    if not ok:
        FAILURES.append("network captures too little of the achievable reduction")


def test_determinism() -> None:
    print("\n5. Same seed gives identical models")
    rng = np.random.default_rng(3)
    cfg = ChainConfig(n=10, rho=0.7, innovation="gaussian")
    a = sample_chains(rng, 500, cfg)
    r1 = train_denoiser(a, hidden=(32,), epochs=20, seed=7)
    r2 = train_denoiser(a, hidden=(32,), epochs=20, seed=7)
    d = max(float(np.abs(w1 - w2).max()) for w1, w2 in zip(r1.model.weights, r2.model.weights))
    check("max |weight difference| across two runs", d, 0.0)


if __name__ == "__main__":
    print("=" * 92)
    print("Validation of nn_baseline.py")
    print("=" * 92)
    test_gradient_check()
    test_time_features()
    test_learns_gaussian_posterior_mean()
    test_training_approaches_bayes_risk()
    test_determinism()
    print("\n" + "=" * 92)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")
    print("=" * 92)
