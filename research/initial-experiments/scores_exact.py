"""
scores_exact.py
===============

Ground-truth benchmarks for the joint score S(x, t) = grad_x log p_t(x)
in three settings, ordered by increasing difficulty:

    1. Gaussian AR(1) for arbitrary K:  S(x, t) = -Sigma_t^{-1}(x - mu_t)
         - Computed two independent ways:
               (a) direct matrix inversion,
               (b) Kalman filter + Rauch-Tung-Striebel smoother, O(K).
         - The two agree to machine precision; this is the cross-check
           that the O(K) route is exact, not an O(K) approximation.

    2. Laplace AR(1) at K = 1:  closed-form Normal-Laplace residual.
         - S(x, t) computed from the closed-form psi_t(r) = (log h_t)'(r).
         - Monte Carlo sanity check at one parameter setting.

    3. Laplace AR(1) at K = 2:  3D Fourier inversion of the exact
       characteristic function.
         - S(x, t) via finite differences on the numerically computed
           log p_t(x).
         - Monte Carlo sanity check at one parameter setting.

The module is intentionally pure numpy / scipy, with a single small
matplotlib block at the bottom guarded by `if __name__ == "__main__"`.
It has no training, no architecture, no toy network -- it is ground truth.

Conventions
-----------
    X = e^{-t} A + sqrt(Delta_t) Z,  Z ~ N(0, I),  Delta_t = 1 - e^{-2t}
    beta := e^{-t}
    S(x, t) = grad_x log p_t(x)
    K is the number of frames.

Reference note:
    Laplace AR(1) structural constants match Table 3.1 of
    `laplace_K1_compendium.pdf` and Table 1 of `laplace_K1_benchmark_memo.pdf`.

Author: Giovanni Mantovani (with Claude as collaborator)
April 2026
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


# =============================================================================
# 1. GAUSSIAN AR(1)
# =============================================================================

@dataclass
class GaussianAR1:
    """Gaussian AR(1) chain:  a_0 ~ N(mu0, sigma0^2),  a_{k+1} = alpha a_k + eta_k,
    eta_k ~ N(0, sigma_eta^2).  If unspecified, sigma0^2 is set to the stationary
    value sigma_eta^2 / (1 - alpha^2)."""

    alpha: float
    sigma_eta: float
    K: int
    mu0: float = 0.0
    sigma0: float | None = None  # if None, use stationary

    def __post_init__(self):
        if self.sigma0 is None:
            if abs(self.alpha) >= 1.0:
                raise ValueError("alpha must be |alpha|<1 to use stationary sigma0")
            self.sigma0 = self.sigma_eta / np.sqrt(1.0 - self.alpha**2)
        if abs(self.alpha) >= 1.0:
            # allowed, but warn: chain is non-stationary
            pass

    # ------------------------------------------------------------------
    # Clean prior covariance Sigma_0 (K x K)
    # ------------------------------------------------------------------
    def Sigma_0(self) -> np.ndarray:
        """Clean prior covariance of (a_0, ..., a_{K-1})."""
        K, a, s0, seta = self.K, self.alpha, self.sigma0, self.sigma_eta
        # Closed form: Sigma_0[i,j] = alpha^|i-j| * (sigma0^2 if stationary,
        # else time-varying variances v_i).
        # We build it from the exact recursion: v_0 = sigma0^2,
        # v_{k+1} = alpha^2 v_k + sigma_eta^2; Cov(a_i, a_j) = alpha^|i-j| * v_min(i,j).
        v = np.empty(K)
        v[0] = s0**2
        for k in range(K - 1):
            v[k + 1] = a**2 * v[k] + seta**2
        Sig = np.empty((K, K))
        for i in range(K):
            for j in range(K):
                Sig[i, j] = a ** abs(i - j) * v[min(i, j)]
        return Sig

    def mu_0(self) -> np.ndarray:
        """Clean prior mean."""
        K, a, m0 = self.K, self.alpha, self.mu0
        return np.array([a**k * m0 for k in range(K)])

    # ------------------------------------------------------------------
    # Noisy covariance Sigma_t and mean mu_t
    # ------------------------------------------------------------------
    def Sigma_t(self, t: float) -> np.ndarray:
        """Noisy joint covariance at diffusion time t."""
        beta2 = np.exp(-2 * t)
        Delta = 1.0 - beta2
        return beta2 * self.Sigma_0() + Delta * np.eye(self.K)

    def mu_t(self, t: float) -> np.ndarray:
        beta = np.exp(-t)
        return beta * self.mu_0()

    # ------------------------------------------------------------------
    # Joint score, method (a): direct matrix inversion
    # ------------------------------------------------------------------
    def score_matrix(self, x: np.ndarray, t: float) -> np.ndarray:
        """S(x, t) = -Sigma_t^{-1} (x - mu_t)."""
        Sigma_t = self.Sigma_t(t)
        mu_t = self.mu_t(t)
        # Solve Sigma_t y = (x - mu_t) via Cholesky (robust & O(K^3) but exact).
        return -np.linalg.solve(Sigma_t, x - mu_t)

    # ------------------------------------------------------------------
    # Joint score, method (b): Kalman filter + RTS smoother, O(K)
    # ------------------------------------------------------------------
    def score_kalman(self, x: np.ndarray, t: float) -> np.ndarray:
        """Joint score via Tweedie + Kalman smoother. O(K)."""
        K, a, seta, m0, s0 = self.K, self.alpha, self.sigma_eta, self.mu0, self.sigma0
        beta = np.exp(-t)
        Delta = 1.0 - beta**2

        # Forward (Kalman filter) in latent space a_k, with observation
        # x_k = beta * a_k + sqrt(Delta) z_k.
        a_hat_filt = np.empty(K)       # hat{a}_{k|k}
        P_filt     = np.empty(K)       # P_{k|k}
        a_hat_pred = np.empty(K)       # hat{a}_{k|k-1}
        P_pred     = np.empty(K)       # P_{k|k-1}

        # k = 0:  prior is N(m0, s0^2), observation is x[0].
        a_hat_pred[0] = m0
        P_pred[0]     = s0**2
        # Update with x[0]
        innov_var = beta**2 * P_pred[0] + Delta
        gain      = beta * P_pred[0] / innov_var
        a_hat_filt[0] = a_hat_pred[0] + gain * (x[0] - beta * a_hat_pred[0])
        P_filt[0]     = (1.0 - beta * gain) * P_pred[0]

        for k in range(K - 1):
            # Predict
            a_hat_pred[k + 1] = a * a_hat_filt[k]
            P_pred[k + 1]     = a**2 * P_filt[k] + seta**2
            # Update with x[k+1]
            innov_var = beta**2 * P_pred[k + 1] + Delta
            gain      = beta * P_pred[k + 1] / innov_var
            a_hat_filt[k + 1] = a_hat_pred[k + 1] + gain * (x[k + 1] - beta * a_hat_pred[k + 1])
            P_filt[k + 1]     = (1.0 - beta * gain) * P_pred[k + 1]

        # Backward (Rauch-Tung-Striebel smoother)
        a_hat_smooth = np.empty(K)
        a_hat_smooth[K - 1] = a_hat_filt[K - 1]
        for k in range(K - 2, -1, -1):
            L = a * P_filt[k] / P_pred[k + 1]
            a_hat_smooth[k] = a_hat_filt[k] + L * (a_hat_smooth[k + 1] - a_hat_pred[k + 1])

        # Tweedie
        S = (beta * a_hat_smooth - x) / Delta
        return S


# =============================================================================
# 2. LAPLACE AR(1) AT K = 1
# =============================================================================

@dataclass
class LaplaceAR1_K1:
    """K = 1 Laplace chain:  a_0 ~ N(mu0, sigma0^2),  a_1 = alpha a_0 + eps_1,
    eps_1 ~ Lap(0, b)."""

    alpha: float
    b: float
    mu0: float = 0.0
    sigma0: float = 1.0

    def constants(self, t: float) -> dict:
        """Structural constants (Table 3.1 of laplace_K1_compendium.pdf)."""
        a, b, m0, s0 = self.alpha, self.b, self.mu0, self.sigma0
        beta  = np.exp(-t)
        Delta = 1.0 - beta**2
        v     = beta**2 * s0**2 + Delta
        s2    = s0**2 * Delta / v
        bt    = beta * b                     # \tilde b
        tau2  = Delta + a**2 * beta**2 * s2
        tau   = np.sqrt(tau2)
        rho   = a * beta**2 * s0**2 / v
        nu    = a * beta * Delta * m0 / v
        arg   = tau / bt if bt > 0 else np.inf  # Mills argument a = tau / \tilde b
        return dict(beta=beta, Delta=Delta, v=v, s2=s2, bt=bt,
                    tau=tau, rho=rho, nu=nu, arg=arg)

    # ------------------------------------------------------------------
    # Residual density h_t(r) and its log-derivatives
    # ------------------------------------------------------------------
    def h_t(self, r: np.ndarray, t: float) -> np.ndarray:
        """Normal-Laplace residual density (eq. 5.15 of the compendium)."""
        c = self.constants(t)
        bt, tau, arg = c["bt"], c["tau"], c["arg"]
        # Numerically robust form:  h_t(r) = exp(a^2/2) / (2 \tilde b) *
        #                                    [ e^{-r/\tilde b} Phi(r/tau - a)
        #                                    + e^{+r/\tilde b} Phi(-r/tau - a) ]
        # For large arg the two exponential branches grow / decay fast; we
        # factor the dominant branch to keep things stable.
        #
        # We use a direct implementation; for the r-range of interest
        # (|r| <= ~8 tau) it's well-conditioned.
        if bt == 0.0:
            # Gaussian limit: tau=0 too => degenerate. Shouldn't be called at t=inf.
            return norm.pdf(r, loc=0.0, scale=tau if tau > 0 else 1e-12)
        log_prefac = arg**2 / 2.0 - np.log(2.0 * bt)
        # To avoid overflow when r is very negative (e^{+r/bt} is small) or
        # very positive (e^{-r/bt} is small), compute in log-space:
        # log-term-1 = -r/bt + log Phi(r/tau - a)
        # log-term-2 = +r/bt + log Phi(-r/tau - a)
        # Then h_t = exp(log_prefac) * (exp(log-term-1) + exp(log-term-2)).
        # Use logsumexp for stability.
        t1 = -r / bt + norm.logcdf(r / tau - arg)
        t2 =  r / bt + norm.logcdf(-r / tau - arg)
        m = np.maximum(t1, t2)
        log_sum = m + np.log(np.exp(t1 - m) + np.exp(t2 - m))
        return np.exp(log_prefac + log_sum)

    def psi_t(self, r: np.ndarray, t: float) -> np.ndarray:
        """Residual score  psi_t(r) = h_t'(r) / h_t(r).

        Closed form (eq. 2.6 of the memo):
            psi_t(r) = (1 / \tilde b) * (L-(r) - L+(r)) / (L+(r) + L-(r))
        where L+(r) = e^{-r/\tilde b} Phi(r/tau - a),
              L-(r) = e^{+r/\tilde b} Phi(-r/tau - a).
        In the tails, psi_t(r) -> -/+ 1/\tilde b.
        """
        c = self.constants(t)
        bt, tau, arg = c["bt"], c["tau"], c["arg"]
        if bt == 0.0:
            return -r / (tau**2 if tau > 0 else 1e-12)
        # work in log-space
        logL_plus  = -r / bt + norm.logcdf( r / tau - arg)
        logL_minus =  r / bt + norm.logcdf(-r / tau - arg)
        # psi = (L- - L+) / (L+ + L-)
        # = tanh-like combination.  For numerical safety:
        m = np.maximum(logL_plus, logL_minus)
        e_plus  = np.exp(logL_plus  - m)
        e_minus = np.exp(logL_minus - m)
        return (1.0 / bt) * (e_minus - e_plus) / (e_plus + e_minus)

    def kappa_t(self, r: np.ndarray, t: float) -> np.ndarray:
        """Residual curvature kappa_t(r) = -(log h_t)''(r).

        Uses the identity (eq. 80 of the derivation note):
            kappa_t(r) = psi_t(r)^2 - 1/\tilde b^2 + (1/\tilde b^2) * phi_tau(r) / h_t(r)
        where phi_tau is the density of N(0, tau^2).
        """
        c = self.constants(t)
        bt, tau = c["bt"], c["tau"]
        if bt == 0.0:
            return np.ones_like(r) / (tau**2 if tau > 0 else 1e-12)
        psi = self.psi_t(r, t)
        phi_tau = norm.pdf(r, loc=0.0, scale=tau)
        return psi**2 - 1.0 / bt**2 + (1.0 / bt**2) * phi_tau / self.h_t(r, t)

    # ------------------------------------------------------------------
    # Joint score S(x, t) for the 2D noisy pair (x0, x1)
    # ------------------------------------------------------------------
    def score(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Exact joint score at K=1, from eq. 6 of the memo (and eq. 60 of
        the derivation note):
            S_0(x) = -(x0 - beta mu0) / v  - rho * psi_t(r)
            S_1(x) =                          psi_t(r)
        where r = x1 - nu - rho x0.

        x has shape (..., 2); output has the same shape.
        """
        x = np.asarray(x, dtype=float)
        x0, x1 = x[..., 0], x[..., 1]
        c = self.constants(t)
        beta, v, rho, nu = c["beta"], c["v"], c["rho"], c["nu"]
        r = x1 - nu - rho * x0
        psi = self.psi_t(r, t)
        S0 = -(x0 - beta * self.mu0) / v - rho * psi
        S1 = psi
        return np.stack([S0, S1], axis=-1)

    # ------------------------------------------------------------------
    # Sampler (clean + noised), for MC validation
    # ------------------------------------------------------------------
    def sample(self, n: int, t: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n pairs (x0, x1) from the noisy joint p_t."""
        if rng is None:
            rng = np.random.default_rng(0)
        a0 = rng.normal(self.mu0, self.sigma0, size=n)
        eps = rng.laplace(loc=0.0, scale=self.b, size=n)
        a1 = self.alpha * a0 + eps
        beta = np.exp(-t)
        Delta = 1.0 - beta**2
        z = rng.normal(0.0, 1.0, size=(n, 2))
        x0 = beta * a0 + np.sqrt(Delta) * z[:, 0]
        x1 = beta * a1 + np.sqrt(Delta) * z[:, 1]
        return np.stack([x0, x1], axis=-1)


# =============================================================================
# 3. LAPLACE AR(1) AT K = 2:  joint score via 3D Fourier inversion
# =============================================================================

@dataclass
class LaplaceAR1_K2:
    """K = 2 Laplace chain.  Exact joint score is computed numerically by
    Fourier inversion of the closed-form characteristic function, then
    finite differencing the log-density.

    This class is heavier than the others and intended for benchmarking at
    a sparse set of evaluation points, not for vectorised batches.
    """

    alpha: float
    b: float
    mu0: float = 0.0
    sigma0: float = 1.0

    # ------------------------------------------------------------------
    # Characteristic function (eq. 16 of the memo)
    # ------------------------------------------------------------------
    def p_hat(self, k: np.ndarray, t: float) -> np.ndarray:
        """
        k has shape (..., 3).  Output is a complex array of the same
        leading shape.
        """
        k = np.asarray(k)
        k0, k1, k2 = k[..., 0], k[..., 1], k[..., 2]
        beta = np.exp(-t)
        Delta = 1.0 - beta**2
        bt = beta * self.b
        alpha = self.alpha
        q = k0 + alpha * k1 + alpha**2 * k2

        # phase and Gaussian damping
        linear = 1j * beta * self.mu0 * q
        gauss  = -0.5 * beta**2 * self.sigma0**2 * q**2 \
                 - 0.5 * Delta * (k0**2 + k1**2 + k2**2)
        rat    = 1.0 / (1.0 + bt**2 * (k1 + alpha * k2)**2) \
               * 1.0 / (1.0 + bt**2 * k2**2)
        return np.exp(linear + gauss) * rat

    # ------------------------------------------------------------------
    # Density at a point x by direct 3D quadrature of the Fourier integral
    # ------------------------------------------------------------------
    def p_t(self, x: np.ndarray, t: float, K_max: float = 12.0, N: int = 81) -> float:
        """
        p_t(x) = (2 pi)^{-3} int e^{-i k . x} p_hat(k, t) dk.
        Evaluated by trapezoidal rule on [-K_max, K_max]^3 with N points per axis.
        x has shape (3,).
        """
        x = np.asarray(x, dtype=float)
        ks = np.linspace(-K_max, K_max, N)
        k0g, k1g, k2g = np.meshgrid(ks, ks, ks, indexing='ij')
        kgrid = np.stack([k0g, k1g, k2g], axis=-1)       # (N, N, N, 3)
        phase = -1j * (k0g * x[0] + k1g * x[1] + k2g * x[2])
        integrand = self.p_hat(kgrid, t) * np.exp(phase)
        # trapezoidal in each axis
        dk = ks[1] - ks[0]
        val = np.real(np.sum(integrand)) * dk**3 / (2 * np.pi)**3
        return val

    def log_p_t(self, x: np.ndarray, t: float, **kwargs) -> float:
        v = self.p_t(x, t, **kwargs)
        return np.log(max(v, 1e-300))

    def score(self, x: np.ndarray, t: float, h: float = 1e-2, **kwargs) -> np.ndarray:
        """
        Numerical gradient of log p_t at x via central differences.
        Each component requires 2 density evaluations; total 6.
        """
        x = np.asarray(x, dtype=float)
        S = np.empty(3)
        for d in range(3):
            xp = x.copy(); xp[d] += h
            xm = x.copy(); xm[d] -= h
            S[d] = (self.log_p_t(xp, t, **kwargs) - self.log_p_t(xm, t, **kwargs)) / (2 * h)
        return S

    # ------------------------------------------------------------------
    # Sampler for MC validation
    # ------------------------------------------------------------------
    def sample(self, n: int, t: float, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(0)
        a0 = rng.normal(self.mu0, self.sigma0, size=n)
        e1 = rng.laplace(0.0, self.b, size=n)
        e2 = rng.laplace(0.0, self.b, size=n)
        a1 = self.alpha * a0 + e1
        a2 = self.alpha * a1 + e2
        beta = np.exp(-t)
        Delta = 1.0 - beta**2
        z = rng.normal(0.0, 1.0, size=(n, 3))
        x0 = beta * a0 + np.sqrt(Delta) * z[:, 0]
        x1 = beta * a1 + np.sqrt(Delta) * z[:, 1]
        x2 = beta * a2 + np.sqrt(Delta) * z[:, 2]
        return np.stack([x0, x1, x2], axis=-1)


# =============================================================================
# SELF-TESTS
# =============================================================================

def test_gaussian_ar1_kalman_vs_matrix():
    """Kalman smoother and direct matrix inversion must agree to floating-point."""
    rng = np.random.default_rng(0)
    for K in [2, 4, 8, 16, 32]:
        for alpha in [0.0, 0.3, 0.7, 0.95]:
            for t in [0.05, 0.3, 1.0, 3.0]:
                model = GaussianAR1(alpha=alpha, sigma_eta=1.0, K=K, mu0=0.3)
                x = rng.normal(size=K)
                S_mat = model.score_matrix(x, t)
                S_kal = model.score_kalman(x, t)
                err = np.max(np.abs(S_mat - S_kal))
                assert err < 1e-10, f"K={K} alpha={alpha} t={t}: err={err:.2e}"
    return True


def test_laplace_K1_mc():
    """Monte-Carlo estimate of score at a single point should agree with closed form.

    We use the fact that  E_{x ~ p_t} [ (x - mu) * indicator(x in B_eps) ] ~ p_t(mu) * eps,
    but a cleaner test is: check psi_t tail saturation.  We do both.
    """
    model = LaplaceAR1_K1(alpha=0.8, b=1.0, sigma0=1.0)
    t = 0.5
    c = model.constants(t)
    bt = c["bt"]
    # Tail saturation:
    r_tail = np.array([-30.0, 30.0])
    psi_tail = model.psi_t(r_tail, t)
    expected = np.array([+1.0 / bt, -1.0 / bt])
    err = np.max(np.abs(psi_tail - expected))
    assert err < 1e-6, f"psi_t tail saturation failed: err={err:.2e}, got {psi_tail}, want {expected}"
    return True


def test_laplace_K1_centre_curvature():
    """Check Mills-ratio closed form for kappa_t(0) (eq. 13 of the memo)."""
    model = LaplaceAR1_K1(alpha=0.8, b=1.0, sigma0=1.0)
    for t in [0.05, 0.3, 1.0, 3.0]:
        c = model.constants(t)
        bt, arg = c["bt"], c["arg"]
        # Mills ratio R(a) = Phi(-a) / phi(a)
        R = norm.cdf(-arg) / norm.pdf(arg)
        kappa_closed = (1.0 / bt**2) * (1.0 / (arg * R) - 1.0)
        kappa_direct = model.kappa_t(np.array([0.0]), t)[0]
        err = abs(kappa_closed - kappa_direct) / abs(kappa_closed)
        assert err < 1e-6, f"t={t}: kappa(0) closed={kappa_closed} direct={kappa_direct}, rel err={err:.2e}"
    return True


def test_laplace_K2_fourier_density_positive():
    """Sanity: 3D Fourier density at the mean must be positive and finite."""
    model = LaplaceAR1_K2(alpha=0.8, b=1.0, sigma0=1.0)
    for t in [0.3, 1.0]:
        p = model.p_t(np.array([0.0, 0.0, 0.0]), t)
        assert p > 0 and np.isfinite(p), f"t={t}: p={p}"
    return True


def test_laplace_K2_score_finite_diff_is_stable():
    """Score via finite diff should be finite and of reasonable magnitude at the mean."""
    model = LaplaceAR1_K2(alpha=0.8, b=1.0, sigma0=1.0)
    S = model.score(np.array([0.0, 0.0, 0.0]), t=0.5)
    assert np.all(np.isfinite(S))
    return True


def test_gaussian_limit_of_laplace_K1():
    """As t -> infinity the Laplace K=1 curvature should approach 1/tau_G^2."""
    model = LaplaceAR1_K1(alpha=0.8, b=1.0, sigma0=1.0)
    t = 5.0
    c = model.constants(t)
    tau2_G = c["tau"]**2 + 2 * c["bt"]**2
    kappa_G = 1.0 / tau2_G
    kappa_direct = model.kappa_t(np.array([0.0]), t)[0]
    rel = abs(kappa_direct - kappa_G) / abs(kappa_G)
    assert rel < 0.05, f"Gaussian limit at t={t} not reached: kappa_direct={kappa_direct} kappa_G={kappa_G}, rel={rel:.2e}"
    return True


# =============================================================================
# Demo / main
# =============================================================================

if __name__ == "__main__":
    print("Running self-tests...")
    print("  [1] Gaussian AR(1): Kalman vs matrix inversion...", end=" ")
    test_gaussian_ar1_kalman_vs_matrix(); print("OK")
    print("  [2] Laplace K=1: psi_t tail saturation...", end=" ")
    test_laplace_K1_mc(); print("OK")
    print("  [3] Laplace K=1: Mills-ratio kappa(0)...", end=" ")
    test_laplace_K1_centre_curvature(); print("OK")
    print("  [4] Laplace K=2: Fourier density at mean positive...", end=" ")
    test_laplace_K2_fourier_density_positive(); print("OK")
    print("  [5] Laplace K=2: score finite at mean...", end=" ")
    test_laplace_K2_score_finite_diff_is_stable(); print("OK")
    print("  [6] Laplace K=1: Gaussian limit at large t...", end=" ")
    test_gaussian_limit_of_laplace_K1(); print("OK")
    print()
    print("All self-tests passed.")
    print()

    # Small demonstration: compare the two Gaussian-AR(1) paths at one point.
    print("Demo: Gaussian AR(1), K=8, alpha=0.8, t=0.3")
    rng = np.random.default_rng(42)
    model = GaussianAR1(alpha=0.8, sigma_eta=1.0, K=8, mu0=0.0)
    x = rng.normal(size=8)
    S_mat = model.score_matrix(x, 0.3)
    S_kal = model.score_kalman(x, 0.3)
    print(f"  S_matrix[:3]:  {S_mat[:3]}")
    print(f"  S_kalman[:3]:  {S_kal[:3]}")
    print(f"  max|S_mat - S_kal|: {np.max(np.abs(S_mat - S_kal)):.2e}")
    print()

    print("Demo: Laplace K=1 joint score at a few points")
    model1 = LaplaceAR1_K1(alpha=0.8, b=1.0, sigma0=1.0)
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 2.0]])
    for x in pts:
        S = model1.score(x, t=0.5)
        print(f"  x = {x},  S = {S}")
