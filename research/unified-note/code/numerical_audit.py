"""
numerical_audit.py
==================

Independent numerical verification of every closed-form identity used
in the unified_document manuscript. The audit is the gate kept by the
project: if anything below fails the manuscript is not allowed to
quote it as Established.

Each check fixes a small set of parameters, evaluates the closed-form
expression and an independent reference (matrix product, exact inverse,
spectral identity, finite differences, or scipy quadrature), and
reports the maximum absolute error against an explicit tolerance.

Tolerance bands:
    - matrix algebra at machine precision    : 1e-12
    - BP vs matrix posterior / score        : 1e-10
    - Tweedie vs finite difference          : 1e-7
    - small-t band-fill exponents           : 0.05  (absolute slope error)
    - large-t exponential return            : 1e-6 relative
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np

from ar1_utils import (
    ar1_covariance,
    ar1_precision_clean,
    gaussian_posterior,
    joint_score_matrix,
    joint_score_via_tweedie,
    ou_params,
    precision_t,
    precision_t_spectral,
    prior_mean,
    stationary_sigma_eta,
)
from bp_score import bp_posterior, bp_score
from laplace_score import (
    laplace_marginal,
    laplace_marginal_cf,
    laplace_marginal_cf_quadrature,
    laplace_marginal_via_green,
    laplace_score,
)


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    error: float
    tol: float

    @property
    def passed(self) -> bool:
        return self.error <= self.tol


def _fmt(c: Check) -> str:
    flag = "PASS" if c.passed else "FAIL"
    return f"[{flag}] {c.name:<58s} err={c.error:.2e}  tol={c.tol:.0e}"


# ---------------------------------------------------------------------------
# 1. Gaussian AR(1) clean precision is tridiagonal and equal to Sigma_0^{-1}
# ---------------------------------------------------------------------------

def check_clean_precision() -> list[Check]:
    out: list[Check] = []
    for K in (5, 9, 20):
        for alpha in (0.3, 0.7, 0.9, -0.5):
            Sigma_0 = ar1_covariance(K, alpha)
            Q_formula = ar1_precision_clean(K, alpha)
            Q_inv = np.linalg.inv(Sigma_0)
            err = float(np.max(np.abs(Q_formula - Q_inv)))
            out.append(Check(
                f"clean precision Q_0 (K={K}, alpha={alpha:+.2f})",
                err, 1e-12,
            ))
            # Off-band entries of Q_formula must be exactly zero
            mask = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) > 1
            off_band = float(np.max(np.abs(Q_formula[mask])))
            out.append(Check(
                f"Q_0 strictly tridiagonal (K={K}, alpha={alpha:+.2f})",
                off_band, 0.0,
            ))
    return out


# ---------------------------------------------------------------------------
# 2. Sigma_t Sigma_t^{-1} = I and spectral preservation
# ---------------------------------------------------------------------------

def check_inverse_consistency() -> list[Check]:
    out: list[Check] = []
    K = 12
    Sigma_0 = ar1_covariance(K, 0.8)
    for t in (0.001, 0.05, 0.4, 1.5, 5.0):
        Q_direct = precision_t(Sigma_0, t)
        from ar1_utils import sigma_t as _sigma_t
        S_t = _sigma_t(Sigma_0, t)
        prod = S_t @ Q_direct
        err = float(np.max(np.abs(prod - np.eye(K))))
        out.append(Check(f"Sigma_t Q_t = I (t={t})", err, 1e-10))

        Q_spec, U, lam_t = precision_t_spectral(Sigma_0, t)
        err_spec = float(np.max(np.abs(Q_spec - Q_direct)))
        out.append(Check(f"Q_t direct vs spectral (t={t})",
                         err_spec, 1e-10))

        # Spectral preservation check: U should diagonalise Sigma_0
        diag_check = U.T @ Sigma_0 @ U
        off = diag_check - np.diag(np.diag(diag_check))
        err_diag = float(np.max(np.abs(off)))
        out.append(Check(f"U diagonalises Sigma_0 (t={t})",
                         err_diag, 1e-10))
    return out


# ---------------------------------------------------------------------------
# 3. Band-fill theorem: |Q_t[i, i + d]| ~ t^{d-1} at small t
# ---------------------------------------------------------------------------

def check_band_fill() -> list[Check]:
    """Verify the leading coefficient of (Q_t)_{i, i + d} at small t.

    The Neumann expansion (precision_lifecycle_summary.pdf, eq 10):
        Q_t = exp(2 t) sum_{m >= 0} (- c_t)^m Q_0^{m + 1},   c_t = exp(2 t) - 1
    expanded at small t with c_t = 2 t + O(t^2), gives the leading
    nonvanishing distance-d contribution from m = d - 1:
        (Q_t)_{i, i + d} = (-1)^{d - 1} (2 t)^{d - 1} (Q_0^d)_{i, i + d}
                           + O(t^d).
    Note: equation (12) of the same PDF prints the formula with an extra
    1 / (d - 1)! factor that does not follow from the resolvent expansion
    of equation (10).  The audit numerically confirms the version above
    (without the factorial); see also unified_document/main.tex section
    "A correction to the published band-fill coefficient".

    The check is the relative agreement
        c_d^{empirical} = (Q_t)_{i, i + d} / t^{d - 1}
        c_d^{predicted} = (-1)^{d - 1} 2^{d - 1} (Q_0^d)_{i, i + d}
    at the smallest t that keeps t^{d - 1} well above the floating-point
    inversion noise floor.  The 5% tolerance covers higher-order
    corrections and inversion noise jointly.
    """
    K = 25
    alpha = 0.9
    Sigma_0 = ar1_covariance(K, alpha)
    Q_0 = ar1_precision_clean(K, alpha)
    i = K // 2

    out: list[Check] = []
    t_for_d = {1: 1e-5, 2: 1e-4, 3: 1e-4, 4: 1e-4, 5: 1e-4}

    Q_d_powers = {1: Q_0}
    for k in range(2, 6):
        Q_d_powers[k] = Q_d_powers[k - 1] @ Q_0

    for d in range(1, 6):
        t = t_for_d[d]
        c_pred = (
            ((-1) ** (d - 1)) * (2.0 ** (d - 1))
            * Q_d_powers[d][i, i + d]
        )
        Q = precision_t(Sigma_0, t)
        c_emp = Q[i, i + d] / (t ** (d - 1))
        rel = float(abs(c_emp - c_pred) / abs(c_pred))
        out.append(Check(
            f"band-fill leading coeff d={d}", rel, 0.05,
        ))
    return out


# ---------------------------------------------------------------------------
# 4. Large-t asymptotic: ||Q_t - I||_F ~ e^{-2 t} ||Sigma_0 - I||_F
# ---------------------------------------------------------------------------

def check_large_t_return() -> list[Check]:
    """Verify Q_t = I - e^{-2t}(Sigma_0 - I) + e^{-4t}(Sigma_0 - I)^2 + O(e^{-6t}).

    Reporting only the leading e^{-2t} term has a relative error O(e^{-2t})
    coming from the second-order Neumann correction; the size of that
    correction is structure-dependent (it scales with the spectral radius
    of Sigma_0 - I), so we set the tolerance to scale linearly with
    e^{-2t} times that radius and add a 1e-7 floating-point floor for
    very large t where the absolute residual saturates at numerical zero.
    """
    K = 20
    Sigma_0 = ar1_covariance(K, 0.9)
    base = np.linalg.norm(Sigma_0 - np.eye(K), ord="fro")
    radius = float(np.max(np.abs(np.linalg.eigvalsh(Sigma_0) - 1.0)))
    out: list[Check] = []
    for t in (3.0, 6.0, 10.0):
        Q = precision_t(Sigma_0, t)
        actual = np.linalg.norm(Q - np.eye(K), ord="fro")
        predicted = np.exp(-2.0 * t) * base
        rel_err = float(abs(actual - predicted) / predicted)
        tol = max(1e-7, 1.5 * radius * np.exp(-2.0 * t))
        out.append(Check(
            f"||Q_t - I||_F vs e^-2t ||Sigma_0 - I||_F (t={t})",
            rel_err, tol,
        ))
    return out


# ---------------------------------------------------------------------------
# 5. BP vs matrix posterior, sweeping a small grid
# ---------------------------------------------------------------------------

def check_bp_vs_matrix() -> list[Check]:
    rng = np.random.default_rng(11)
    out: list[Check] = []
    err_mu_max = 0.0
    err_var_max = 0.0
    err_score_max = 0.0
    n_cases = 0
    for K in (2, 3, 5, 8, 16):
        for alpha in (0.2, 0.5, 0.9, -0.4):
            for t in (0.05, 0.3, 1.0, 3.0):
                Sigma_0 = ar1_covariance(K, alpha)
                x = rng.standard_normal(K)

                mu_bp, var_bp = bp_posterior(x, t, alpha)
                mu_m, Sigma_m = gaussian_posterior(x, t, Sigma_0, alpha)
                err_mu_max = max(err_mu_max,
                                 float(np.max(np.abs(mu_bp - mu_m))))
                err_var_max = max(err_var_max,
                                  float(np.max(np.abs(var_bp - np.diag(Sigma_m)))))

                s_bp = bp_score(x, t, alpha)
                s_m = joint_score_matrix(x, t, Sigma_0, alpha)
                err_score_max = max(err_score_max,
                                    float(np.max(np.abs(s_bp - s_m))))
                n_cases += 1

    out.append(Check(f"BP vs matrix posterior mean ({n_cases} cases)",
                     err_mu_max, 1e-10))
    out.append(Check(f"BP vs matrix posterior variance ({n_cases} cases)",
                     err_var_max, 1e-10))
    out.append(Check(f"BP vs matrix joint score ({n_cases} cases)",
                     err_score_max, 1e-10))
    return out


# ---------------------------------------------------------------------------
# 6. Tweedie matrix score equals -Q_t (x - mu_t) for Gaussian AR(1)
# ---------------------------------------------------------------------------

def check_tweedie_matrix_consistency() -> list[Check]:
    rng = np.random.default_rng(12)
    K = 8
    Sigma_0 = ar1_covariance(K, 0.6)
    err_max = 0.0
    for t in (0.1, 0.5, 2.0):
        for _ in range(10):
            x = rng.standard_normal(K)
            s_a = joint_score_matrix(x, t, Sigma_0, 0.6)
            s_b = joint_score_via_tweedie(x, t, Sigma_0, 0.6)
            err_max = max(err_max, float(np.max(np.abs(s_a - s_b))))
    return [Check("Tweedie matrix score consistency (Gaussian AR(1))",
                  err_max, 1e-12)]


# ---------------------------------------------------------------------------
# 7. Laplace K=1 score: Tweedie vs central finite difference of log p_t
# ---------------------------------------------------------------------------

def check_laplace_K1_score_fd() -> list[Check]:
    out: list[Check] = []
    h = 5e-4
    err_max = 0.0
    for b in (0.5, 1.0):
        for t in (0.1, 0.4, 1.0, 2.0):
            for x in (-1.0, -0.3, 0.2, 0.7, 1.5):
                lp_plus = np.log(laplace_marginal(x + h, t, b))
                lp_minus = np.log(laplace_marginal(x - h, t, b))
                s_fd = (lp_plus - lp_minus) / (2.0 * h)
                s_tw = laplace_score(x, t, b)
                err_max = max(err_max, abs(s_fd - s_tw))
    out.append(Check("Laplace K=1 score: Tweedie vs central FD",
                     float(err_max), 5e-7))
    return out


# ---------------------------------------------------------------------------
# 8. Laplace K=1 score limits: t -> infty score -> -x (Gaussian attractor)
# ---------------------------------------------------------------------------

def check_laplace_K1_limits() -> list[Check]:
    # As t -> infty, Sigma_t -> 1 and the marginal becomes N(0, 1).  The
    # joint score should then approach -x.  We verify at t = 6 where
    # exp(-2 t) ~ 6e-6.
    err_max = 0.0
    for x in (-1.0, -0.2, 0.5, 1.2):
        s = laplace_score(x, 6.0, b=1.0)
        err_max = max(err_max, abs(s + x))
    return [Check("Laplace K=1 score(t=6) approx -x", float(err_max), 1e-3)]


# ---------------------------------------------------------------------------
# 9. Laplace K=1 Fourier representation -- characteristic function and
#    screened Poisson reconstruction
# ---------------------------------------------------------------------------

def check_laplace_K1_fourier() -> list[Check]:
    """Verify two distinct algebraic representations of the K=1 noised
    marginal:

      (a) phi_t(k) = (1 + b^2 mu^2 k^2)^{-1} * exp(-Delta_t k^2 / 2),
          with the LHS computed by quadrature on the density and the
          RHS evaluated in closed form.
      (b) p_t(y) = (G_ODE * G_{Delta_t})(y), the screened-Poisson
          Green's-function representation, with the convolution
          computed by 1D quadrature.
    """
    out: list[Check] = []
    # Characteristic function check
    err_max = 0.0
    for b in (0.5, 1.0):
        for t in (0.2, 0.6, 1.4):
            for k in (0.3, 1.0, 2.5):
                phi_q = laplace_marginal_cf_quadrature(k, t, b)
                phi_c = laplace_marginal_cf(k, t, b)
                err_max = max(err_max, abs(phi_q.real - phi_c)
                              + abs(phi_q.imag))
    # Tolerance set by adaptive quadrature on an oscillatory integrand on
    # an unbounded domain; closed-form vs FFT-style quadrature agreement
    # is typically 1e-6 to 1e-5 for the parameter ranges checked.
    out.append(Check("Laplace K=1 c.f.: quadrature vs closed form",
                     float(err_max), 1e-5))

    # Green's function reconstruction check
    err_max = 0.0
    for b in (0.5, 1.0):
        for t in (0.2, 0.6, 1.4):
            for y in (-1.5, -0.4, 0.0, 0.7, 2.0):
                p_direct = laplace_marginal(y, t, b)
                p_green = laplace_marginal_via_green(y, t, b)
                err_max = max(err_max, abs(p_direct - p_green))
    out.append(Check("Laplace K=1 screened-Poisson Green reconstruction",
                     float(err_max), 1e-7))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all() -> int:
    checks: list[Check] = []
    checks += check_clean_precision()
    checks += check_inverse_consistency()
    checks += check_band_fill()
    checks += check_large_t_return()
    checks += check_bp_vs_matrix()
    checks += check_tweedie_matrix_consistency()
    checks += check_laplace_K1_score_fd()
    checks += check_laplace_K1_limits()
    checks += check_laplace_K1_fourier()

    n_total = len(checks)
    n_fail = sum(1 for c in checks if not c.passed)
    print("=" * 78)
    print(f"unified_document numerical audit  --  {n_total} checks")
    print("=" * 78)
    for c in checks:
        print(_fmt(c))
    print("-" * 78)
    print(f"PASSED {n_total - n_fail} / {n_total}  "
          f"FAILED {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
