"""
Belief propagation for the Gaussian AR(1) diffusion model.

This script compares two mathematically equivalent ways of computing the
score of the noisy distribution p_t(x):

1. Matrix route:
       x ~ N(0, Sigma_t), so score(x,t) = - Sigma_t^{-1} x.

2. BP route:
       compute the posterior marginals p(a_i | x_1,...,x_n) by Gaussian
       belief propagation on the AR(1) chain, then use
       score_i(x,t) = -(x_i - alpha_t E[a_i | x]) / Delta_t.

Running this file creates numerical checks and figures in the chosen output
folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray


def diffusion_params(t: float) -> Tuple[float, float]:
    """Return alpha_t = exp(-t) and Delta_t = 1 - exp(-2t)."""
    if t <= 0:
        raise ValueError("This implementation expects t > 0 because Delta_t must be positive.")
    alpha = float(np.exp(-t))
    Delta = float(1.0 - alpha**2)
    if Delta <= 0:
        raise ValueError("Delta_t is not positive. Use t > 0.")
    return alpha, Delta


def ar1_covariance(n: int, rho: float) -> Array:
    """Clean stationary AR(1) covariance: Sigma0[i,j] = rho^{|i-j|}."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if abs(rho) >= 1:
        raise ValueError("Stationary AR(1) requires |rho| < 1.")
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample_clean_ar1(n: int, rho: float, seed: int = 0) -> Array:
    """Sample a clean stationary AR(1) chain with marginal variance one."""
    if abs(rho) >= 1:
        raise ValueError("Stationary AR(1) requires |rho| < 1.")
    rng = np.random.default_rng(seed)
    q = 1.0 - rho**2
    a = np.empty(n)
    a[0] = rng.normal()
    for i in range(1, n):
        a[i] = rho * a[i - 1] + np.sqrt(q) * rng.normal()
    return a


def sample_noisy_ar1(n: int, rho: float, t: float, seed: int = 0) -> Tuple[Array, Array]:
    """Sample clean a and noisy x from x = alpha a + sqrt(Delta) z."""
    alpha, Delta = diffusion_params(t)
    rng = np.random.default_rng(seed + 1)
    a = sample_clean_ar1(n=n, rho=rho, seed=seed)
    x = alpha * a + np.sqrt(Delta) * rng.normal(size=n)
    return a, x


def score_by_precision_matrix(x: Array, rho: float, t: float) -> Dict[str, Array]:
    """
    Direct Gaussian route.

    Clean prior:
        a ~ N(0, Sigma0)

    Forward OU noising:
        x = alpha a + sqrt(Delta) z, z ~ N(0, I)

    Hence:
        x ~ N(0, Sigma_t), Sigma_t = alpha^2 Sigma0 + Delta I.

    Score:
        score(x,t) = - Sigma_t^{-1} x.

    Posterior mean:
        E[a | x] = alpha Sigma0 Sigma_t^{-1} x.

    Posterior covariance:
        Cov(a | x) = Sigma0 - alpha^2 Sigma0 Sigma_t^{-1} Sigma0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    alpha, Delta = diffusion_params(t)

    Sigma0 = ar1_covariance(n, rho)
    Sigma_t = alpha**2 * Sigma0 + Delta * np.eye(n)

    precision_times_x = np.linalg.solve(Sigma_t, x)
    score = -precision_times_x

    posterior_mean = alpha * Sigma0 @ precision_times_x

    # Solve Sigma_t Y = Sigma0 to avoid explicitly inverting Sigma_t.
    Sigma_t_inv_Sigma0 = np.linalg.solve(Sigma_t, Sigma0)
    posterior_cov = Sigma0 - alpha**2 * Sigma0 @ Sigma_t_inv_Sigma0
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)  # numerical symmetrization
    posterior_var = np.diag(posterior_cov)

    return {
        "score": score,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "posterior_cov": posterior_cov,
        "Sigma0": Sigma0,
        "Sigma_t": Sigma_t,
    }


def score_by_gaussian_bp(x: Array, rho: float, t: float) -> Dict[str, Array]:
    """
    Gaussian BP route on the AR(1) chain.

    Messages are represented in information form:
        m(a) propto exp(-0.5 lambda a^2 + h a),
    where lambda is the precision and h is precision times mean.

    The posterior marginal at node i is
        b_i(a_i) propto L_i(a_i) ell_i(a_i; x_i) R_i(a_i).

    Its mean gives the denoiser E[a_i | x], and the score is
        score_i = -(x_i - alpha E[a_i | x]) / Delta.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if abs(rho) >= 1:
        raise ValueError("Stationary AR(1) requires |rho| < 1.")

    alpha, Delta = diffusion_params(t)
    q = 1.0 - rho**2

    # Likelihood ell_i(a_i; x_i) = N(x_i; alpha a_i, Delta), seen as a function of a_i.
    # Information-form parameters are:
    #   lambda_l = alpha^2 / Delta,
    #   h_l_i    = alpha x_i / Delta.
    lambda_l = alpha**2 / Delta
    h_l = alpha * x / Delta

    # Forward messages L_i.
    lambda_L = np.zeros(n)
    h_L = np.zeros(n)
    lambda_L[0] = 1.0  # prior N(0,1)
    h_L[0] = 0.0

    for i in range(n - 1):
        # M_i^right(a_i) propto L_i(a_i) ell_i(a_i; x_i).
        lambda_msg = lambda_L[i] + lambda_l
        h_msg = h_L[i] + h_l[i]

        # If a_i has mean h_msg/lambda_msg and variance 1/lambda_msg,
        # then a_{i+1} = rho a_i + sqrt(q) noise has mean and variance:
        #   rho h_msg/lambda_msg,
        #   q + rho^2/lambda_msg.
        lambda_L[i + 1] = 1.0 / (q + rho**2 / lambda_msg)
        h_L[i + 1] = lambda_L[i + 1] * rho * h_msg / lambda_msg

    # Backward messages R_i.
    lambda_R = np.zeros(n)
    h_R = np.zeros(n)
    lambda_R[-1] = 0.0  # constant message R_n(a_n) = 1
    h_R[-1] = 0.0

    for i in range(n - 1, 0, -1):
        # M_i^left(a_i) propto R_i(a_i) ell_i(a_i; x_i).
        lambda_msg = lambda_R[i] + lambda_l
        h_msg = h_R[i] + h_l[i]

        # Integrate N(a_i; rho a_{i-1}, q) M_i^left(a_i) da_i.
        # The result is Gaussian-shaped in a_{i-1}, with:
        denom = q * lambda_msg + 1.0
        lambda_R[i - 1] = rho**2 * lambda_msg / denom
        h_R[i - 1] = rho * h_msg / denom

    # Beliefs b_i(a_i) propto L_i(a_i) ell_i(a_i; x_i) R_i(a_i).
    lambda_b = lambda_L + lambda_l + lambda_R
    h_b = h_L + h_l + h_R

    posterior_mean = h_b / lambda_b
    posterior_var = 1.0 / lambda_b
    score = -(x - alpha * posterior_mean) / Delta

    return {
        "score": score,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "lambda_L": lambda_L,
        "h_L": h_L,
        "lambda_R": lambda_R,
        "h_R": h_R,
        "lambda_b": lambda_b,
        "h_b": h_b,
        "lambda_l": np.full(n, lambda_l),
        "h_l": h_l,
    }


def compare_methods(x: Array, rho: float, t: float) -> Dict[str, float | Array | Dict[str, Array]]:
    """Run both methods and return outputs plus numerical discrepancies."""
    bp = score_by_gaussian_bp(x, rho, t)
    mat = score_by_precision_matrix(x, rho, t)

    score_err = np.abs(bp["score"] - mat["score"])
    mean_err = np.abs(bp["posterior_mean"] - mat["posterior_mean"])
    var_err = np.abs(bp["posterior_var"] - mat["posterior_var"])

    return {
        "bp": bp,
        "matrix": mat,
        "score_err": score_err,
        "mean_err": mean_err,
        "var_err": var_err,
        "max_score_error": float(np.max(score_err)),
        "max_mean_error": float(np.max(mean_err)),
        "max_var_error": float(np.max(var_err)),
    }


def save_plots(a_true: Array, x: Array, comparison: Dict[str, object], outdir: Path) -> None:
    """Create plots comparing BP and matrix outputs."""
    outdir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(1, len(x) + 1)
    bp = comparison["bp"]
    mat = comparison["matrix"]

    plt.figure(figsize=(10, 5))
    plt.plot(idx, a_true, marker="o", label="true clean a")
    plt.plot(idx, x, marker="o", label="observed noisy x")
    plt.plot(idx, bp["posterior_mean"], marker="o", label="posterior mean by BP")
    plt.plot(idx, mat["posterior_mean"], linestyle="--", label="posterior mean by matrix")
    plt.xlabel("index i")
    plt.ylabel("value")
    plt.title("Clean signal, noisy observation, and posterior mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "clean_noisy_posterior.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(idx, bp["score"], marker="o", label="score by BP")
    plt.plot(idx, mat["score"], linestyle="--", label="score by precision matrix")
    plt.xlabel("index i")
    plt.ylabel("score component")
    plt.title("Score comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "score_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(idx, comparison["score_err"] + 1e-18, marker="o", label="absolute score difference")
    plt.semilogy(idx, comparison["mean_err"] + 1e-18, marker="o", label="absolute posterior mean difference")
    plt.semilogy(idx, comparison["var_err"] + 1e-18, marker="o", label="absolute posterior variance difference")
    plt.xlabel("index i")
    plt.ylabel("absolute difference, log scale")
    plt.title("Numerical differences between BP and matrix formulas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "absolute_differences.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(idx, bp["lambda_L"], marker="o", label="left-message precision")
    plt.plot(idx, bp["lambda_R"], marker="o", label="right-message precision")
    plt.plot(idx, bp["lambda_l"], marker="o", label="local likelihood precision")
    plt.plot(idx, bp["lambda_b"], marker="o", label="belief precision")
    plt.xlabel("index i")
    plt.ylabel("precision")
    plt.title("Information-form precisions in BP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bp_message_precisions.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(idx, bp["posterior_var"], marker="o", label="posterior variance by BP")
    plt.plot(idx, mat["posterior_var"], linestyle="--", label="posterior variance by matrix")
    plt.xlabel("index i")
    plt.ylabel("posterior variance")
    plt.title("Posterior marginal variances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "posterior_variance.png", dpi=200)
    plt.close()


def write_summary(
    a_true: Array,
    x: Array,
    rho: float,
    t: float,
    comparison: Dict[str, object],
    outdir: Path,
) -> None:
    """Write a plain-text numerical summary."""
    alpha, Delta = diffusion_params(t)
    lines = []
    lines.append("Gaussian AR(1) diffusion BP check")
    lines.append("==================================")
    lines.append(f"n                 = {len(x)}")
    lines.append(f"rho               = {rho:.8f}")
    lines.append(f"t                 = {t:.8f}")
    lines.append(f"alpha = exp(-t)   = {alpha:.12f}")
    lines.append(f"Delta = 1-alpha^2 = {Delta:.12f}")
    lines.append("")
    lines.append("Numerical agreement")
    lines.append("-------------------")
    lines.append(f"max |score_BP - score_matrix|      = {comparison['max_score_error']:.16e}")
    lines.append(f"max |mean_BP  - mean_matrix|       = {comparison['max_mean_error']:.16e}")
    lines.append(f"max |var_BP   - var_matrix|        = {comparison['max_var_error']:.16e}")
    lines.append("")
    lines.append("First five entries")
    lines.append("------------------")
    lines.append("i      a_true          x              mean_BP         score_BP")
    bp = comparison["bp"]
    for i in range(min(5, len(x))):
        lines.append(
            f"{i+1:<6d} {a_true[i]: .8f}   {x[i]: .8f}   "
            f"{bp['posterior_mean'][i]: .8f}   {bp['score'][i]: .8f}"
        )
    text = "\n".join(lines) + "\n"
    (outdir / "results_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def run_experiment(n: int, rho: float, t: float, seed: int, outdir: Path) -> Dict[str, object]:
    """Run a complete experiment and save plots plus summary."""
    a_true, x = sample_noisy_ar1(n=n, rho=rho, t=t, seed=seed)
    comparison = compare_methods(x=x, rho=rho, t=t)
    save_plots(a_true=a_true, x=x, comparison=comparison, outdir=outdir)
    write_summary(a_true=a_true, x=x, rho=rho, t=t, comparison=comparison, outdir=outdir)
    return {"a_true": a_true, "x": x, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gaussian BP and precision-matrix scores.")
    parser.add_argument("--n", type=int, default=40, help="sequence length")
    parser.add_argument("--rho", type=float, default=0.7, help="AR(1) correlation parameter")
    parser.add_argument("--t", type=float, default=0.8, help="diffusion time, must be positive")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures_bp_ar1"),
        help="folder where figures and summary are written",
    )
    args = parser.parse_args()
    run_experiment(n=args.n, rho=args.rho, t=args.t, seed=args.seed, outdir=args.outdir)


if __name__ == "__main__":
    main()
