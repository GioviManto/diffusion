"""Experiment 06 -- EM through exact BP: does it recover the prior? (Layer 5)

This is the correctness layer for the learning story. Nothing here compares
learning methods; it establishes that maximizing the marginal likelihood of a
chain prior *through* belief propagation behaves exactly as the theory says.

Part 1 (monotonicity and recovery). Laplace-innovation chain, theta = (rho, b)
  deliberately mis-initialized, observed only through the OU channel. Check
  that the exact marginal log-likelihood increases at every EM iteration -- the
  defining guarantee, and the sharpest available test of the whole pipeline,
  since an error anywhere in the forward-backward recursion, the pairwise
  accumulation, or the M-step destroys it -- and that the iterates reach the
  truth from many random starts.

Part 2 (the price of noising). The same fit repeated at single noise levels.
  At t -> 0 the latent chain is effectively observed and EM degenerates to
  plain MLE; as t grows the OU channel destroys information. Fisher's identity
  gives the per-chain score grad_theta log p_t(x) from one BP pass, hence the
  observed information J_t = E[g g^T] and the Cramer-Rao yardstick, so the
  estimator is measured against its own information budget rather than an
  arbitrary reference. The Gaussian chain is used here because its kernel is
  smooth: the Laplace kernel's rho-derivative carries a sign discontinuity that
  would contaminate J_t with quadrature error (Part 5).

Part 3 (misspecification -- the case the project actually cares about). The
  generative innovation is Laplace; the fitted kernel is a C-component mixture
  that has never heard of Laplace. "We know the data is Markov plus noise, but
  not the model behind it." We track the recovered innovation law -- variance
  and excess kurtosis, the project's non-Gaussianity knob -- against C and
  against the number of observed chains, plus the denoiser error the fitted
  kernel induces at every noise level.

Part 4 (rate). Parameter error versus number of observed chains for the two
  smooth kernels, confirming the parametric N^{-1/2} scaling that the whole
  efficiency argument rests on.

Part 5 (a discretization caveat, honestly reported). The Laplace M-step is
  exact for the *discretized* model, and that exact solution is quantized: the
  minimizer of a weighted sum of |u_k - rho u_j| sits at one of the ratios
  u_k / u_j, and on a uniform grid through the origin those ratios pile up on
  low-denominator rationals. We measure the resulting lattice snapping against
  grid size and against the true rho. Smooth kernels -- Gaussian and mixture,
  whose M-steps are ratios of smooth moments -- are free of it, which is why
  Parts 2-4 rely on them.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import make_grid
from src.denoiser import bp_posterior_mean, evaluate_denoiser
from src.em import e_step_multi, fit_em, q_gradient
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel, MixtureInnovationKernel
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 32
RHO_TRUE = 0.8
GRID_A = 8.0
GRID_M = 401
T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)


def noisy_groups(A: np.ndarray, t_values, rng: np.random.Generator):
    """Split chains evenly across noise levels; one noise draw per chain.

    Deliberately stingy: each clean chain is observed exactly once, at exactly
    one noise level, and never seen clean. This is strictly less information
    than the score-matching baseline of exp_07 receives.
    """
    parts = np.array_split(rng.permutation(len(A)), len(t_values))
    groups = []
    for t, idx in zip(t_values, parts):
        alpha, delta = alpha_delta(t)
        sub = A[idx]
        X = alpha * sub + np.sqrt(delta) * rng.standard_normal(sub.shape)
        groups.append((X, alpha, delta))
    return groups


# ----------------------------------------------------------------------------
# Part 1: monotonicity, recovery, initialization robustness
# ----------------------------------------------------------------------------

def part1_monotonicity(grid, weights, n_chains: int, n_inits: int, out):
    prior = LaplaceAR1(RHO_TRUE)
    rng = rng_for("exp06-p1", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = noisy_groups(A, T_TRAIN, rng)

    rows, traces = [], []
    init_rng = rng_for("exp06-p1-init")
    for k in range(n_inits):
        rho0 = float(init_rng.uniform(-0.5, 0.95))
        b0 = float(init_rng.uniform(0.1, 1.5))
        fitted, trace = fit_em(
            LaplaceAR1Kernel(rho0, b0), grid, weights, groups, n_iters=80
        )
        traces.append(trace)
        rows.append({
            "init": k,
            "rho_init": rho0,
            "b_init": b0,
            "rho_hat": fitted.rho,
            "b_hat": fitted.b,
            "rho_true": RHO_TRUE,
            "b_true": prior.b,
            "rho_err": abs(fitted.rho - RHO_TRUE),
            "b_err": abs(fitted.b - prior.b),
            "logL_start": trace.log_evidence[0],
            "logL_end": trace.log_evidence[-1],
            "monotone_violation": trace.monotone_violation,
            "iters": len(trace.log_evidence),
            "sec_per_iter": float(np.mean(trace.seconds)),
        })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    for tr in traces:
        ax[0].plot(tr.log_evidence, lw=1.2, alpha=0.85)
        th = np.array(tr.theta)
        ax[1].plot(th[:, 0], th[:, 1], lw=1.0, alpha=0.85, marker=".", ms=3)
    ax[0].set_xlabel("EM iteration")
    ax[0].set_ylabel(r"exact $\log p_t(x)$")
    ax[0].set_title(f"Marginal log-likelihood, {n_inits} random inits")
    ax[1].plot(RHO_TRUE, prior.b, "k*", ms=14, label="truth")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$b$")
    ax[1].set_title("Parameter trajectories")
    ax[1].legend()
    save_figure(fig, out / "em_monotonicity.png")
    return rows


# ----------------------------------------------------------------------------
# Part 2: the statistical price of noising, against the Fisher information
# ----------------------------------------------------------------------------

def part2_price_of_noise(grid, weights, n_chains, t_values, n_rep, n_info, out):
    q_true = 1.0 - RHO_TRUE**2
    truth = GaussianAR1Kernel(RHO_TRUE, q_true)
    prior = GaussianAR1(RHO_TRUE)
    log_k_true = truth.log_transition_matrix(grid)
    grad_true = truth.grad_log_transition_matrix(grid)

    rows = []
    for t in t_values:
        alpha, delta = alpha_delta(t)

        # Observed information at the truth, from an independent sample. One BP
        # pass per chain gives the exact per-chain score by Fisher's identity.
        rng_i = rng_for("exp06-p2-info", t)
        A_i = np.stack([prior.sample(rng_i, N_SITES) for _ in range(n_info)])
        X_i = alpha * A_i + np.sqrt(delta) * rng_i.standard_normal(A_i.shape)
        g = np.stack([
            q_gradient(
                e_step_multi(grid, weights, log_k_true, [(x[None, :], alpha, delta)]),
                grad_true,
            )
            for x in X_i
        ])
        j_mat = g.T @ g / n_info
        try:
            crlb = np.sqrt(np.diag(np.linalg.inv(j_mat)) / n_chains)
        except np.linalg.LinAlgError:
            crlb = np.full(2, np.nan)

        errs = []
        for r in range(n_rep):
            rng = rng_for("exp06-p2", t, r)
            A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
            X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
            fitted, _ = fit_em(
                GaussianAR1Kernel(0.2, 0.8), grid, weights,
                [(X, alpha, delta)], n_iters=120,
            )
            errs.append([fitted.rho - RHO_TRUE, fitted.q - q_true])
        errs = np.array(errs)
        rows.append({
            "t": t,
            "alpha": alpha,
            "delta": delta,
            "n_chains": n_chains,
            "n_rep": n_rep,
            "rho_rmse": float(np.sqrt(np.mean(errs[:, 0] ** 2))),
            "rho_std": float(np.std(errs[:, 0], ddof=1)),
            "rho_bias": float(np.mean(errs[:, 0])),
            "q_rmse": float(np.sqrt(np.mean(errs[:, 1] ** 2))),
            "q_std": float(np.std(errs[:, 1], ddof=1)),
            "q_bias": float(np.mean(errs[:, 1])),
            "rho_crlb": float(crlb[0]),
            "q_crlb": float(crlb[1]),
            "fisher_rho": float(j_mat[0, 0]),
            "fisher_q": float(j_mat[1, 1]),
            "fisher_logdet": float(np.linalg.slogdet(j_mat)[1]),
        })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    ts = [r["t"] for r in rows]
    ax[0].loglog(ts, [r["rho_std"] for r in rows], "o-", label=r"realized sd $\hat\rho$")
    ax[0].loglog(ts, [r["rho_crlb"] for r in rows], "s--", label="Cramer-Rao")
    ax[0].loglog(ts, [abs(r["rho_bias"]) for r in rows], "^:", label=r"$|$bias$|$")
    ax[0].set_xlabel("noise level $t$")
    ax[0].set_ylabel(r"error in $\rho$")
    ax[0].set_title(f"Estimator vs information budget ($N={n_chains}$)")
    ax[0].legend()
    ax[1].loglog(ts, [r["fisher_rho"] for r in rows], "o-", label=r"$J_t[\rho,\rho]$")
    ax[1].loglog(ts, [r["fisher_q"] for r in rows], "s-", label=r"$J_t[q,q]$")
    ax[1].set_xlabel("noise level $t$")
    ax[1].set_ylabel("Fisher information per chain")
    ax[1].set_title("Information destroyed by the OU channel")
    ax[1].legend()
    save_figure(fig, out / "price_of_noising.png")
    return rows


# ----------------------------------------------------------------------------
# Part 3: misspecified (mixture) kernel on Laplace data
# ----------------------------------------------------------------------------

def part3_misspecified(grid, weights, n_chains, n_components, t_eval, sizes, out):
    prior = LaplaceAR1(RHO_TRUE)
    rng_test = rng_for("exp06-p3-test")
    A_test = np.stack([prior.sample(rng_test, N_SITES) for _ in range(256)])
    X_test = {}
    m_ref = {}
    for t in t_eval:
        alpha, delta = alpha_delta(t)
        X_test[t] = alpha * A_test + np.sqrt(delta) * rng_test.standard_normal(
            A_test.shape
        )
        m_ref[t] = bp_posterior_mean(prior, grid, weights, X_test[t], t)

    rng = rng_for("exp06-p3", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = noisy_groups(A, T_TRAIN, rng)

    rows, fitted_kernels = [], {}
    for c in n_components:
        k0 = MixtureInnovationKernel.init(
            c, rho=0.3, var=0.8, rng=rng_for("exp06-p3-init", c)
        )
        fitted, trace = fit_em(k0, grid, weights, groups, n_iters=120)
        fitted_kernels[c] = fitted
        mom = fitted.innovation_moments
        for t in t_eval:
            ev = evaluate_denoiser(
                bp_posterior_mean(fitted, grid, weights, X_test[t], t),
                m_ref[t], X_test[t], t,
            )
            rows.append({
                "n_components": c, "n_chains": n_chains, "t": t,
                "rho_hat": fitted.rho, **mom,
                "logL_end": trace.log_evidence[-1],
                "monotone_violation": trace.monotone_violation,
                **ev,
            })

    # How much data does the heavy tail need? Fixed C, growing N.
    c_fix = max(n_components)
    rate_rows = []
    for n in sizes:
        rng_n = rng_for("exp06-p3-size", n)
        A_n = np.stack([prior.sample(rng_n, N_SITES) for _ in range(n)])
        fitted, trace = fit_em(
            MixtureInnovationKernel.init(
                c_fix, rho=0.3, var=0.8, rng=rng_for("exp06-p3-init", c_fix)
            ),
            grid, weights, noisy_groups(A_n, T_TRAIN, rng_n), n_iters=120,
        )
        ev = evaluate_denoiser(
            bp_posterior_mean(fitted, grid, weights, X_test[t_eval[0]], t_eval[0]),
            m_ref[t_eval[0]], X_test[t_eval[0]], t_eval[0],
        )
        rate_rows.append({
            "n_chains": n, "n_components": c_fix, "rho_hat": fitted.rho,
            **fitted.innovation_moments,
            "monotone_violation": trace.monotone_violation,
            "score_rel_l2_at_t0": ev["score_rel_l2"],
        })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    e = np.linspace(-3.0, 3.0, 601)
    ax[0].plot(e, np.exp(-np.abs(e) / prior.b) / (2 * prior.b),
               "k-", lw=2, label="true Laplace innovation")
    for c, ker in fitted_kernels.items():
        dens = sum(
            p * np.exp(-0.5 * (e - m) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
            for p, m, s2 in zip(ker.pi, ker.mu, ker.s2)
        )
        ax[0].plot(e, dens, lw=1.3, label=f"EM mixture, $C={c}$")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 3.0)
    ax[0].set_xlabel(r"innovation $\varepsilon$")
    ax[0].set_ylabel("density")
    ax[0].set_title(f"Innovation law recovered from noisy data\n($N={n_chains}$ chains, never seen clean)")
    ax[0].legend()

    ns = [r["n_chains"] for r in rate_rows]
    ax[1].semilogx(ns, [r["innovation_excess_kurtosis"] for r in rate_rows], "o-")
    ax[1].axhline(3.0, color="k", ls="--", lw=1, label="true excess kurtosis")
    ax[1].set_xlabel("number of observed noisy chains $N$")
    ax[1].set_ylabel("recovered excess kurtosis")
    ax[1].set_title(f"Heavy tail emerges with data ($C={c_fix}$)")
    ax[1].legend()
    save_figure(fig, out / "recovered_innovation.png")
    return rows, rate_rows


# ----------------------------------------------------------------------------
# Part 4: N^{-1/2} rate for the smooth kernels
# ----------------------------------------------------------------------------

def part4_rate(grid, weights, sizes, n_rep, out):
    rows = []
    gauss_prior = GaussianAR1(RHO_TRUE)
    lap_prior = LaplaceAR1(RHO_TRUE)
    q_true = 1.0 - RHO_TRUE**2

    for n_chains in sizes:
        g_err, m_err = [], []
        for r in range(n_rep):
            rng = rng_for("exp06-p4-g", n_chains, r)
            A = np.stack([gauss_prior.sample(rng, N_SITES) for _ in range(n_chains)])
            fit_g, _ = fit_em(
                GaussianAR1Kernel(0.3, 0.8), grid, weights,
                noisy_groups(A, T_TRAIN, rng), n_iters=120,
            )
            g_err.append([fit_g.rho - RHO_TRUE, fit_g.q - q_true])

            rng = rng_for("exp06-p4-m", n_chains, r)
            A = np.stack([lap_prior.sample(rng, N_SITES) for _ in range(n_chains)])
            fit_m, _ = fit_em(
                MixtureInnovationKernel.init(
                    4, rho=0.3, var=0.8, rng=rng_for("exp06-p4-init")
                ),
                grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=120,
            )
            m_err.append([
                fit_m.rho - RHO_TRUE,
                fit_m.innovation_moments["innovation_var"] - q_true,
            ])
        g_err, m_err = np.array(g_err), np.array(m_err)
        rows.append({
            "n_chains": n_chains,
            "n_edges": n_chains * (N_SITES - 1),
            "gauss_rho_rmse": float(np.sqrt(np.mean(g_err[:, 0] ** 2))),
            "gauss_q_rmse": float(np.sqrt(np.mean(g_err[:, 1] ** 2))),
            "mixture_rho_rmse": float(np.sqrt(np.mean(m_err[:, 0] ** 2))),
            "mixture_var_rmse": float(np.sqrt(np.mean(m_err[:, 1] ** 2))),
        })

    fig, ax = new_figure()
    ns = np.array([r["n_chains"] for r in rows], dtype=float)
    for key, style, lbl in (
        ("gauss_rho_rmse", "o-", r"Gaussian kernel, $\rho$"),
        ("gauss_q_rmse", "s-", r"Gaussian kernel, $q$"),
        ("mixture_rho_rmse", "^-", r"mixture kernel, $\rho$"),
        ("mixture_var_rmse", "v-", r"mixture kernel, innovation var"),
    ):
        ax.loglog(ns, [r[key] for r in rows], style, label=lbl)
    ax.loglog(ns, rows[0]["gauss_rho_rmse"] * np.sqrt(ns[0] / ns), "k--", lw=1,
              label=r"$N^{-1/2}$")
    ax.set_xlabel("number of observed noisy chains $N$")
    ax.set_ylabel("RMSE")
    ax.set_title("Parametric rate of the EM estimator")
    ax.legend()
    save_figure(fig, out / "em_rate.png")
    return rows


# ----------------------------------------------------------------------------
# Part 5: the Laplace M-step is exact but lattice-quantized
# ----------------------------------------------------------------------------

def part5_quantization(grid_sizes, rho_values, n_chains, out):
    """Snapping of the Laplace rho-estimate onto the grid's ratio lattice.

    The exact M-step minimizes sum Xi[k,j] |u_k - rho u_j| over rho, a convex
    piecewise-linear function whose minimizer must be a breakpoint, i.e. one of
    the ratios u_k / u_j. On a uniform grid through the origin those ratios are
    rationals m / l, and a low-denominator value such as 4/5 has many aliases
    (8/10, 12/15, ...) that pool weight onto it. So rho_hat lands on simple
    rationals; which one it lands on depends on the data, so a truth that
    happens to sit on the lattice is sometimes recovered to machine precision
    and sometimes not, at the same grid size.

    Reported as a limitation of *this kernel on this grid*, not of EM: the
    smooth kernels' M-steps are ratios of moments and vary continuously.

    The seed deliberately excludes the grid size. An earlier version keyed on
    it, which made every grid draw a different dataset and confounded the
    across-M trend with resampling -- the one thing this part exists to
    measure. Now the data is fixed per rho_true and the grid is the only thing
    that varies down each column.
    """
    rows = []
    for rho_true in rho_values:
        prior = LaplaceAR1(rho_true)
        rng_data = rng_for("exp06-p5", rho_true)
        A = np.stack([prior.sample(rng_data, N_SITES) for _ in range(n_chains)])
        groups = noisy_groups(A, T_TRAIN, rng_data)
        for m_size in grid_sizes:
            grid, weights = make_grid(GRID_A, m_size)
            fitted, trace = fit_em(
                LaplaceAR1Kernel(0.3, 0.8), grid, weights, groups, n_iters=80,
            )
            rows.append({
                "rho_true": rho_true,
                "grid_size": m_size,
                "dx": float(grid[1] - grid[0]),
                "rho_hat": fitted.rho,
                "rho_err": abs(fitted.rho - rho_true),
                "b_hat": fitted.b,
                "b_err": abs(fitted.b - prior.b),
                "monotone_violation": trace.monotone_violation,
            })

    fig, ax = new_figure()
    for rho_true in rho_values:
        sub = [r for r in rows if r["rho_true"] == rho_true]
        ax.loglog([r["grid_size"] for r in sub], [r["rho_err"] + 1e-17 for r in sub],
                  "o-", label=rf"$\rho^*={rho_true}$")
        ax.loglog([r["grid_size"] for r in sub], [r["b_err"] for r in sub],
                  "s--", alpha=0.5, label=rf"$b$ error, $\rho^*={rho_true}$")
    ax.set_xlabel("grid size $M$")
    ax.set_ylabel("absolute error")
    ax.set_title("Laplace M-step: lattice snapping in $\\rho$, none in $b$")
    ax.legend(fontsize=8)
    save_figure(fig, out / "laplace_quantization.png")
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_06_em_parameter_recovery",
        "EM through exact BP: monotonicity, recovery, information, rate.",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 201, "n_chains": 128, "n_inits": 3,
        "t_price": (0.1, 0.4), "n_rep": 3, "n_info": 100, "n_chains_price": 256,
        "components": (2, 4), "t_eval": (0.2, 0.8),
        "sizes_kurtosis": (128, 512), "sizes_rate": (64, 256), "n_rep_rate": 2,
        "quantization_grids": (201, 401), "quantization_rhos": (0.8, 0.77),
        "n_chains_quantization": 128,
    }
    full = {
        "grid_size": GRID_M, "n_chains": 1024, "n_inits": 6,
        "t_price": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6), "n_rep": 10, "n_info": 400,
        "n_chains_price": 256,
        "components": (2, 3, 5, 8), "t_eval": (0.1, 0.2, 0.4, 0.8, 1.6),
        "sizes_kurtosis": (128, 256, 512, 1024, 2048),
        "sizes_rate": (64, 128, 256, 512, 1024), "n_rep_rate": 4,
        "quantization_grids": (201, 401, 801, 1601),
        "quantization_rhos": (0.8, 0.77, 0.813), "n_chains_quantization": 256,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    def p1(grid, weights, out):
        write_csv(out / "monotonicity.csv", part1_monotonicity(
            grid, weights, cfg["n_chains"], cfg["n_inits"], out))

    def p2(grid, weights, out):
        write_csv(out / "price_of_noising.csv", part2_price_of_noise(
            grid, weights, cfg["n_chains_price"], cfg["t_price"],
            cfg["n_rep"], cfg["n_info"], out))

    def p3(grid, weights, out):
        mis_rows, kurt_rows = part3_misspecified(
            grid, weights, cfg["n_chains"], cfg["components"],
            cfg["t_eval"], cfg["sizes_kurtosis"], out)
        write_csv(out / "misspecified_mixture.csv", mis_rows)
        write_csv(out / "kurtosis_vs_n.csv", kurt_rows)

    def p4(grid, weights, out):
        write_csv(out / "em_rate.csv", part4_rate(
            grid, weights, cfg["sizes_rate"], cfg["n_rep_rate"], out))

    def p5(grid, weights, out):
        write_csv(out / "laplace_quantization.csv", part5_quantization(
            cfg["quantization_grids"], cfg["quantization_rhos"],
            cfg["n_chains_quantization"], out))

    parts = {
        "monotonicity": ("monotonicity and recovery", p1),
        "price_of_noising": ("price of noising vs Fisher information", p2),
        "misspecified": ("misspecified mixture kernel", p3),
        "rate": ("sample-size rate", p4),
        "quantization": ("Laplace lattice quantization", p5),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, cfg["grid_size"])

    # One params file per selection, so parallel array tasks do not race for
    # the same filename and each output stays traceable to its own invocation.
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "grid_half_width": GRID_A,
        "t_train": T_TRAIN, "quick": args.quick, "parts": list(selected),
        "overrides": args.set, **cfg, **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(grid, weights, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
