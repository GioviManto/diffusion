"""Experiment 07 -- Learning the denoiser: EM through BP vs a score network.

The claim under test is Marc's: that regressing a handful of BP parameters by
expectation maximization learns the denoiser far more efficiently than training
a network to represent it. "Efficiently" is not one number, so we measure four
and let them disagree where they do.

The two estimators target the *same* object, the posterior mean
m_i(x, t) = E[a_i | x], which fixes the score through the exact OU identity.

  EM-BP     learns the transition kernel K_theta of the clean chain by exact
            EM, then computes m by belief propagation. The learned object lives
            on R x R and carries no dependence on t whatsoever.
  DSM net   learns (x, t) -> m directly by denoising score matching, the
            vanilla diffusion recipe, with no knowledge that the chain is
            Markov.

Fairness. The comparison is set up to favour the network wherever there was a
choice:
  - Data budget is the number of *clean chains* N. The network trains on paired
    (a, x) with a fresh noise draw at every gradient step, so it may consume
    unlimited noise realizations of those N chains. EM sees one noisy
    realization of each chain, at one noise level, and never sees a clean chain.
  - Both are evaluated on the same held-out chains against the same reference:
    exact grid BP under the true prior.
  - The network's objective is the one whose minimizer is exactly the target,
    so it is not handicapped by a surrogate loss.

Part 1 (sample efficiency). Test error vs N, the headline curve.
Part 2 (capacity). Test error vs network size at fixed N, to check the network
  is not merely undertrained or too small -- the comparison must not hinge on a
  badly chosen architecture.
Part 3 (transfer across the noise schedule). Both methods evaluated at noise
  levels *outside* the training schedule. EM-BP has no schedule to leave: its
  parameters are properties of the clean chain, and BP supplies every t exactly.
  The network must extrapolate in t.
Part 4 (cost accounting, honestly). Training seconds, parameter counts, and
  inference seconds per chain. The last one is where BP loses, and it is
  reported as prominently as the wins.
"""

from __future__ import annotations

import time

import numpy as np

from common import experiment_parser, provenance
from src.bp_grid import make_grid
from src.denoiser import (
    bp_posterior_mean,
    dsm_posterior_mean,
    evaluate_denoiser,
    train_dsm_denoiser,
)
from src.em import fit_em
from src.kernels import MixtureInnovationKernel
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 32
RHO_TRUE = 0.8
GRID_A = 8.0
GRID_M = 401
T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)
N_COMPONENTS = 4
N_TEST = 256


def make_test_set(prior, grid, weights, t_values, n_test: int):
    """Held-out chains plus the exact BP reference denoiser at each level."""
    rng = rng_for("exp07-test")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_test)])
    bundle = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        bundle[t] = (X, bp_posterior_mean(prior, grid, weights, X, t))
    return A, bundle


def noisy_groups(A: np.ndarray, t_values, rng: np.random.Generator):
    """One noise draw per chain, chains split evenly across noise levels."""
    parts = np.array_split(rng.permutation(len(A)), len(t_values))
    groups = []
    for t, idx in zip(t_values, parts):
        alpha, delta = alpha_delta(t)
        sub = A[idx]
        groups.append(
            (alpha * sub + np.sqrt(delta) * rng.standard_normal(sub.shape), alpha, delta)
        )
    return groups


def score_both(kernel, net, grid, weights, bundle, t_values, tag: dict):
    """Evaluate both denoisers on the shared test bundle."""
    rows = []
    for t in t_values:
        X, m_ref = bundle[t]
        rows.append({**tag, "method": "em_bp", "t": t,
                     **evaluate_denoiser(
                         bp_posterior_mean(kernel, grid, weights, X, t),
                         m_ref, X, t)})
        rows.append({**tag, "method": "dsm_net", "t": t,
                     **evaluate_denoiser(
                         dsm_posterior_mean(net, X, t), m_ref, X, t)})
    return rows


# ----------------------------------------------------------------------------
# Part 1: sample efficiency
# ----------------------------------------------------------------------------

def part1_sample_efficiency(grid, weights, sizes, hidden, n_steps, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, T_TRAIN, N_TEST)
    rows, cost_rows = [], []

    for n_chains in sizes:
        rng = rng_for("exp07-p1", n_chains)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

        t0 = time.perf_counter()
        kernel, trace = fit_em(
            MixtureInnovationKernel.init(
                N_COMPONENTS, rho=0.3, var=0.8, rng=rng_for("exp07-init")
            ),
            grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=120,
        )
        em_seconds = time.perf_counter() - t0

        dsm = train_dsm_denoiser(
            A, T_TRAIN, rng_for("exp07-net", n_chains),
            hidden=hidden, n_steps=n_steps,
        )

        tag = {"n_chains": n_chains}
        rows += score_both(kernel, dsm.net, grid, weights, bundle, T_TRAIN, tag)
        cost_rows.append({
            "n_chains": n_chains,
            "em_seconds": em_seconds,
            "em_iters": len(trace.log_evidence),
            "em_n_params": int(len(kernel.theta)),
            "em_monotone_violation": trace.monotone_violation,
            "net_seconds": dsm.seconds,
            "net_n_params": dsm.n_params,
            "net_grad_steps": dsm.n_grad_steps,
            "net_final_dsm_loss": dsm.loss_history[-1],
        })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    for method, style in (("em_bp", "o-"), ("dsm_net", "s-")):
        for j, key in enumerate(("score_rel_l2", "mean_rel_l2")):
            agg = []
            for n in sizes:
                vals = [r[key] for r in rows
                        if r["method"] == method and r["n_chains"] == n]
                agg.append(float(np.mean(vals)))
            ax[j].loglog(sizes, agg, style, label=method)
    for j, ttl in enumerate(("relative score error", "relative posterior-mean error")):
        ax[j].set_xlabel("number of training chains $N$")
        ax[j].set_ylabel(ttl)
        ax[j].set_title(f"{ttl}, averaged over $t$")
        ax[j].legend()
    save_figure(fig, out / "sample_efficiency.png")
    return rows, cost_rows


# ----------------------------------------------------------------------------
# Part 2: is the network merely too small or undertrained?
# ----------------------------------------------------------------------------

def part2_capacity(grid, weights, n_chains, archs, step_counts, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, T_TRAIN, N_TEST)
    rng = rng_for("exp07-p1", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

    kernel, _ = fit_em(
        MixtureInnovationKernel.init(
            N_COMPONENTS, rho=0.3, var=0.8, rng=rng_for("exp07-init")
        ),
        grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=120,
    )
    em_err = float(np.mean([
        evaluate_denoiser(
            bp_posterior_mean(kernel, grid, weights, bundle[t][0], t),
            bundle[t][1], bundle[t][0], t,
        )["score_rel_l2"]
        for t in T_TRAIN
    ]))

    rows = []
    for hidden in archs:
        for n_steps in step_counts:
            dsm = train_dsm_denoiser(
                A, T_TRAIN, rng_for("exp07-p2", str(hidden), n_steps),
                hidden=hidden, n_steps=n_steps,
            )
            err = float(np.mean([
                evaluate_denoiser(
                    dsm_posterior_mean(dsm.net, bundle[t][0], t),
                    bundle[t][1], bundle[t][0], t,
                )["score_rel_l2"]
                for t in T_TRAIN
            ]))
            rows.append({
                "n_chains": n_chains, "hidden": str(hidden), "n_steps": n_steps,
                "net_n_params": dsm.n_params, "net_seconds": dsm.seconds,
                "net_score_rel_l2": err,
                "em_score_rel_l2": em_err,
                "em_n_params": int(len(kernel.theta)),
                "ratio_net_over_em": err / em_err,
            })

    fig, ax = new_figure()
    for hidden in archs:
        sub = [r for r in rows if r["hidden"] == str(hidden)]
        ax.loglog([r["n_steps"] for r in sub], [r["net_score_rel_l2"] for r in sub],
                  "o-", label=f"net {hidden}, {sub[0]['net_n_params']} params")
    ax.axhline(em_err, color="k", ls="--", lw=1.5,
               label=f"EM-BP ({len(kernel.theta)} params)")
    ax.set_xlabel("gradient steps")
    ax.set_ylabel("relative score error")
    ax.set_title(f"Network capacity and training budget ($N={n_chains}$)")
    ax.legend(fontsize=8)
    save_figure(fig, out / "capacity.png")
    return rows


# ----------------------------------------------------------------------------
# Part 3: transfer to unseen noise levels
# ----------------------------------------------------------------------------

def part3_transfer(grid, weights, n_chains, t_probe, hidden, n_steps, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, t_probe, N_TEST)
    rng = rng_for("exp07-p1", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

    kernel, _ = fit_em(
        MixtureInnovationKernel.init(
            N_COMPONENTS, rho=0.3, var=0.8, rng=rng_for("exp07-init")
        ),
        grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=120,
    )
    dsm = train_dsm_denoiser(
        A, T_TRAIN, rng_for("exp07-net", n_chains), hidden=hidden, n_steps=n_steps
    )

    rows = score_both(kernel, dsm.net, grid, weights, bundle, t_probe,
                      {"n_chains": n_chains})
    for r in rows:
        r["in_training_schedule"] = bool(
            any(abs(r["t"] - t) < 1e-12 for t in T_TRAIN)
        )

    fig, ax = new_figure()
    for method, style in (("em_bp", "o-"), ("dsm_net", "s-")):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["t"])
        ax.loglog([r["t"] for r in sub], [r["score_rel_l2"] for r in sub],
                  style, label=method)
    for t in T_TRAIN:
        ax.axvline(t, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("noise level $t$  (dotted = training schedule)")
    ax.set_ylabel("relative score error")
    ax.set_title(f"Transfer across the noise schedule ($N={n_chains}$)")
    ax.legend()
    save_figure(fig, out / "transfer.png")
    return rows


# ----------------------------------------------------------------------------
# Part 4: inference cost -- where BP loses
# ----------------------------------------------------------------------------

def part4_inference_cost(grid, weights, batch_sizes, hidden, out):
    prior = LaplaceAR1(RHO_TRUE)
    rng = rng_for("exp07-p4")
    kernel = MixtureInnovationKernel.init(
        N_COMPONENTS, rho=0.8, var=0.36, rng=rng_for("exp07-init")
    )
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(max(batch_sizes))])
    dsm = train_dsm_denoiser(A[:64], T_TRAIN, rng, hidden=hidden, n_steps=50)

    rows = []
    for b in batch_sizes:
        alpha, delta = alpha_delta(0.4)
        X = alpha * A[:b] + np.sqrt(delta) * rng.standard_normal((b, N_SITES))
        t0 = time.perf_counter()
        bp_posterior_mean(kernel, grid, weights, X, 0.4)
        bp_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        dsm_posterior_mean(dsm.net, X, 0.4)
        net_sec = time.perf_counter() - t0
        rows.append({
            "batch": b, "grid_size": len(grid),
            "bp_seconds": bp_sec, "net_seconds": net_sec,
            "bp_per_chain_ms": 1e3 * bp_sec / b,
            "net_per_chain_ms": 1e3 * net_sec / b,
            "slowdown": bp_sec / net_sec,
        })
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_07_em_vs_score_network",
        "EM-learned BP denoiser vs a denoising-score-matching network.",
    )
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, 201 if args.quick else GRID_M)

    if args.quick:
        sizes = (64, 256)
        hidden, n_steps = (64, 64), 1500
        archs, step_counts = ((32, 32), (128, 128)), (1000, 4000)
        cap_n = 256
        t_probe = (0.05, 0.2, 0.8, 3.0)
        batches = (32, 128)
    else:
        sizes = (32, 64, 128, 256, 512, 1024, 2048)
        hidden, n_steps = (128, 128), 20000
        archs = ((32, 32), (128, 128), (256, 256), (512, 512))
        step_counts = (1000, 5000, 20000, 50000)
        cap_n = 1024
        t_probe = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2)
        batches = (32, 128, 512)

    write_json(out / "params.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "grid_size": len(grid),
        "grid_half_width": GRID_A, "t_train": T_TRAIN,
        "n_components": N_COMPONENTS, "n_test": N_TEST, "sizes": sizes,
        "net_hidden": hidden, "net_steps": n_steps, "archs": [str(a) for a in archs],
        "step_counts": step_counts, "capacity_n": cap_n, "t_probe": t_probe,
        "inference_batches": batches, "quick": args.quick, **provenance(),
    })

    print("Part 1: sample efficiency ...", flush=True)
    rows, cost_rows = part1_sample_efficiency(
        grid, weights, sizes, hidden, n_steps, out
    )
    write_csv(out / "sample_efficiency.csv", rows)
    write_csv(out / "training_cost.csv", cost_rows)
    print("Part 2: network capacity and training budget ...", flush=True)
    write_csv(out / "capacity.csv",
              part2_capacity(grid, weights, cap_n, archs, step_counts, out))
    print("Part 3: transfer across the noise schedule ...", flush=True)
    write_csv(out / "transfer.csv",
              part3_transfer(grid, weights, cap_n, t_probe, hidden, n_steps, out))
    print("Part 4: inference cost ...", flush=True)
    write_csv(out / "inference_cost.csv",
              part4_inference_cost(grid, weights, batches, hidden, out))
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
