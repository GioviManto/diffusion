"""Experiment 16 -- does the score advantage survive into the generated samples?

The question, and why it is the one that matters
------------------------------------------------
Layer 5 established that EM through exact BP learns a *score* far more efficiently than a
trained network: ~10x lower relative score error against a vanilla MLP at matched data,
~4x against a locality-respecting CNN. Every one of those numbers is **pointwise** -- given
the same noisy x, how far is one denoiser from another.

Jerome's objection in the call of 29 July is that this is not what diffusion is judged on:

    "what's important with diffusion is that the error on the score, the relationship
     between the score and the distribution of data that you end up generating at the end
     of the diffusion process, is not trivial, because you accumulate the errors of the
     scores, you go backwards, and not all times are equal. So in particular, if you
     accumulate errors at intermediate times, which is what seems to be happening, you may,
     in terms of sampling, have something that's very wrong."

And he gave the reason a pointwise metric cannot settle it: at small t the denoiser barely
acts (alpha_t -> 1, "the denoiser is an identity matrix in principle"), at large t noise
dominates whatever it does, so the generated law is governed by the intermediate times. An
error averaged over the schedule sees none of that structure.

There is a concrete reason to expect the pointwise ranking *not* to transfer. exp_03 and the
cluster sweep both find the Gaussian-closure gap concentrated at **small** t -- 0.20 at
t=0.08 falling to 9e-5 at t=2.4 for the Laplace chain. That is precisely the regime Jerome
argues is cheap. So a method could look much worse pointwise and cost almost nothing in
samples.

Equally, there is a concrete reason to expect real damage: exp_05 ran the reverse SDE under
the Gaussian score on this same chain and found it reproduces second-order statistics while
flattening the innovation excess kurtosis to **0.12 against a true 2.7-2.9**. Covariance
preserved, shape destroyed.

This experiment settles which happens for the *learned* scores. It is the missing link
between "EM-BP learns a better score" and Marc's publishable claim, "we can do very efficient
learning of the denoiser using expectation maximization relative to a vanilla neural
network".

Design
------
One prior (Laplace AR(1), rho=0.85, n=32 -- the exp_07 setting, so the pointwise numbers are
directly comparable). Four score functions, all evaluated inside the *same* reverse
integrator with *common random numbers*:

    exact    grid BP under the true prior            -- the reference. Not a competitor:
                                                        this is the true score, so its
                                                        samples define the target.
    em_bp    grid BP under an EM-learned kernel      -- 13 parameters, fitted from N noisy
                                                        chains, never shown a clean sample.
    mlp      vanilla DSM network                     -- ~25k parameters, same N.
    cnn      weight-shared local head                -- ~6k parameters, same N.

Both networks are trained in both standard parameterizations and the better is taken per
noise level, which is the treatment exp_07 settled on: eps wins at low noise, x0 at high, and
reporting one alone misleads in whichever direction is chosen.

What is measured, and why all of it
-----------------------------------
`src/sample_metrics.py` carries the argument in full. In short, four families:

  1. MSE(m_hat, m*)  -- pure method error, the primary pointwise metric.
  2. MSE(m_hat, a)   -- the denoising risk, against its Bayes floor.
  3. relative score error -- kept for continuity, de-emphasised. It equals (1) reweighted by
     alpha_t/delta_t, which spans a factor of 65 across this schedule.
  4. distributional statistics of the generated samples -- innovation excess kurtosis
     against its bootstrap SE, histogram KL, covariance profile. The only family that sees
     accumulation.

Parts
-----
generate   The main comparison: generate from each arm at several N, measure the laws.
pointwise  The metric ladder at fixed noise levels, so the pointwise and distributional
           rankings can be put side by side and any disagreement between them attributed.
steps      Integrator control. Halving the step size must not change the ranking; if it
           does, we are measuring discretization rather than score quality.

The `steps` part is not optional garnish. A claim that arm A generates better samples than
arm B is worthless if the gap is an Euler-Maruyama artifact, and this is the cheapest
possible guard against reporting one.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance
from src.bp_grid import make_grid
from src.denoiser import bp_posterior_mean, dsm_posterior_mean, train_dsm_denoiser
from src.em import fit_em
from src.kernels import MixtureInnovationKernel
from src.local_head import local_posterior_mean, train_local_head
from src.priors import LaplaceAR1
from src.reverse import reverse_sde, time_grid
from src.sample_metrics import compare_distributions, pointwise_ladder
from src.utils import ensure_dir, rng_for, write_csv, write_json

# ---------------------------------------------------------------------------
# Settings. Overridable from the command line with --set, so a cluster run that
# scales anything stays self-describing through params.json.
# ---------------------------------------------------------------------------

SETTINGS = {
    # Data model. n=32 and rho=0.85 match exp_07 so the pointwise numbers line up.
    "n_sites": 32,
    "rho": 0.85,
    # Training budgets. The interesting region is the small end, where exp_07 finds
    # EM-BP on 32 chains beating the network on 4096.
    "sizes": (32, 128, 512, 2048),
    # Noise levels the networks are trained across, and probed at.
    "t_train": (0.1, 0.2, 0.4, 0.8, 1.6),
    "t_probe": (0.1, 0.2, 0.4, 0.8, 1.6),
    # Reverse integration. t_max large enough that p_T is effectively N(0,I);
    # t_min bounded away from 0 because the score diverges as delta -> 0.
    "t_max": 3.0,
    "t_min": 0.02,
    "n_steps": 200,
    "n_generate": 2000,
    # Grid BP resolution. M=401, A=8 is the calibrated default (exp_01: worst relative
    # error 9.2e-15); it is not a free parameter and should not be lowered to save time
    # without re-running that calibration.
    "grid_m": 401,
    "grid_a": 8.0,
    # Learning budgets.
    "em_iters": 40,
    "em_components": 4,
    "net_hidden": (128, 128),
    "net_steps": 6000,
    "cnn_hidden": (64, 64),
    "cnn_steps": 6000,
    "cnn_radius": 6,
    # Replication. Every headline number is a mean over seeds with a standard error;
    # single-seed structure has dissolved twice in this project.
    "n_seed": 4,
    # Shifts the seed range, so a cluster array can run one seed per task and still cover
    # distinct seeds: task k passes seed_offset=k with n_seed=1.
    "seed_offset": 0,
    # Integrator control.
    "steps_ladder": (100, 200, 400),
}

QUICK = {
    "sizes": (32, 128),
    "n_generate": 300,
    "n_steps": 60,
    "n_seed": 1,
    "em_iters": 8,
    "net_steps": 400,
    "cnn_steps": 400,
    "grid_m": 201,
    "steps_ladder": (60, 120),
}

PARTS = ("generate", "pointwise", "steps")


# ---------------------------------------------------------------------------
# Score functions: one builder per arm, all with the signature (X, t) -> score
# ---------------------------------------------------------------------------

def sample_chains(prior, rng: np.random.Generator, n_chains: int, n_sites: int) -> np.ndarray:
    """Draw a batch. `prior.sample` returns a single chain, so stack -- matching exp_07."""
    return np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])


def _score(X: np.ndarray, means: np.ndarray, t: float) -> np.ndarray:
    """Tweedie, applied to a batch. The single place the identity is used."""
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    return -(X - alpha * means) / delta


def make_exact_arm(prior, grid, weights):
    """The true score. Defines the target the other arms are judged against."""
    def fn(X, t):
        return _score(X, bp_posterior_mean(prior, grid, weights, X, float(t)), float(t))
    return fn


def make_em_arm(kernel, grid, weights):
    """Grid BP under the learned kernel -- structurally identical to `exact`.

    The *only* difference from the reference arm is which transition matrix goes in, which
    is what makes this comparison clean: any gap is attributable to the learned parameters
    and not to a different inference algorithm.
    """
    def fn(X, t):
        return _score(X, bp_posterior_mean(kernel, grid, weights, X, float(t)), float(t))
    return fn


def make_network_arm(res_eps, res_x0, better_at):
    """Oracle over the two parameterizations, chosen per noise level.

    `better_at` maps a probed t to "eps" or "x0". Giving the network the better of the two
    at every level is deliberately generous: exp_07 found eps wins at low noise and x0 at
    high, by large factors (at t=1.6, 1.19 against 0.156), so a fixed choice would make the
    baseline look bad for a reason unrelated to the question being asked.
    """
    def fn(X, t):
        t = float(t)
        res = res_eps if better_at(t) == "eps" else res_x0
        return _score(X, dsm_posterior_mean(res, X, t), t)
    return fn


def make_cnn_arm(res_eps, res_x0, better_at):
    def fn(X, t):
        t = float(t)
        res = res_eps if better_at(t) == "eps" else res_x0
        return _score(X, local_posterior_mean(res, X, t), t)
    return fn


# ---------------------------------------------------------------------------
# Fitting all four arms at one data budget
# ---------------------------------------------------------------------------

def fit_arms(prior, grid, weights, n_chains, seed, cfg):
    """Fit every learned arm on the *same* N noisy chains.

    The shared training set is the point: EM and the networks see identical data, so the
    comparison is about what each does with it rather than about sampling luck. Noise is
    drawn fresh per training example for the networks (the standard diffusion recipe) while
    EM sees one noisy realisation per chain -- that asymmetry favours the networks and is
    recorded rather than corrected, because correcting it would depart from how each method
    is normally used.
    """
    rng = rng_for("exp16-fit", seed, n_chains)
    A = sample_chains(prior, rng, n_chains, cfg["n_sites"])

    # --- EM arm: one noisy realisation per chain, at each training noise level.
    groups = []
    for t in cfg["t_train"]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        groups.append((X, alpha, delta))

    kernel, trace = fit_em(
        MixtureInnovationKernel.init(
            cfg["em_components"], rho=0.3, var=0.8,
            rng=rng_for("exp16-eminit", seed, n_chains),
        ),
        grid, weights, groups, n_iters=cfg["em_iters"],
    )

    # --- Network arms, both parameterizations.
    nets = {}
    for mode in ("eps", "x0"):
        nets[("mlp", mode)] = train_dsm_denoiser(
            A, cfg["t_train"], rng_for("exp16-mlp", seed, n_chains, mode),
            hidden=cfg["net_hidden"], n_steps=cfg["net_steps"], parameterization=mode,
        )
        nets[("cnn", mode)] = train_local_head(
            A, cfg["t_train"], cfg["cnn_radius"],
            rng_for("exp16-cnn", seed, n_chains, mode),
            hidden=cfg["cnn_hidden"], n_steps=cfg["cnn_steps"], parameterization=mode,
        )

    return kernel, nets, trace


def choose_parameterization(prior, grid, weights, nets, cfg, seed):
    """Pick eps or x0 per noise level, by held-out posterior-mean error.

    Selected on a *fresh* sample, not on the generated data, so the oracle cannot leak the
    comparison it is part of.
    """
    rng = rng_for("exp16-select", seed)
    A = sample_chains(prior, rng, 256, cfg["n_sites"])
    best = {}
    for t in cfg["t_probe"]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_star = bp_posterior_mean(prior, grid, weights, X, t)
        for family, predict in (("mlp", dsm_posterior_mean), ("cnn", local_posterior_mean)):
            errs = {
                mode: float(np.mean((predict(nets[(family, mode)], X, t) - m_star) ** 2))
                for mode in ("eps", "x0")
            }
            best[(family, t)] = min(errs, key=errs.get)
    return best


def _selector(best, family, t_probe):
    """Nearest-probed-level lookup, so the arm is defined at every integrator time."""
    def fn(t):
        nearest = min(t_probe, key=lambda u: abs(u - t))
        return best[(family, nearest)]
    return fn


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def run_generate(prior, grid, weights, cfg, out_dir):
    """Generate from every arm and measure the resulting laws."""
    rows = []
    times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps"])
    n = cfg["n_sites"]
    sigma_true = prior.rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            kernel, nets, trace = fit_arms(prior, grid, weights, n_chains, seed, cfg)
            best = choose_parameterization(prior, grid, weights, nets, cfg, seed)

            arms = {
                "exact": make_exact_arm(prior, grid, weights),
                "em_bp": make_em_arm(kernel, grid, weights),
                "mlp": make_network_arm(nets[("mlp", "eps")], nets[("mlp", "x0")],
                                        _selector(best, "mlp", cfg["t_probe"])),
                "cnn": make_cnn_arm(nets[("cnn", "eps")], nets[("cnn", "x0")],
                                    _selector(best, "cnn", cfg["t_probe"])),
            }

            # A reference sample from the forward model -- the yardstick. Never a
            # reverse-generated one, which would fold the integrator's error into the target.
            a_ref = sample_chains(prior, rng_for("exp16-ref", seed), cfg["n_generate"], n)

            for name, fn in arms.items():
                # Common random numbers: identical initial noise and identical Brownian
                # increments across arms, so differences are the score and nothing else.
                rng = rng_for("exp16-gen", seed, n_chains)
                x_init = rng.standard_normal((cfg["n_generate"], n))
                a_gen = reverse_sde(x_init, fn, times, rng)

                c = compare_distributions(
                    a_gen, a_ref, prior.rho, sigma_true,
                    innov_kurtosis_true=prior.innovation_excess_kurtosis,
                    innov_variance_true=prior.q, name=name, seed=seed,
                )
                rows.append({
                    "n_chains": n_chains, "seed": seed, "arm": name,
                    "innov_kurtosis": c.innov_kurtosis,
                    "innov_kurtosis_se": c.innov_kurtosis_se,
                    "kurtosis_gap_in_se": c.kurtosis_gap_in_se(),
                    "innov_variance": c.innov_variance,
                    "innov_kl": c.innov_kl,
                    "cov_frobenius_rel": c.cov_frobenius_rel,
                    "cov_worst_lag_abs": c.cov_worst_lag_abs,
                    "marginal_var": c.marginal_var,
                    "notes": "; ".join(c.notes),
                })
                print(f"  N={n_chains:5d} seed={seed} {name:6s} "
                      f"kurt={c.innov_kurtosis:6.3f}+-{c.innov_kurtosis_se:.3f} "
                      f"(true {c.innov_kurtosis_true:.2f}, gap {c.kurtosis_gap_in_se():6.1f} se)  "
                      f"KL={c.innov_kl:.4f}  cov_lag={c.cov_worst_lag_abs:.4f}", flush=True)

    write_csv(out_dir / "generation.csv", rows)
    return rows


def run_pointwise(prior, grid, weights, cfg, out_dir):
    """The metric ladder at fixed noise levels, for side-by-side with `generate`."""
    rows = []
    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            kernel, nets, _ = fit_arms(prior, grid, weights, n_chains, seed, cfg)
            best = choose_parameterization(prior, grid, weights, nets, cfg, seed)

            rng = rng_for("exp16-point", seed, n_chains)
            A = sample_chains(prior, rng, 512, cfg["n_sites"])
            for t in cfg["t_probe"]:
                alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                m_star = bp_posterior_mean(prior, grid, weights, X, t)

                preds = {
                    "em_bp": bp_posterior_mean(kernel, grid, weights, X, t),
                    "mlp": dsm_posterior_mean(nets[("mlp", best[("mlp", t)])], X, t),
                    "cnn": local_posterior_mean(nets[("cnn", best[("cnn", t)])], X, t),
                }
                for name, m_hat in preds.items():
                    d = pointwise_ladder(m_hat, m_star, A, alpha, delta)
                    rows.append({"n_chains": n_chains, "seed": seed, "arm": name,
                                 "t": t, **d})
    write_csv(out_dir / "pointwise.csv", rows)
    return rows


def run_steps(prior, grid, weights, cfg, out_dir):
    """Integrator control: does the ranking survive a change of step size?

    If halving the step changes which arm wins, the comparison is measuring
    Euler-Maruyama discretization rather than score quality, and nothing in `generate`
    should be believed.
    """
    rows = []
    n = cfg["n_sites"]
    sigma_true = prior.rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    n_chains = cfg["sizes"][-1]

    kernel, nets, _ = fit_arms(prior, grid, weights, n_chains, 0, cfg)
    best = choose_parameterization(prior, grid, weights, nets, cfg, 0)
    arms = {
        "exact": make_exact_arm(prior, grid, weights),
        "em_bp": make_em_arm(kernel, grid, weights),
        "mlp": make_network_arm(nets[("mlp", "eps")], nets[("mlp", "x0")],
                                _selector(best, "mlp", cfg["t_probe"])),
        "cnn": make_cnn_arm(nets[("cnn", "eps")], nets[("cnn", "x0")],
                            _selector(best, "cnn", cfg["t_probe"])),
    }
    a_ref = sample_chains(prior, rng_for("exp16-ref", 0), cfg["n_generate"], n)

    for n_steps in cfg["steps_ladder"]:
        times = time_grid(cfg["t_max"], cfg["t_min"], n_steps)
        for name, fn in arms.items():
            rng = rng_for("exp16-steps", n_steps)
            x_init = rng.standard_normal((cfg["n_generate"], n))
            a_gen = reverse_sde(x_init, fn, times, rng)
            c = compare_distributions(
                a_gen, a_ref, prior.rho, sigma_true,
                innov_kurtosis_true=prior.innovation_excess_kurtosis,
                innov_variance_true=prior.q, name=name, seed=0,
            )
            rows.append({"n_steps": n_steps, "arm": name,
                         "innov_kurtosis": c.innov_kurtosis,
                         "innov_kurtosis_se": c.innov_kurtosis_se,
                         "innov_kl": c.innov_kl,
                         "cov_worst_lag_abs": c.cov_worst_lag_abs})
            print(f"  steps={n_steps:4d} {name:6s} kurt={c.innov_kurtosis:6.3f} "
                  f"KL={c.innov_kl:.4f}", flush=True)

    write_csv(out_dir / "steps.csv", rows)
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    parser = experiment_parser("exp_16_sampling_validation", __doc__)
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = PARTS if args.only is None else tuple(p.strip() for p in args.only.split(","))
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown part(s): {sorted(unknown)}; choose from {PARTS}")

    out_dir = ensure_dir(args.output_dir)
    grid, weights = make_grid(cfg["grid_a"], cfg["grid_m"])
    prior = LaplaceAR1(cfg["rho"])

    for part in parts:
        print(f"\n=== part: {part} ===", flush=True)
        if part == "generate":
            run_generate(prior, grid, weights, cfg, out_dir)
        elif part == "pointwise":
            run_pointwise(prior, grid, weights, cfg, out_dir)
        elif part == "steps":
            run_steps(prior, grid, weights, cfg, out_dir)

    write_json(out_dir / f"params_{'_'.join(parts)}.json",
               {"settings": cfg, "parts": list(parts), **provenance()})


if __name__ == "__main__":
    main()
