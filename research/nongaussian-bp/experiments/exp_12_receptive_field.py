"""Experiment 12 -- How big a receptive field does a score head actually need?

Two things this settles, both of which have been outstanding.

**B15, currently classified HEURISTIC.** The ledger reads the Gaussian locality
law as an architecture prescription: a local score head is near-optimal once its
receptive-field radius satisfies r >~ xi log(1/eps). Nothing in the project ever
trained such a head, so the reading was inferred rather than measured. Here it is
measured: sweep r and find where the error stops improving.

**The structured-architecture baseline.** exp_07 compares EM-BP against a fully
connected MLP, which carries no locality prior at all, and the compendium flags
this as the most important unasked question -- a reviewer will want to know
whether an architecture that *does* encode sequence structure closes the gap. A
weight-shared window predictor is exactly a 1-D CNN, so this supplies that
baseline, with r = n giving a global head for reference.

The quantitative prediction being tested comes from exp_11. The radius-r
*estimator* error decays as q^r with q measured per family, and exp_11 found q
is larger (less local) for non-Gaussian chains at low noise -- 1.12x to 1.46x,
implying a receptive field 1.12x to 1.46x wider. So the prediction is not merely
"error saturates" but "it saturates later for non-Gaussian data, by a factor the
locality law names in advance".

Care taken: the error is reported both over all sites and restricted to interior
sites (at least r from either end), because the locality statement is about the
interior and zero padding contaminates the edges. A prescription validated on
the padded number would be validating an edge artifact.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import grid_bp_batch, make_grid
from src.denoiser import dsm_posterior_mean, train_dsm_denoiser
from src.local_head import (
    interior_slice,
    local_posterior_mean,
    train_local_head,
)
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import GaussianAR1, GaussianMixtureAR1, LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 33
GRID_A = 8.0
GRID_M = 401
T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)
N_TEST = 256


def _families(rho):
    return [GaussianAR1(rho), LaplaceAR1(rho), GaussianMixtureAR1(rho, kappa=0.9)]


def _reference(prior, grid, weights, A_test, t_values):
    """Exact-BP reference denoiser on held-out chains."""
    rng = rng_for("exp12-testnoise", prior.name)
    log_k = prior.log_transition_matrix(grid)
    bundle = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A_test + np.sqrt(delta) * rng.standard_normal(A_test.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        bundle[t] = (X, m_ref)
    return bundle


def _score_error(m_hat, m_ref, X, t, sl):
    """Relative score error, restricted to a slice of sites."""
    alpha, delta = alpha_delta(t)
    s_hat = -(X[:, sl] - alpha * m_hat[:, sl]) / delta
    s_ref = -(X[:, sl] - alpha * m_ref[:, sl]) / delta
    return float(np.linalg.norm(s_hat - s_ref) / np.linalg.norm(s_ref))


def part1_sweep(cfg, out):
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []
    for prior in _families(cfg["rho"]):
        rng = rng_for("exp12", prior.name)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
        rng_test = rng_for("exp12-test", prior.name)
        A_test = np.stack([prior.sample(rng_test, N_SITES) for _ in range(cfg["n_test"])])
        bundle = _reference(prior, grid, weights, A_test, T_TRAIN)

        for radius in cfg["radii"]:
            for mode in cfg["parameterizations"]:
                head = train_local_head(
                    A, T_TRAIN, radius,
                    rng_for("exp12-head", prior.name, radius, mode),
                    hidden=cfg["hidden"], n_steps=cfg["n_steps"],
                    parameterization=mode,
                )
                for t in T_TRAIN:
                    X, m_ref = bundle[t]
                    m_hat = local_posterior_mean(head, X, t)
                    sl_all = slice(None)
                    sl_in = interior_slice(N_SITES, radius)
                    rows.append({
                        "family": prior.name,
                        "excess_kurtosis": prior.innovation_excess_kurtosis,
                        "radius": radius,
                        "receptive_field": 2 * radius + 1,
                        "parameterization": mode,
                        "t": t,
                        "n_params": head.n_params,
                        "score_rel_l2_all": _score_error(m_hat, m_ref, X, t, sl_all),
                        "score_rel_l2_interior": _score_error(m_hat, m_ref, X, t, sl_in),
                        "seconds": head.seconds,
                    })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    t_lo, t_hi = T_TRAIN[0], T_TRAIN[-1]
    for j, t in enumerate((t_lo, t_hi)):
        for prior in _families(cfg["rho"]):
            sub = {}
            for r in cfg["radii"]:
                vals = [x["score_rel_l2_interior"] for x in rows
                        if x["family"] == prior.name and x["radius"] == r
                        and x["t"] == t]
                sub[r] = min(vals)  # best parameterization at this radius
            ax[j].semilogy(list(sub), list(sub.values()), "o-", label=prior.name, ms=4)
        ax[j].set_xlabel("receptive-field radius $r$")
        ax[j].set_ylabel("relative score error (interior)")
        ax[j].set_title(f"$t={t}$")
        ax[j].legend(fontsize=8)
    save_figure(fig, out / "receptive_field_sweep.png")
    return rows


def part2_vs_global(cfg, out):
    """Local head vs the fully connected MLP of exp_07, at equal data."""
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []
    for prior in _families(cfg["rho"]):
        rng = rng_for("exp12", prior.name)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
        rng_test = rng_for("exp12-test", prior.name)
        A_test = np.stack([prior.sample(rng_test, N_SITES) for _ in range(cfg["n_test"])])
        bundle = _reference(prior, grid, weights, A_test, T_TRAIN)

        for mode in cfg["parameterizations"]:
            glob = train_dsm_denoiser(
                A, T_TRAIN, rng_for("exp12-global", prior.name, mode),
                hidden=cfg["global_hidden"], n_steps=cfg["n_steps"],
                parameterization=mode,
            )
            best_r = cfg["compare_radius"]
            head = train_local_head(
                A, T_TRAIN, best_r,
                rng_for("exp12-head", prior.name, best_r, mode),
                hidden=cfg["hidden"], n_steps=cfg["n_steps"], parameterization=mode,
            )
            for t in T_TRAIN:
                X, m_ref = bundle[t]
                sl = interior_slice(N_SITES, best_r)
                rows.append({
                    "family": prior.name, "t": t, "parameterization": mode,
                    "arch": "local_cnn", "radius": best_r,
                    "n_params": head.n_params,
                    "score_rel_l2_interior": _score_error(
                        local_posterior_mean(head, X, t), m_ref, X, t, sl),
                })
                rows.append({
                    "family": prior.name, "t": t, "parameterization": mode,
                    "arch": "global_mlp", "radius": N_SITES,
                    "n_params": glob.n_params,
                    "score_rel_l2_interior": _score_error(
                        dsm_posterior_mean(glob, X, t), m_ref, X, t, sl),
                })
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_12_receptive_field",
        "How big a receptive field does a score head need? (tests B15)",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 301, "rho": 0.85, "n_chains": 256, "n_test": 128,
        "radii": (0, 1, 2, 4), "hidden": (64, 64), "global_hidden": (64, 64),
        "n_steps": 2000, "parameterizations": ("eps",), "compare_radius": 4,
    }
    full = {
        "grid_size": GRID_M, "rho": 0.85, "n_chains": 1024, "n_test": N_TEST,
        "radii": (0, 1, 2, 3, 4, 6, 8, 12, 16),
        "hidden": (64, 64), "global_hidden": (128, 128),
        "n_steps": 20000, "parameterizations": ("eps", "x0"), "compare_radius": 6,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "sweep": ("receptive-field sweep",
                  lambda o: write_csv(o / "receptive_field.csv", part1_sweep(cfg, o))),
        "vs_global": ("local head vs fully connected MLP",
                      lambda o: write_csv(o / "local_vs_global.csv",
                                          part2_vs_global(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "t_train": T_TRAIN, "grid_half_width": GRID_A,
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
