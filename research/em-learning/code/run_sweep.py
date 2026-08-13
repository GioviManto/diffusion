"""Sweep driver for the HPC. Shardable for SLURM job arrays.

Experiments
-----------
``identifiability``
    The headline EM study. For each (family, rho, n, t, n_chains) cell: fit (rho,q) by
    multistart EM, record the restart spread, and profile the likelihood in rho to get the
    ridge width and curvature. This is what tells us at which diffusion times the model can
    be learned at all.

``sample_efficiency``
    Score error of the EM-learned score against the exact score, as a function of the number
    of training chains. This is the EM arm of the headline paper figure; the neural-network
    arm is added once that baseline is written and validated.

``nonlinearity_gap``
    Gaussian closure (= the LMMSE denoiser, by the closure proposition) against exact grid
    BP, across families, rho and t. Quantifies how much nonlinearity the exact score carries
    over the best linear one. Expensive: grid BP is O(n M_grid^2).

Usage
-----
    python run_sweep.py --experiment identifiability --shard 0 --nshards 40 --out DIR

Each shard writes one CSV; ``merge_shards.py`` concatenates them.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from chain_models import (
    ChainConfig,
    ar1_covariance,
    noise_chains,
    ou_coefficients,
    sample_chains,
)
from bp_core import (
    Grid,
    build_transition_matrix,
    exact_gaussian_posterior_mean,
    gaussian_closure_bp,
    grid_bp,
    likelihood_matrix,
    score_from_mean,
)
from em_parametric import (
    NoisyDataset,
    em_fit_with_spread,
    em_fit_multistart,
    fisher_curvature,
    profile_likelihood_ridge,
    restart_spread,
    ridge_width,
)

FAMILIES = ("gaussian", "laplace", "student", "mixture", "uniform")
T_VALUES = (0.08, 0.15, 0.30, 0.50, 0.80, 1.20, 1.80, 2.40)


# -----------------------------------------------------------------------------

def cells_identifiability() -> list[dict]:
    rows = []
    for fam, rho, n, t, m in itertools.product(
        FAMILIES,
        (0.50, 0.70, 0.85, 0.95),
        (20, 40, 80),
        T_VALUES,
        (250, 1000, 4000),
    ):
        # Replicates give error bars on every reported quantity. The sweep is cheap
        # (worst cell ~56 s), so thin single-seed cells would be a false economy.
        for rep in range(3):
            rows.append({"family": fam, "rho": rho, "n": n, "t": t,
                         "n_chains": m, "rep": rep})
    return rows


def run_identifiability(cell: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cfg = ChainConfig(n=cell["n"], rho=cell["rho"], innovation=cell["family"])
    q_true = cfg.innovation_variance
    a = sample_chains(rng, cell["n_chains"], cfg)
    x, alpha, delta = noise_chains(rng, a, cell["t"])
    data = NoisyDataset(x, alpha, delta, cell["t"])

    # One set of restarts serves both the point estimate and the spread. Running
    # em_fit_multistart and restart_spread separately duplicated every EM fit.
    fit, spread = em_fit_with_spread(data, n_restarts=6, max_iter=300)

    # Symmetric, rho-independent scan bracket. The earlier asymmetric bracket
    # (rho-0.6, rho+0.14) had a ceiling on the measurable ridge width that varied with rho,
    # which made widths incomparable across cells. A fixed span keeps the censoring
    # threshold identical everywhere, and the flags below record when it still bites.
    lo = max(0.005, cell["rho"] - 0.45)
    hi = min(0.995, cell["rho"] + 0.45)
    prof = profile_likelihood_ridge(data, cell["rho"], q_true, np.linspace(lo, hi, 41))
    width, width_censored = ridge_width(prof)
    curv, peak_at_edge = fisher_curvature(prof)

    return {
        **cell,
        "rho_true": cell["rho"],
        "q_true": q_true,
        "rho_hat": fit.rho,
        "q_hat": fit.q,
        "rho_abs_err": abs(fit.rho - cell["rho"]),
        "q_rel_err": abs(fit.q - q_true) / q_true,
        "loglik": fit.loglik,
        "em_iters": fit.n_iter,
        "em_converged": int(fit.converged),
        "restart_rho_std": spread["rho_std"],
        "restart_rho_range": spread["rho_max"] - spread["rho_min"],
        "restart_loglik_range": spread["loglik_range"],
        "ridge_width": width,
        # Censored widths are LOWER BOUNDS (the region hit the end of the scan) and must be
        # excluded from cross-cell comparisons rather than averaged in.
        "ridge_width_censored": int(width_censored),
        # nan when the profile peak sits at the scan edge; do NOT read that as zero curvature.
        "profile_curvature": curv,
        "profile_peak_at_edge": int(peak_at_edge),
        "profile_scan_lo": lo,
        "profile_scan_hi": hi,
        "profile_peak_rho": float(prof["rho"][int(np.argmax(prof["loglik"]))]),
    }


# -----------------------------------------------------------------------------

def cells_sample_efficiency() -> list[dict]:
    rows = []
    for fam, rho, m, t_eval in itertools.product(
        FAMILIES, (0.70, 0.85), (60, 125, 250, 500, 1000, 2000, 4000), (0.15, 0.50, 1.20)
    ):
        # The sample-efficiency curve is the headline figure, so it gets the most
        # replicates: the interesting regime is small M, where scatter is largest.
        for rep in range(5):
            rows.append({"family": fam, "rho": rho, "n_chains": m,
                         "t_eval": t_eval, "rep": rep})
    return rows


def run_sample_efficiency(cell: dict, seed: int) -> dict:
    """Train EM on chains at a low diffusion time, then score at t_eval.

    The point of the experiment: EM learns the *generative parameters* once, and BP then
    yields a score at any t. We therefore train at a single low-noise t and evaluate
    elsewhere, which is the operating mode a network cannot match without retraining.
    """
    rng = np.random.default_rng(seed)
    n = 40
    t_train = 0.15
    cfg = ChainConfig(n=n, rho=cell["rho"], innovation=cell["family"])

    a_tr = sample_chains(rng, cell["n_chains"], cfg)
    x_tr, al_tr, de_tr = noise_chains(rng, a_tr, t_train)
    fit = em_fit_multistart(
        NoisyDataset(x_tr, al_tr, de_tr, t_train), n_restarts=6, max_iter=300
    )

    # Held-out evaluation at a different diffusion time.
    a_te = sample_chains(rng, 400, cfg)
    x_te, al, de = noise_chains(rng, a_te, cell["t_eval"])

    sigma0_true = ar1_covariance(cfg)
    m_ref = exact_gaussian_posterior_mean(x_te, sigma0_true, al, de)
    s_ref = score_from_mean(x_te, m_ref, al, de)

    res = gaussian_closure_bp(x_te, fit.rho, fit.q, al, de)
    s_em = score_from_mean(x_te, res.means, al, de)

    rel = float(np.linalg.norm(s_em - s_ref) / np.linalg.norm(s_ref))
    return {
        **cell,
        "t_train": t_train,
        "rho_hat": fit.rho,
        "q_hat": fit.q,
        "rho_abs_err": abs(fit.rho - cell["rho"]),
        "n_params_learned": 2,
        # NOTE: s_ref here is the LMMSE/Gaussian-BP score built from the TRUE (rho,q).
        # For non-Gaussian families that is not the exact score; it isolates the error due
        # to *parameter estimation* alone. The separate gap to the exact score is measured
        # by the nonlinearity_gap experiment and must not be conflated with this number.
        "rel_score_err_vs_true_params": rel,
    }


# -----------------------------------------------------------------------------

def cells_nonlinearity_gap() -> list[dict]:
    rows = []
    for fam, rho, t in itertools.product(
        FAMILIES, (0.50, 0.70, 0.85, 0.95), T_VALUES
    ):
        rows.append({"family": fam, "rho": rho, "t": t, "rep": 0})
    return rows


def run_nonlinearity_gap(cell: dict, seed: int) -> dict:
    """Exact grid BP vs Gaussian closure: how much nonlinearity does the true score carry?"""
    rng = np.random.default_rng(seed)
    n, n_trials = 40, 200
    # 'uniform' has jump discontinuities and converges at only O(h); 'student' needs a wide
    # interval for its tail. Both are handled by resolution/extent rather than ignored.
    grid_hi = 20.0 if cell["family"] == "student" else 12.0
    grid_m = 1201 if cell["family"] == "uniform" else 801

    cfg = ChainConfig(n=n, rho=cell["rho"], innovation=cell["family"])
    grid = Grid.build(-grid_hi, grid_hi, grid_m)
    K = build_transition_matrix(grid, cfg.rho, cfg)
    prior = np.exp(-0.5 * grid.points ** 2 / cfg.stationary_variance)
    alpha, delta = ou_coefficients(cell["t"])

    rels, coss, worst_b = [], [], 0.0
    for _ in range(n_trials):
        a = sample_chains(rng, 1, cfg)
        x, _, _ = noise_chains(rng, a, cell["t"])
        ell = likelihood_matrix(grid, x[0], alpha, delta)
        ref, _ = grid_bp(grid, K, ell, prior)
        worst_b = max(worst_b, ref.worst_boundary_mass)
        s_ref = score_from_mean(x[0], ref.means, alpha, delta)[0]

        res = gaussian_closure_bp(x, cfg.rho, cfg.innovation_variance, alpha, delta)
        s_lin = score_from_mean(x, res.means, alpha, delta)[0]

        rels.append(float(np.linalg.norm(s_lin - s_ref) / np.linalg.norm(s_ref)))
        coss.append(float(s_lin @ s_ref / (np.linalg.norm(s_lin) * np.linalg.norm(s_ref))))

    return {
        **cell,
        "n": n,
        "n_trials": n_trials,
        "grid_size": grid_m,
        "grid_halfwidth": grid_hi,
        "rel_score_gap_mean": float(np.mean(rels)),
        "rel_score_gap_std": float(np.std(rels, ddof=1)),
        "score_cosine_mean": float(np.mean(coss)),
        "worst_boundary_mass": worst_b,
    }


# -----------------------------------------------------------------------------

EXPERIMENTS = {
    "identifiability": (cells_identifiability, run_identifiability),
    "sample_efficiency": (cells_sample_efficiency, run_sample_efficiency),
    "nonlinearity_gap": (cells_nonlinearity_gap, run_nonlinearity_gap),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--limit", type=int, default=0, help="debug: cap number of cells")
    args = p.parse_args()

    make_cells, run_cell = EXPERIMENTS[args.experiment]
    cells = make_cells()

    # Deterministically shuffle before sharding.
    #
    # The cell list is a nested product whose inner loops cycle with small periods (3
    # replicates x 3 chain counts = 9). Plain modulo sharding therefore aliases badly: with
    # a shard count sharing a factor with that period, one shard receives every expensive
    # cell and blows its walltime while others idle. Shuffling with a fixed seed decorrelates
    # cost from shard index for *any* shard count, and stays reproducible.
    order = np.random.default_rng(12345).permutation(len(cells))
    cells = [cells[i] for i in order]

    # Index carried through for seeding is the post-shuffle position, so a given cell keeps
    # the same seed regardless of how many shards are used.
    mine = [(i, c) for i, c in enumerate(cells) if i % args.nshards == args.shard]
    if args.limit:
        mine = mine[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{args.experiment}_shard{args.shard:03d}.csv"

    print(f"[{args.experiment}] shard {args.shard}/{args.nshards}: "
          f"{len(mine)} of {len(cells)} cells -> {out_csv}", flush=True)

    rows, t0 = [], time.time()
    for k, (idx, cell) in enumerate(mine, 1):
        try:
            rows.append(run_cell(cell, args.seed + idx))
        except Exception as exc:                      # keep the shard alive
            print(f"  CELL FAILED {cell}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({**cell, "error": f"{type(exc).__name__}: {exc}"})
        if k % 5 == 0 or k == len(mine):
            el = time.time() - t0
            print(f"  {k}/{len(mine)} cells, {el:.0f}s elapsed, "
                  f"{el / k:.1f}s/cell, ETA {(len(mine) - k) * el / k:.0f}s", flush=True)

    if rows:
        keys: list[str] = []
        for r in rows:
            for kk in r:
                if kk not in keys:
                    keys.append(kk)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    meta = {
        "experiment": args.experiment,
        "shard": args.shard,
        "nshards": args.nshards,
        "n_cells": len(mine),
        "n_failed": sum(1 for r in rows if "error" in r),
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "host": os.uname().nodename,
    }
    (outdir / f"{args.experiment}_shard{args.shard:03d}.meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"done: {len(rows)} rows, {meta['n_failed']} failed, "
          f"{meta['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
