"""Diagnostics the revision needs and the existing outputs do not contain.

Three products, all cheap on one CPU, all written as CSV so the figures are
generated from committed data rather than from numbers typed out of prose.

1. ``boundary`` -- truncation diagnostic.
   The note previously carried a [pending] marker here: the boundary mass was
   computed inside the recursion and thrown away, so no aggregate could be
   quoted. We recompute it explicitly. For each (t, A, N_g) we run one forward
   pass and record the fraction of each normalised message that sits in the two
   edge cells of the grid, taking the worst case over sites. This is the
   quantity that bounds the truncation error: mass that leaves [-A, A] is
   discarded rather than transported.

   Reported alongside it is the normalisation residual |1 - \\int b| of the
   single-site beliefs, which is the other half of the same story: quadrature
   error at fixed support.

2. ``emtrace`` -- per-iteration marginal log-likelihood.
   ``outputs/exp_06/monotonicity.csv`` records only the first and last value, so
   monotonicity could be asserted but not plotted. ``fit_em`` already returns the
   whole trace; we persist it, for both the closed-form-M-step Laplace kernel and
   the generalised-EM mixture kernel, from several initialisations.

3. ``density`` -- the learned innovation law.
   Fits the mixture kernel at several data budgets and evaluates the fitted
   innovation density on a common grid, next to the true Laplace density. This
   is the direct picture of what is being estimated; the existing outputs record
   only its moments.

Usage
-----
    python experiments/exp_18_revision_diagnostics.py --parts boundary,emtrace,density
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bp_grid import make_grid  # noqa: E402
from src.em import fit_em  # noqa: E402
from src.kernels import MixtureInnovationKernel  # noqa: E402
from src.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from src.priors import LaplaceAR1  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "exp_18"

RHO = 0.85
N_SITES = 32
T_VALUES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def _rng(*tag) -> np.random.Generator:
    return np.random.default_rng(abs(hash(("exp18",) + tag)) % (2**32))


# ---------------------------------------------------------------------------
# 1. Truncation and normalisation diagnostics
# ---------------------------------------------------------------------------

def part_boundary() -> None:
    """Boundary mass and normalisation residual over (t, A, N_g)."""
    prior = LaplaceAR1(RHO)
    rows = []
    for half_width in (4.0, 6.0, 8.0, 10.0):
        for n_grid in (201, 401, 801):
            grid, weights = make_grid(half_width, n_grid)
            log_k = prior.log_transition_matrix(grid)
            K = np.exp(log_k)
            for t in T_VALUES:
                alpha, delta = alpha_delta(t)
                rng = _rng("boundary", t, half_width, n_grid)
                a = prior.sample(rng, N_SITES)
                x = alpha * a + np.sqrt(delta) * rng.standard_normal(N_SITES)
                log_ell = log_likelihood_matrix(grid, x, alpha, delta)
                ell = np.exp(log_ell)

                # Forward sweep, recording the edge-cell mass of every message.
                edge = np.zeros(len(grid), dtype=bool)
                edge[0] = edge[-1] = True
                worst_edge = 0.0
                L = np.exp(-0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi))
                for i in range(N_SITES - 1):
                    incoming = L * ell[i] * weights
                    out = K @ incoming
                    mass = float(np.sum(out * weights))
                    L = out / mass
                    dens = L * weights
                    worst_edge = max(worst_edge, float(dens[edge].sum() / dens.sum()))

                # Beliefs are renormalised by construction, so their normalisation
                # carries no information. The informative quantity is the quadrature
                # of the kernel itself: each column of K should integrate to one.
                #
                # Two residuals, because they measure different things. The maximum
                # over all columns is attained at the edge, where the transition
                # density is centred at rho*(+/-A) and its tail leaves the grid --
                # that is truncation, and it responds to A, not to N_g. The maximum
                # over interior columns (|u| <= A/2) has no truncated tail, so what
                # remains is quadrature, and it responds to the spacing.
                col_mass = (K * weights[:, None]).sum(axis=0)
                interior = np.abs(grid) <= 0.5 * half_width
                resid = float(np.max(np.abs(col_mass - 1.0)))
                resid_interior = float(np.max(np.abs(col_mass[interior] - 1.0)))

                rows.append(
                    {
                        "t": t,
                        "half_width": half_width,
                        "n_grid": n_grid,
                        "spacing": float(grid[1] - grid[0]),
                        "worst_boundary_mass": worst_edge,
                        "kernel_norm_residual_max": resid,
                        "kernel_norm_residual_interior": resid_interior,
                    }
                )
                print(
                    f"A={half_width:>5} Ng={n_grid:>4} t={t:<5} "
                    f"boundary={worst_edge:.3e} resid_max={resid:.3e} "
                    f"resid_int={resid_interior:.3e}"
                )
    _write_csv(OUT / "boundary.csv", rows)


# ---------------------------------------------------------------------------
# 2. EM objective trace
# ---------------------------------------------------------------------------

def part_emtrace() -> None:
    """Persist the whole marginal log-likelihood trace, several initialisations."""
    prior = LaplaceAR1(RHO)
    grid, weights = make_grid(8.0, 401)
    rows = []
    n_chains, t_train = 512, (0.1, 0.2, 0.4, 0.8, 1.6)

    rng = _rng("emtrace", "data")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = []
    for t in t_train:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        groups.append((X, alpha, delta))

    for n_comp in (4, 8):
        for init_id, rho0 in enumerate((0.0, 0.3, 0.6, -0.4)):
            kernel = MixtureInnovationKernel.init(
                n_comp, rho=rho0, var=0.8, rng=_rng("emtrace", n_comp, init_id)
            )
            fitted, trace = fit_em(kernel, grid, weights, groups, n_iters=60)
            mom = fitted.innovation_moments
            for it, (ll, sec) in enumerate(zip(trace.log_evidence, trace.seconds)):
                rows.append(
                    {
                        "n_components": n_comp,
                        "init_id": init_id,
                        "rho_init": rho0,
                        "iteration": it,
                        "log_evidence": ll,
                        "seconds": sec,
                        "rho_hat": float(trace.theta[it][0]),
                    }
                )
            print(
                f"C={n_comp} init rho0={rho0:>5}: violation={trace.monotone_violation:.3e} "
                f"rho_hat={fitted.rho:.4f} mean={mom['innovation_mean']:.2e} "
                f"exkurt={mom['innovation_excess_kurtosis']:.3f}"
            )
    _write_csv(OUT / "em_trace.csv", rows)


# ---------------------------------------------------------------------------
# 3. Learned innovation density
# ---------------------------------------------------------------------------

def part_density() -> None:
    """Fitted innovation density at several budgets, against the truth."""
    prior = LaplaceAR1(RHO)
    grid, weights = make_grid(8.0, 401)
    e = np.linspace(-3.0, 3.0, 601)
    b = prior.b
    true = np.exp(-np.abs(e) / b) / (2.0 * b)

    rows = [{"e": float(v), "budget": 0, "n_components": 0, "arm": "true",
             "density": float(d)} for v, d in zip(e, true)]
    summary = []

    for n_chains in (128, 512, 2048):
        for n_comp in (4, 8):
            rng = _rng("density", n_chains, n_comp)
            A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
            groups = []
            for t in (0.1, 0.2, 0.4, 0.8, 1.6):
                alpha, delta = alpha_delta(t)
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                groups.append((X, alpha, delta))
            kernel = MixtureInnovationKernel.init(
                n_comp, rho=0.3, var=0.8, rng=_rng("density", "init", n_chains, n_comp)
            )
            fitted, trace = fit_em(kernel, grid, weights, groups, n_iters=60)

            dens = np.zeros_like(e)
            for w, m, s2 in zip(fitted.pi, fitted.mu, fitted.s2):
                dens += w * np.exp(-0.5 * (e - m) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
            rows += [
                {"e": float(v), "budget": n_chains, "n_components": n_comp,
                 "arm": "fitted", "density": float(d)}
                for v, d in zip(e, dens)
            ]
            mom = fitted.innovation_moments
            summary.append(
                {
                    "n_chains": n_chains,
                    "n_components": n_comp,
                    "rho_hat": float(fitted.rho),
                    "rho_true": RHO,
                    "innovation_mean": mom["innovation_mean"],
                    "innovation_var": mom["innovation_var"],
                    "innovation_var_true": prior.q,
                    "innovation_excess_kurtosis": mom["innovation_excess_kurtosis"],
                    "innovation_excess_kurtosis_true": 3.0,
                    "monotone_violation": trace.monotone_violation,
                    "iters": len(trace.log_evidence),
                }
            )
            print(
                f"M={n_chains:>5} C={n_comp}: rho={fitted.rho:.4f} "
                f"mean={mom['innovation_mean']:+.2e} "
                f"exkurt={mom['innovation_excess_kurtosis']:.3f} (true 3.0)"
            )

    _write_csv(OUT / "innovation_density.csv", rows)
    _write_csv(OUT / "innovation_summary.csv", summary)


# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


PARTS = {"boundary": part_boundary, "emtrace": part_emtrace, "density": part_density}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="boundary,emtrace,density")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = [p.strip() for p in args.parts.split(",") if p.strip()]
    for name in chosen:
        print(f"\n=== {name} ===")
        PARTS[name]()
    (OUT / "params.json").write_text(
        json.dumps(
            {
                "rho": RHO,
                "n_sites": N_SITES,
                "t_values": list(T_VALUES),
                "parts": chosen,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "blas_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
