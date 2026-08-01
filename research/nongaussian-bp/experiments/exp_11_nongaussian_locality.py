"""Experiment 11 -- Is the locality law second-order universal? (question F5)

For the Gaussian chain the project already has an exact locality law (ledger
G12): the RMS error of the radius-r estimator -- the one that predicts a_i from
only x_{i-r..i+r} -- decays exactly as q^r, with

    q = (J_d - sqrt(J_d^2 - 4 beta^2)) / (2 |beta|),

an explicit function of rho and t through the posterior precision. It is the
statement behind B15, the reading that a CNN score head with receptive field
r >~ xi log(1/eps) is near-optimal.

F5 asks for the non-Gaussian analogue, "basis-independent". The sharp version of
that question, and the one this experiment tests, is:

    Is the locality decay rate a property of the second-order structure alone --
    so that every innovation law with the same (rho, q) shares the Gaussian
    rate -- or does it depend on the shape of the innovation?

The two outcomes mean quite different things. If the rate is second-order
universal, then B15's receptive-field prescription transfers to non-Gaussian
data unchanged, and the architecture reading is safe. If the rate depends on
shape, then heavy-tailed or bimodal data needs a different receptive field than
its covariance-matched Gaussian would suggest, and B15 is only a Gaussian
statement.

Method. For each prior and noise level, compute the exact posterior mean two
ways with grid BP: on the full chain, and on the window [i-r, i+r] treated as a
chain in its own right (which is exactly what an estimator with access to only
those observations can do). The decay of ||m^(r) - m^full|| in r is the locality
law. Fitting log-error against r gives the rate, which is compared against the
Gaussian chain at matched (rho, q).

Everything is measured at the centre site of the window to avoid contaminating
the result with edge effects.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import grid_bp_batch, make_grid
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import (
    GaussianAR1,
    GaussianMixtureAR1,
    LaplaceAR1,
    StudentTAR1,
    UniformAR1,
)
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 41  # odd, so there is a well-defined centre site
CENTRE = N_SITES // 2
GRID_A = 8.0


def _families(rho: float):
    """Priors sharing the same (rho, q) but differing beyond second moments."""
    return [
        GaussianAR1(rho),
        LaplaceAR1(rho),
        StudentTAR1(rho, nu=5.0),
        UniformAR1(rho),
        GaussianMixtureAR1(rho, kappa=0.6),
        GaussianMixtureAR1(rho, kappa=0.9),
    ]


def locality_curve(prior, grid, weights, X, t, radii):
    """Error of the radius-r estimator at the centre site, against full-chain BP."""
    alpha, delta = alpha_delta(t)
    log_k = prior.log_transition_matrix(grid)

    m_full, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
    m_ref = m_full[:, CENTRE]

    out = []
    for r in radii:
        lo, hi = CENTRE - r, CENTRE + r + 1
        m_win, _ = grid_bp_batch(grid, weights, log_k, X[:, lo:hi], alpha, delta)
        err = float(np.sqrt(np.mean((m_win[:, r] - m_ref) ** 2)))
        out.append((r, err))
    return out


def _fit_rate(radii, errors):
    """Geometric decay rate q from a log-linear fit, with its standard error."""
    r = np.asarray(radii, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = e > 0
    if ok.sum() < 3:
        return np.nan, np.nan
    r, e = r[ok], np.log(e[ok])
    A = np.vstack([r, np.ones(len(r))]).T
    beta, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ beta
    s2 = resid @ resid / max(len(r) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(np.exp(beta[0])), float(np.exp(beta[0]) * np.sqrt(cov[0, 0]))


def part1_universality(cfg, out):
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []
    for rho in cfg["rhos"]:
        for prior in _families(rho):
            rng = rng_for("exp11", prior.name, rho)
            A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
            for t in cfg["t_values"]:
                alpha, delta = alpha_delta(t)
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                curve = locality_curve(prior, grid, weights, X, t, cfg["radii"])
                q_hat, q_se = _fit_rate(*zip(*curve))
                for r, err in curve:
                    rows.append({
                        "family": prior.name,
                        "excess_kurtosis": prior.innovation_excess_kurtosis,
                        "rho": rho,
                        "t": t,
                        "radius": r,
                        "rms_error": err,
                        "fitted_rate": q_hat,
                        "fitted_rate_se": q_se,
                    })

    # Ratio of each family's rate to the Gaussian rate at matched (rho, t).
    for row in rows:
        ref = next(
            x["fitted_rate"] for x in rows
            if x["family"] == "gaussian" and x["rho"] == row["rho"]
            and x["t"] == row["t"]
        )
        row["rate_over_gaussian"] = row["fitted_rate"] / ref if ref > 0 else np.nan

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    rho0, t0 = cfg["rhos"][0], cfg["t_values"][len(cfg["t_values"]) // 2]
    for prior in _families(rho0):
        sub = sorted(
            [r for r in rows if r["family"] == prior.name
             and r["rho"] == rho0 and r["t"] == t0],
            key=lambda r: r["radius"],
        )
        ax[0].semilogy([r["radius"] for r in sub], [r["rms_error"] for r in sub],
                       "o-", label=prior.name, ms=3)
    ax[0].set_xlabel("window radius $r$")
    ax[0].set_ylabel("RMS error at the centre site")
    ax[0].set_title(rf"Locality decay, $\rho={rho0}$, $t={t0}$")
    ax[0].legend(fontsize=7)

    for prior in _families(rho0):
        sub = [r for r in rows if r["family"] == prior.name and r["rho"] == rho0
               and r["radius"] == cfg["radii"][0]]
        sub.sort(key=lambda r: r["t"])
        ax[1].semilogx([r["t"] for r in sub], [r["rate_over_gaussian"] for r in sub],
                       "o-", label=prior.name, ms=3)
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("noise level $t$")
    ax[1].set_ylabel("fitted rate / Gaussian rate")
    ax[1].set_title("Is the rate second-order universal?")
    ax[1].legend(fontsize=7)
    save_figure(fig, out / "locality_universality.png")
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_11_nongaussian_locality",
        "Does the Gaussian locality law survive beyond second order?",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 301, "n_chains": 32, "rhos": (0.85,),
        "t_values": (0.1, 0.4), "radii": (1, 2, 3, 4, 5),
    }
    full = {
        "grid_size": 401, "n_chains": 128, "rhos": (0.5, 0.85, 0.95),
        "t_values": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
        "radii": (1, 2, 3, 4, 5, 6, 7, 8),
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "universality": ("locality rate vs innovation shape",
                         lambda o: write_csv(o / "locality.csv",
                                             part1_universality(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "centre": CENTRE, "grid_half_width": GRID_A,
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
