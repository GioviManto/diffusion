#!/usr/bin/env python3
"""Summarise the exp_16 continuous-training generation rerun across every data budget.

WHY THIS EXISTS. `CLAIMS_TO_UPDATE.md` section 10 -- the audit's most serious item, and
the one flagged as able to move the paper's headline -- was written while job 618378 had
returned only its N=32 cells. It marks itself "not-yet-writable" and asks that nothing be
acted on until the remaining budgets land. They landed on 2026-08-08; the document was
never revisited, so for three days the most consequential open question in the revision was
answerable from data already on disk and nobody had looked.

That is the failure this guards against: not a wrong number, but a finished computation
that no one noticed had finished. Run it after any generation sweep returns.

The reading is deliberately mechanical -- medians across seeds, distance to the analytic
target, and a blow-up count -- because the interesting comparisons here are between arms at
matched N, and a mean is not robust to the MLP's small-data excursions (one seed at 44.3
would drag a 4-seed mean anywhere you like).

    python tools/summarize_generation_rerun.py [--root <dir>] [--csv <out.csv>]

Reads only, unless --csv is given.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys

# AR-filtered residual excess kurtosis of the true innovation law. The whole table is
# distances to this number; it is not a fitted quantity.
TARGET = 1.9098

ARMS = ["exact", "em_bp", "mlp", "cnn"]

# |excess kurtosis| beyond this is a diverged integration, not a bad fit. Chosen well above
# the target so it cannot fire on a merely poor arm.
PATHOLOGICAL = 5.0


def repo_root() -> str:
    """research/nongaussian-bp, resolved from this file rather than the caller's cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "research", "nongaussian-bp")


def load(root: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(root, "gen2_n*_seed*", "generation.csv"))):
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="directory holding gen2_n*_seed* (default: outputs/exp_16)")
    ap.add_argument("--csv", default=None, help="also write the summary table here")
    args = ap.parse_args()

    root = args.root or os.path.join(repo_root(), "outputs", "exp_16")
    rows = load(root)
    if not rows:
        print(f"no generation.csv under {root}/gen2_n*_seed*", file=sys.stderr)
        return 1

    n_dirs = len(glob.glob(os.path.join(root, "gen2_n*_seed*")))
    budgets = sorted({int(r["n_chains"]) for r in rows})
    print(f"{len(rows)} rows from {n_dirs} cells;  budgets {budgets};  target {TARGET}\n")

    hdr = (f"{'N':>6} {'arm':>7} {'seeds':>5} {'kurtosis per seed':>34} "
           f"{'median':>8} {'|med-tgt|':>10} {'blowups':>8} {'cov_rel':>8}")
    print(hdr)
    print("-" * len(hdr))

    summary: dict[tuple[int, str], tuple[float, int, float]] = {}
    out_rows = []
    for n in budgets:
        for arm in ARMS:
            cells = [r for r in rows if int(r["n_chains"]) == n and r["arm"] == arm]
            if not cells:
                continue
            ks = [float(r["innov_kurtosis"]) for r in cells]
            covs = [float(r["cov_frobenius_rel"]) for r in cells]
            med, cov = st.median(ks), st.median(covs)
            blow = sum(1 for k in ks if abs(k) > PATHOLOGICAL)
            summary[(n, arm)] = (med, blow, cov)
            per_seed = " ".join(f"{k:7.3f}" for k in sorted(ks, reverse=True)[:4])
            print(f"{n:>6} {arm:>7} {len(ks):>5} {per_seed:>34} {med:>8.3f} "
                  f"{abs(med - TARGET):>10.3f} {blow:>8} {cov:>8.3f}")
            out_rows.append({"n_chains": n, "arm": arm, "n_seeds": len(ks),
                             "median_kurtosis": med, "dist_to_target": abs(med - TARGET),
                             "blowups": blow, "median_cov_frobenius_rel": cov})
        print()

    print("=" * len(hdr))
    print("\n1. Dissociation -- distance to target, CNN vs EM-BP:")
    for n in budgets:
        if (n, "cnn") in summary and (n, "em_bp") in summary:
            dc = abs(summary[(n, "cnn")][0] - TARGET)
            de = abs(summary[(n, "em_bp")][0] - TARGET)
            print(f"   N={n:>5}: cnn {dc:6.3f}   em_bp {de:6.3f}   -> "
                  f"{'CNN closer' if dc < de else 'EM-BP closer'}")

    print("\n2. MLP blow-ups (|excess kurtosis| > 5):")
    for n in budgets:
        if (n, "mlp") in summary:
            med, blow, _ = summary[(n, "mlp")]
            print(f"   N={n:>5}: {blow} of 4 seeds   (median {med:.3f})")

    print("\n3. Second-order fidelity -- median relative covariance error:")
    for n in budgets:
        line = "  ".join(f"{a}:{summary[(n, a)][2]:.3f}" for a in ARMS if (n, a) in summary)
        print(f"   N={n:>5}: {line}")

    print("\n4. Network-free arms (must not blow up at any budget):")
    for arm in ("exact", "em_bp"):
        meds = "  ".join(f"N={n}:{summary[(n, arm)][0]:.3f}"
                         for n in budgets if (n, arm) in summary)
        print(f"   {arm:>6}: {meds}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
