"""Experiment 23 -- what CIFAR-10 wavelet coefficients actually look like.

This experiment is a gate, not a result. Everything downstream assumes two
things about natural-image wavelet coefficients, and both are assumptions the
project has so far only tested on data it simulated itself:

1. **Non-Gaussianity.** The package's premise is that the innovation law is
   heavy-tailed and that this is what a second-order closure misses. If CIFAR
   subband coefficients came back approximately Gaussian, the premise would not
   apply to images and the honest move would be to stop. `subbands` measures
   excess kurtosis per subband with a bootstrap error over *images*, against a
   variance-matched Gaussian surrogate that says what "approximately Gaussian"
   looks like at this sample size.

2. **Cross-scale dependence of a form the kernel can represent.** This is the
   part that is easy to get wrong. `src/kernels.py` offers linear-autoregressive
   kernels, K(a'|a) = phi(a' - rho a), whose entire dependence on the parent is a
   shift of the innovation's location. The classical wavelet literature (Crouse,
   Nowak, Baraniuk 1998) reports the opposite structure: parent and child
   coefficients are close to *linearly uncorrelated* while their *magnitudes* are
   strongly dependent -- a large parent predicts a large-variance child, of
   either sign. A linear-AR kernel cannot express that at all: it would fit
   rho ~ 0 and collapse to a factorised heavy-tailed model, which would score
   well on marginals and capture no hierarchy whatsoever.

   `crossscale` measures both, so the choice of kernel family is made on
   evidence. The decisive number is `std_ratio_q4_q1`: the standard deviation of
   the child given a top-quartile |parent|, over the same given a bottom-quartile
   |parent|. At 1 there is no magnitude dependence; well above 1 a linear-AR
   kernel is the wrong family and a scale-mixture kernel is required.

Parts
-----
subbands    Per-subband scale, excess kurtosis, bootstrap SE, Gaussian surrogate.
crossscale  Parent-child linear correlation vs magnitude dependence, per level.
"""

from __future__ import annotations

import numpy as np

from common import (
    PACKAGE_ROOT, apply_overrides, experiment_parser, provenance, select_parts,
)
from src.hierarchy import TreeIndex
from src.image_data import load_cifar10_luminance
from src.sample_metrics import bootstrap_se, excess_kurtosis
from src.utils import ensure_dir, rng_for, write_csv, write_json
from src.wavelet import ORIENTATIONS, images_to_tree

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    "levels": 5,
    "n_images": 10000,
    "n_boot": 200,
    "seed": 0,
}


def _archive(settings):
    """Resolve the archive against the package root, not the caller's cwd."""
    from pathlib import Path

    p = Path(settings["archive"])
    return p if p.is_absolute() else PACKAGE_ROOT / p


def _load(settings):
    split = load_cifar10_luminance(
        _archive(settings), "train", settings["n_images"], settings["seed"]
    )
    qt, nodes, scaling = images_to_tree(split.images, settings["levels"])
    return split, qt, nodes, scaling


# ----------------------------------------------------------------------------

def part_subbands(settings, out_dir):
    """Marginal shape of every subband, against a variance-matched Gaussian."""
    split, qt, nodes, scaling = _load(settings)
    depth = qt.node_depth
    rng = rng_for("exp23", "surrogate", settings["seed"])
    rows = []

    for oi, orient in enumerate(ORIENTATIONS):
        for d in range(qt.depth + 1):
            block = nodes[:, oi, depth == d]          # (B, 4**d)
            sd = float(block.std())
            kurt = excess_kurtosis(block)
            se = bootstrap_se(
                block, excess_kurtosis, n_boot=settings["n_boot"], seed=settings["seed"]
            )
            # Same shape, same variance, genuinely Gaussian: the null against
            # which the measured kurtosis has to be judged.
            surrogate = rng.standard_normal(block.shape) * sd
            rows.append({
                "orientation": orient,
                "tree_depth": d,
                "subband_side": qt.subband_side(d),
                "n_coefficients": int(block.size),
                "std": sd,
                "std_over_pixel_std": sd,   # pixels are standardised to unit variance
                "excess_kurtosis": kurt,
                "excess_kurtosis_se": se,
                "excess_kurtosis_gaussian_surrogate": excess_kurtosis(surrogate),
                "kurtosis_in_se": kurt / se if se > 0 else float("nan"),
            })

    rows.append({
        "orientation": "LL",
        "tree_depth": -1,
        "subband_side": 1,
        "n_coefficients": int(scaling.size),
        "std": float(scaling.std()),
        "std_over_pixel_std": float(scaling.std()),
        "excess_kurtosis": excess_kurtosis(scaling),
        "excess_kurtosis_se": bootstrap_se(
            scaling, excess_kurtosis, n_boot=settings["n_boot"], seed=settings["seed"]
        ),
        "excess_kurtosis_gaussian_surrogate": float("nan"),
        "kurtosis_in_se": float("nan"),
    })

    write_csv(out_dir / "subbands.csv", rows)
    detail = [r for r in rows if r["orientation"] != "LL"]
    print(f"[subbands] {len(detail)} detail subbands over {len(split.images)} images")
    print(f"[subbands] excess kurtosis range "
          f"{min(r['excess_kurtosis'] for r in detail):.2f} .. "
          f"{max(r['excess_kurtosis'] for r in detail):.2f}")
    worst = min(abs(r["kurtosis_in_se"]) for r in detail)
    print(f"[subbands] weakest subband is {worst:.0f} bootstrap SE from Gaussian")
    return rows


# ----------------------------------------------------------------------------

def part_crossscale(settings, out_dir):
    """Linear correlation vs magnitude dependence across a scale boundary."""
    _, qt, nodes, _ = _load(settings)
    ti = TreeIndex(depth=qt.depth, branching=4)
    depth = qt.node_depth
    rows = []

    for oi, orient in enumerate(ORIENTATIONS):
        for d in range(qt.depth):
            parent_block = nodes[:, oi, depth == d]        # (B, 4**d)
            child_block = nodes[:, oi, depth == d + 1]     # (B, 4**(d+1))
            # Child j at level d+1 descends from parent j // 4 -- the breadth-first
            # rule, verified against the spatial map in tests/test_wavelet.py.
            parent = np.repeat(parent_block, 4, axis=1).ravel()
            child = child_block.ravel()

            # Scale-free versions of both variables.
            p_s = parent / parent.std()
            c_s = child / child.std()

            rho_lin = float(np.corrcoef(p_s, c_s)[0, 1])
            rho_abs = float(np.corrcoef(np.abs(p_s), np.abs(c_s))[0, 1])
            eps = 1e-12
            rho_log = float(np.corrcoef(np.log(p_s**2 + eps), np.log(c_s**2 + eps))[0, 1])

            # The decisive diagnostic: child spread conditioned on parent magnitude.
            q = np.quantile(np.abs(p_s), [0.25, 0.5, 0.75])
            lo = np.abs(p_s) <= q[0]
            hi = np.abs(p_s) >= q[2]
            std_lo, std_hi = float(c_s[lo].std()), float(c_s[hi].std())

            rows.append({
                "orientation": orient,
                "parent_depth": d,
                "child_depth": d + 1,
                "n_pairs": int(parent.size),
                "corr_linear": rho_lin,
                "corr_abs": rho_abs,
                "corr_log_sq": rho_log,
                "child_std_given_parent_q1": std_lo,
                "child_std_given_parent_q4": std_hi,
                "std_ratio_q4_q1": std_hi / std_lo if std_lo > 0 else float("nan"),
            })

    write_csv(out_dir / "crossscale.csv", rows)
    lin = np.array([abs(r["corr_linear"]) for r in rows])
    ratio = np.array([r["std_ratio_q4_q1"] for r in rows])
    print(f"[crossscale] |linear corr|: median {np.median(lin):.4f}, max {lin.max():.4f}")
    print(f"[crossscale] magnitude ratio q4/q1: median {np.median(ratio):.3f}, "
          f"max {ratio.max():.3f}")
    if lin.max() < 0.1 <= np.median(ratio) - 1.0:
        print("[crossscale] VERDICT: dependence is in the magnitude, not the mean. "
              "A linear-AR kernel is the wrong family; use a scale-mixture kernel.")
    return rows


PARTS = {"subbands": part_subbands, "crossscale": part_crossscale}


def main() -> None:
    parser = experiment_parser(
        "exp_23_wavelet_statistics",
        "CIFAR-10 wavelet coefficient statistics: the gate for the image extension.",
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings["n_images"] = 1000
        settings["n_boot"] = 40

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts), "provenance": provenance()})


if __name__ == "__main__":
    main()
