"""Experiment 24 -- fit the wavelet hidden Markov tree to CIFAR-10 and score it.

This is the first time the package's exact-BP machinery is applied to data it did
not generate itself. Everything is measured on a held-out split with the
normalisation and the subband scales estimated on training data only.

What this experiment establishes, and what it deliberately does not:

* It establishes that the pipeline runs end to end on real images -- transform,
  per-subband standardisation, exact per-orientation BP with a per-scale noise
  level, per-level M-step -- with exactly monotone EM and a finite exact held-out
  likelihood in *pixel* coordinates.
* It establishes the **Gaussian tree baseline**: the second-order closure that
  every non-Gaussian result has to beat. This is baseline (i) of the plan, and it
  is worth having on its own.
* It does **not** yet test the project's central claim on images. That needs a
  kernel whose conditional *scale* depends on the parent, because
  `exp_23 --only crossscale` shows the cross-scale dependence in CIFAR is largely
  in the magnitude, which a linear-autoregressive kernel cannot represent. Until
  that kernel exists, `mixture` below is reported as what it is: a linear-AR
  model with a flexible innovation *shape*, which improves the marginals and not
  the hierarchy.

Parts
-----
fit       Fit per-scale kernels, record the EM trace and the fitted parameters.
denoise   Held-out denoising MSE against the raw observation and the Gaussian
          closure, across noise levels the model was not fitted at.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    PACKAGE_ROOT, apply_overrides, experiment_parser, provenance, select_parts,
)
from src.image_data import load_cifar10_luminance
from src.kernels import GaussianAR1Kernel, MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel
from src.noising import alpha_delta
from src.utils import ensure_dir, rng_for, write_csv, write_json
from src.wavelet_model import fit_wavelet_tree

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    "levels": 5,
    "n_train": 2000,
    "n_test": 500,
    "t_train": (0.3, 0.6, 1.0),
    "t_eval": (0.2, 0.4, 0.8, 1.5),
    "n_iters": 25,
    "t_resolve": 0.05,
    "half_width": 8.0,
    "n_components": 4,
    "families": ("gaussian", "mixture"),
    "chunk": 64,
    "seed": 0,
}


def _archive(settings):
    from pathlib import Path

    p = Path(settings["archive"])
    return p if p.is_absolute() else PACKAGE_ROOT / p


def _splits(settings):
    train = load_cifar10_luminance(
        _archive(settings), "train", settings["n_train"], settings["seed"]
    )
    test = load_cifar10_luminance(
        _archive(settings), "test", settings["n_test"], settings["seed"],
        stats=(train.mean, train.std),
    )
    return train, test


def _factory(family: str, settings):
    if family == "gaussian":
        return lambda d, rng: GaussianAR1Kernel(rho=0.2, q=0.8)
    if family == "mixture":
        return lambda d, rng: MixtureInnovationKernel.init(
            settings["n_components"], rho=0.2, var=0.8, rng=rng
        )
    if family == "scale_mixture":
        return lambda d, rng: ScaleMixtureKernel.init(
            settings["n_components"], rho=0.2, var=0.8, rng=rng
        )
    raise ValueError(f"unknown family {family!r}")


def _fit(family, settings, train):
    t0 = time.perf_counter()
    model, trace = fit_wavelet_tree(
        train.images, levels=settings["levels"], t_train=list(settings["t_train"]),
        kernel_factory=_factory(family, settings), n_iters=settings["n_iters"],
        half_width=settings["half_width"], t_resolve=settings["t_resolve"],
        chunk=settings["chunk"],
    )
    return model, trace, time.perf_counter() - t0


def _aligned(rows):
    """Give every row the same keys, in first-seen order.

    `write_csv` takes its header from the first row alone, and `csv.DictWriter`
    raises on any later row carrying a key that header does not list. The rho
    columns differ by family by construction -- a scalar-rho family has no
    per-component ones and vice versa -- and `families` normally starts with
    `gaussian`, so the scale_mixture row appended after it is exactly the one
    that would raise. Missing entries are left empty rather than filled with a
    number, because "this family has no such parameter" is not a value.
    """
    headers = list(dict.fromkeys(key for row in rows for key in row))
    return [{h: row.get(h, "") for h in headers} for row in rows]


def part_fit(settings, out_dir):
    train, test = _splits(settings)
    rows, traces = [], []

    for family in settings["families"]:
        model, trace, secs = _fit(family, settings, train)
        ll = model.log_likelihood_images(test.images, settings["t_eval"][1])
        row = {
            "family": family,
            "n_train": settings["n_train"],
            "seconds": secs,
            "em_iters": len(trace.log_evidence),
            "final_log_evidence": trace.log_evidence[-1],
            "monotone_violation": trace.monotone_violation,
            "heldout_loglik_per_image": ll / len(test.images),
        }
        for d in range(model.depth):
            k = model.kernels[0][d]
            # `d` is already the detail level: `kernels[orientation][d]` holds one
            # kernel per (orientation, level). What varies *inside* one kernel is
            # the mixture component. GaussianAR1Kernel and MixtureInnovationKernel
            # carry a scalar `rho` shared by every component; ScaleMixtureKernel
            # carries one `rho_c` per component, shape (C,), and float() on that
            # array is what killed job 627175 on 2026-08-14. Give a vector kernel
            # one column per component instead of collapsing it -- differing
            # components are the whole point of the family, and a mean over them
            # would report a number the model never uses.
            rho = np.atleast_1d(getattr(k, "rho", np.nan)).astype(float)
            if rho.size == 1:
                row[f"rho_d{d}"] = float(rho[0])
            else:
                for c, value in enumerate(rho):
                    row[f"rho_d{d}_c{c}"] = float(value)
        rows.append(row)
        traces.extend(
            {"family": family, "iteration": i, "log_evidence": v}
            for i, v in enumerate(trace.log_evidence)
        )
        print(f"[fit] {family}: {secs:.0f}s, {len(trace.log_evidence)} iters, "
              f"monotone violation {trace.monotone_violation:.3g}, "
              f"held-out loglik/image {ll / len(test.images):.2f}")

    write_csv(out_dir / "fit.csv", _aligned(rows))
    write_csv(out_dir / "em_trace.csv", traces)
    return rows


def part_denoise(settings, out_dir):
    """Held-out denoising, including at noise levels not used for fitting.

    The point of the construction is that one fitted kernel is a denoiser at
    *every* t with no refitting, so t_eval deliberately straddles t_train.
    """
    train, test = _splits(settings)
    rng = rng_for("exp24", "denoise", settings["seed"])
    rows = []

    fitted = {f: _fit(f, settings, train)[0] for f in settings["families"]}

    for t in settings["t_eval"]:
        alpha, delta = alpha_delta(t)
        noisy = alpha * test.images + np.sqrt(delta) * rng.standard_normal(
            test.images.shape
        )
        row = {
            "t": t,
            "alpha": alpha,
            "delta": delta,
            "fitted_at_this_t": t in settings["t_train"],
            "mse_raw_rescaled": float(np.mean((noisy / alpha - test.images) ** 2)),
        }
        for family, model in fitted.items():
            hat = model.denoise_images(noisy, t, chunk=settings["chunk"])
            row[f"mse_{family}"] = float(np.mean((hat - test.images) ** 2))
            row[f"loglik_per_image_{family}"] = (
                model.log_likelihood_images(noisy, t, chunk=settings["chunk"])
                / len(test.images)
            )
        rows.append(row)
        print(f"[denoise] t={t}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k.startswith("mse_")
        ))

    write_csv(out_dir / "denoise.csv", rows)
    return rows


PARTS = {"fit": part_fit, "denoise": part_denoise}


def main() -> None:
    parser = experiment_parser(
        "exp_24_wavelet_fit",
        "Fit the wavelet HMT to CIFAR-10 and score it on held-out images.",
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings.update(
            n_train=300, n_test=100, n_iters=6, t_resolve=0.3,
            t_train=(0.5,), t_eval=(0.4, 0.8),
        )

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts), "provenance": provenance()})


if __name__ == "__main__":
    main()
