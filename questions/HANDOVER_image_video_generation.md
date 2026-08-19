# Handover: images, video, reverse diffusion, and sample metrics

**For a fresh chat.** Written 19 August 2026 from the state of `main` at that
date. This is the exploratory branch of the project — none of it feeds the
paper, the workshop or the thesis, all of which are about 1-D Markov chains and
are close to done. You have room to be wrong here in a way the main line does
not.

Repo root: `/Users/gloriabagnato/Code/Thesis/Diffusion`.
Package: `research/nongaussian-bp/`. Run everything from there; the venv is
`./.venv/bin/python`.

---

## 1. What the project claims, in one paragraph

Coordinatewise Gaussian noising multiplies a prior by **unary** factors. Those
reweight node potentials without creating edges, so the posterior factor graph is
whatever the prior's was. If the prior is a chain, the posterior is a chain — a
tree — and sum-product returns the **exact** score. That is the whole mechanism,
and it is why this project can compute a target that is unavailable almost
everywhere else.

## 2. Why images break it, and the route around

The pixel lattice has **loops**. Sum-product on a loopy graph is not exact, so
the central claim dies on images taken at face value.

A multiscale wavelet decomposition restores a tree: parent coefficient → four
children, no cycles. That substrate is **built and verified** —
`src/wavelet.py`, `src/wavelet_bp.py`, `src/wavelet_model.py`,
`src/wavelet_stats.py`, with `exp_23`/`exp_24`/`exp_25`.

## 3. The measurement that changed the plan — read this before writing code

The plan was to reuse `MixtureInnovationKernel` per scale. **That family is
wrong for wavelets**, and it is wrong in a way that will look like a working
result if you do not check for it.

The kernel is linear-autoregressive, `K(a'|a) = φ(a' − ρa)`, so the parent's
entire influence is a **shift of the innovation's location**. Its conditional
variance does not depend on the parent at all. But measured on CIFAR
(`outputs/exp_23_wavelet_statistics/crossscale.csv`, 2.56 M pairs per
orientation at the finest scale boundary):

| orientation | linear corr | Q4/Q1 std ratio | linear-AR null | **excess** |
|---|---|---|---|---|
| HL | 0.452 | 3.65 | 1.32 | **2.33** |
| LH | 0.482 | 3.29 | 1.36 | **1.92** |
| HH | **0.148** | 2.89 | **1.03** | **1.86** |

Wavelet children depend on their parent's **magnitude**, strongly, and a
linear-AR kernel cannot express any of it. HH is the sharp case: linear
correlation is only 0.15, so a linear-AR fit returns ρ ≈ 0 and collapses to an
independent model, while the magnitude dependence — the actual structure — is
untouched.

**The null matters.** The raw Q4/Q1 ratio is not a pure measure of magnitude
dependence: conditioning a child on a *set* of parent values also picks up the
spread of the conditional mean across that set, so a perfectly homoscedastic
AR(1) already scores 1.31 at ρ = 0.45. The quantity that carries the argument is
the **excess** over that null — `scale_kernel.linear_ar_magnitude_ratio`,
verified against simulation to three decimals. An earlier draft omitted the null
and the number looked much better than it was.

So: a scale-mixture / magnitude-dependent kernel is the thing to build.
`src/scale_kernel.py` exists and is the start of it.

## 4. Known bug, still open

`exp_24_wavelet_fit.py:125` does `float(getattr(k, "rho", np.nan))`. The
scale-mixture kernel's `rho` is a **vector**, not a scalar, so this raises. Job
`627175` died on it in 8 minutes. It was left unfixed because wavelets are out
of scope for the thesis — fix it before running `exp_24`.

## 5. What exists for generation and metrics

**Reverse dynamics** — `src/reverse.py`: `reverse_sde`, `probability_flow_ode`,
`time_grid`, `nested_brownian_path` (shared Brownian increments across
resolutions, so a coarse and a fine integration of the same path are comparable),
`denoising_readout`. Driven by `exp_05_reverse_dynamics.py`.

**Sample metrics** — `src/sample_metrics.py`: `compare_distributions`,
`histogram_kl`, `covariance_error`, `excess_kurtosis`, `bootstrap_se`,
`ar_residuals`, `residual_autocorrelation`. All 1-D / second-order. **There is
no IS and no FID**, because nothing so far needed an Inception network.

**Video** — `src/video_bp.py` (caterpillar BP), `src/video_model.py`,
`src/video_data.py`, `exp_26_video.py`.

## 6. IS and FID — what you actually have to build

Both need a pretrained Inception-v3 and both are conventions as much as metrics.
Getting them *comparable to published numbers* is mostly about matching
conventions, not about the maths:

- **FID** wants pool3 activations (2048-d), images resized to 299×299 with
  **bilinear** interpolation, and the standard `torchvision` or TF-Slim weights.
  Numbers from different Inception weights are not comparable, and this is the
  single most common source of FID disagreement in the literature.
- **FID is biased at small n.** It is a plug-in estimator of a Fréchet distance
  between fitted Gaussians, so it falls as the sample grows and comparing two
  models at different n is meaningless. Fix n (50 k is the convention) or report
  the curve.
- **IS is weaker than it looks** and says nothing about diversity within a class.
  If you report it, report FID beside it.
- This project's venv is numpy/cupy, **not** torch. Adding torch to
  `research/nongaussian-bp/.venv` for this is fine — but keep it out of the
  paper's dependency path, and pin it, because the cluster has a load-bearing
  `cupy < 14` constraint (14.x passes matmuls and fails reductions).

Sanity checks before trusting any number: FID(real, real) on two disjoint halves
of the same dataset should be small but **not zero** — that value is your noise
floor. FID should increase monotonically under added Gaussian blur or noise.

## 7. Two traps this project has already fallen into

**Do not compare estimators at a fixed budget and call it a comparison of
estimators.** The optimisation budget is a regularisation knob here, not a
convergence detail: the finite-sample maximiser is not at the truth, so more
ascent is not more accuracy. This has bitten three times — EM's outer loop, the
mixture M-step's inner sweeps, and Rung 4a's gradient ascent. If two arms have
different cap-hit rates you are comparing budgets, not methods.

**Do not report a quantile as a bound.** A boundary statistic quoted as
"the truncation residual is 1.4e-8" was the 90th percentile over chains; the
worst chain was 1.2e-6, 84× larger. Say which statistic you computed.

## 8. Cluster

`ssh 3164542@lnode02-da.hpc.unibocconi.it` — use the login node directly;
`hpc.unibocconi.it` frequently NXDOMAINs on a healthy VPN.

- Working dir mirrors the repo: `~/nongaussian-bp/research/nongaussian-bp`.
- GPU: `medium_gpuh200` (6 h), `short_gpuh200`, `debug_gpuh200` (needs
  `--qos=debug`). cupy 13.6.0, inside the pin.
- `MODE=gpu` in `hpc/bocconi_frozen.sbatch` asserts the device is reachable and
  performs a real cupy **reduction** before running the parity suite, aborting on
  any of the three. Keep all of it. The parity suite on its own gates nothing:
  every test in it skips when there is no device, and **an all-skipped pytest run
  exits 0** — job `631477` passed that way on an H200 it could not use. The
  reduction is separate from the import check because cupy JIT-compiles reduction
  kernels: against the system-default CUDA 13.3 headers that compile fails while
  cuBLAS keeps working, so matmuls pass and every `.sum()`/`.max()` in the
  recursion dies mid-sweep. **Source `~/nongaussian-bp/gpu_env.sh`** — it pins
  `CUDA_PATH` to 12.4, and without it cupy cannot see the card at all.
- **`--propagate=NONE` is mandatory.** The login node has `ulimit -t 600` and
  Slurm propagates the submitter's rlimits, so without it a 6-hour job is killed
  by SIGXCPU after ten minutes of CPU time. An array died to this on 18 Aug and
  the logs showed nothing — `sacct` showed `TotalCPU` pinned at exactly 10:00.
- QOS: 30 jobs submitted, 10 running. Array tasks count individually.
- Stamp provenance before rsync: `hpc/stamp_revision.sh` writes `REVISION`,
  which `provenance()` falls back to because the deployed tree has no `.git`.

## 9. Suggested first moves

1. Fix the `exp_24` rho-vector bug and get one scale-mixture fit to run.
2. Build FID with the conventions above; validate it with the real-vs-real noise
   floor and the blur monotonicity check **before** pointing it at any sample.
3. Only then generate. Reverse-integrate under the fitted wavelet model and
   compare against a DDPM baseline at matched n.

Do not start at step 3. The generation study in the thesis got demoted to a
recorded negative result precisely because it rested on fits that had not
converged, and no amount of careful sampling repairs that.

## 10. Where to read more

- `research/nongaussian-bp/IMAGE_EXTENSION_STAGE1.md` — the full Stage-1 report.
  Dated 8 Aug; its efficiency figures are superseded, its wavelet content is not.
- `overleaf/compendium/` chapter 11 — the claim ledger. **Check it before citing
  any number**: it says what is settled, conditional, censored or withdrawn.
- `overleaf/paper/` — the main line, for the method these extensions build on.
