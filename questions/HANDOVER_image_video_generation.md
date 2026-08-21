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

## 4. The rho-vector bug: fixed, and now exercised

`exp_24_wavelet_fit.py:125` used to do `float(getattr(k, "rho", np.nan))`. The
scale-mixture kernel's `rho` is a **vector**, so this raised; job `627175` died
on it in 8 minutes.

Fixed in `7477a92` (17 Aug) — two days before the first draft of this document
was written, which still listed it as open. The fix gives a vector kernel one
`rho_d{d}_c{c}` column per component rather than collapsing it, with `_aligned`
reconciling the ragged header across families.

**First actually exercised 19 Aug**, job `631488`
(`MODE=imagefit_scale_calibrate`, n_train=200, n_iters=5, 22 min). Until then no
scale-mixture fit had ever run: both `fit.csv` files on disk were dated 12 Aug
and held only `gaussian`/`mixture` rows. `outputs/exp_24_scale_calibrate/fit.csv`
now has 27 columns — 4 scalar `rho_d{d}` for the scalar families, 16
`rho_d{d}_c{c}` for scale_mixture, non-applicable cells empty. The components
differ substantially (depth 1 spans 0.209–0.454), so the old `float()` collapse
would have reported ~0.317, a number the model never uses.

Calibration timings, which is what full-size walltimes should be sized from:
gaussian 176 s, mixture 227 s, scale_mixture 183 s. Cost is linear in `n_train`
and `n_iters`, so full size (2000/25) is ×50: ≈2.4 h, 3.2 h, 2.5 h, ≈8.1 h total.
**That does not fit `medium_cpu`'s 6 h cap** — use `compute`, or the GPU path in
§4a.

Denoising MSE from that run, held out:

| t | gaussian | mixture | scale_mixture |
|---|---|---|---|
| 0.2 | 0.1177 | 0.1200 | **0.1118** |
| 0.4 | 0.1760 | 0.1768 | **0.1693** |
| 0.8 | 0.2667 | 0.2655 | **0.2602** |
| 1.5 | 0.4233 | 0.4228 | **0.4207** |

scale_mixture wins at every level and the margin grows as t falls, which is the
direction §3 predicts; `mixture` is at best level with `gaussian`, which is the
"marginals, not hierarchy" claim showing up in a measurement. **Do not quote
any of this.** It is 5 EM iterations at n_train=200. Trap 1 in §7 is exactly
this situation, and `rho` alone needs ~25 iterations to settle. The full-size
run is job `631500`.

## 4a. The quadtree runs on a GPU as of 19 Aug 2026

`src/backend.py` was wired into `bp_grid.py` and `denoiser.py` only, so
`BP_DEVICE=gpu` was a **silent no-op for every image and video experiment** —
exp_23 through exp_26 all go through `wavelet_tree_bp`, which was pure numpy.
The recursion is now parameterised by its array module the same way the chain
is, through `wavelet_model.fit_wavelet_tree(..., xp=)` and
`video_bp(..., xp=)`; `xp=None` at the application layer reads `BP_DEVICE`,
while `wavelet_tree_bp` itself defaults to numpy so an unrelated environment
variable cannot silently move a library call to a device.

`wavelet_tree_bp` returns **numpy** whatever device it ran on, unlike
`grid_bp_batch`. The M-step and every experiment downstream are host code, and
the results are small (O(B × n_nodes), O(M×M)) beside an O(B × n_nodes × M²)
recursion, so converting at that boundary left all callers unchanged.

Parity coverage is in `tests/test_backend_parity.py`: posterior means, log
evidence *and* the per-level Xi, on uniform and per-depth meshes, at deltas
spanning decades. Xi is included deliberately — a device disagreement that left
the means intact but perturbed the expected transition counts would corrupt
every fitted kernel while looking fine in a denoising plot.

`MODE=gpu_gate` in `hpc/bocconi_wavelet.sbatch` checks the *device*, not
pytest's exit code: every parity test skips without a card, an all-skipped run
exits 0, and job `631477` did several hours of CPU work wearing a GPU job's name
that way. Gate before any GPU run.

**Verified**, job `631499` on an H200 NVL, 35 s: 17/17 parity tests passed with
**no skips**, and on the real exp_24 problem shape (per-depth grids
[1593, 771, 349, 151, 65], batch 256) one E-step went

| device | time | log evidence |
|---|---|---|
| CPU (numpy) | 3.44 s | −1.436776565e+05 |
| GPU (H200 NVL) | **0.08 s** | −1.436776565e+05 |

**43× faster, rel mean difference 2.4e-16, rel log-evidence difference exactly
zero.** Note the speedup is on the E-step only — the M-step stays on the host,
so the end-to-end fit gains less than 43×.

## 5. What exists for generation and metrics

**Reverse dynamics** — `src/reverse.py`: `reverse_sde`, `probability_flow_ode`,
`time_grid`, `nested_brownian_path` (shared Brownian increments across
resolutions, so a coarse and a fine integration of the same path are comparable),
`denoising_readout`. Driven by `exp_05_reverse_dynamics.py`.

**Sample metrics** — `src/sample_metrics.py`: `compare_distributions`,
`histogram_kl`, `covariance_error`, `excess_kurtosis`, `bootstrap_se`,
`ar_residuals`, `residual_autocorrelation`. All 1-D / second-order.

**IS and FID** — `src/fid.py`, built 19 Aug; see §6.

**Video** — `src/video_bp.py` (caterpillar BP), `src/video_model.py`,
`src/video_data.py`, `exp_26_video.py`.

## 6. IS and FID — built, `src/fid.py`

Both are conventions at least as much as statistics, and the conventions are
where the number goes wrong. All four are fixed explicitly in the module:
pool3 (2048-d), **bilinear** resize to 299×299, torchvision ImageNet weights
recorded in `ActivationStats.weights` so a stored number carries its own
provenance, and ImageNet mean/std normalisation.

The module refuses rather than warns in the two cases that produce a
plausible-looking wrong answer:

- **Mismatched n.** FID is a plug-in estimator of a Fréchet distance between
  *fitted* Gaussians, so the fit error inflates it and it falls as n grows.
  Comparing two models at different n is decided by the sample sizes.
  `fid_from_stats` raises; use `bias_curve` if the sizes genuinely differ.
- **n ≤ 2048.** Below the activation dimension the sample covariance is
  singular, so the estimate is rank-deficient, not merely noisy.
  `fid_from_samples` raises unless `allow_small_n=True` (tests only).
- Mismatched Inception weights also raise: torchvision and TF-Slim numbers are
  not comparable, which is the commonest source of FID disagreement in the
  literature.

`inception_score` takes the conventional 10-way split, and note that the split
is not cosmetic — p(y) is estimated *within* each split, so IS at different
`n_splits` is not comparable either. IS says nothing about within-class
diversity; `tests/test_fid.py` asserts that blindness directly so it cannot be
forgotten in reporting. Report FID beside it or not at all.

**Environment.** torch lives in a **separate** `.venv-metrics` on the cluster
(torch 2.6.0+cu124, torchvision 0.21.0+cu124), *not* in the package venv. That
venv has cupy 13.6 against the CUDA 12.4 module and the pin is load-bearing for
every exact-BP number in the project; a second CUDA stack in the same
environment is a risk with no upside, since nothing in the metrics path imports
`src.wavelet*`. `MODE=fid_validate` uses the right interpreter. Cost ≈ 6 GB
against a home quota that was at 151 G of 180 G on 19 Aug.

**One trap, found the hard way.** Every reference FID implementation writes
`scipy.linalg.sqrtm(A, disp=False)` and unpacks a 2-tuple. `disp` was removed
from scipy, and both the laptop and the cluster are on **scipy 1.18**, where
that call raises `TypeError`. Use the single-argument form.

### Validating it — `exp_30_fid_validation.py`, run this before scoring anything

Three parts, each checked against an answer known in advance. `floor` is
FID(real, real) on two disjoint halves: small but **not zero**, and its value is
the resolution of every later comparison. `blur` is FID against a
progressively blurred copy, which must *increase* — assumption-free, and it
catches the resize, channel and normalisation errors that are otherwise
invisible. `bias` is FID against n, which must *fall*.

The blur check earns its place: the first version of the blur used numpy's
`reflect` padding, which drops the edge sample where `scipy.ndimage`'s `reflect`
repeats it, and disagreed with `ndimage.gaussian_filter` by up to 0.13 on
[0, 1] data. Interior pixels were exact, so an interior-only check would have
passed — and Inception sees the edges. `mode="symmetric"` is correct; there is a
regression test at machine precision.

**Results**, job `631514`, H200 NVL, 4 min 44 s, 20 000 CIFAR luminance images
(`outputs/exp_29_fid_validation/` — the directory name predates the exp_30
rename):

- **Floor: FID(real, real) = 4.5011 ± 0.0339** at n = 10 000 per half. *This is
  the resolution of every FID comparison made at that n.* A model gap smaller
  than ~4.5 is not a gap.
- **Blur: monotone.** 4.48 → 10.01 → 72.45 → 174.99 → 332.22 for
  σ = 0, 0.5, 1, 2, 4. The pipeline is wired correctly.
- **Bias: 17.71 (n=2500) → 8.90 (n=5000) → 4.48 (n=10 000).** It halves as n
  doubles — a clean 1/n, both ratios 1.99. This is the concrete form of "you
  cannot compare two models at different n": at n=2500 the *same distribution*
  scores 17.7 against itself.

One trap in the bias table, since it is the row most likely to be quoted: at the
largest n the subsample is the whole sample, so the repeats draw the same set
and the spread is identically zero. `bias_curve` now reports `resampled: False`
and `fid_std: nan` there rather than a 0.0000 that looks like a measurement.

**The extrapolation was tested and it over-predicts.** Job `632871` measured the
floor directly on the full 50 000-image train split: **1.8024 ± 0.0111 at
n = 25 000 per half**, against 1.99 extrapolated from the n <= 10 000 fit — 10.2%
high. The reason was the one given in advance: the local exponent steepens from
-0.9228 over 625->10 000 (d/n 3.28 -> 0.205) to **-0.9921** over
10 000 -> 25 000 (d/n 0.205 -> 0.082), approaching the asymptotic 1/n as the
Marchenko-Pastur ratio falls. **Use ~0.91 for the floor at the conventional
n = 50 000**, from the local exponent one doubling out, not 1.04 from the global
fit. Extrapolate this curve forwards only one doubling at a time.

### What the bias actually is — `--only biaslaw`, job `631527`

FID splits exactly into a mean term and a covariance (Bures) term, and only the
first has a known law: for two independent size-n samples from one distribution,
`E |mu1 - mu2|^2 = 2 Tr(Sigma) / n`, exactly, no Gaussianity needed. Measured on
CIFAR pool3 (d = 2048, **Tr(Sigma) = 188.56**):

| n | FID | mean term | predicted `2Tr(S)/n` | covariance term | mean share |
|---|---|---|---|---|---|
| 625 | 57.79 | 0.523 | 0.603 | 57.27 | 0.9% |
| 1250 | 33.74 | 0.267 | 0.302 | 33.47 | 0.8% |
| 2500 | 17.64 | 0.140 | 0.151 | 17.50 | 0.8% |
| 5000 | 8.96 | 0.079 | 0.075 | 8.88 | 0.9% |
| 10000 | 4.47 | 0.037 | 0.038 | 4.44 | 0.8% |

**The bias is ~99% covariance estimation.** The mean term is under 1% throughout
and matches its closed form. That is not a curiosity — it says the part of the
bias with a provable 1/n law is negligible, and the part that carries it is
estimating ~2.1 M covariance parameters from n samples at n/d between 0.3 and 5.

Which is why the fitted law is **n^(-0.93)**, not n^(-1) (max |log residual|
0.059). A shallower-than-1/n exponent is what the Marchenko-Pastur regime
predicts when d/n is not small, and it should steepen towards −1 as n grows. So
the extrapolations — 1.99 at n = 25 000, 1.04 at n = 50 000 — are extrapolating
the covariance term *across a change in its own regime*, and are likely slight
over-estimates. Job `632682` measures n = 25 000 directly against the predicted
1.99, which is the test of whether the law can be pushed to 50 000 at all.

### The consequence for step 4, which is the reason any of this was worth doing

**At n = 10 000 the floor is 4.47, and that is very probably larger than the
difference the generation study is trying to detect.** The calibration puts
scale_mixture ahead of the Gaussian closure by ~5% in denoising MSE at the most
favourable t. Percent-level quality differences on CIFAR are worth roughly
0.5–2.5 FID units for models in the usual 10–50 range — at or under the floor.
A comparison run at n = 10 000 would very likely return "undecided" and read as
"close".

So: run step 4 at n = 50 000 (also the conventional number, and the full CIFAR
train set as reference), where the floor is ~1. Even then the margin is thin.

And note what that implies about instrument choice. This project computes an
**exact** held-out log-likelihood in pixel coordinates — that is its entire
comparative advantage over any DDPM baseline, and it carries no 4.47 noise
floor. Leading the generation study on FID would be adopting the blunter
instrument in precisely the place where the project owns the sharper one. Report
FID because reviewers expect it; decide on the likelihood.

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

## 9. Where the work stands, and what is next

Steps 1 and 2 of the original plan are done. The ordering below is unchanged,
and it is still the point of this section.

1. ~~Fix the `exp_24` rho-vector bug and get one scale-mixture fit to run.~~
   Done — §4. Fix was already in `main`; the *run* was the missing half, and
   job `631488` closed it.
2. ~~Build FID and validate it before pointing it at any sample.~~ Built and
   tested (`src/fid.py`, 30 tests); validation is `exp_30_fid_validation.py`,
   §6. Blur monotonicity passes.
3. **Here now: the converged full-size fit.** Job `631500` (`compute`, 18 h,
   n_train=2000, n_iters=25, three families) is the number to quote — nothing
   from the calibration is. A GPU arm runs the same settings into a separate
   directory once `MODE=gpu_gate` passes, which also cross-checks the new
   device path at production size rather than only on parity-test shapes.
4. **Only then generate.** Reverse-integrate under the fitted wavelet model and
   compare against a DDPM baseline at matched n.

Do not start at step 4, and note that step 3 is now the easier one to wave
through: the fit *runs*, so the failure mode is no longer a crash but a fit
that is quietly short of convergence. The generation study in the thesis got
demoted to a recorded negative result for exactly that reason, and no amount of
careful sampling repairs it. Check the EM trace before using the fit — `rho`
alone needs ~25 iterations, and innovation shape takes ~120.

Before quoting any FID, run `exp_30` at the n you will report at, and check the
model gap against the floor from that same run.

## 10. Where to read more

- `research/nongaussian-bp/IMAGE_EXTENSION_STAGE1.md` — the full Stage-1 report.
  Dated 8 Aug; its efficiency figures are superseded, its wavelet content is not.
- `overleaf/compendium/` chapter 11 — the claim ledger. **Check it before citing
  any number**: it says what is settled, conditional, censored or withdrawn.
- `overleaf/paper/` — the main line, for the method these extensions build on.
