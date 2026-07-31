# BP-informed diffusion scores for (approximately) Markovian sequences

Continuation of the `bp_markov_diffusion_gaussian_approx` project. Research question:

> How can (approximate) Markovianity of sequence data be exploited to compute the
> diffusion score in a more informed way than a black-box network, and what does
> this reveal about reverse-diffusion dynamics?

## Headline results of this iteration

1. **Audit** (`audit/AUDIT_NOTE.md`): the previous package reproduces bit-for-bit,
   but its grid-projected Gaussian BP suffers a boundary-collapse artifact at
   weakly informative t that contaminated all large-t results; and its "Gaussian
   message error" is mathematically identical to the covariance-matched Gaussian
   *model* error (equivalence proposition), which changes the interpretation.
2. **Grid BP is spectrally accurate** (exp_01): errors sit at the 1e-15 floor
   until the likelihood width ~ sqrt(2t) drops below ~2 grid steps; truncation
   matters only for A <= 4. Reference default: M = 401, A = 8.
3. **The corrected Gaussian baseline** (analytic information-form BP, exp_02/03)
   fails exactly where theory predicts: error ~ (alpha_t/Delta_t) x mean error,
   worst at small t, growing with |innovation excess kurtosis| (bimodal worst),
   and *decreasing* with stronger correlation rho.
4. **Approximate Markovianity** (exp_04): for AR(1)+global-latent priors the
   non-Markov score correction is *exactly rank one* (Woodbury); a residual MLP
   on top of the frozen Markov-BP score beats a direct score MLP by ~20x in
   sample and parameter efficiency on exact-score supervision.
5. **Reverse dynamics** (exp_05): the Gaussian score reproduces second-order
   statistics of the Laplace chain but completely washes out heavy-tailed
   innovations (excess kurtosis 0.12 vs 2.7-2.9); the score error is dynamically
   stable in L2 (trajectory divergence 0.11 despite pointwise deviation 0.49 at
   small t) yet distributionally decisive in higher moments.

## Layer 5: learning the prior (exp_06, exp_07)

Everything above hands BP the true prior. Layer 5 removes it: the transition
kernel of the clean chain becomes an unknown parameter, only *noised* sequences
are observed, and the kernel is estimated by maximum marginal likelihood with
EM. The theory is written up in `report/em_bp_learning.tex`.

The structural points, in one place:

- **The E-step is exact.** The posterior of a chain under sitewise likelihood
  factors is a chain, chains are trees, so BP gives exact pairwise marginals.
  No variational bound, no loopy approximation.
- **The whole E-step is one matrix.** All of it compresses into an `M x M`
  matrix `Xi` — the continuum analogue of Baum-Welch's expected transition
  counts — which is sufficient and independent of how the kernel is
  parameterized. The M-step never touches the data again, and observations at
  different noise levels simply add into the same `Xi`.
- **No autodiff, anywhere.** Fisher's identity makes `<Xi, grad log K>` the
  exact gradient of the marginal log-likelihood, so BP is never
  differentiated through: it supplies the expectation. Verified against finite
  differences of the exact evidence.
- **One fit serves every noise level.** The learned kernel lives on `R x R` and
  has no `t` in it; the noise level enters only through the likelihood factors
  inside BP. A score network must instead learn a function on `R^n x R_+`
  across the whole schedule.

Four kernel rungs, increasing in expressivity, all consuming the same `Xi`:
Gaussian AR(1) and Laplace AR(1) (closed-form M-steps), a mixture-innovation
kernel that has never heard of the Laplace density it must recover, and a
mixture-density-network kernel whose gradient touches the network only at the
`M` grid points regardless of dataset size.

Two discretization artifacts are recorded rather than smoothed over, both
specific to the non-smooth Laplace kernel and absent for the smooth ones: its
`rho`-gradient loses accuracy to a `sign` discontinuity under trapezoidal
quadrature, and its exact M-step is quantized onto the grid's ratio lattice.
Reported rates therefore use the smooth kernels.

The comparison in exp_07 is set up to favour the network: the data budget is
counted in *clean* chains, of which the network gets paired `(a, x)` with a
fresh noise draw every gradient step while EM gets one noisy realization each
and never sees a clean chain; width and training budget are swept; and both
standard parameterizations (noise prediction and clean-signal prediction) are
trained and reported. The baseline is a vanilla MLP, as specified — a
temporal-convolution or U-Net baseline would carry a locality prior of its own
and is the natural next comparison, not something settled here.

## Layout

    src/          core library (priors, noising, grid BP, Gaussian BP, exact
                  scores, Markov approximations, numpy MLP, reverse samplers,
                  EM + parameterized kernels + denoiser comparison)
    experiments/  exp_01 ... exp_07, all with --quick smoke mode
    outputs/      CSV + JSON + PNG per experiment (committed results)
    notebooks/    executed analysis notebooks 01-04
    tests/        pytest suite (30 tests)
    report/       updated_report.tex / .pdf   (Layers 1-4)
                  em_bp_learning.tex / .pdf   (Layer 5 theory)
    audit/        Layer-1 audit note

## Reproduce

Requirements: Python >= 3.12, numpy, scipy, matplotlib, pandas (analysis),
pytest (tests), jupyter/nbformat (notebooks). `pip install -r requirements.txt`.

```bash
python -m pytest tests/ -q                     # verify the core
python experiments/exp_01_grid_validation.py   # each writes to outputs/<name>/
python experiments/exp_02_laplace_gaussian_message_error.py
python experiments/exp_03_nongaussian_innovation_sweep.py
python experiments/exp_04_approx_markovianity.py
python experiments/exp_05_reverse_dynamics.py
python experiments/exp_06_em_parameter_recovery.py
python experiments/exp_07_em_vs_score_network.py
```

`tests/test_em_bp.py` runs many small EM fits and takes ~6 minutes; the rest of
the suite is seconds. Experiments 06 and 07 take roughly an hour each at full
settings — use `--quick` for a minutes-scale smoke run.

Every experiment: deterministic seeds via `src.utils.rng_for` (common random
numbers across compared methods), full parameter dump to `params.json`,
tabular results to CSV, figures to PNG. Add `--quick` to any experiment for a
minutes-scale smoke run.

## Conventions

Forward process: dX = -X dt + sqrt(2) dW, so x_t = e^{-t} a + sqrt(1-e^{-2t}) z.
Score identity: s(x,t) = -(x - alpha_t E[a|x]) / Delta_t.
Error identity: s_hat - s_ref = (alpha_t/Delta_t)(m_hat - m_ref), verified to
machine precision in every experiment (column `identity_residual`).

## Note on reproducibility (fixed 2026-07-31)

`src/utils.rng_for` derived its seeds from Python's builtin `hash`, which is
salted per process for strings (PEP 456). Because every call mixes in a string
tag, **no experiment in this package was bit-reproducible from a fresh
interpreter**, contrary to what this README and the module docstring claimed.
The paired "common random numbers" property that the method comparisons rely on
held *within* each run, so previously reported numbers stand as measurements —
they were simply not reproducible. `rng_for` now uses a fixed digest, verified
identical across processes. Experiments 06 and 07 were run under the fix;
outputs committed for exp_01–exp_05 predate it and have not been regenerated.
