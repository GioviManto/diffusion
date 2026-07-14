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

## Layout

    src/          core library (priors, noising, grid BP, Gaussian BP, exact
                  scores, Markov approximations, numpy MLP, reverse samplers)
    experiments/  exp_01 ... exp_05, all with --quick smoke mode
    outputs/      CSV + JSON + PNG per experiment (committed results)
    notebooks/    executed analysis notebooks 01-04
    tests/        pytest suite (14 tests)
    report/       updated_report.tex / .pdf
    audit/        Layer-1 audit note

## Reproduce

Requirements: Python >= 3.12, numpy, scipy, matplotlib, pandas (analysis),
pytest (tests), jupyter/nbformat (notebooks). `pip install -r requirements.txt`.

```bash
python -m pytest tests/ -q                     # verify the core (fast)
python experiments/exp_01_grid_validation.py   # each writes to outputs/<name>/
python experiments/exp_02_laplace_gaussian_message_error.py
python experiments/exp_03_nongaussian_innovation_sweep.py
python experiments/exp_04_approx_markovianity.py
python experiments/exp_05_reverse_dynamics.py
```

Every experiment: deterministic seeds via `src.utils.rng_for` (common random
numbers across compared methods), full parameter dump to `params.json`,
tabular results to CSV, figures to PNG. Add `--quick` to any experiment for a
minutes-scale smoke run.

## Conventions

Forward process: dX = -X dt + sqrt(2) dW, so x_t = e^{-t} a + sqrt(1-e^{-2t}) z.
Score identity: s(x,t) = -(x - alpha_t E[a|x]) / Delta_t.
Error identity: s_hat - s_ref = (alpha_t/Delta_t)(m_hat - m_ref), verified to
machine precision in every experiment (column `identity_residual`).
