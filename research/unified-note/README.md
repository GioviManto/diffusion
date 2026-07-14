# Unified working note

`unified_document/` is the consolidated reference for the joint-score
programme on AR(1) chains under Ornstein--Uhlenbeck diffusion. It
collapses five session-summary PDFs (under `Research/Relevant/`) into a
single LaTeX manuscript with reproducible code and figures.

## What is here

```
unified_document/
├── main.tex                              -- unified manuscript (~25 pages)
├── main.pdf                              -- compiled output
├── README.md                             -- this file
├── RESEARCH_INDEX.md                     -- 5-tier map of Research/
├── code/
│   ├── ar1_utils.py                      -- Gaussian AR(1) + OU benchmark
│   ├── bp_score.py                       -- belief propagation / Kalman
│   ├── laplace_score.py                  -- Laplace K=1 warm-up by quadrature
│   ├── numerical_audit.py                -- 53 closed-form-vs-reference checks
│   └── generate_figures.py               -- produces all 8 PNGs deterministically
└── figures/
    ├── fig01_sigma_vs_precision.png      -- Sigma_0 dense vs Q_0 tridiagonal
    ├── fig02_factor_graph.png            -- chain factor graph (tree, no cycles)
    ├── fig03_precision_annotated.png     -- Q_t with values in cells, 6 t values
    ├── fig04_band_fill.png               -- |Q_t[i,i+d]| ~ t^{d-1} log-log
    ├── fig05_bp_audit.png                -- BP vs matrix joint score residuals
    ├── fig06_K1_density.png              -- Laplace vs Gaussian noised marginals
    ├── fig07_K1_score.png                -- Laplace vs Gaussian K=1 joint score
    └── fig08_K1_score_limits.png         -- K=1 score asymptotes
```

## Manuscript structure

The note has eleven numbered sections plus reproducibility:

1. **Introduction and motivation** — score as denoiser, factor graph
   as inductive bias.
2. **Setup, notation, and the OU forward channel** — full SDE
   integration to the discrete-time channel.
3. **The score is a denoiser: Tweedie's identity** — full derivation,
   three equivalent readings (denoiser form, expected-noise form,
   nonlinearity tracks the prior).
4. **Bayesian view of the chain posterior** — prior-likelihood
   factorisation, the marginalisation problem and the O(K^2) cost.
5. **Factor graph and tree structure** — counting argument, global
   Markov property.
6. **Belief propagation: from sum--product to Kalman** — Convention A
   derivation, Gaussian closure (three lemmas with proofs), Kalman
   filter as forward pass, RTS smoother as backward pass, O(K)
   complexity.
7. **Act I — Gaussian AR(1) benchmark** — full derivation pipeline:
   stationary Sigma_0, tridiagonal Q_0, noised covariance, three forms
   of Q_t, spectral preservation, annotated precision lifecycle,
   corrected band-fill theorem, large-t asymptotic.
8. **Act II — Laplace K=1 warm-up** — Tweedie carries through,
   nonlinearity emerges, two limits.
9. **Act III — Laplace K≥2** — chain factor graph survives, forward
   message clean, backward message structural; open structural
   questions enumerated.
10. **Numerical audit summary** — 53 / 53 PASSED.
11. **Closing argument: the factor graph as inductive bias** — the
    structure-aware vs universal-approximator framing, Bayesian
    operator's intuition, convolutional analogy, programme directions.

## Coloured derivation boxes

Every non-trivial claim in the manuscript is paired with a coloured
box that contains its full derivation:

* **Blue** — Derivation / Computation: full proofs and explicit
  algebraic computations.
* **Beige** — Bayesian view: interpretive commentary on what each
  identity says probabilistically.
* **Green** — Intuition: the high-level take-away after the
  derivation.
* **Purple** — Numerical audit: link between a claim and the
  corresponding `numerical_audit.py` check.

## What was deliberately cut

The cuts requested for the consolidation are applied as follows:

* `Gaussian_session_summary.pdf` -- section 6 on the Hessian field
  (Stein identity, `H_t` formula, limits, within-quadrant precision)
  is omitted.
* `session_summary_K1_warmup.pdf` -- section 3.3 (truncated-Gaussian
  closed-form derivation) and Appendix B (Mills-ratio asymptotic) are
  omitted; quadrature in `laplace_score.py` is sufficient for every
  quantitative claim.
* `precision_lifecycle_summary.pdf` -- only results with algebraic
  proof or numerical verification at the audit's tolerances are
  retained. The published band-fill formula (eq. 12 of that PDF)
  contains an extraneous `1 / (d - 1)!` factor; we use and verify the
  corrected version `(-1)^{d-1} (2 t)^{d-1} (Q_0^d)_{i, i+d}`.
* `bp_session_summary.pdf` -- the BP code (`code/bp_score.py`) is
  rewritten from scratch under Convention A and validated; only
  identities confirmed by `numerical_audit.py` are asserted.
* `numerical_experiments.pdf` -- section 6 on Conjecture 15 (HPC
  pending, regime-dependent at quick resolution) and figures 15-17
  (K = 1 proxy mislabeled as K = 2) are omitted.

## How to reproduce

```bash
cd Score_Diffusion/Research/unified_document/code
python3 numerical_audit.py     # MUST report 53/53 PASSED
python3 generate_figures.py    # writes ../figures/fig0{1..8}_*.png
cd ..
tectonic main.tex              # preferred: single-binary, downloads packages
# or:  pdflatex main.tex && pdflatex main.tex   # second pass for the TOC
```

The bundled `main.pdf` was compiled with `tectonic` (single-binary
LaTeX engine, available via `brew install tectonic`). Running
`pdflatex` twice produces an equivalent PDF with the same content;
fine-grained spacing may differ by a few points because the two
engines load fonts and apply microtype differently.

The Python stack is `numpy + scipy + matplotlib` only; no neural
network or stochastic training code appears anywhere.

## Audit status

As of compile date, `numerical_audit.py` reports **53 / 53 PASSED**.
The audit is the gate: any claim labelled `[Established]` in `main.tex`
either has an algebraic proof in the manuscript or a corresponding
check in the audit, with explicit tolerances. Any claim that fails the
audit must be retagged `[Open]` before compilation.

## Notation

| Symbol | Meaning |
|---|---|
| `K` | sequence length (number of frames) |
| `a = (a_0, ..., a_{K-1})` | clean Markov sequence |
| `x = (x_0, ..., x_{K-1})` | OU-corrupted sequence at diffusion time `t` |
| `mu = exp(-t)` | OU signal-attenuation factor |
| `Delta_t = 1 - exp(-2t)` | OU noise variance |
| `Sigma_0` | covariance of clean chain |
| `Sigma_t = mu^2 Sigma_0 + Delta_t I` | covariance of noised chain |
| `Q_t = Sigma_t^{-1}` | precision of noised chain |
| `S(x, t)` | **joint** score = `grad_x log P_t(x)` |
| `S_k(x, t)` | k-th component of joint score |
