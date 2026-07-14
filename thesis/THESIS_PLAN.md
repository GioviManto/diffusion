# Thesis Plan — DSBA MSc Research Thesis (Bocconi, A.Y. 2025-26)

**Working title:** *Exact Scores for Structured Data: Statistical Mechanics, Belief
Propagation, and the Gaussian Closure in Diffusion Models on Markov Chains*

**Deadline context:** graduation appeal October 2026, thesis ready ~September 2026.

## Binding format rules (guide.pdf §10.3 + writing guide)

- Single PDF ≤ 10 MB, filename `TS<studentID>.pdf` (ID added by Giovanni at upload).
- First 4 pages: blank / blank / acknowledgements-or-blank / blank. Content (TOC first)
  starts at page 5; page numbering included. **No title page, no abstract in the body**
  (printed automatically by the university system).
- **No student name/ID anywhere in the content. No Bocconi seal.**
- A4; left/right margins 2.5 cm; 26–30 lines per page; body 12 pt; numbered pages;
  thesis starts on odd (right) page.
- Citations: author–date (Harvard) in text + full alphabetical bibliography
  (natbib, `agsm`-like style). URLs get [last access] dates.
- Research-thesis structure: Introduction (gaps → research questions → contributions),
  literature review, method, analysis, discussion, conclusions, bibliography.
- General rule in guide is ~50 pages; user explicitly targets ~100 pages (research
  thesis, math-heavy, benchmark: thesis_achilli_final.pdf ~160 pp).
- DO NOT cite `Generative_diffusion_updated_notes_MM.pdf` (unofficial lecture notes) —
  cite published Mézard/Biroli/Bonnaire works instead.

## Chapter plan (page budgets → ~102 body pages + appendices)

1. **Introduction** (5) — topic; gap (joint score for sequential data; continuous BP
   messages infinite-dimensional; Gaussian closure quality unknown on chains);
   research questions RQ1–RQ4; contributions; structure.
2. **From Hamiltonian Mechanics to Statistical Mechanics** (12) — Hamiltonian
   mechanics, phase space; Liouville theorem (toolbox proof); ensembles;
   Boltzmann distribution (toolbox max-ent derivation); free energy; Ising +
   complex systems, energy landscapes.
3. **Stochastic Processes and MCMC** (10) — Markov chains, stationarity, detailed
   balance, ergodicity; AR(1) introduced HERE as running example; Metropolis–Hastings
   (toolbox detailed-balance proof), Gibbs; Langevin dynamics bridge.
4. **SDEs, Itô Calculus, OU, Fokker–Planck** (10) — Brownian motion; Itô integral +
   lemma (toolbox); OU exact solution (toolbox integrating factor); Fokker–Planck
   (toolbox derivation); stationary = Boltzmann; Anderson time reversal.
5. **Energy-Based Models and Markov Random Fields** (8) — EBMs, partition function;
   MRFs, Hammersley–Clifford; hidden Markov models (noisy chain posterior planted);
   factor graphs.
6. **Diffusion Models** (14) — DDPM (Sohl-Dickstein 2015, Ho 2020, ELBO toolbox);
   score SDE (Song–Ermon 2019, Song et al. 2021), probability-flow ODE; score
   matching (Hyvärinen 2005, Vincent 2011 toolbox); Tweedie/Miyasawa (toolbox);
   stat-phys of diffusion (Biroli–Mézard 2023 JSTAT; Biroli–Bonnaire–De Bortoli–
   Mézard Nat. Comm. 2024 dynamical regimes; Achilli et al. memorization);
   gap: structured sequential data → RQs.
7. **The Gaussian Chain Solved Exactly** (14) — joint vs marginal score (Mézard
   correction 2026-03-12); linear algebra toolbox (spectral theorem, Toeplitz,
   tridiagonal); Σ0 = α^{|i-j|}σ∞², Q0 tridiagonal (toolbox derivation);
   Σt = e^{-2t}Σ0 + Δt·I, shared eigenbasis, spectral lifecycle; three forms of Qt;
   band-fill law **corrected — NO 1/(d-1)! factor**:
   (Qt)[i,i+d] = (−1)^{d−1}(2t)^{d−1}(Q0^d)[i,i+d] + O(t^d);
   bulk closed forms: J_d = e^{-2t}/Δt + (1+α²)/(1−α²)·(1/σ_η² scaled), β = −α/σ_η²-scaled,
   V = 1/√(J_d²−4β²), covariance (J⁻¹)_{i,i+d} = q^d·V with
   q = (J_d−√(J_d²−4β²))/(2|β|), q→α as t→∞; locality law q^r; K=2 worked example.
   Sources: research/gaussian-bp/main.tex §§4–8, 10–12; research/unified-note Act I.
8. **Beyond Gaussianity: Laplace Innovations** (8) — K=1: nonlinear score, Fourier/
   convolution form, x-dependent curvature; K≥2: Markov-structured joint score,
   boundary message closed form, general messages no clean form; innovation
   coordinates. Source: unified-note Acts II–III.
9. **Score as Posterior Expectation: BP and AMP** (16) — Bayesian inversion;
   posterior chain factorisation; factor graph = tree; sum–product (Mézard–Montanari
   Ch. 14); BP exact on trees (toolbox chain proof); continuous messages problem
   (Jerome 5 Jun call); Gaussian BP = Kalman, machine-precision equality with
   matrix score; bulk cavity fixed point λ* = (J_d+√(J_d²−4β²))/2 — BP never breaks
   (AM–GM argument); AMP/TAP: same score, different variance, breakdown
   t_c(α) = −½log(g/(1+g)), g = (2√2α−1−α²)/(1−α²), **α_c = √2−1 ≈ 0.4142**;
   discriminant 4β² vs 8β² (factor-2 closure); weak coupling V_amp−V = 2β⁴/J_d⁵;
   non-Gaussian: grid BP vs Gaussian-projected BP on Laplace chain — ~20% rel.
   score error at small t → <1% at large t, direction accurate; truncated
   inference (12%) beats truncated matrix (21%) at small t; architectural
   implications (locality/CNN analogy).
   Sources: research/bp-from-scratch, research/gaussian-bp §§9–13,
   research/nongaussian-bp/report, research/bp-generalization jpegs, meetings docx.
10. **Discussion and Conclusions** (5) — summary per RQ; contributions; limitations;
    future work (discrete data model, hybrid BP+neural residual, reverse dynamics).

Appendices: A Gaussian identities; B code listings (chain_formulas.py, bp_score.py,
bp_gaussian.py, exact grid BP excerpts); C reproducibility (audits 72/72 + 53/53,
repo map); D figure sources.

## Figures available (copy into thesis/figures/)

- research/gaussian-bp/figures: fig_band_fill, fig_precision_lifecycle,
  fig_local_vs_full, fig_tridiag_loss, fig_bulk_variance, fig_spectral, fig_bp_vs_amp
- research/unified-note/figures (8), research/bp-from-scratch/figures
- research/nongaussian-bp/outputs/exp_01..05 (Laplace closure error, reverse dynamics)
- research/initial-experiments/early-figures (fig1–4, score fields)
- meetings/board sketches + research/bp-generalization/general-BP*.jpeg (scans)

## Research questions (fixed)

- RQ1: What is the exact joint score of a Markov chain under OU corruption, and how
  does its structure (precision matrix) evolve with diffusion time?
- RQ2: Can message passing compute this score exactly and locally, and how does the
  Gaussian case connect BP to classical Kalman smoothing?
- RQ3: When BP messages are forced into a Gaussian family (AMP/TAP closure), what is
  lost — on the Gaussian chain (nothing: score exact; variance differs, with an
  exact phase boundary α_c = √2−1) and on non-Gaussian chains (quantified closure error)?
- RQ4: What do exact scores and their locality imply for the architecture of learned
  score models on structured data?

## Build

- `tectonic main.tex` in thesis/ (tectonic at /opt/homebrew/bin/tectonic; natbib+bibtex).
- 12 pt, onehalfspacing (~28 lines/page), geometry hmargin=2.5cm.

## Progress tracker (updated 2026-07-14)

- [x] Plan, skeleton, preamble
- [x] Ch1–Ch10 written (137 pp. compiled)
- [x] Appendices A–C (Gaussian identities, code listings, reproducibility)
- [x] References (verified entries only; unofficial notes not cited)
- [x] Format audit: pages 1/2/4 blank, p.3 acknowledgements, content from p.5
      (odd); A4, 2.5 cm side margins, ~29 lines/page; 1.63 MB ≤ 10 MB
- [x] Numerical audits re-run on this machine: 72/72 and 55/55 PASS;
      Laplace closure medians + cosine re-verified from experiment CSVs
- [x] docs/: RESULT_LEDGER.md, ADVISOR_BRIEFING.md, SOURCE_AUDIT.md
- [ ] Push GitHub repo `diffusion` (private); old repos: user decides archiving
- [ ] Giovanni: fill acknowledgements (p. 3), set final title, add student ID
      to filename at upload (TS<ID>.pdf), review Ch1 RQ wording
- [ ] Optional next experiments: mixture messages; discrete chain (see briefing)
