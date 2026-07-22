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

## REVISION v2 (2026-07-19) — advisor/author feedback round

Source: `Things-to-change-thesis.pdf` (Desktop/Diffusion). Directives:
less redundant; less textbook, more thesis; purpose stated before every deep
dive; way fewer pages; less AI-sounding; formulas must fit toolboxes;
"Tweedie" → score as posterior expectation; §7.4–7.5 rebuilt around the
spectral form + tridiagonality-loss study (paper-style white-background
charts); Laplace restricted to 100%-certain material; BP theory + Kalman
explained extensively, O(K) claim corrected (only for closed families);
cancel §9.4 (AMP) and bulk fixed point from body → appendix; cancel all code
listings; long derivations & nice-to-have theory → appendices; INSIST on
stat-mech↔AI story (Hopfield Nobel, EBMs/LeCun, Ho, Song, flows, Biroli–
Bonnaire–Mézard, cascade of phase transitions NeurIPS 2024 — Giovanni's
coursework: Mantovani_Slides_SM.pdf + SM_Paper_Notes.pdf, 41002 course).

### New chapter map (8 chapters, target ~100 pp total)

1. Introduction (trimmed, purpose-driven)
2. Statistical Mechanics and the Origins of Learning Machines
   — condensed physics core (Hamiltonian→Liouville→ensembles→Boltzmann→
   free energy→Ising; long proofs → App. B) + THE AI STORY: spin glasses
   (SK), Hopfield 1982 + Nobel 2024, Amit–Gutfreund–Sompolinsky capacity,
   Boltzmann machines (Ackley–Hinton–Sejnowski), RBMs (Smolensky, Hinton CD),
   deep belief nets, Gardner space of interactions, Mézard 2017 PRE
   (Hopfield ↔ RBM ↔ message passing — the bridge to Ch. 7).
3. Stochastic Processes, SDEs, and the Ornstein–Uhlenbeck Channel
   — merged old ch3+ch4, condensed: Markov chains, AR(1), MCMC (MH/Gibbs/
   Langevin), Itô, OU, Fokker–Planck, Anderson reversal. Purpose-first.
4. From Energy-Based Models to Diffusion and Flows (deep lit review)
   — EBMs (LeCun 2006 tutorial deep), score matching (Hyvärinen; Vincent),
   DDPM (Sohl-Dickstein; Ho deep), score SDE (Song deep), probability-flow
   ODE → normalizing flows → flow matching; stat-phys of diffusion
   (Biroli–Mézard 2023; Biroli–Bonnaire–De Bortoli–Mézard 2024; Ghio 2024;
   Achilli memorization; Raya–Ambrogioni); CASCADE OF PHASE TRANSITIONS
   (Bachtis–Biroli–Decelle–Seoane 2024): RBM training as effective cooling,
   Curie–Weiss/Mattis mapping, two critical times, FSS — then the gap + RQs.
5. The Gaussian Chain Solved Exactly (reworked §7.4/7.5: spectral theorem
   first → spectral form → tridiagonality-loss study across t; "score as
   posterior expectation" naming; K=2 kept; band-fill kept brief)
6. Laplace Innovations (restricted: K=1 exact results, screened Poisson,
   where closed forms stop; innovation coordinates brief)
7. Score as Posterior Expectation: BP and Kalman (extensive BP theory:
   factor graphs + HMM here, where used; sum–product; Kalman filter/RTS
   derived properly; Gaussian closure lemmas; machine-precision result;
   complexity stated honestly — O(K) updates only under a closed family;
   non-Gaussian closure experiments kept; NO AMP, NO bulk fixed point)
8. Discussion and Conclusions
Appendices: A Gaussian identities; B Long derivations (Liouville, max-ent,
Itô, ELBO, denoising SM proof, band-fill Neumann, transfer matrix);
C AMP/TAP and the bulk fixed point (moved from body — incl. t_c(α),
α_c=√2−1); D Reproducibility (repo map + audits; NO code listings).

### Figure policy
Paper-style: white background, no grid clutter, serif-compatible, consistent
palette (matplotlib default-white + explicit style block), vector PDF.

## Progress tracker (updated 2026-07-19, revision v2)

- [x] Revision v2 executed in full (see REVISION v2 section above):
      8 chapters, 119 pp., AI/stat-mech story + EBM→diffusion→flows +
      cascade section written; Ch5 spectral rework; Ch7 BP+Kalman full
      derivation, honest O(K); AMP → App. C; code listings cancelled;
      "Tweedie" → score–posterior identity; \nocite{*} removed (all 76
      bib entries cited; unofficial ZK-2021 notes dropped)
- [x] All figures regenerated paper-style (vector PDF, white background)
      by `thesis/figures/make_figures.py`
- [x] Format audit v2: pages 1/2/4 blank, p.3 acknowledgements, TOC from
      p.5, Ch1 from p.9 (odd); A4; 0.69 MB ≤ 10 MB; no undefined refs
- [x] docs/ updated: RESULT_LEDGER (renumbered), ADVISOR_BRIEFING (v2
      changelog), SOURCE_AUDIT (v2 sources incl. cascade coursework)
- [ ] Giovanni: fill acknowledgements (p. 3), set final title, add student ID
      to filename at upload (TS<ID>.pdf), review Ch1 RQ wording
- [ ] Giovanni: archive (not delete) old GitHub repos after sending Jérôme
      the new link
- [ ] Optional next experiments: mixture messages; discrete chain (see briefing)

---

## REVISION v3 (2026-07-21) — condensation pass

Standing editorial contract received 2026-07-21 (senior-editor prompt +
complaints: too long for the information given, humbler RQs, linear
narrative, no ping-pong, one-place-per-topic, formal charts, cut
redundancy, keep bulk-fixed-point only if 100% verified).

Executed: 119 pp → 68 pp; 8 chapters → 5 (1 Intro · 2 Model/identities/
context/research-development · 3 Gaussian complete incl. BP+Kalman+
locality+truncation · 4 Laplace complete incl. grid reference+closure
measurements · 5 Conclusions); appendices A Gaussian identities ·
B AMP (independently re-verified 2026-07-21, kept per the 100% rule) ·
C reproducibility. Figures 11 → 7. RQ1–RQ4 replaced by the author's
modest versions. Em dashes ~300 → 0; banned phrases removed. v2 sources
archived at ~/Code/Thesis/archive/thesis-v2-chapters/.

Editorial registers (standing documents, update each pass):
`thesis/editorial/DECISIONS.md`, `CHANGELOG.md`, `CORRECTNESS.md`,
`FIGURES.md`.

User TODOs unchanged: acknowledgements text (p.3), final title,
TS<studentID>.pdf rename at upload, archive old GitHub repos after
Jérôme has the new link.

## REVISION v4 (2026-07-22)
Author requested re-expansion: full intro + related work back, deep
background on stat mech/AI, EBMs, diffusion/flows, the Mezard-Biroli-
Achilli-Garnier-Brun-Bonnaire line (sources/papers + main-sources), deep
explanations of all tools (stationarity, Markov, Gaussian, OU, spectral
theorem, inverses, MRF, BP, AMP, CNN analogy), preliminary work from
github.com/GioviManto/Score_Diffusion (toy models + board sketches), no
O(K) talk, no AI-styled charts. Delivered: 9-chapter structure, 124 pp,
see thesis/editorial/CHANGELOG.md v4.0. User TODOs unchanged
(acknowledgements, final title, TS<studentID> rename, old-repo
archival after Jerome has the new link).
