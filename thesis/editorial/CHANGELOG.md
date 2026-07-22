# Change log

## v3.0 (2026-07-21) — condensation and consolidation pass

Result: 119 pp -> 68 pp (-43%), 8 chapters -> 5, 11 figures -> 7, clean
build (0 overfull boxes, 0 undefined references), format audit green
(blank pp.1/2/4, acknowledgements p.3, TOC from p.5, body starts p.9 odd).
Every v2 chapter file and the 119-pp PDF archived at
`~/Code/Thesis/archive/thesis-v2-chapters/` before removal.

### Merged
- Old Ch2 (stat mech / AI story) + Ch3 (stochastic tools) + Ch4
  (generative models), 40 pp -> new Ch2 "Model, Score Identities, and
  Research Context", 8 pp. Kept: Markov/AR(1) definitions + toolbox, OU
  channel kernel and covariance map, score/reverse-SDE statement, score
  matching in one paragraph, full derivation of the score-posterior
  identity, condensed related work, NEW research-development section.
- Old Ch5 (Gaussian) + Gaussian parts of old Ch7 (factor graphs,
  sum-product, closure, Kalman) + the truncation experiment -> new Ch3
  "The Gaussian Chain", 18 pp. The complete Gaussian case is now in one
  chapter (matrix score, spectral lifecycle, BP, Kalman, bulk, locality,
  truncation), as the contract requires.
- Old Ch6 (Laplace) + experimental sections of old Ch7 (grid BP,
  closure measurements, innovation sweep) -> new Ch4 "Beyond
  Gaussianity", 9 pp.

### Deleted (with reason)
- Hamiltonian mechanics, Liouville, ensembles, heat-bath and max-ent
  derivations of Boltzmann, free-energy toolbox, Ising section, spin-glass
  narrative, Hopfield/Boltzmann-machine/RBM history sections, stat-mech
  dictionary table (old Ch2): background not used by any later
  derivation; reduced to a cited lineage paragraph in Ch2 related work.
- MCMC/Metropolis-Hastings/Gibbs/Langevin sections and Ito/Fokker-Planck
  toolboxes (old Ch3): not used by the research chapters; OU kernel now
  cited, Anderson's theorem stated with citation.
- DDPM/ELBO toolbox, implicit/denoising score-matching toolboxes, EBM
  section, flows section (old Ch4): compressed to one paragraph each
  inside Ch2; the score-posterior identity keeps its full derivation
  (it is used everywhere).
- Cascade-of-phase-transitions section incl. one-mode-RBM toolbox
  (old §4.8): reduced to one focused paragraph in Ch2 related work
  (background rule: cited, not re-derived). REVERSIBLE if the author
  wants the coursework featured more prominently.
- Appendix B "Long derivations" (MH detailed-balance proof): orphaned
  once MCMC was cut; deleted.
- Chapter summaries/preambles, "two movements" structure talk, closing
  "black box -> dial" remark, all supervision quotations in the body.
- Figures fig_spectral, fig_precision_lifecycle, fig08_K1_score_limits
  (content stated in equations); fig06+fig07 merged into fig_laplace_k1.

### Rewritten
- Introduction: problem reached on p.1-2; the four modest RQs supplied by
  the author adopted verbatim (lightly typeset); contributions restated
  with restrained verbs; reproducibility framed as methodological
  strength, not novelty; scope-and-limitations section added.
- Conclusions: findings grouped as derived-exactly / measured /
  comparison-lesson; "suggests but does not establish" section; future
  work cut to three grounded items.
- Global style: em dashes ~300 -> 2 occurrences -> 0 in body; banned
  phrases removed (mother theorem, punchline, miracle, two movements,
  old-fashioned, home turf, celebrated, missing axis...); ping-pong
  cross-references removed; grid BP consistently described as a
  validated numerical approximation.

### Kept after verification (not weakened)
- Appendix B (ex-C) AMP/TAP bulk fixed point: every formula
  independently re-verified 2026-07-21 (see CORRECTNESS.md) plus repo
  audits 72/72 and 55/55; per the author's rule it stays in the
  appendix, trimmed (supervision quotes and editorialising removed).

### Reintroduced from earlier material (logged per contract)
- Research-development section (Ch2 §2.7): toy models TM1-TM7 summary,
  the per-frame -> joint score correction, the continuous-message
  bottleneck, the discarded rank-two conjecture. Sources: archived
  toy-model notes and project memory; no new claims, no dates, no quoted
  conversations.

## v3.1 (2026-07-21) — proofread pass

Page-by-page visual QA of the 68-pp build (all pages rendered; sample of
13 inspected in detail across every chapter and appendix). One defect
found and fixed: Ch4 experimental sections used $\rho$ for the chain
correlation and $n$ for the chain length where the rest of the thesis
uses $\alpha$ and $K$; renamed throughout Ch4 (values unchanged; for the
AR(1) chain the experiment's correlation parameter is the AR
coefficient). No other prose, layout, figure, or toolbox issues found.
Rebuild clean: 68 pp, 0 overfull, 0 undefined references.

## v4.0 (2026-07-22) — re-expansion pass (author request)

Result: 68 pp -> 124 pp, 5 chapters -> 9, 7 figures -> 8, clean build
(0 overfull boxes, 0 undefined references), body starts p. 11 (odd).
Author's instruction: restore the full introduction and related work, go
deep on statistical mechanics / AI history, EBMs, diffusion and flows,
and the Mezard-Biroli-Achilli-Garnier-Brun-Bonnaire literature; explain
deeply every tool used (stationarity, Markov chains, Gaussian density,
OU, spectral theorem, inverses, MRFs, BP, AMP, CNN locality analogy);
discuss the Score_Diffusion preliminary work and toy models; remove all
O(K) complexity talk; fix remaining AI-styled charts. The earlier
problem was ping-pong and filler, not depth, so the restored chapters
keep one-topic-one-place discipline and contain no em-dash decoration.

### Restored from the v2 archive (logged per contract; reintroduction
### explicitly requested by the author)
- Ch. 2 Statistical Mechanics and the Origins of Learning Machines
  (Hamiltonian mechanics, Liouville, ensembles, Boltzmann two ways, free
  energy, Ising, spin glasses, Hopfield, Boltzmann machines, statistical
  mechanics of learning) — from v2 ch02, em-dashes removed, ping-pong
  trimmed.
- Ch. 3 Stochastic Processes, SDEs, and the OU Channel — from v2 ch03
  (incl. MCMC/Langevin, Ito, Fokker-Planck, Anderson), PLUS a new formal
  stationarity section (strict/weak, Toeplitz consequence). "Mother
  theorem" title replaced by "Anderson's theorem". The MH detailed-
  balance proof reference now cites Robert-Casella (the v2 appendix
  derivation stays deleted).
- Ch. 4 From Energy-Based Models to Diffusion and Flows — from v2 ch04
  with the full cascade section (one-mode RBM toolbox) restored, and the
  statistical-physics-of-diffusion review EXPANDED with newly cited
  papers (all metadata verified against the PDFs in sources/):
  achilli2026speciation, garnierbrun2026biased, bonnaire2025memorize,
  sclocchi2025phase, mei2024unets, holderrieth2025flow,
  ronneberger2015unet, krzakala2024statphys, achilli2026thesis,
  lai2025principles.

### New chapters
- Ch. 5 Gaussian Vectors, Graphical Models, and Message Passing:
  multivariate Gaussian in both parametrisations, precision-zeros =
  conditional independence, spectral theorem + Neumann/Woodbury, MRFs +
  Hammersley-Clifford, factor graphs, sum-product + tree exactness,
  Gaussian closure lemmas (moved here from the Gaussian chapter),
  classical Kalman as Gaussian sum-product, TAP/AMP genealogy with the
  CLT logic spelled out, CNN/receptive-field/U-Net locality section
  (Mei's U-Nets-as-BP).
- Ch. 6 Preliminary Work: initial dynamic-objects framing + propagator
  hypothesis, the circle/rotating-data models from the first supervision
  meetings (board sketches used as source material only), toy models
  TM1-TM7 described individually with the Score_Diffusion repo cited in
  a footnote, the joint-score correction, minimal-model choice,
  continuous-message bottleneck, discarded rank-two conjecture.

### Rewritten / edited
- Ch. 1 Introduction restored to full v2 depth (topic/motivation,
  three-strand positioning, methodology, structure) while KEEPING the v3
  humble RQ1-RQ4 verbatim and the restrained contribution list.
- Ch. 7 (ex Ch. 3) Gaussian chain: generic MRF/factor-graph/sum-product/
  closure-lemma material moved to Ch. 5 and replaced by pointers; all
  O(K)/O(1)/O(K^3)/O(KM^2) language removed thesis-wide and replaced by
  plain-language cost statements (author: "avoid the bullshit of O(k)").
- Ch. 8 (ex Ch. 4) Laplace and Ch. 9 (ex Ch. 5) conclusions: same O(K)
  cleanup; content unchanged.

### Figures
- fig_bulk_variance, fig_bp_vs_amp redesigned in sober journal style:
  no shaded "phase" regions, no colour-referencing titles, no rotated
  labels; the AMP phase diagram is now a black closed-form boundary
  curve with open-circle iteration checkpoints and text-labelled phases.
  Captions updated to match.
- NEW fig_toymodel_score (Ch. 6): two-frame toy model joint density +
  joint score field, computed from closed-form Gaussian-mixture
  expressions, black-and-white contours/quiver, low-density region
  masked.

### Not changed
- App. B AMP results and all numerical claims (no new claims introduced;
  the new related-work text states only what the cited papers' abstracts
  and statements support). App. A, App. C unchanged except the O(1)
  phrase in App. B prose.
