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
