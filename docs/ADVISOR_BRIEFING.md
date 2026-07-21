# Briefing for Jérôme — state of the thesis (updated 2026-07-21)

One-page summary of what is proved, what is measured, what is speculative,
and what I propose to do next. Full claim-by-claim classification in
`RESULT_LEDGER.md`; thesis draft in `thesis/main.pdf` (68 pp., compiles
with one `tectonic main.tex`; all numerical audits re-run on this machine:
72/72 and 55/55 pass).

## Revision of 2026-07-21 (condensation round, v3)

On Giovanni's second feedback round ("too long for the information we
give") the draft was condensed 119 pp → 68 pp and restructured to 5
chapters: 1 Introduction (modest RQ1–RQ4) · 2 Model + score identities +
related work + research development (the old three background chapters,
40 pp → 8 pp; the AI-history and cascade material reduced to a cited
related-work discussion) · 3 the complete Gaussian case in one chapter
(matrix score, spectral lifecycle, BP, Kalman, bulk, locality,
truncation) · 4 the complete non-Gaussian case in one chapter (Laplace,
grid reference, Gaussian-projection measurements) · 5 conclusions.
Appendices: A Gaussian identities · B AMP/TAP bulk fixed point (every
formula independently re-verified before keeping it, per the "100% sure
or delete" rule) · C reproducibility. A new research-development section
records the toy models, the per-frame→joint score correction, and the
continuous-message bottleneck. Figures cut 11 → 7, one per distinct
result. Editorial registers in `thesis/editorial/`.

## Revision of 2026-07-19 (author feedback round, v2 — superseded by v3)

The draft was restructured on Giovanni's written feedback: 8 chapters
instead of 10, ~119 pp. instead of 137. The literature review now carries
the full statistical-mechanics→AI story (spin glasses → Hopfield/Nobel →
Boltzmann machines/RBMs → statistical mechanics of learning) and a deep
review of the generative line (LeCun's EBM tutorial → score matching →
Ho's DDPM → Song's score SDE → flow matching), including a detailed
section on *Cascade of phase transitions in the training of energy-based
models* (Bachtis–Biroli–Decelle–Seoane, NeurIPS 2024) built on Giovanni's
coursework for Mézard's 41002 course. In the research part: Ch. 5 now
develops the spectral theorem first and studies the precision-matrix
lifecycle entirely through the spectral form (when/how/how-much
tridiagonality is lost); BP theory and the Kalman filter are derived in
full where used; the complexity claim is stated honestly (O(K) *updates*
always, O(K) *cost* only under a closed message family); "Tweedie" is now
"the score–posterior identity" throughout; the AMP/TAP analysis
(t_c(α), α_c = √2−1) moved intact to Appendix C; code listings were
dropped; every figure was regenerated in a uniform paper style (vector
PDF, white background) by one script, `thesis/figures/make_figures.py`.

## What we have shown exactly (closed form + audit)

1. **Gaussian AR(1) + OU, solved end to end.** Joint score `S = −Qt·x`;
   Tweedie identity for arbitrary priors; posterior precision
   `J = (e^{−2t}/Δt)I + Q0` stays tridiagonal for all t; full lifecycle of
   `Qt` including the corrected band-fill law (no `1/(d−1)!` — the earlier
   note's factor was a transcription error, caught by coefficient audit).
2. **Bulk closed forms.** Posterior variance `V = 1/√(J_d²−4β²)`,
   covariance `q^d·V` with an explicit `q(α,t)`; `q → α` as t → ∞. The
   locality error of a radius-r estimator decays exactly as `q^r`.
3. **BP = Kalman, by algebra not CLT.** Gaussian messages are closed under
   the chain updates (two lemmas); Convention-A BP reproduces the matrix
   score to 1e−14 at O(K). Your first suggestion from the June call —
   "run Gaussian message passing on the solved case and see if we recover
   what we had" — holds to machine precision.
4. **Your CLT objection, made exact.** On the chain, BP/AMP/mean-field all
   return the *same exact score* (the mean fixed point is
   closure-independent). The *variance* closure is where AMP pays: the
   whole BP↔AMP difference is one factor of 2 in a discriminant
   (`J_d²−4β²` vs `J_d²−8β²`), giving an exact breakdown time `t_c(α)` and
   a critical coupling **α_c = √2−1 ≈ 0.414**. Below α_c AMP never breaks;
   above it, it has no variance fixed point past t_c. "A CLT on two data
   points is usually not right" is now a theorem with a phase diagram.

## What is verified numerically (no closed form)

5. **The Gaussian closure cost on non-Gaussian chains.** Against
   grid-quadrature BP (spectrally accurate, validated on the Gaussian case
   to 1e−15): Laplace chain (ρ=0.85) median relative score error **0.39 at
   t=0.02 → 0.17 at t=0.08 → 3.5e−2 at t=0.4 → 7.6e−5 at t=2.4**.
   Direction is much more robust than magnitude: median cosine 0.92 even at
   t=0.02, ≥0.99 for t≥0.12.
6. **What makes closure hard.** Sweeping innovation families at fixed
   covariance: bimodality is the worst case (0.94 at t=0.05), heavy tails
   intermediate; and error *decreases* with chain correlation ρ —
   informative neighbours Gaussianise the local tilted posterior.
7. **Truncation.** At equal locality budget (range 1), truncating the
   *inference* beats truncating the *matrix*: 12.3% vs 21.0% at t=0.05;
   both peak at intermediate t and die at large t. Cutting the algorithm
   is better than cutting the answer.

## What is speculative / deliberately not claimed

- The old "Laplace K=2 Hessian is approximately rank-2" conjecture is
  regime-dependent under quick tests and is **not** asserted anywhere.
- Architecture implications (local/CNN score heads sized by `q(α,t)`) are
  readings of exact inference results — no training experiments were run.
- Approximate-Markov (chain + global latent, hybrid BP + neural residual)
  and reverse-dynamics experiments exist in preliminary form but are not
  consolidated into the thesis claims.

## Most convincing artefacts (in order)

1. The BP-vs-matrix machine-precision agreement across 80 configurations.
2. The AMP phase boundary: closed-form `t_c(α)` matching the fixed-point
   iteration at all 180 scanned points, and α_c = √2−1.
3. The Laplace closure curve with its grid-error budget 10⁴–10⁷ below the
   measured effect (the reference is trustworthy where used).
4. The band-fill coefficient audit that caught the factorial error.

## Recommended next step (my proposal)

**Mixture messages** (2-component Gaussian mixtures in the same sweeps).
It is the cheapest experiment that (a) directly attacks the worst case we
measured (bimodal beliefs), and (b) separates message-representation error
from model-mismatch error, which the single-Gaussian experiment cannot do
by construction. Second choice: the discrete-alphabet chain you sketched
(exact vector messages, no closure at all) as a clean control.

## Open questions for you

1. Is the α_c = √2−1 phase boundary worth developing further (e.g. other
   graph topologies, small-world perturbations of the chain), or should
   the thesis close the AMP thread here?
2. For the thesis defence: do you prefer the emphasis on the exact
   Gaussian/AMP results (theorem-style) or on the closure-error
   measurements (experiment-style)?
3. The reverse-dynamics question (where along the generation trajectory do
   score errors matter) — in scope for this thesis or explicitly future
   work?
4. Is the current framing of the locality/architecture reading (receptive
   field `r ≳ ξ log(1/ε)`, ξ = 1/log(1/q)) a claim you are comfortable
   putting before the committee, given that no network was trained?
