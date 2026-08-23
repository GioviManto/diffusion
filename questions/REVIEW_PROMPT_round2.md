# Second review request — what changed, and what did not

Repository: <https://github.com/GioviManto/diffusion>
Branch `main`, commits `0456687..e162220` (twelve commits, all pushed).

You reviewed this repository and returned a report in two roles, NeurIPS
reviewer and Bocconi supervisor, with a verdict of reject / do-not-submit and a
numbered list of blocking issues, code defects and required experiments. This
is the response to that report. Please review it in the same two roles.

**The single thing I most want checked:** your central empirical criticism was
that the 7–20× headline could not distinguish the value of Markov structure from
the weakness of the neural baseline. That is now measured, and **the measurement
substantially agrees with you** — most of the margin was architectural. Section
1 below is the result. I would like to know whether the revised claim is one you
would accept, and whether the way it is now framed overclaims in some new way.

Please be as adversarial as before. Where I have answered a point by scoping a
claim rather than by doing the experiment, say so.

---

## 1. Your Priority 1 / issue 3.1 — the structured baseline

You asked for a full-receptive-field structured baseline under saturated tuning
before the headline could stand.

Two corrections to the premise first, both of which were my errors, not yours:

- The paper claimed the headline comparison already included "a weight-shared
  one-dimensional convolution". **It did not** — `exp_07` trains only the MLP.
  You assumed the conv arm was present but weak; it was absent.
- The experiment you asked for largely existed already (`exp_12`), unrun at
  scale and unreported.

It has now been run at frozen scale: 16 seeds, radius swept to r=16, which at
33 sites is a window covering **every** site, with radius and parameterisation
selected on a disjoint validation bundle.

| n_seq | MLP | CNN (tuned) | EM–BP | MLP/EM | **CNN/EM** |
|---|---|---|---|---|---|
| 32 | 0.5305 | 0.1157 | 0.0763 | 6.8 | **1.82 ± 0.10** |
| 128 | 0.3639 | 0.0761 | 0.0386 | 10.4 | **2.63 ± 0.23** |
| 512 | 0.1852 | 0.0590 | 0.0180 | 11.2 | **4.04 ± 0.25** |
| 2048 | 0.1252 | 0.0520 | 0.0131 | 12.5 | **5.54 ± 0.38** |

You were right. Most of the headline gap is architectural. The claim is now that
against a tuned convolution the margin is **1.8–5.5×**, not 7–20×.

What I think survives, and want you to attack: at n=32 the convolution recovers
nearly all the advantage (1.82× against the MLP's 6.8×), and all seven cells
where the CNN wins sit at the two smallest sizes. So the structural advantage is
not a constant the locality prior fails to supply — it is something the
convolution **fails to acquire as data grows**, widening monotonically to 5.54×.
Is that reading defensible, or am I rescuing a weakened result with a story?

Still not done: no transformer, no dilated stack, no temporal U-Net, and the
budget-saturation curves you asked for (Priority 2) are running now, not merged.

- [tab-structured.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/tab-structured.tex)
- [tools/make_tab_structured.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/tools/make_tab_structured.py)
- [exp_12_receptive_field.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/experiments/exp_12_receptive_field.py)

## 2. Your Priority 6 — the headline under other estimands

You asked for the ratio under alternative aggregations. Recomputed on the same
1,712 cells, no retraining:

| n_seq | per-cell *(reported)* | ratio-of-means | geometric | median | paired 95% CI |
|---|---|---|---|---|---|
| 32 | 6.96 | 6.85 | 6.06 | 6.79 | [6.11, 8.07] |
| 512 | 8.99 | 8.57 | 7.89 | 8.20 | [8.06, 9.95] |
| 2048 | **15.49** | **10.03** | 13.19 | 12.86 | [12.97, 18.85] |
| 8192 | 20.21 | 16.06 | 18.79 | 19.51 | [18.33, 22.01] |

The conclusion is robust — every estimand at every size exceeds 6.1× — but the
estimand we report is **systematically the largest of the four**, by up to 5.5 at
n=2048. That is Jensen, not an error, but quoting only the largest of five was a
choice presented as a fact, so all five are now in the appendix. Resolving over
noise levels also shows the ratio peaks near t≈0.5 with a worst single cell of
**3.6×**, not 7×.

- [tab-aggregation.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/tab-aggregation.tex)
- [tools/make_aggregation_robustness.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/tools/make_aggregation_robustness.py)

## 3. Your issue 3.3 — Proposition 2 (LMMSE)

Restated with every hypothesis you listed: ρ≠0 with the ρ=0 case handled
separately, finite second moments, the projection operator Π_G defined, **where**
it is applied (it does not commute with the likelihood multiplication), the
initial-message condition, and an explicit statement that the grid plays no part.
Proved by induction over both recursions.

I also accepted your correction that "noising Gaussianises the posterior" is
false — the posterior approaches the prior; the noised marginal Gaussianises and
Tweedie's prefactor attenuates. Fixed in paper and workshop.

- [prop-lmmse.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/prop-lmmse.tex)
- [prop-lmmse-proof.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/prop-lmmse-proof.tex)

## 4. Your issue 3.6 / Priority 7 — Fisher's identity

Split into a discrete theorem (what the 10⁻⁹ check certifies) and a continuum
one with domination hypotheses. Index orientation now matches the code's
child-first Ξ.

You noted the finite-difference check is not independent. Added a brute-force
control that enumerates the whole posterior on a tiny grid from the model
definition, sharing nothing with `e_step` but the kernel and the grid. It checks
evidence, Ξ **including orientation**, and the gradient against a difference of
the enumerated evidence. Verified it has teeth: transposing Ξ moves the gradient
from `[-4.75, -1.01]` to `[-1.22, +1.51]`.

- [prop-fisher.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/prop-fisher.tex) · [prop-fisher-scope.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/shared/sections/prop-fisher-scope.tex)
- [tests/test_fisher_bruteforce.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/tests/test_fisher_bruteforce.py)

## 5. Your issues 3.4 and 3.5 — log domain, and the ECM gain

**3.4.** You were right: the appendix claimed log-domain recursions and "no
message underflows"; the implementation is scaled linear-domain. Described
accurately now. Your Priority 5 sweep then measured how far the 10⁻¹⁴ figure is
from a bound:

| worst rel. score error at M=401, A=8 | Gaussian | bimodal | Laplace | Student |
|---|---|---|---|---|
| ordinary draws | 2e-13 | 1e-15 | 1e-5 | **9.9e-4** |
| tail-conditioned | 2e-7 | 1e-11 | **1.7e-2** | 8.3e-3 |

10⁻¹⁴ is the *best* cell, not a bound. Partial underflow is real: up to 0.27% of
likelihood entries round to exactly zero at the production grid. No reported
number moves — all sit in the ordinary-draw regime, resolved to 10⁻³ — but the
discretisation claim now has a measured scope.

**3.5.** The negative-Q-gain defect was real and is fixed. On whether it
contaminated the frozen table: `inner_q_gain` was not recorded, so your proposed
audit was impossible as stated. I instrumented a replay of the production fit
instead — **1,764 inner gains, none negative**, including across the 41-of-60
outer iterations that stop early at n=2048. The table is uncontaminated. The
flag *is* hot, though: `inner_converged` is true in 53% of certified cells and
96% at n=8192, against a docstring claiming it is "normally False".

- [src/em.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/src/em.py) · [src/kernels.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/src/kernels.py)
- [tools/make_grid_convergence.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/tools/make_grid_convergence.py)

## 6. Your issues 3.2, 3.7, 3.10 — audit, Hellinger, ring

**3.2.** The audit row withdrew capacity saturation flatly while §corr-capacity
withdraws only the *shape-based attribution*, and ch11 preserved the
evidence-based result. Both documents now draw that line the same way. I did not
delete the evidence-based saturation, because the correction section does not
withdraw it — please push back if you think that is self-serving.

**3.7.** You were right that the floor was the expression, not the metric.
Switched to `½Σw(√p−√q)²`; identity is now exactly 0 and reported values move by
1.2e-15. The test that asserted the artifact is replaced.

**3.10.** The theorem already said "single-frame"; the audit summary row did not,
and now does. On your point that the machine-precision zero is tautological —
correct, the function takes no ψ argument — added paired sample-level controls:
cross-ψ one-frame energy distances (2.6–4.7e-3) fall **below** the same-ψ
resampling floor (1.5e-2), while the lag-one joint tracks sin ψ to 0.002.

- [ch11-claim-audit.tex](https://github.com/GioviManto/diffusion/blob/main/overleaf/compendium/chapters/ch11-claim-audit.tex)
- [src/metrics.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/src/metrics.py)
- [tests/test_ring_blindness_controls.py](https://github.com/GioviManto/diffusion/blob/main/research/nongaussian-bp/tests/test_ring_blindness_controls.py)

## 7. Your code-defect list (C1–C2, H1–H5, M1–M5, L1–L2)

All closed. The ones where the fix differs from what you proposed:

- **M2 (ρ unconstrained).** Deliberately still unconstrained: the update is the
  exact maximiser of Q, and clamping trades that for a silently biased M-step
  the monotonicity check could never catch. The defect is interpretive, so
  `is_covariance_stationary` and a warning were added instead. It fired
  immediately in a test that transposes grids.
- **H2 (duplicated recursion).** Not refactored — a refactor of the hot path
  three weeks from submission trades a hypothetical divergence for a real chance
  of causing one. Instead the property is pinned: the two must agree on evidence
  and site-1 posterior across four families and three noise levels. They agree
  to 1e-12.
- **H4.** A reader-side scan found **three more** declared-but-unread fields
  beyond the two you knew about: `em_loglik_tol` (correct only by coincidence —
  the library default matched), `em_shape_tol` (the stopping rule the appendix
  describes still does not exist), and `innovation`.
- **H3.** `BP_DEVICE=gpu` now raises. Both GPU batch scripts also gated with
  `pytest || exit`, which exits 0 when the whole module skips.

## 8. Thesis structure — your Role B section 7

Main text **110 → 104pp**; appendices 49 → 55. Nothing deleted, relocated:
Chapter 2 25→20 (the Ising/Hopfield/Boltzmann block from five sections to one,
derivations to Appendix D), ring 16→14 (proofs to a new Appendix H).

Also added, filling a hole your report implies but does not name: the
introduction had **no strand on neural score approximation at all**, so Chapter 9
arrived as an experiment rather than an answer to a stated question. It now has
one, framed as amortised inference. And the related-work engagement you asked
for — assumed-density filtering, EP, Gaussian-sum, grid HMM filtering, particle
methods — is in, with the distinction stated.

**Not done, and I would rather you judge these than have me claim otherwise:**

- Main text is 104pp, not 100. The remaining 4pp would come from Chapter 5
  (17pp against your 12–14 allocation) or the ring (14pp against your 8–10).
- Your item 3 (merge repeated diffusion derivations) and item 4 (merge the
  Gaussian and Laplace setup) — not attempted.
- Appendix B ("why not AMP") is still 6pp, not 2. Left because appendices are
  now explicitly allowed to run long.
- Priority 3 (ECM audit) done; Priority 4 (resolve the 16 excluded cells at
  M=801) and Priority 2 (budget saturation) are **running on the cluster, not
  merged** — so the headline table still excludes those 16 cells and the budget
  is still right-censored.

## Questions

1. Is the revised structured-baseline claim acceptable, or does the
   "fails to acquire as data grows" reading overclaim?
2. Given the aggregation table, should the abstract quote the ratio-of-means
   range rather than ours?
3. Is preserving the evidence-based capacity saturation defensible?
4. With the discretisation scope now measured, is "exact grid-BP reference"
   still the right phrase?
5. Where would you take the last 4 pages from?
