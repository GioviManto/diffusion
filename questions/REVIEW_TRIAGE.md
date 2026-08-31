# Triage of the ChatGPT Pro reviewer pass (30 Aug 2026)

> **STATUS: all twelve blocking items closed.** Commits `a8548ff`,
> `9bac337`, `86e3b81`, `bf007d3`, `b233b79`, `d696bd5`, `9f97afa`,
> `40c27ea`, `c8b4694`. All four documents build clean; paper 9 body pages,
> thesis 181. Verified by grep that no surviving text asserts the generative
> dissociation, capacity saturation past C=8, the 9-14x data claim, or a
> 2-4x structured-baseline figure.
>
> Blocking item 12 (missing title page and abstract) was **not a defect**:
> `thesis/main.tex` implements the university's own format, "Guide to the
> University" section 10.3 -- no title page and no abstract in this file,
> four preliminary pages, content from page 5. The system prepends them.
>
> Two things turned up that the reviewer did not flag and that mattered more
> than several that were. `build.sh` could never detect an undefined
> reference -- tectonic keeps the engine log to itself unless `--print` is
> passed -- so the check silently matched nothing for every document, always;
> that is why `rem:asymmetry` rendered as "??" until an outside reader saw
> it. Fixing the gate exposed four dangling refs and a duplicate label. And
> the sample-efficiency figure was plotting the `nseq=8192` point the table
> had withdrawn for unrecoverable provenance, at the far right where it
> carries most weight, under a caption asserting the two could not disagree.
>
> **Still open, and it needs your decision: length.** See the bottom.

Source: two reviews requested via `questions/PROMPT_single.md` — a NeurIPS-AC
review of paper+workshop (score 3/6 weak reject) and a Bocconi committee
review of the thesis (not ready, major revisions). Full text is in the chat
transcript, not duplicated here. This file is the action plan, grouped by
what kind of fix each item needs and whether it can be done right now.

**Overall reading:** the reviewer's central complaint matches what we already
suspected (development-log tone leaking into production docs), but the
review also caught several things that are wrong independent of tone —
direct contradictions between sections, and a mismatch between the paper and
the thesis on what the actual algorithm is. Those have to be fixed no matter
what we decide about register.

---

## Group A — Real contradictions / defects, fixable now, no new data needed

These are blocking regardless of the tone discussion. Each is a factual
inconsistency inside the documents as they stand today.

1. **Thesis Ch9.4 vs Ch9.8 vs Ch10**: the pointwise/generative dissociation
   result is withdrawn in 9.4, then reasserted as one of four headline
   findings in 9.8, then correctly re-withdrawn in Ch10. Pick one status and
   make 9.8 agree with 9.4/10.
2. **Thesis Ch9.3 vs Ch9.8 vs Ch10**: same pattern for the mixture-capacity
   claim — 9.3 is careful ("failure to resolve ≠ demonstration of none"),
   9.8 asserts "capacity beyond ~8 buys nothing," Ch10 withdraws it again.
   Same fix: make 9.8 match 9.3/10, not contradict them.
3. **Thesis Ch6 vs Ch8 chain definition**: Ch6's Laplace chain has
   `a_{-1}=0`; Ch8 claims to reuse "the same chain (6.1)" but actually
   defines a different stationary-variance construction. Pick one
   definition, use it everywhere, and check which simulations/claims
   actually used which.
4. **EM inner-sweep count: paper says up to 16, thesis says 4** (thesis
   §8/Appendix H region). This is exactly the kind of budget-cap confound
   the whole review cycle has been about — recurring between the two
   documents themselves this time. Reconcile against what
   `experiments/exp_3*` actually run and correct whichever document is
   stale.
5. **"9–14× less data needed" (thesis Ch9 summary, Ch10) is not the same
   estimand as the paper's 7.0–17.6× error-ratio-at-equal-M**, and the
   paper explicitly declines to quote a data-equivalence number because the
   curves don't cross in range. Either compute the crossing/interpolation
   properly, or drop "less data" framing everywhere and standardize on the
   error-ratio framing the paper already uses correctly.
6. **Thesis Ch3: "no network is trained in this thesis"** — stale, Ch8/9
   train MLP/CNN baselines. Fix the sentence.
7. **Thesis Appendix C: "no GPU, no neural network framework, no stochastic
   training"** — same staleness, plus the reproducibility chapter generally
   documents the Gaussian/non-Gaussian/ring packages and not the Ch8/9
   pipeline. Needs a pass to cover the actual final experiment suite,
   HPC/CUDA environment, frozen configs.
8. **Thesis Ch1 roadmap says "three appendices," rendered thesis has ten.**
   One-line fix, but a good canary for "later chapters appended without a
   final integration pass" — worth treating as a prompt to actually do that
   integration pass rather than just patching the count.
9. **Fig 9.1 caption references "two network baselines," figure shows one
   curve.** Fix caption or figure, whichever is stale.
10. **Grid spacing possibly off-by-one**: thesis writes `2A/M` for M points
    including endpoints; that's normally `2A/(M-1)`. Check `src/` grid
    construction against the written formula and correct whichever is
    wrong — this one matters because it feeds Prop 2 / the M-step, not just
    exposition.
11. **t=0.9 threshold claim**: intro/conclusion claim median error
    `< 1e-3` for t≳0.9; Ch6's own table/plot looks closer to `~1e-2` at
    that point, and the actual measured claim seems to be "no tested trial
    exceeds 0.1 error" (a max, not a median). State the real number,
    explicitly, at t=0.9.
12. **Theorem 5.15 overclaimed as an exact theorem**; the actual support is
    a fitted slope matching `log q` within 0.2% over r=2..13, not a proof
    that survives finite-window boundary effects. Either produce the real
    proof/bound, or re-title it a numerical proposition/conjecture with the
    fit as evidence.
13. **Missing title page / abstract in the rendered thesis PDF.** Could be
    a template issue (Bocconi may prepend a cover on official submission)
    or a real gap in `overleaf/thesis/`. Check `overleaf/build.sh` output
    and the thesis's title-page include before assuming it's just a
    packaging artifact.

## Group B — Blocked on already-running experiments (nothing to do but wait)

These are the review's two biggest scientific complaints, and both are
already mid-fix on the cluster right now:

- **"Structure-aware neural baseline is uncertified"** (paper App N,
  thesis 9.2/9.8, Blocking 3) — this is exactly `exp_31_structured_baseline`,
  currently running in three parallel lanes (jobs 641509–641514). Once
  merged, this replaces the disclaimed CNN table and gives a real number
  for the "residual against a structure-aware baseline" claim in the
  abstract.
- **"Capacity beyond C=8 buys nothing" is not established** (paper §... ,
  thesis 9.3/9.8, Blocking 2) — this is `exp_32_capacity_equivalence` with
  the corrected `em_cap=1200`, also running now (jobs 641515–641520). Once
  merged and the contrast is regenerated manually (not via `afterany`,
  per the lesson in `README_exp32_capacity.md`), this gives the honest
  answer instead of the withdrawn one.

Do not touch the prose for these two claims until the reruns land — fixing
the wording now would just require a second pass.

## Group C — Register / development-log language (the original complaint)

The reviewer exhaustively quoted every instance in both documents — 22 in
the paper, ~54 in the thesis. Full list is in the review text (not
reproduced here to keep this file short). Pattern: strip narration of
"an earlier draft/version claimed X, that was wrong, here is the
correction" down to just the corrected, current fact. Keep hedges that are
honest scope statements (the reviewer flagged which ones to keep, e.g.
Ch9.3's "failure to resolve is not equivalence," the ρ-slope
not-yet-asymptotic caveat, the MLP-comparison asymmetry disclosure).

Concrete instruction that generalizes across all ~76 instances: **delete
the "earlier draft/version said X, it was wrong because Y" clause; keep
only the current correct statement**, unless the wrong prior claim is
itself pedagogically necessary (rare — the reviewer didn't flag any case
where it was).

Move to `overleaf/compendium/` outright (not just reworded) rather than cut:
- Thesis Appendix I (withdrawn-results log) — reviewer explicitly says this
  is compendium material, full stop.
- Thesis Chapter 4 (toy-model chronology) — keep only the joint-score
  correction insight in the Ch1 intro, move the rest.

## Group D — Structural / length

- Thesis is 178pp; reviewer's estimate of a right-sized version is
  120–140pp including bibliography. Sources of the cut: Ch2 background
  (currently spans stationary action → phase transitions → generative
  modeling in one arc — reviewer says the "one theorem, many stationarity
  principles" framing is rhetorically attractive but not mathematically
  tight, cut it back), Appendices D–F (re-derive things Ch2 already
  covers), Appendix B (AMP/TAP — interesting but peripheral, reviewer says
  don't headline it unless the relation to the main thesis is sharpened).
- Ch7 (rotating ring) needs an explicit framing decision: second major
  contribution, or a shorter side study. Either is defensible per the
  reviewer, but it currently reads as neither. **This needs your/Marc's
  call, not mine.**
- Ch9 needs to be substantively rebuilt around only the claims that survive
  Group A's contradiction fixes — not just re-worded.

## Group E — Paper-specific (on top of the above)

- Headline estimand choice (mean-of-ratios vs ratio-of-paired-means):
  paper reports 7.0–17.6 (mean of ratios, the largest defensible number);
  Appendix P's own robust range is 6.6–12.4. Reviewer wants either a
  predeclared justification for picking the mean-of-ratios, or leading with
  the robust range instead of the max.
- "Exact score" language in the abstract should be qualified — functionally
  exact, numerically validated-approximate in the non-Gaussian experiments.
- Appendix N (disclaimed structured-baseline table) — once exp_31 lands,
  this whole appendix's caveat-drenched framing goes away; don't spend time
  rewriting it now, just replace it wholesale when the data arrives (Group
  B).
- Related-work: reviewer wants explicit positioning against deterministic-
  grid/quadrature filters for continuous-state HMMs, EP/assumed-density
  filtering, Gaussian-sum smoothing, HMM kernel identifiability, amortized
  smoothing-operator networks. This is a real gap, not a phrasing issue —
  needs actual citations added.

---

## What is actually left

**Group B — waiting on the cluster.** `exp_32` lane a is complete (70/70 at
the corrected `em_cap=1200`); lanes b and c are running. `exp_31` — the
structured-baseline measurement, which is the review's single biggest
scientific objection — has not started: all six shards are PENDING behind
priority and node-drain. Every document now says plainly that its results
are not included, so nothing is blocked on it except the stronger claim
itself.

**Group D — length. THE ONE OPEN DECISION.** Thesis is 181pp against the
reviewer's suggested 120–140. Measured page counts:

    body            Ch1 10   Ch2 20   Ch3 6   Ch4 3   Ch5 17
                    Ch6 10   Ch7 15   Ch8 9   Ch9 12  Ch10 6   = 108pp
    appendices      A 3   B 6 (AMP/TAP)   C 5   D 12 (statmech)
                    E 8 (stochastic)  F 6 (diffusion)  G 6  H 3  I 2  = 51pp

The reviewer wants Chapter 2 cut back and Appendices D–F reduced to "only
derivations needed for self-containment". I checked what actually depends on
them: **Appendices D, E and F are referenced from Chapter 2 and nowhere
else** — no research chapter (5–10) cites any of them. On that evidence the
reviewer is right that they are background supporting background, and
26 pages is the largest single block available.

Appendix B (AMP/TAP) is a different case and I would keep it: it is cited
from Ch1, Ch5, Ch6, Ch10, App C and App G, so it is genuinely woven in, and
the reviewer's objection there was only to headlining it.

**Why I have not done this cut.** It is large, hard to reverse, and turns on
what the thesis is *for* rather than on any defect — and the specific
material at risk is the statistical-mechanics background, in a thesis being
examined with Marc Mézard involved. Cutting a statistical physicist's
framing because a reviewer persona recommended it is exactly the kind of
call that should not be made on my own judgement. The options:

1. **Cut hard** (~40pp): Ch2 to ~12pp, Appendices D–F to ~12pp combined.
   Lands near the reviewer's 140. Highest risk to the physics framing.
2. **Cut the appendices only** (~14pp): leave Ch2 intact, reduce D–F to what
   Ch2 actually needs. Lands ~167. Low risk, since nothing but Ch2 cites
   them.
3. **Leave it.** Length was explicitly filed by the reviewer under
   "important but non-blocking". Nothing about the thesis's correctness
   depends on it, and everything blocking is now fixed.

My recommendation is **2** — it removes the duplication the reviewer
identified without touching the argument or the framing, and it is the only
one of the three that is clearly right regardless of what Marc wants the
thesis to look like.

**Chapter 7 framing.** The reviewer says the rotating ring currently reads
as neither a second major contribution nor a short side study, and that
either is defensible. They explicitly call this your and Marc's call, not
theirs.

## Suggested order of operations, given the 16 Sept deadline

1. **Now, while exp_31/exp_32 finish on the cluster**: fix everything in
   Group A (self-contained, no data dependency, removes the worst
   "this document contradicts itself" risk before you show Marc/Jérôme
   anything tomorrow).
2. **Also now**: start the Group C mechanical strip (the "earlier draft
   said X, withdrawn" clause deletions) — this is high-volume but
   low-judgment, and independent of the reruns.
3. **When exp_31/exp_32 land** (currently running, ETA depends on queue —
   will report): fill in Group B with real numbers, replace Appendix N,
   settle the capacity and structured-baseline claims in Ch9/paper §7 for
   good.
4. **Group D structural cuts and the Ch7 framing decision**: needs your
   input on scope — flagging, not starting without a steer.
5. **Group E related-work additions**: lowest urgency of the substantive
   items, but real work (finding and reading citations) — worth carving
   out time even under deadline pressure since it's a "novelty" objection,
   not a wording one.
