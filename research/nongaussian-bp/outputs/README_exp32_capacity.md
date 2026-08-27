# exp_32_capacity — first run (em_cap=400) is confounded; do not cite it

Job chain `635645→635646→637047` (sweep) + `635648` (contrast), commit
`99acb49`, clean provenance, 160/160 cells complete. The predeclared C=16 vs
C=8 contrast came back **resolved, not equivalent, in the direction "C=16 is
worse"** at both N=128 and N=512, on both held-out log-evidence per edge and
schedule-level score risk. See `capacity_contrasts.csv` in this directory for
the exact numbers.

## Why this run cannot be trusted as-is

Every one of the 320 fits in this sweep (16 seeds x 2 sizes x 5 components x
2 inits) hit the `em_cap=400` iteration ceiling without satisfying the formal
`tol=1e-9` convergence criterion — `em_converged` is `False` in every row.

Checked directly (not just inferred from the flag): at iteration 400 on a
representative fit, C=8's held-out evidence moved 0.025 nats over the last
100 iterations (per-edge gain 9.9e-9 — a hair past the tolerance, essentially
flat). C=16's moved **0.77 nats over the same 100 iterations** (per-edge gain
6.9e-8, thirty times larger). C=16 was still climbing substantially when the
cap cut it off; C=8 was not.

This is the capacity-vs-convergence-rate confound the whole review is about,
recurring in a different place. It is not the same failure as the withdrawn
shape-based capacity claim (that one was about the fitted innovation SHAPE
needing ~2000 iterations while evidence settles by ~40; this one is about
held-out EVIDENCE itself not yet having settled for the higher-capacity model
at a shared iteration cap) — but the shape of the mistake is identical: a
shared budget that is generous enough for the simpler model and not for the
richer one, read as a property of the models rather than of the budget.

## What is NOT in question

- The sweep's own provenance and completeness: clean, 160/160, verified.
- That the extra capacity is being used: effective component count rises from
  6.85 (C=8) to 13.33 (C=16) at both sizes, and the narrowest fitted
  component stays well-resolved (s_min/h 3.6-4.4, comfortably above the ~2
  floor) at both capacities. This is not an unused-capacity or
  under-resolved-component artefact.
- The statistical design itself: 16 paired seeds, validation-selected
  initialisation, predeclared equivalence region, paired bootstrap. All of
  that is sound; what it was fed was under-converged at C=16.

## The fix

`em_cap` raised in `experiments/exp_32_capacity_equivalence.py`, sized by
directly measuring how many iterations C=16 needs to reach a per-edge gain
comparable to C=8's at 400 (see the module docstring's correction note and
the commit that raised the cap for the exact number). Rerun queued; this
directory's numbers are superseded once it lands.

**Do not** report "capacity is not equivalent past C=8" from the em_cap=400
run. **Do not** report "capacity saturates by C=8" either — that claim is
still what was withdrawn from the thesis, and this run does not restore it.
The honest state until the corrected run lands is: unresolved, again, for a
new and now-understood reason.
