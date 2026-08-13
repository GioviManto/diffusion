# CORRECTION — this audit was written against the wrong tree

**Read `questions/QUESTIONS_AND_ANSWERS.md` on
`origin/claude/em-bp-denoiser-learning-e07ike` instead.** That document is the real
answer to "what do we have for Jérôme's and Marc's asks", it is more complete than
anything below, and it predates this file.

## What happened

I ran `git branch -a` without fetching, saw only `main` and `origin/main`, and concluded
the research branch did not exist. It does — 49 commits ahead of `main`, with `main`
containing nothing it lacks. The audit that used to be in this file was built entirely
from the untracked local directory `research/em-learning/`, which is a **redundant
reimplementation** of `research/nongaussian-bp/` on that branch.

Nearly every "gap" reported here was already closed:

| I said | Actually |
|---|---|
| Reverse diffusion not done | `src/reverse.py` + `exp_05_reverse_dynamics.py`; kurtosis 0.12 vs true 2.7–2.9 |
| MLP-vs-BP on Laplace not done | `exp_07_em_vs_score_network.py`, 24 architectures, 4-seed replicates |
| Depth/architecture question not started | `exp_12_receptive_field.py` — answered as N2 |
| No paper | `report/em_bp_learning.tex`, 14 pp, five propositions, compiles |
| EM claim is Gaussian-only and shaky | 13 params vs 25,248, ≥64× data gap, with three controls |

The branch also has 10 test files, built PDFs, and a 50 KB compendium.

## What this directory does still contribute

One thing, and it is worth keeping: a **5,530-row cluster sweep** (48 CPU-hours, 0 failed
cells) in `outputs/`, covering 5 families × 4 ρ × 3 n × 3 M × 3 reps. It is largely
duplicative of `exp_03` / `exp_06` / `exp_07`, but at much larger scale, and it
**independently reproduces** the branch's central correction: the Gaussian-closure error
decays monotonically in `t` with no large-`t` blowup (0.200 → 9.2e-5 for Laplace), which
is what `C-i` reports after the information-form fix. Independent replication on a
separate implementation is real evidence; treat it as that, not as new science.

The merged CSVs are in `outputs/`. Everything else here should be considered superseded.

## The actual remaining gaps

From `QUESTIONS_AND_ANSWERS.md`, whose own accounting is: every original question closed
or bounded, four partial, none open.

| | Gap | Why it matters |
|---|---|---|
| **E4** | Overleaf write-up has no results section | the only item with the 16 Sep deadline on it |
| F4 | reverse dynamics never run with a *learned* score | now unblocked; also where BP's 211–320× inference cost bites |
| F3 | learned kernel + rank-one correction | turns "the prior is Markov" into a measured approximation |
| R5 | neural message approximators need a hard prior | partial, evidence on three sides |
