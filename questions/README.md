# `questions/` — the project's own open questions, tracked and answered

One document lives here: **[`QUESTIONS_AND_ANSWERS.md`](QUESTIONS_AND_ANSWERS.md)**.

It collects every question this project has posed to itself and gives each a
status — ANSWERED, PARTIAL, or OPEN — with the evidence named or the gap stated.
Questions come from four places:

| tag | source |
|---|---|
| **[R]** | `research/gaussian-ar1-bp/markov_gaussian_approx/report/bp_markov_diffusion_gaussian_approx.pdf`, §"Interpretation and next steps" |
| **[F]** | `docs/RESULT_LEDGER.md`, §"Explicitly future" (F1–F5) |
| **[E]** | The Marc/Jérôme email of 2026-07-30 |
| **[L]** | Questions that arose during the Layer-5 (EM + BP) work |

## Keeping it honest

Two rules the document follows, both learned the hard way in this project:

1. **PARTIAL always names the gap.** "Mostly done" without saying what is missing is how an open question quietly becomes a closed one.
2. **A question that was answered *wrongly* stays visible.** Where a claim was made and later overturned by better data, the document records both — see R1 (a rate that 4 replicates could not resolve), L3 (a lattice attractor mislocated by a seeding confound), and L4 (a kurtosis convergence that did not survive a full run).

## Related documents

- `docs/EM_BP_LEARNING_COMPENDIUM.md` — what was built and measured in Layer 5.
- `docs/AGENT_HANDOFF_EM_BP.md` — how to continue without reintroducing known bugs.
- `docs/RESULT_LEDGER.md` — claim-by-claim classification across all layers.
