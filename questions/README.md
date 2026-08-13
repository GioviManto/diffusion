# `questions/` — the project's own open questions, tracked and answered

One document lives here: **[`QUESTIONS_AND_ANSWERS.md`](QUESTIONS_AND_ANSWERS.md)**.

It collects every question this project has posed to itself and gives each a
status — ANSWERED, PARTIAL, or OPEN — with the evidence named or the gap stated.
Questions come from five places:

| tag | source |
|---|---|
| **[R]** | `research/gaussian-ar1-bp/markov_gaussian_approx/report/bp_markov_diffusion_gaussian_approx.pdf`, §"Interpretation and next steps" |
| **[F]** | `docs/RESULT_LEDGER.md` (archived Aug 2026), §"Explicitly future" (F1–F5) |
| **[E]** | The Marc/Jérôme email of 2026-07-30 |
| **[C]** | The 2026-07-29 call with Jérôme — the grid-BP / Gaussian-BP concerns |
| **[L]** | Questions that arose during the Layer-5 (EM + BP) work |

## Keeping it honest

Two rules the document follows, both learned the hard way in this project:

1. **PARTIAL always names the gap.** "Mostly done" without saying what is missing is how an open question quietly becomes a closed one.
2. **A question that was answered *wrongly* stays visible.** Where a claim was made and later overturned by better data, the document records both — see R1 (a rate that 4 replicates could not resolve), L3 (a lattice attractor mislocated by a seeding confound), and L4 (a kurtosis convergence that did not survive a full run).

## Related documents

The three below are archived (Aug 2026, superseded by the more complete
`research/nongaussian-bp/` documentation); kept as citations for what they originally covered.

- `docs/EM_BP_LEARNING_COMPENDIUM.md` — what was built and measured in Layer 5. See `research/nongaussian-bp/compendium/` for the current version.
- `docs/AGENT_HANDOFF_EM_BP.md` — how to continue without reintroducing known bugs. See `research/nongaussian-bp/REPRODUCIBILITY.md`.
- `docs/RESULT_LEDGER.md` — claim-by-claim classification across all layers. See `research/nongaussian-bp/CLAIMS_TO_UPDATE.md`.
