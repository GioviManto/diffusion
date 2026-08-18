# research — code and the experimental record

`nongaussian-bp/` is the live package. Everything else is earlier work kept for
provenance: the results in the documents trace back through these, and several
of them are the independently-written implementations that current code is
checked against.

| Folder | What it was for | Status |
|---|---|---|
| [`nongaussian-bp/`](nongaussian-bp) | the current package — grid BP, EM, kernels, all frozen experiments | **live** |
| `board-3problems/` | the three board problems, incl. the audited rotating-ring implementation | reference |
| `gaussian-ar1-bp/` | Gaussian AR(1) chain, closed forms and BP | superseded |
| `gaussian-bp/` | earlier Gaussian BP work | superseded |
| `bp-from-scratch/` | BP built up from first principles | superseded |
| `bp-generalization/` | generalising beyond the Gaussian case | superseded |
| `experiment1-rotating-ring/` | the first ring experiments | superseded |
| `initial-experiments/` | first exploration | superseded |
| `notebook-scans/` | scans of handwritten derivations | reference |
| `session-summaries/` | working notes per session | reference |
| `unified-note/` | an earlier attempt at one unified write-up | superseded |

"Superseded" means the current package does the same thing better, not that the
folder is worthless — `board-3problems` in particular is the implementation the
ring port is checked against bit-for-bit, so it is load-bearing for a test.

Older material that nothing depends on is in `../archive/`, on disk but not
tracked.

## The one thing to know before running anything

Every experiment imports `nongaussian-bp/experiments/frozen_config.py`. It fixes
ρ, the grid, the noise schedule, the replicate count and the iteration budgets in
one place, so no run can quietly diverge from the configuration the documents
describe. `tools/check_paper.sh` fails if an experiment sets any of them locally.
