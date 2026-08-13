# Diffusion — MSc Research Thesis (DSBA, Bocconi)

Score-based diffusion for structured data: exact joint scores for Markov chains, Gaussian
belief propagation and AMP, Gaussian-closure error beyond the Gaussian case, and — the most
recent line of work — EM-learned non-Gaussian innovation kernels, compared against trained
neural denoisers both pointwise and generatively.

## Start here

Two documents are the canonical output of this project. Everything else in `research/` is the
working material behind them.

| Document | What it is |
|---|---|
| [`thesis/main.tex`](thesis/main.tex) → `thesis/main.pdf` | The Bocconi MSc thesis, 12 chapters + 3 appendices (178 pp). Chapters 1–5 are background; 6–11 are the research, ending with the EM/non-Gaussian material (Ch. 10–11); 12 is discussion and conclusions. |
| [`research/nongaussian-bp/paper/main.tex`](research/nongaussian-bp/paper/main.tex) → `paper/main.pdf` | The standalone NeurIPS-format paper on the EM/BP estimator (24 pp). Self-contained Overleaf mirror at `research/nongaussian-bp/overleaf/` — see that folder's `README.md` to share it with Marc and Jérôme; regenerate it with `./overleaf/sync.sh`. |

For a guided walkthrough of the theory *and* the code behind Chapters 10–11 — runnable,
executed notebooks that re-derive every number from committed data rather than transcribing
it — start at [`research/nongaussian-bp/notebooks/README.md`](research/nongaussian-bp/notebooks/README.md).
For what is and is not established, in one place: `research/nongaussian-bp/CLAIMS_TO_UPDATE.md`
and `research/nongaussian-bp/audit/AUDIT_NOTE.md`.

## Layout

| Path | Content |
|---|---|
| `thesis/` | The official Bocconi thesis (LaTeX + figures + compiled PDF) |
| `research/nongaussian-bp/` | The main research line: EM-learned non-Gaussian innovation kernels, exact chain BP, 27 experiments, HPC sweeps, notebooks, paper. See its own `README.md`. |
| `research/experiment1-rotating-ring/` | The rotating-ring dynamic object behind thesis Chapter 9 — exact Cartesian surrogate, numerical validation, locality diagnostics |
| `research/board-3problems/` | The three board problems from supervision, each developed in its simplest content-preserving setting |
| `research/unified-note/` | Consolidated working note (main.pdf, code, 8 figures, 53/53 audit) |
| `research/gaussian-bp/` | 31-page Gaussian BP + AMP note, 3 executed notebooks, closed forms |
| `research/bp-from-scratch/` | Independent equation-by-equation BP derivation from the 5 Jun call |
| `research/gaussian-ar1-bp/` | Gaussian AR(1) package (BP = precision-matrix score to machine precision) |
| `research/bp-generalization/` | General (non-Gaussian) BP formulation sketches + code |
| `questions/` | Every open question the project posed, with status and evidence |
| `research/session-summaries/` | Five session-summary PDFs + problem formulation |
| `research/initial-experiments/` | Initial diffusion-setup experiments and early figures |
| `research/notebook-scans/` | Handwritten derivation notebook scans |
| `tools/` | Repo-wide auditors (provenance checking, cross-run summarisation) |
| `sources/` *(local only)* | Key papers and textbooks (gitignored) |
| `meetings/` *(local only)* | Call recordings/transcripts and board sketches (gitignored) |
| `thesis/editorial/`, `thesis/official-guides/` *(local only)* | Thesis revision notes and the university's own guide PDFs — kept for reference, gitignored |

Full project history (superseded packages, toy models, transcripts, retired drafts) is
preserved outside this repo in `../archive/`, organised by the date each batch was retired.

## Verifying a claim

`research/nongaussian-bp/tools/check_all.sh` runs everything this project can check locally in
one command: the test suite, provenance audits, notebook re-execution and freshness, sweep
completeness, and cross-process reproducibility. `./check_all.sh --quick` skips the slow parts
(notebooks, reproducibility) for a one-minute sanity check.
