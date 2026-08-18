# Overleaf handover — four documents, one folder

Everything needed to compile all four documents is in this folder. Nothing
resolves a path outside it, so uploading the folder to Overleaf as a single
project works with no rearranging. Set the main document per compile.

| Root | What it is | Ready to send? |
|---|---|---|
| `workshop.tex` | 4pp workshop version | Yes |
| `paper.tex` | the full paper, 9pp body | **No** — one placeholder |
| `compendium.tex` | every derivation and experiment, including cut ones | Yes |
| `thesis.tex` | the thesis | Yes, as a draft |

All four build with `tectonic`, with zero undefined references and zero
undefined citations. `./build.sh` compiles all four and reports page counts.

## The paper is not ready to send

`paper.tex` renders a loud red `[PENDING: ...]` marker where a number is not yet
earned: the numerical-values table in `appendix.tex` needs regenerating from the
frozen batch. It is called out here rather than quietly shipped, and the marker
is red on purpose — it should be impossible to send by accident.

`workshop.tex` no longer inputs anything unfilled, so it is sendable. `build.sh`
attributes placeholders per root by following `\input` transitively, so it will
say which documents are actually affected rather than listing every file.

The efficiency table both share (`sections/tab-efficiency.tex`) is now filled
from the run in which **both** arms have their optimisation budget chosen on a
validation bundle — the network its parameterisation and training length, EM–BP
its iteration count. One caveat is stated in its caption rather than hidden: at
the largest sample size both arms select the largest budget the grid offers
(33% and 61% of cells), so that row's ratio is bounded by the grid rather than
by the estimators, and the range over the other six sizes is quoted alongside it.

The compendium and thesis are allowed to carry provisional results, and both say
which results are provisional.

## What is shared, and why editing matters

The four documents deliberately share files. This is the mechanism that stops
them disagreeing with each other, which is a failure this project has already
had:

- `notation.tex` — every root inputs it
- `sections/` — `model`, `prop-lmmse`, `prop-fisher`, `ring-model`,
  `thm-blindness`, `tab-efficiency` are inputted by **both** `paper.tex` and
  `workshop.tex`
- `references.bib` — merged from the paper's and the thesis's bibliographies
  (132 entries); the paper's version wins on the 14 keys that were in both,
  because those carry the `eprint` and `doi` fields
- `figures/` — the union of both figure directories

**Editing a file in `sections/` changes both the paper and the workshop.** That
is intended. If you want them to differ, the text has to move out of `sections/`
and into the roots.

## Regenerating

This folder is assembled by `tools/make_handover.sh` in the repo. Re-running it
rebuilds the folder from the repo sources and **overwrites edits made here**.

```
tools/make_handover.sh           # rebuild
tools/make_handover.sh --check   # report drift, change nothing
```

So: revise here freely for Overleaf, but before anyone re-runs the generator,
run `--check` to see what would be lost, and copy real edits back to the repo
sources under `research/nongaussian-bp/` and `thesis/`.

This README and `build.sh` live at `tools/handover-assets/` in the repo and are
copied in by the generator — edit them there, not here, or a rebuild will
discard the change.
