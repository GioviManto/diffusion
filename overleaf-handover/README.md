# Overleaf handover — four documents, one folder

Everything needed to compile all four documents is in this folder. Nothing
resolves a path outside it, so uploading the folder to Overleaf as a single
project works with no rearranging. Set the main document per compile.

| Root | What it is | Body | Ready to send? |
|---|---|---|---|
| `workshop.tex` | the analytical result: exact scores + LMMSE closure | 3pp of 4 | Yes |
| `paper.tex` | the full paper | 9pp of 9 | Yes |
| `compendium.tex` | every derivation and experiment, including cut ones | 53pp | Yes |
| `thesis.tex` | the thesis | 169pp | Yes, as a draft |

All four build with `tectonic`, with zero undefined references, zero undefined
citations, and no unfilled placeholders. `./build.sh` compiles all four and
reports page counts; it attributes placeholders per root by following `\input`
transitively, so it names the documents actually affected rather than every file.

## What changed in the last revision

Both production documents were rescoped after an external audit found them
carrying more questions than their page counts could defend.

**The rotating ring is out of the paper and the workshop.** It changes the state
space, the dynamical model and the estimand, so calling it the next rung of the
chain derivation was a presentational move rather than a real continuity. It
lives in the compendium (ch12) and the thesis (ch09), both of which carry a more
complete version than the paper ever did — the paper's copy predated the
normaliser fix. Cutting it took the paper body from 10pp to 8pp.

**The neural comparison stayed, reframed.** The audit recommended cutting it too,
but on a stale reading: it saw a README from before the symmetric rerun and
concluded the protocol was still asymmetric in selection. It is not — both arms
now choose their optimisation budget on a disjoint validation bundle. What
survives of that criticism is about framing, and is now stated plainly wherever
the number appears: the two arms are asymmetric in *supplied information* by
construction, so the measurement is of what a correct Markov assumption buys on
a family where it holds, not a ranking of learning algorithms.

**Efficiency numbers now come from macros, not typing.** The abstract read
"between 8 and 14" for weeks after the regenerated table said 7.3–15.7. Every
figure the prose quotes is now defined in `sections/efficiency-numbers.tex`,
generated with the table, so the two cannot disagree. Related: the table said
"12 free parameters", right at C=4 and wrong since the config moved to C=8 —
it is 24, and now derived rather than written.

One caveat is stated in the table caption rather than hidden: at the largest
sample size both arms select the largest budget the grid offers (33% and 61% of
cells), so that row's ratio is bounded by the grid rather than by the estimators,
and the range over the other six sizes is quoted alongside it.

The compendium and thesis are allowed to carry provisional results, and both say
which results are provisional.

## How the page counts are measured, and why they moved

Both documents now carry `\label{LastMainPage}` immediately before the
bibliography, so LaTeX reports where main content ends. The gate previously found
the bibliography by looking through extracted text for the word "References" next
to a "[1]", which **undercounted by one page** in both documents — it stops at the
page before the one where references appear, which is wrong whenever the
bibliography starts on the same page the main text ends. So the paper is at 9 of 9,
not the 8 the old check reported: compliant, but with no slack left.

The workshop has 1pp of genuine headroom. If you want to spend it, the place is
the functional-BP exposition: what a message *is* as a function on the reals, and
why the grid is a representation of it rather than the thing itself. That is the
part readers of the earlier draft found compressed.

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
