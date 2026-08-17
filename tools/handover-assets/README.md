# Overleaf handover — four documents, one folder

Everything needed to compile all four documents is in this folder. Nothing
resolves a path outside it, so uploading the folder to Overleaf as a single
project works with no rearranging. Set the main document per compile.

| Root | What it is | Ready to send? |
|---|---|---|
| `workshop.tex` | 4pp workshop version | **No** — see placeholders |
| `paper.tex` | the full paper, 9pp body | **No** — see placeholders |
| `compendium.tex` | every derivation and experiment, including cut ones | Yes |
| `thesis.tex` | the thesis | Yes, as a draft |

All four build with `tectonic`, with zero undefined references and zero
undefined citations. `./build.sh` compiles all four and reports page counts.

## Two documents are NOT ready to send

`paper.tex` and `workshop.tex` each render a loud red `[PENDING: ...]` marker
where a number is not yet earned. They are listed here rather than quietly
shipped, and the marker is red on purpose — it is meant to be impossible to
send by accident.

1. **`sections/tab-efficiency.tex`** — the efficiency table. The numbers in it
   came from runs at an EM budget of 120 iterations; the frozen config says
   400, and the budget turned out to change the answer's direction (worse at
   small n, better at large n). The 16-seed validation-selected rerun has since
   landed, but it is not filled in here, because the protocol it used is still
   asymmetric: EM's stopping point is chosen on a validation bundle while the
   network's training length is fixed at 20,000 steps. That asymmetry favours
   EM, and this table is the paper's headline claim, so it should not go in
   until both arms are selected the same way.

2. **`appendix.tex`** (numerical-values table) — needs regenerating from the
   frozen batch.

The compendium and thesis are unaffected: both are allowed to carry provisional
results, and both say which results are provisional.

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
