# shared — what more than one document uses

Single-sourced on purpose. **Editing anything here changes every document that
inputs it.** The four documents disagreeing with each other is a failure this
project has already had — a stale copy of the ring model outlived the corrected
one for weeks because it sat in a folder nothing compiled.

| File / folder | Used by |
|---|---|
| `notation.tex` | all four |
| `references.bib` | all four |
| `figures/` | all four |
| `sections/model.tex` | paper, workshop |
| `sections/prop-lmmse.tex` | paper, workshop |
| `sections/prop-fisher.tex` | paper |
| `sections/tab-efficiency.tex` | paper |
| `sections/efficiency-numbers.tex` | paper |

`check_paper.sh` fails if a file in `sections/` is reached by neither production
document, because an unreachable copy is one that drifts and then gets pasted
back in.

## Generated files — do not hand-edit

`sections/tab-efficiency.tex` and `sections/efficiency-numbers.tex` are written
by `research/nongaussian-bp/tools/make_tab_efficiency.py` from the frozen
outputs. The second defines the macros (`\ratiolo`, `\ratiohisub`,
`\nfreeparams`, …) that the prose cites instead of typing figures, so the
abstract cannot hold a number the table has moved past.

Figures are written by `research/nongaussian-bp/tools/make_figures.py`. Both
`.pdf` (used by LaTeX) and `.png` (for quick viewing) are produced; only the PDFs
are included.

## references.bib

132 entries, merged once from what were the paper's and the thesis's separate
files. They shared 14 keys for the same works, formatted differently; the paper's
entries won because they carry `eprint`/`archivePrefix` and DOI fields. Add new
references here, never to a per-document copy.
