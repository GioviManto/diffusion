# overleaf — the four documents

Everything that gets written up, in one place. Each document has its own folder
and compiles from inside it, so `../shared/...` resolves the same way here as it
does in Overleaf when you set that file as the main document. Upload this folder
as a project and pick the main file per compile; nothing needs rearranging.

| Folder | What it is | Body | Ready to send? |
|---|---|---|---|
| `workshop/` | the analytical result: exact scores + LMMSE closure | 3pp of 4 | Yes |
| `paper/` | the full paper | 9pp of 9 | Yes |
| `compendium/` | every derivation and experiment, including the cut ones | 54pp | Internal only |
| `thesis/` | the thesis | 169pp | Yes, as a draft |

```bash
./build.sh
```

Compiles all four, reports total and main-content pages, and fails on undefined
references, over-limit bodies, or unfilled `\needsdata` markers.

## shared/ is shared on purpose

`shared/` holds what more than one document uses: `notation.tex`, the merged
`references.bib`, the propositions and efficiency table in `sections/`, and every
figure. **Editing a file there changes every document that inputs it.** That is
the point — the four disagreeing with each other is a failure this project has
already had, and single-sourcing is the fix.

To make two documents differ, move the text out of `shared/` into the roots.
Do not make a second copy.

## Two files are generated — do not hand-edit them

`shared/sections/tab-efficiency.tex` and `shared/sections/efficiency-numbers.tex`
come from `research/nongaussian-bp/tools/make_tab_efficiency.py`. The prose cites
macros (`\ratiolo`, `\nfreeparams`, …) rather than typed numbers, so the abstract
cannot hold a figure the table has moved past. It did exactly that for three
weeks: the abstract said "between 8 and 14" against a table reading 7.3–15.7.

Figures come from `research/nongaussian-bp/tools/make_figures.py`.

## What is checked before anything is sent

`research/nongaussian-bp/tools/check_paper.sh` gates the paper and the workshop:
page limits read from `\label{LastMainPage}` in the `.aux`, no unfilled
`\needsdata`, no code references, no hand-typed ratio ranges, and no orphaned
shared section. The compendium and thesis are deliberately **not** gated — they
are allowed to carry provisional results, and both say which results those are.
