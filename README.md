# Diffusion — exact scores and transition learning for Markov data

MSc thesis project. Submission 16 September 2026, defence October 2026.

**The result.** A locally specified non-Gaussian Markov law induces a *globally*
dependent diffusion score, yet that score is computable exactly by local
inference on the latent chain — and the same inference yields the pairwise
posterior statistics needed to learn the transition when it is unknown.

Coordinatewise noising multiplies the prior by *unary* factors. Those reweight
node potentials without creating edges, so the posterior factor graph is still
the prior's chain; a chain is a tree; sum-product on a tree is exact. That is the
whole mechanism, and it is why the exact score is available here when it is
unavailable almost everywhere else.

## Start here

```bash
cd overleaf && ./build.sh                 # build all four documents
cd overleaf && ./make_upload_bundle.sh    # Overleaf-ready zips in overleaf/upload/
```

`build.sh` compiles in place and checks each document for undefined references,
undefined citations and the paper's nine-page body limit.

```bash
cd research/nongaussian-bp && .venv/bin/python -m pytest -q   # 445 passed, 27 skipped (~14 min)
cd research/nongaussian-bp && ./tools/check_all.sh --quick    # structural checks (~1 min)
```

`make_upload_bundle.sh` produces one self-contained folder and zip per document
under `overleaf/upload/`. Upload `upload/thesis.zip` to Overleaf via **New
Project -> Upload Project**; `main.tex` is the main document. The flattening is
not cosmetic: the working tree keeps one copy of the figures, notation and
generated numbers in `shared/` and reaches them with `../shared/`, and Overleaf's
bibtex will not read through `..` -- it finds no `.bib`, silently drops every
citation, and still compiles. Each bundle is verified by being compiled inside
its own directory before the zip is written.

| Folder | What it holds |
|---|---|
| [`overleaf/`](overleaf) | the four documents — paper, workshop, compendium, thesis |
| [`research/`](research) | the code and the experimental record |
| [`questions/`](questions) | open questions for advisors, and planning documents |
| [`tools/`](tools) | repository-level scripts |
| `sources/` | reference PDFs — local only, not tracked |
| `meetings/` | supervision notes and call transcripts — local only, not tracked |
| `archive/` | superseded work — local only, not tracked |

The last three are gitignored: they hold third-party PDFs and private notes
that do not belong in a public history. Everything needed to rebuild every
document and reproduce every number is tracked.

## Where things are single-sourced

Two rules carry most of the weight, both because breaking them has cost real
time on this project:

**Documents share, they do not copy.** `overleaf/shared/` holds the notation,
the bibliography, the figures and the propositions that more than one document
uses. Four documents drifting apart is the failure this structure prevents.

**Numbers are generated, not typed.** The efficiency table and the macros the
prose quotes come from `research/nongaussian-bp/tools/make_tab_efficiency.py`,
run against frozen outputs. The abstract once carried "between 8 and 14" for
three weeks after the table read 7.3–15.7; the fix was to make it impossible
rather than to check for it.

Similarly, every experiment imports `experiments/frozen_config.py` rather than
setting ρ, the grid, or an iteration budget locally.

## Before sending anything

```bash
cd research/nongaussian-bp && tools/check_paper.sh
```

Page limits, placeholders, code references, hand-typed ratios, orphaned shared
sections, replicate counts, and stray configuration. It gates the paper and the
workshop only — the compendium and thesis are allowed to carry provisional
results, and both label them.
