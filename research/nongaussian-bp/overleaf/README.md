# Overleaf project

Self-contained: `main.tex` plus `appendix.tex`, `notation.tex`, `bibliography.bib`,
`neurips_2023.sty` and every figure it references. It compiles from this directory alone, with no
path into the rest of the repository, so it can be dropped into Overleaf as is.

Verified: compiles clean from a copy outside the repo, **zero undefined references or citations**,
22 pages.

## Getting it into Overleaf

**Upload the zip** (simplest, works on the free plan). In the repo root:

```bash
cd research/nongaussian-bp && zip -r overleaf-bp-diffusion.zip overleaf -x '*.DS_Store'
```

Then Overleaf → New Project → Upload Project → pick the zip. Overleaf makes `main.tex` the root
automatically; if it does not, right-click `main.tex` → *Set as main file*.

**Or sync from GitHub** (keeps it updated, needs a premium plan): Overleaf → New Project → Import
from GitHub → `GioviManto/diffusion`, branch `paper-revision-2026-08-07`. Set the root document to
`research/nongaussian-bp/overleaf/main.tex`.

## Settings

- Compiler: **pdfLaTeX** (Overleaf's default). `natbib` is loaded explicitly with the
  `nonatbib` option to the NeurIPS style, which is the combination that works.
- Bibliography is BibTeX; Overleaf runs it automatically. If citations show as `[?]` on the first
  compile, hit Recompile once more.

## Sharing with Marc and Jérôme

Share → invite by email, **Can edit** so they can comment inline. Overleaf's *Review* mode gives
them tracked changes and comments, which is the closest thing to what they asked for.

## Keeping it in sync with the repo

This directory is a copy, not a symlink — the source of truth is `paper/`. After editing `paper/`,
refresh with:

```bash
cd research/nongaussian-bp && cp paper/main.tex paper/appendix.tex paper/notation.tex paper/bibliography.bib paper/neurips_2023.sty overleaf/ && cp paper/figures/*.pdf overleaf/figures/
```

If instead you edit in Overleaf, copy the changed files back into `paper/` so the committed
version stays authoritative. Do not let the two diverge.
