# thesis — the MSc thesis

**This folder is self-contained.** Zip it and upload it to Overleaf on its own;
nothing outside it is needed to compile. `figures/`, `sections/` and
`references.bib` are local copies of the shared assets, refreshed by
`./sync-assets.sh`.

```bash
tectonic main.tex        # builds main.pdf, runs bibtex automatically
```

Submission 16 September 2026, defence October 2026. 144 pages at present.

## Uploading to Overleaf

Upload the whole folder. Overleaf needs `main.tex` as the root document; it
picks that up automatically. Files it does not need but that are harmless to
include: `tools/`, `sync-assets.sh`, this README, and the build products
(`main.aux`, `main.bbl`, …), which Overleaf regenerates.

If the file-count limit is a problem, the folder compiles without `tools/`.

## Structure

Twelve chapters in three parts, following the order the research happened in.

**Part 0 — context.** Chapter 1 states the problem and the six research
questions. Chapter 2 is the background: statistical mechanics and the Boltzmann
distribution, energy-based models, generative diffusion and the reverse-time
SDE, the distinction between the exact, empirical and fitted score, the
speciation and collapse transitions, and the reading of U-Nets as belief
propagation. Chapter 3 fixes the model and the estimand.

**Part I — exact scores under known dynamics.** Chapter 4 is the rotating ring,
placed first because it is where the work started and because its
zero-Fisher-information result is the argument for studying the joint score at
all. Chapters 5 and 6 solve the Gaussian chain twice, by linear algebra and by
belief propagation, and prove the two agree. Chapter 7 leaves Gaussianity.

**Part II — learning dynamics that are not known.** Chapters 8 and 9 recover
the transition law from noised sequences by EM, first two parameters and then a
whole kernel. Chapter 10 measures what the structural assumption is worth
against trained networks and where it stops paying. Chapter 11 concludes.

Every proof and derivation is in the body: the Gaussian toolkit in Chapter 3,
the ring derivations in Chapter 4, the Gaussian-closure proof in Chapter 7. The
two appendices carry supporting material only --- reproducibility and
provenance (A), and the aggregation robustness of the reported ratios (B),
worked in full in the companion compendium, chapter 19.

`./check.sh` builds and reports every defect class. It checks the build's exit
status before reading `main.pdf`, because tectonic leaves the previous PDF in
place when a run fails and a check that skips that step will report a stale
document as clean.

## Figures

Six figures are built by this folder and live only here:

```bash
python3 tools/make_thesis_figures.py          # all of them
python3 tools/make_thesis_figures.py fig_em_diagnostics
```

`tools/figstyle.py` holds the palette and layout conventions — one place, so
every figure looks like it came from the same hand. Legends never sit inside
the data area, panels use constrained layout, and widths are fixed to the
text block so nothing is rescaled by LaTeX.

The other twelve figures are shared with the paper, the workshop note and the
compendium; `sync-assets.sh` keeps the local copies current and never
overwrites the six built here.

## What is not in this document

The AMP/TAP analysis of the bulk fixed point, the textbook linear-algebra
appendix, and the extended statistical-mechanics and stochastic-calculus
derivations are in the companion compendium. They answer none of the six
research questions, and at roughly fifty pages they lowered the fraction of the
thesis that is this work.

Two studies are withdrawn rather than reported: the reverse-generation
comparison, whose fits used a fixed iteration budget shorter than the
innovation shape needs to settle, and an earlier mixture-capacity design at too
few seeds. The capacity question was rerun to a convergence criterion at
sixteen seeds and **is** reported, in Section 10.3. The reverse-generation
question is left open, and Chapter 11 says so.

## editorial/

Notes about the document rather than the science: changelog, figure inventory,
structural decisions, correctness checklist.
