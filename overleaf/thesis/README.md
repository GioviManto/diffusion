# thesis — the MSc thesis

Compile `main.tex` from inside this folder. Chapters and appendices are in
`chapters/`; `preamble.tex` holds the packages and the `\graphicspath` pointing
at `../shared/figures/`.

Submission 16 September 2026, defence October 2026. 169 pages at present.

## Structure

Two parts and a case study:

- **Exact scores under known dynamics** — the Gaussian chain solved exactly,
  BP/Kalman/RTS equivalence, locality and truncation, the non-Gaussian chain,
  Gaussian message closure and what it discards.
- **Learning dynamics from noised data** — identifiability, the pairwise
  statistic and Fisher's identity, EM and generalised EM, and what supplying the
  Markov structure buys against trained denoisers.
- **The rotating ring** (ch09) is a self-contained case study, not a research
  question: what joint observations identify that no single-frame marginal can.

Six research questions, stated in ch01 and answered one by one in ch12.

## Two lines of work are reported as unresolved

The mixture-capacity sweep and the reverse-generation study both rest on fits
that the convergence analysis showed had not settled. They are recorded as
methodological negative results, not findings, and neither supports a conclusion
in ch12. That is deliberate and it should stay that way unless they are rerun
with certified estimators.

## editorial/

Notes about the document rather than the science: changelog, figure inventory,
structural decisions, correctness checklist. Read its README first — those four
files were restored from git history after an untracked-file deletion and may be
missing edits made between 21 July and 18 August 2026.
