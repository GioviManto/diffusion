# Round-four review — prescriptive pass before submission

You are reviewing an MSc thesis in Data Science (Bocconi, DSBA) that is **eleven
working days from deposit** (16 September 2026; defence in October). Supervisors:
Marc Mézard and Jérôme Garnier-Brun.

Three previous committee rounds have happened. Round three returned "not ready to
upload" for the thesis and "not ready for submission" for the paper. **Every
blocking item from those rounds has been closed.** This round is not a re-audit.
It is the last pass, and it has one job: turn whatever is still wrong into text I
can paste.

---

## 1. How to answer — this is the part that matters

You cannot run the code, reach the cluster, or edit the repository. So do not
write commentary that someone then has to interpret. **Every finding must arrive
as something mechanically applicable.**

For each item, give exactly this:

```
### [BLOCKING | OPTIONAL] <one-line title>
FILE:    overleaf/thesis/chapters/ch11-nongaussian-em-results.tex
ANCHOR:  "the first six words of the sentence you are replacing"
REASON:  one or two sentences. Why it is wrong, not why it could be better.
REPLACE WITH:
<the literal LaTeX. Complete. Compiles as-is. Uses the macros that already
 exist rather than typing numbers.>
PAGES:   +0.2 (your estimate of the page-count delta)
```

Rules that make this usable:

- **Never write a number literally** if a macro exists for it. The numeric
  macros are listed in §5. If you need a number that has no macro, say
  `NEEDS MACRO: <what it should contain>` and I will add it to the generator.
  A hand-typed number is the single defect this project has spent three rounds
  removing; do not reintroduce one.
- **Do not propose anything requiring a new experiment.** There is no time. If
  you believe a claim needs data we do not have, the correct output is
  replacement text that narrows the claim to what the existing data supports.
- **Quote the anchor exactly** as it appears in the source I give you, so I can
  find it with a literal string search.
- If you have no findings in a section, write `no change` and move on. Padding
  a review with restatements of what is already right costs me time I do not
  have.
- Rank everything **BLOCKING** (would embarrass at defence, or is factually
  wrong) versus **OPTIONAL** (would improve it). Give a **total page delta** at
  the end. The thesis is 172 pages with no hard limit; the companion paper is at
  a hard 9-page body limit and cannot grow.

---

## 2. What you are being given

- `thesis.pdf` — 172 pp, ten chapters, six appendices.
- `paper.pdf` — 28 pp, 9-page body, NeurIPS format. **Hard 9-page limit.**
- `workshop.pdf` — 6 pp, 3-page body.
- `compendium.pdf` — 106 pp. Unbounded companion holding derivations, the full
  development record, and the claim audit. **It is not examined.** Anything you
  want removed from the thesis for length or tone can be sent here rather than
  deleted — say so explicitly when you propose a cut.
- The research package: experiment code, frozen outputs, the generators that
  write every number, and the provenance gate.

---

## 3. State of play — do not re-litigate these

Round three's blocking items are closed. In particular:

- **Provenance.** Every run cited is now certified. The last uncertified result
  (§9.5, non-Markov robustness) was rerun from clean git-archive deployments —
  15 cells, empty dirty lists. The Gaussian arm reproduced the old numbers
  *exactly*, which is the evidence that the previously-uncommitted `src/em.py`
  never mattered. Appendix C sets out the certification routes.
- **The "2000 iterations" error** (wrong by ~10×, in three documents) is
  replaced by measured settling times from a 16-seed sweep.
- **Every number is generated.** No results table or quoted figure is typed.
  Thirteen generators read frozen CSVs and write LaTeX macros; the build fails
  on a dirty source tree.
- **Layout.** Zero overfull boxes in all four documents.
- **Bibliography.** One shared file; 21 double-entered works removed.
- **Notation.** `M` no longer means three things; figure axes no longer bake
  symbols that differ between documents; all 17 figures share one text font and
  one math font.

If you think one of these is still broken, say so — but check the PDF first, and
quote the page.

---

## 4. What I actually want from this round

### 4.1 Figures for the chapters that have none — the main request

Four chapters carry **zero figures**:

| Chapter | Content | Length |
|---|---|---|
| 1 Introduction | motivation, RQs, contributions | 12 pp |
| 2 From Stationary Action to Generative Diffusion | the physics lineage: Ising → RBM → statistical mechanics of learning → OU → time reversal → the diffusion idea → speciation → cascade of transitions | 18 pp, 14 sections |
| 3 Graphical Models and Message Passing | MRFs, factor graphs, sum–product, locality in learned architectures | 6 pp |
| 10 Discussion and Conclusions | findings, limitations, future work | 8 pp |

Chapter 2 is the worst case: eighteen pages of dense physics with not one
picture, and it is the chapter that carries the whole motivating narrative.

**I intend to reproduce figures from the source papers, with citation, inside
the thesis only** — the convention Beatrice Achilli's thesis uses, and standard
for a deposited thesis. The companion paper will not carry them.

Tell me, concretely:

1. **Which specific published figures to reproduce**, by paper and figure
   number, for each of §2.3 (statistical mechanics of learning), §2.10 (the
   diffusion idea), §2.12 (statistical physics of generative diffusion —
   speciation, memorisation), §2.13 (learning as a cascade of second-order
   transitions). Candidate sources, all already cited:
   - `biroli2023generative` — Biroli & Mézard, *J. Stat. Mech.* 2023
   - `biroli2024dynamical` — Biroli, Bonnaire, de Bortoli, Mézard,
     *Nature Communications* 2024 — dynamical regimes, speciation
   - `achilli2026speciation` — Achilli, Benedetti, Biroli, Mézard, 2026
   - `achilli2024losing` — Achilli et al., geometric memorisation, 2024
   - `bonnaire2025memorize` — Bonnaire, Urfin, Biroli, Mézard, NeurIPS 2025
   - `bachtis2024cascade` — the RBM cascade, NeurIPS 2024
   - `sclocchi2025phase` — Sclocchi, Favero, Wyart, *PNAS* 2025
   - `kadkhodaie2024generalization` — ICLR 2024, geometry-adaptive
   Pick the **fewest figures that carry the argument** — I would rather have
   four excellent ones than twelve decorative ones. For each, say what it shows
   and why that section needs it.

2. **Which of those to recreate rather than reproduce.** Some of these are
   simple enough to recompute from the published closed forms and plot in the
   thesis's own style, which looks better and enters the reproducibility map.
   Tell me which are in that category and give the equation to plot.

3. **The exact caption** for each, in the attribution form appropriate for a
   deposited thesis, plus the `\label` and the section it goes in.

4. **Whether any figure is needed in Ch 1, 3 or 10 at all**, or whether those
   chapters are fine as text. Do not invent a need.

### 4.2 The one thing I could not settle myself

§9.5 now reports two innovation laws. On the rank-one mechanism they agree in
shape but not at the margin: at β=1 the Gaussian arm leads at every noise level
(worst ratio 1.06), while the Laplace arm leads against the CNN (worst 1.32) but
lands *on* the break-even line against the MLP at the largest diffusion time
(0.998). I have written this as "the mechanism carries across the innovation
law; the size of the margin it leaves does not quite."

Is that the right reading, and is it stated at the right strength? Give
replacement text if not.

### 4.3 Register and concision

The thesis benchmarks against Achilli's, which is clear and unpadded. Find
sentences that are stacked clauses rather than necessary precision, and give the
split version. I have already cut the worst; find what I missed. Do **not**
propose cutting scientific content to shorten — send it to the compendium
instead, and say so.

### 4.4 Defence exposure

List the five questions a Bocconi committee is most likely to ask that the
current text does not already answer, with the answer I should be ready to give.
Only questions the document leaves open — not ones it already addresses.

---

## 5. Macros you must use instead of typing numbers

Defined in `overleaf/shared/sections/*-numbers.tex`, written by the generators.
Selection:

```
efficiency      \ratiolo \ratiohi \nsizesused \biggestn
structured      \structratiolo \structratiohi \structseeds \structcells
                \structcapem \structcapwindow
screening       \screenwinnerwidth \screenwindowparams \screenconfigs
capacity        \capcomps \capheadsize \capothersize \capseeds
                \capiterlimit \capcappedcells \capcappedtotal \capsminfloor
convergence     \convshapemed \convshapemax \convrhomed \convivarmed
                \convratio \convcap \convseeds \convbudget \convunsettled
non-Markov      \nmbetamax \nmgammamax \nmcrossover \nmlastholding
                \nmbetaminratio \nmgammaminratio \nmemworsterr
                \nmrhoclean \nmrhobetamax \nmcontrolbeta \nmcontrolgamma
                \nmcldevbeta \nmcldevgamma \nmcldevfactor
                \nmlapcontrol \nmlapbetaminratio \nmlapbetaminratiocnn
grid            \gdinterior \gdedge \gdchains \gdlevels
notation        \nseq (training sequences) \ngrid (grid points)
                \corr (AR coefficient) \ivar (innovation variance)
```

Note `\nseq` renders `N` in the thesis and `M` in the paper; `\corr` renders
`\alpha` in the thesis and `\rho` in the paper. Write the macro, never the
letter.

---

## 6. Output order

1. BLOCKING items, most severe first, in the format of §1.
2. The figure plan (§4.1) — this is the priority of this round.
3. OPTIONAL items.
4. Five defence questions with answers.
5. Total page delta, and a one-line verdict: ready to deposit, or not, and if
   not, the shortest path to ready.
