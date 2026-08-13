# Experiment 1 — Joint score dynamics on a rotating ring

Revised research note for the **first** of the three board problems formulated with
Marc Mézard. Self-contained and built to drop into the thesis.

- `main.pdf` / `main.tex` — the note (47 pp).
- `figures/` — 18 figures, PDF (for LaTeX) + PNG (for inspection).
- `tables/` — LaTeX tables, all generated from `data/` and `validation.json`.
- `data/` — canonical numerical results (16 CSVs).
- `code/` — model, diagnostics, figure and table generation, validation.
- `PREVIOUS_VERSION.pdf` / `PREVIOUS_VERSION_main.tex` — the version this revises,
  kept for diffing.

## Rebuild

```bash
python code/make_tables.py     # tables from data/ + validation.json
python code/make_figures.py    # all 18 figures (accepts a name filter, e.g. "fig04")
python code/validate.py        # numerical checks -> validation.json
tectonic -X compile main.tex   # or: latexmk -pdf main.tex
```

`code/compute_surrogate_full_windows.py` regenerates the surrogate window sweep.
The heavier pipelines that produced `data/` are preserved as `code/run_*_original.py`.

## What changed in this revision

**Structure.** Three background/setup sections, then Part I (exactly solvable surrogate)
and Part II (the board problem, polar process). The research question is unchanged:
*how can the Markov structure of the clean rotating trajectory be exploited to compute
and understand the dynamics of the joint diffusion score?*

**New background (§1).** Why diffusion models matter and where industry uses them;
Brownian motion from Einstein to Wiener; total variation vs quadratic variation and why
Riemann–Stieltjes fails; the Itô integral, its martingale property and the Itô isometry;
Itô's lemma and the product rule, i.e. the difference between an ODE and an SDE; Langevin
dynamics, Fokker–Planck, and the Boltzmann stationary law; the OU process and why
diffusion models use it; the non-equilibrium-thermodynamics reading of Sohl-Dickstein et
al. (arXiv:1503.03585) together with the H-theorem as the second law for Fokker–Planck;
static fluctuation–dissipation; first- vs second-order phase transitions and the cascade
picture. Five numbered **toolboxes** carry the derivations.

**Derivations now shown in full.**

| Asked for | Where |
|---|---|
| Complete VP-OU channel solution, in a toolbox | Toolbox 2, §2.4 |
| The field/covariance identity, step by step | Toolbox 3, §2.7 (5 steps + Hessian symmetry check) |
| Itô chain rule and Itô isometry stated as used | Toolboxes 1–2, §1.2, referenced at each use |
| Rotation matrix shown and explained | §4.2 + fig05 |
| Rotations preserve N(0,I) | §4.3, proved twice (covariance and density) |
| U_ψ orthogonal; density and score transformation | §4.4, both halves proved |
| Meaning of "gauge transformation" | §4.4, dedicated paragraph |
| Unrolling Y_k = A + σ Σ η_r | §4.5, first steps + telescoping sum |
| Why the interior clean score is a second difference | §5.1, two-incident-edges counting + fig07 |
| Y \| A=a ~ N(1⊗a, K⊗I₂) notation | §5.2, definition box with a written-out T=3 example |
| Force from a potential = −V′(r) | §7.3 + fig01(b) |
| Integrating factor e^{κu}, mean zero, Itô-isometry variance | Toolbox 6, §7.4 (6 steps) |
| Angular integration and wrapping | Toolbox 7, §7.5 |
| Polar coordinates introduced properly, with a visualisation | §7.1 + fig01 |
| Taylor expansion order by order, and its validity | §9.1, §9.3 + fig13 |

**Metrics (§3).** Rewritten. Intensity `I_off`, weighted mean lag `ℓ̄`, and the functional
receptive field `L_ε` are separated, and a new combined diagnostic is introduced:

```
Ξ  = I_off · ℓ̄  = Σ_{ℓ≥1} ℓ C_t(ℓ)        absolute weighted reach
Ξ̃ = Ξ / C_t(0)                            relative weighted reach  ← headline
```

`Ξ̃` responds to reach *and* intensity and factorises exactly into "how much" × "how far".
Measured, it is **non-monotone in diffusion time** (polar model: 1.55 → 2.26 at t=0.7 →
0.43 at t=2) and turns in the same region as the independently measured receptive radius
`L_5%`, which neither the normalised range (monotone up) nor the intensity (monotone down)
can indicate. Every diagnostic is reported against its structureless finite-chain ceiling.
Definitions live in one place, `code/metrics.py`.

**Removed / compressed.** Bessel machinery moved out of the main line into one footnote
plus Appendix A; the simulation sections condensed (17 figures → 18 but with 5 new ones and
4 merges: model-overview, workflow and model-comparison figures dropped or folded in);
exponential-fit lengths reported only where they are meaningful.

**Claims made more modest.** No "proves", no "robust"; the single parameter set, the
28–36 trajectories, the discretised latent space and the oracle radial/tangential
projections are stated in the protocol paragraphs and in §13. The finite-size caveat is
built into the diagnostics rather than added as a remark.

## Verification

`code/validate.py` → `validation.json`, status PASS:

| check | discrepancy | tolerance |
|---|---|---|
| posterior covariance vs finite differences (polar) | 6.6e-09 | 1e-07 |
| analytic surrogate Jacobian vs finite differences | 9.2e-06 | 1e-04 |
| 41×128 vs 51×192 response grid | 3.0e-11 | 1e-07 |
| rotation gauge round trip | 2.2e-16 | 1e-12 |
| clean score on an exact co-rotating ring path | 0 (exact) | 1e-12 |

Every number quoted in the prose was checked against the CSVs; no number in the text or
tables is typed by hand.

## Relation to the thesis

This note is the **standalone, reproducible deliverable** behind
Chapter 9 of `thesis/` ("The rotating ring: a two-dimensional dynamic object").
It is deliberately **not cited** by the thesis: it is unpublished, and the
chapter presents the work as the thesis's own. The thesis refers to this folder
only through the reproducibility appendix, as the package that regenerates the
chapter's numbers, figures, and validation.

What the thesis takes, and what it leaves here:

| | thesis Chapter 9 | this note |
|---|---|---|
| notation | thesis notation (`K` frames, `e^{-t}`, `\Delta_t`, `S`) | note notation (`T` frames, `m_t`) |
| figures | 7, in the thesis paper style, one per distinct result | 18, including intermediate diagnostics |
| background | in the chapters that own each topic (Ch. 1--4) | Section 1, self-contained |
| derivations | same, in the thesis `toolbox` environment | same, in tcolorbox toolboxes |
| tables | inline, verified against these CSVs | generated by `code/make_tables.py` |

The background of Section 1 here is deliberately redundant with thesis
Chapters 2 to 4, so that the note reads on its own. The thesis instead received
only the material it was missing: the divergence of the total variation (Ch. 3),
the static linear-response identity and the first- against second-order
transition distinction (Ch. 2), the H-theorem and the
non-equilibrium-thermodynamics reading (Ch. 4), and two deployment examples
(Ch. 1).

Thesis-side figures are produced by `thesis/figures/make_ring_figures.py`, which
imports the model code and reads the CSVs of this package.
