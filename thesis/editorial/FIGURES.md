# Figure register

All figures: `thesis/figures/make_figures.py` (matplotlib, white background,
vector PDF, serif/cm mathtext, colorblind-safe navy/red/olive palette, no
top/right spines). Only fig_laplace_closure reads stored data (per-trial CSV
of nongaussian-bp experiment 02); everything else is recomputed from closed
forms or quadrature at build time. Deterministic (seed fixed where sampling
is used).

## Retained (v3)

| File | Chapter | Distinct claim it supports | Status |
|---|---|---|---|
| fig_band_fill.pdf | Ch3 | measured (Q_t)_{i,i+d} matches the no-factorial band-fill law with slopes d-1, no fitted parameters | kept |
| fig_tridiag_loss.pdf | Ch3 | off-tridiagonal mass of Q_t rises, peaks at intermediate t, decays as e^{-2t} | kept |
| fig_local_vs_full.pdf | Ch3 | truncation error decays exactly as q(alpha,t)^r; non-monotone in t | kept |
| fig_laplace_k1.pdf | Ch4 | Laplace noised marginal and score interpolate between prior shape and Gaussian attractor | NEW (merges old fig06+fig07; fig08 dropped, limits stated in text) |
| fig_laplace_closure.pdf | Ch4 | Gaussian-closure score error vs t with grid error budget on identical data | kept |
| fig_bulk_variance.pdf | App B | three closures in closed form; AMP branch terminates at t_c | kept |
| fig_bp_vs_amp.pdf | App B | mean error flat vs variance error growing; (alpha,t) existence phase diagram vs t_c curve | kept |

## Removed (v3)

| File | Reason |
|---|---|
| fig_spectral.pdf | content stated in one equation (eigenvectors frozen, eigenvalues flow affinely); decorative |
| fig_precision_lifecycle.pdf | heatmap triptych restating what fig_band_fill + fig_tridiag_loss quantify |
| fig06_K1_density.pdf, fig07_K1_score.pdf | merged into fig_laplace_k1.pdf |
| fig08_K1_score_limits.pdf | limits are exact statements in the text; plot added nothing |

## v4 (2026-07-22)
- `fig_bulk_variance.pdf` — RESTYLED: black/gray line styles, dotted t_c
  marker with horizontal annotation; shaded no-fixed-point region removed.
- `fig_bp_vs_amp.pdf` — RESTYLED: right panel is now a line phase diagram
  (closed-form boundary + open-circle iteration checks + text phase
  labels) instead of a two-colour imshow; title no longer references a
  colour.
- `fig_toymodel_score.pdf` — NEW (Ch. 6): two-frame toy model
  (trimodal mixture prior, additive dynamics), noised joint density and
  joint score field at t = 0.15; exact mixture formulas, quiver masked
  where density < 1.5% of max. Distinct claim: the joint score of a
  minimal dynamic object is visibly coupled across frames.
All 8 figures produced by thesis/figures/make_figures.py (matplotlib,
vector PDF, white background, colourblind-safe or monochrome).

## v5 (2026-07-26) — Chapter 9, rotating ring

All produced by `thesis/figures/make_ring_figures.py`, same style contract as
`make_figures.py` (matplotlib, white background, vector PDF, serif with cm
mathtext, navy/red/olive palette, no top/right spines). Closed-form panels are
recomputed at build time; measured panels read the curated CSVs of
`research/experiment1-rotating-ring/data/`. Deterministic (seeds fixed).

| File | Distinct claim it supports |
|---|---|
| fig_ring_model.pdf | the model: polar coordinates and moving frame, the confining force as -V'(r), one clean trajectory, and the same trajectory after the channel |
| fig_ring_joint_vs_marginal.pdf | as graphical models, the joint score conditions on all observations while the per-frame marginal severs every cross-frame path |
| fig_ring_gauge.pdf | a known rotation is an orthogonal change of frame; after de-rotation the circulation is gone and a ring-anchored random walk remains |
| fig_ring_surrogate_response.pdf | the exact surrogate response splits into a near-diagonal chain term and a rank-one anchor term |
| fig_ring_diagnostics.pdf | reach rises to the finite-chain ceiling while intensity falls; the weighted reach is non-monotone and turns where the measured receptive radius does |
| fig_ring_score_field.pdf | the joint conditional field is displaced towards the inferred phase; the one-frame marginal field stays ring-symmetric |
| fig_ring_taylor.pdf | the linearisation is accurate while the posterior angular spread stays below ~0.32 rad, and degrades after it |

Eighteen further figures exist in the standalone note
(`research/experiment1-rotating-ring/figures/`) and are deliberately not
carried into the thesis: the register rule is one figure per distinct result,
and the note's extra panels are either duplicates of the seven above or
intermediate diagnostics whose content is stated in Tables 9.2 and 9.3.
