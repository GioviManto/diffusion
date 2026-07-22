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
