# BP Markov diffusion experiment outputs

Configuration used for this run:

- n = 50
- rho = 0.85
- grid interval = [-12.0, 12.0]
- validation grid sizes = (151, 301, 601)
- Laplace convergence coarse grid size = 301
- reference grid size = 601
- n_trials = 50
- seed = 11
- diffusion times = (0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4)
- boundary mass tolerance = 1e-06 over 3 edge cells

Grid diagnostics: worst boundary mass 5.262e-10 at belief b[5] (tol 1.0e-06)

The Gaussian validation compares grid BP with the exact precision-matrix score, and
checks that Gaussian closure reproduces that score to machine precision. The retained
grid-projection ablation shows what the earlier discretised projection was costing.

The Laplace convergence compares grid BP at two resolutions.

The Laplace Gaussian-message experiment compares Gaussian closure with reference grid BP.
Because closure coincides exactly with the linear (LMMSE) denoiser, the reported error is
the excess error of the best linear estimator over the exact one. The column
`closure_vs_lmmse_rel_gap_mean` verifies that identity numerically.
