# BP Markov diffusion experiment outputs

Configuration used for this run:

- n = 50
- rho = 0.85
- grid interval = [-8.0, 8.0]
- validation grid sizes = (101, 201, 401)
- Laplace convergence coarse grid size = 401
- reference grid size = 801
- n_trials = 16
- seed = 11
- diffusion times = (0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4)

The Gaussian validation compares grid BP with the exact precision-matrix score.
The Laplace convergence compares grid BP at two resolutions.
The Laplace Gaussian-message experiment compares Gaussian projected BP with reference grid BP.
