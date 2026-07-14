# When does the Gaussian score baseline work?

Median relative score error of the analytic Gaussian baseline (== covariance-matched Gaussian model) vs fine-grid BP.

| family | kurt. | rho | small t (min) | mid t (0.2) | large t (max) |
|---|---|---|---|---|---|
| gauss_mix_kappa0.3 | -0.18 | 0.5 | 0.068 | 0.026 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.5 | 0.405 | 0.118 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.5 | 0.939 | 0.346 | 0.000 |
| laplace | +3.00 | 0.5 | 0.520 | 0.233 | 0.000 |
| student_t_nu5 | +6.00 | 0.5 | 0.317 | 0.159 | 0.000 |
| student_t_nu8 | +1.50 | 0.5 | 0.196 | 0.082 | 0.000 |
| gauss_mix_kappa0.3 | -0.18 | 0.85 | 0.051 | 0.006 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.85 | 0.255 | 0.036 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.85 | 0.714 | 0.100 | 0.000 |
| laplace | +3.00 | 0.85 | 0.354 | 0.080 | 0.000 |
| student_t_nu5 | +6.00 | 0.85 | 0.241 | 0.059 | 0.000 |
| student_t_nu8 | +1.50 | 0.85 | 0.138 | 0.030 | 0.000 |
| gauss_mix_kappa0.3 | -0.18 | 0.95 | 0.024 | 0.002 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.95 | 0.142 | 0.010 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.95 | 0.368 | 0.024 | 0.000 |
| laplace | +3.00 | 0.95 | 0.270 | 0.038 | 0.000 |
| student_t_nu5 | +6.00 | 0.95 | 0.135 | 0.035 | 0.000 |
| student_t_nu8 | +1.50 | 0.95 | 0.115 | 0.012 | 0.000 |