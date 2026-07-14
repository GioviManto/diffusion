# Grid parameter recommendation (from exp_01 heatmap, rho = 0.95)

M =  101, A = 4: worst rel. error = 1.018e-05
M =  101, A = 5: worst rel. error = 9.551e-07
M =  101, A = 6: worst rel. error = 1.368e-04
M =  101, A = 7: worst rel. error = 2.564e-03
M =  101, A = 8: worst rel. error = 1.711e-02
M =  201, A = 4: worst rel. error = 9.958e-06
M =  201, A = 5: worst rel. error = 9.604e-10
M =  201, A = 6: worst rel. error = 9.278e-15
M =  201, A = 7: worst rel. error = 3.693e-14
M =  201, A = 8: worst rel. error = 9.294e-11
M =  401, A = 4: worst rel. error = 9.903e-06
M =  401, A = 5: worst rel. error = 9.429e-10
M =  401, A = 6: worst rel. error = 8.909e-15
M =  401, A = 7: worst rel. error = 9.326e-15
M =  401, A = 8: worst rel. error = 9.862e-15
M =  801, A = 4: worst rel. error = 9.889e-06
M =  801, A = 5: worst rel. error = 9.385e-10
M =  801, A = 6: worst rel. error = 9.204e-15
M =  801, A = 7: worst rel. error = 9.435e-15
M =  801, A = 8: worst rel. error = 9.233e-15

Cheapest configuration with worst-case error < 1e-4 over tested t: M = 101, A = 4.

Caveats for the working default:
- This calibration is Gaussian; heavy-tailed innovations (Laplace,
  Student-t) put more posterior mass in the tails, so the half-width
  should NOT be reduced to the Gaussian-optimal value. We adopt
  M = 401, A = 8 as the reference configuration for all non-Gaussian
  experiments (error floor ~1e-15 in the Gaussian calibration, cost
  O(n M^2) still negligible), and M = 801, A = 8 for convergence checks.
- The binding constraint at very small t is the likelihood width
  sqrt(Delta)/alpha ~ sqrt(2t) versus the grid step 2A/(M-1); the
  `resolution_ok` column flags settings with < 3 grid points per
  likelihood standard deviation.