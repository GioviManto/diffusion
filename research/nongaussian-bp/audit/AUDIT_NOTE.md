# Audit note: `bp_markov_diffusion_gaussian_approx_package`

Date: 2026-07-12.
Auditor environment: Python 3.12, numpy 2.4.2, matplotlib 3.10.8, macOS (Darwin 23.1.0).

## 1. Reproduction status

- `--mode quick` executes cleanly end to end.
- `--mode report --n-trials 12 --seed 11` (the configuration documented in the report,
  Section "Numerical results") **reproduces the packaged CSV outputs bit-for-bit up to
  machine precision**: the largest absolute deviation across every mean-valued column of
  `gaussian_grid_validation.csv`, `laplace_grid_convergence.csv`, and
  `laplace_gaussian_message_error.csv` is `2.1e-14` (attributable to numpy version /
  BLAS reduction-order differences). The pipeline is deterministic given the seed.
- The notebook is a results viewer over the CSVs/PNGs; it loads and renders correctly.

Verdict: the packaged results are exactly reproducible.

## 2. Mathematical consistency (report vs code)

Checked line by line:

- **Posterior chain factorization** (report Prop. 1) — correct, and the code's message
  recursions implement exactly Eqs. (left-message)/(right-message): with the convention
  `K[out, in] = K(a_out | a_in)`, the forward update `L[i+1] = K @ (L[i] * ell[i] * w)`
  and backward update `R[i-1] = K.T @ (R[i] * ell[i] * w)` are the correct trapezoidal
  quadrature discretizations. Beliefs `b ∝ L * ell * R` match Eq. (chain-belief).
- **Score identity** `s_i = -(x_i - α_t m_i)/Δ_t` — derivation in the report is a correct
  Tweedie/denoising identity for the OU kernel; the code's `score_from_mean` matches.
- **Error identity** `ŝ - s = (α_t/Δ_t)(m̂ - m)` — verified numerically in the CSVs:
  `score_error_identity_rel_residual_mean ≈ 1e-15` in all rows. Exact as claimed.
- **Gaussian closure algebra** (information-form updates) — correct.
- Message normalization after every step is legitimate (messages are defined up to
  positive constants; beliefs are renormalized at the end).

Verdict: the mathematics in the report and its implementation are consistent.

## 3. Findings (defects and weaknesses)

### F1 (major, numerical): boundary collapse of Gaussian-projected messages at large t

At weakly informative times (t ≳ 1.3 with ρ = 0.85), the backward pass of
`gaussian_projected_bp` can **collapse onto the grid boundary**. Diagnosed example
(Gaussian prior!, t = 1.8, seed 123, trial 2): `R_mean ≈ -7.68` (grid edge at -8) with
`R_var ≈ 0.08` for all early sites, producing a projected posterior mean of `-7.09`
where the exact value is `-0.17` (per-site error ≈ 6.9).

Mechanism: when ℓ is nearly flat as a function of a (large t), the outgoing message
function is a near-flat ramp whose maximizer lies outside [-A, A]. Moment-matching that
ramp *truncated to the grid* produces a mean near the boundary and a spuriously small
variance; the next `K^T` update pushes the mass further toward the edge (the integrand
in a is maximized at a = a'/ρ, i.e. outside the grid), creating positive feedback.

Consequences:
- The large-t rows (t ≥ 1.3) of `laplace_gaussian_message_error.csv` (e.g. posterior
  mean MSE 2.22 ± 7.47 at t = 1.3) mix genuine multimodality failures with this
  numerical artifact.
- The same artifact is visible in the *Gaussian* sanity check itself
  (`posterior_mean_mse_gauss_projected_mean` up to 5.0 at t = 1.8), where Gaussian
  projection should be essentially exact. This confirms it is an implementation
  artifact, not closure error.

Remedy (implemented in the new package): represent Gaussian messages analytically in
information form (precision λ ≥ 0, information h) and perform the ADF update in closed
form; a flat message is λ = 0 exactly, and no truncated moment-matching of near-flat
functions ever occurs.

### F2 (conceptual, important): single-Gaussian "message approximation" ≡ "model approximation" for linear transitions

For chains with **linear** transitions a_i = ρ a_{i-1} + ε_i (any zero-mean innovation
with variance q) and Gaussian OU likelihoods, moment-matched single-Gaussian BP is
mathematically identical to *exact* Gaussian BP on the Gaussian AR(1) model with the
same second-order structure. Reason: incoming Gaussian × Gaussian likelihood is exactly
Gaussian, and the transition step maps a Gaussian N(m̃, ṽ) to a density whose first two
moments are (ρ m̃, ρ² ṽ + q) *regardless of the innovation shape*; similarly for the
backward pass, where the message is a convolution (Gaussian ⊛ innovation) evaluated at
ρa, whose moment-match again depends only on (ρ, q). Hence the moment projection
discards all innovation information beyond variance.

Consequence: what the existing package measures as "Gaussian message approximation
error" on the Laplace chain **is** the model approximation error of the
covariance-matched Gaussian prior (up to grid-truncation numerics — which is exactly
where finding F1 lives). The distinction between message-level and model-level
approximation becomes real only for richer message families (e.g. Gaussian mixtures) or
nonlinear transitions. The new package makes this explicit, implements the analytic
information-form Gaussian BP as the honest "Gaussian baseline", verifies the
equivalence numerically, and reserves "message approximation error" for families where
it is genuinely distinct.

### F3 (minor, statistical): grid sizes are compared on different random trials

In `run_gaussian_grid_validation`, a single `rng` is consumed sequentially over
`(grid_size, t, trial)`, so different grid sizes see *different* data. The grid-size
comparison is therefore confounded with Monte Carlo noise (harmless here because grid
errors differ by orders of magnitude, but wrong methodology in general). The new
package uses common random numbers: data depends only on `(rho, t, trial)`, never on
grid parameters.

### F4 (minor, robustness): linear-domain likelihoods can underflow to all-zero rows

`likelihood_matrix` computes N(x_i; α u, Δ) in the linear domain. For small t and an
x_i whose preimage x_i/α falls outside the grid, an entire row can underflow to 0,
making `normalize_density` raise `FloatingPointError`. Never triggered at the packaged
settings (t ≥ 0.08, A = 8), but it is the binding constraint for pushing to smaller t.
Remedy: compute log-likelihood rows and subtract the per-row maximum before
exponentiating (a per-site constant rescaling of ℓ_i is absorbed by message
normalization and belief renormalization, so this is exact).

### F5 (minor): ad-hoc numerical floors

`normal_pdf` clamps var at 1e-14, `density_moments` floors variances at 1e-10. These
hide, rather than flag, resolution failures when the likelihood width √Δ/α approaches
the grid spacing (relevant at very small t). The new package instead *checks* the
resolution condition (grid step vs likelihood width) and reports it in experiment
metadata.

### F6 (observation): grid BP itself is accurate at the packaged settings

The Gaussian validation shows relative score errors of order 1e-7 to 1e-4 for
M ∈ {101, 201, 401} on [-8, 8] over t ∈ [0.08, 2.4] — consistent with trapezoidal
quadrature + tail truncation both being tiny at these settings. The Laplace 201-vs-401
self-convergence (~1e-4 → 1e-6) supports using fine-grid BP as reference in
non-Gaussian chains. Layer 2 of the new package quantifies exactly where this breaks
(small t, large ρ, small A).

## 4. Overall verdict

The current implementation is mathematically sound and exactly reproducible. Its two
substantive limitations are (F1) an unstable grid-mediated implementation of Gaussian
projected BP in the weak-likelihood regime, which contaminates the large-t results, and
(F2) the unrecognized equivalence of single-Gaussian projection with the matched
Gaussian model, which changes the interpretation (not the values) of the Laplace
experiments. Both are corrected in the new package; all downstream experiments use the
analytic information-form Gaussian BP as the Gaussian baseline and log-domain grid BP
as the numerical reference.
