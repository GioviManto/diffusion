# Result Ledger

Every substantive claim of the thesis, classified. Categories:
**EXACT** (proved in closed form, audited numerically) · **NUMERICAL**
(measured against a validated exact reference; no closed form) ·
**HEURISTIC** (mechanistically argued, not proved) · **SPECULATIVE**
(conjecture, explicitly not asserted) · **FUTURE** (stated as open).

Revision v2 (2026-07-19): chapters renumbered (Gaussian 7→5, Laplace 8→6,
BP 9→7); the AMP/TAP results (B5, B7–B9) moved intact to Appendix C of
the thesis at the author's request — their classification is unchanged.

All audits re-executed on this machine on 2026-07-14:
`research/gaussian-bp/code/numerical_audit.py` → **72/72 PASS**;
`research/unified-note/code/numerical_audit.py` → **55/55 PASS**.

## Gaussian chain (thesis Ch. 7)

| # | Claim | Class | Evidence |
|---|-------|-------|----------|
| G1 | Stationary covariance `(Σ0)_ij = α^{|i−j|}` (unit variance norm.) | EXACT | derivation + audit |
| G2 | Clean precision Q0 exactly tridiagonal, entries explicit | EXACT | Hessian derivation + audit |
| G3 | `Σt = e^{−2t}Σ0 + Δt·I`; joint score `S = −Qt·x` | EXACT | audit err ~1e−16 |
| G4 | Tweedie: `S_k = (e^{−t}E[a_k|x] − x_k)/Δt`, any prior | EXACT | proof + two-route agreement 2.7e−15 |
| G5 | Posterior precision `J = (e^{−2t}/Δt)I + Q0` tridiagonal ∀t | EXACT | completing the square |
| G6 | K=2 closed forms, coupling `r = α e^{−2t}` | EXACT | audit 1.8e−15 |
| G7 | Three forms of Qt (SNR / spectral / resolvent); shared eigenbasis ∀t | EXACT | audit |
| G8 | Band-fill law `(Qt)_{i,i+d} = (−1)^{d−1}(2t)^{d−1}(Q0^d)_{i,i+d} + O(t^d)` — **no 1/(d−1)! factor** (corrects earlier note) | EXACT | Neumann derivation; coefficient audit <0.9% for d≤5; factorial version rejected (86–800% error) |
| G9 | Large-t return `‖Qt − I‖_F ≈ e^{−2t}‖Σ0 − I‖_F` | EXACT | expansion + audit |
| G10 | Bulk variance `V = 1/√(J_d²−4β²)`; covariance `(J⁻¹)_{i,i+d} = q^d·V`, `q = (J_d−√(J_d²−4β²))/(2|β|)` | EXACT | transfer-matrix proof; audit 8e−13 (K=400) |
| G11 | Limits `q→0` (t→0), `q→α` (t→∞) | EXACT | algebra |
| G12 | Locality: RMS error of radius-r estimator decays exactly as `q^r` (rate) | EXACT (rate) / NUMERICAL (prefactor) | slope audit 0.2% |

## Laplace case (thesis Ch. 8)

| # | Claim | Class | Evidence |
|---|-------|-------|----------|
| L1 | K=1 noised marginal limits (Laplace at t→0, N(0,1) at t→∞) | EXACT | sifting/dominated convergence |
| L2 | K=1 score limits: −sign(x)/b (rescaled, t→0); −x (t→∞) | EXACT + audit (7.4e−6 at t=6) |
| L3 | Characteristic-function factorisation | EXACT | audit 2.9e−6 (quadrature floor) |
| L4 | Screened Poisson: `(1 − b²μ²∂²)p_t = G_Δt`, screening length `bμ = be^{−t}` | EXACT | derivation + Green reconstruction audit 1.9e−9 |
| L5 | Curvature `H_t(x)` is x-dependent field (Var[a|x] non-constant) | EXACT (structure) | Tweedie differentiation |
| L6 | Score from chain messages: two 1D recursions + one 1D integral per frame | EXACT | Tweedie + sum–product |
| L7 | Boundary message = `p_t^{(1)}(x' − μα a)` in closed form | EXACT | innovation substitution |
| L8 | General messages have no clean closed form (≥2 kinks per integrand) | EXACT (structural argument) | Remark in Ch. 8 |
| L9 | Innovation coordinates: prior factorises iid; likelihood densifies (LᵀL dense) — coupling must live somewhere | EXACT | unit-Jacobian change of variables |
| L10 | Laplace K=2 Hessian approximately rank-2 | **SPECULATIVE — NOT ASSERTED** (regime-dependent under quick tests; pending HPC) | conjecture15 note |

## BP (thesis Chs. 5 and 7) / AMP (thesis App. B)

| # | Claim | Class | Evidence |
|---|-------|-------|----------|
| B1 | Factor graph of noisy chain is a tree (3K−1 edges) | EXACT | edge count |
| B2 | Convention-A sweeps compute exact marginals in O(K); Convention B double-counts | EXACT | derivation; audit (A: 1e−14; naive B: O(1) error) |
| B3 | Gaussian messages closed under updates — algebra, **no CLT anywhere** | EXACT | induction on two lemmas |
| B4 | BP = Kalman filter + RTS smoother; matches matrix score to 1.2e−14 | EXACT | identification + audit (80 configs) |
| B5 | Bulk cavity fixed point `λ* = (J_d+√(J_d²−4β²))/2` exists ∀(α,t) — BP never breaks | EXACT | AM–GM margin |
| B6 | Mean field, BP, AMP share the same fixed-point means ⇒ same exact score | EXACT | linear-system argument; audit 1.7e−12 |
| B7 | AMP variance: `V = (J_d−√(J_d²−8β²))/4β²`, exists iff `J_d ≥ 2√2|β|` — factor-2 discriminant vs BP | EXACT | derivation + audit 2.5e−12 |
| B8 | Breakdown time `t_c(α) = −½log(g/(1+g))`, `g = (2√2|α|−1−α²)/(1−α²)`; critical coupling `α_c = √2−1 ≈ 0.4142` | EXACT | derivation; t_c vs bisection 6.6e−6; 180-pt scan exact |
| B9 | Weak coupling: `V_AMP − V = 2β⁴/J_d⁵ + …` (AMP overestimates) | EXACT | series; ratio→1 within 0.4% |
| B10 | Grid BP spectrally accurate; validated regime `√(2t) ≳ 3dx`; M=401, A=8 working config | NUMERICAL | exp_01 calibration vs exact Gaussian score |
| B11 | Laplace closure error (ρ=0.85): median rel. score error 0.39 @ t=0.02 → 7.6e−5 @ t=2.4; P(err>0.1)=1 for t≤0.08, 0 for t≥0.9 | NUMERICAL | exp_02 CSV, medians re-verified 2026-07-14 |
| B12 | Score direction robust: median cosine 0.92 @ t=0.02, ≥0.99 for t≥0.12, worst trial 0.78 | NUMERICAL | exp_02 `score_cosine`, re-verified 2026-07-14 |
| B13 | Innovation sweep: noise level dominates; bimodal worst (0.94 @ t=0.05); error **decreases** with ρ | NUMERICAL | exp_03 |
| B14 | Truncated inference beats truncated matrix at small t: 12.3% vs 21.0% @ t=0.05 (b=r=1, K=40, α=0.8); coincide at large t | NUMERICAL (deterministic traces, seed-free) | bp-from-scratch exp 1 |
| B15 | Architecture reading: local/CNN score heads near-optimal outside mid-diffusion window; receptive field `r ≳ ξ log(1/ε)` | HEURISTIC | reading of G12/B14 |

## Explicitly future

| # | Item |
|---|------|
| F1 | Mixture-of-Gaussians message closure (separate representation vs model error) |
| F2 | Discrete-alphabet chain (exact vector messages, no closure) |
| F3 | Approximate Markovianity: chain + global latent; hybrid BP + learned residual (preliminary exp_04 runs exist, not consolidated) |
| F4 | Reverse-SDE dynamics under exact/closed/truncated scores (preliminary exp_05 runs exist, not consolidated) |
| F5 | Non-Gaussian locality laws (basis-independent statement) |
