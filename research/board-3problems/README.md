# The three board problems

The three panels circled in blue on Mézard's board are three problems. Each is
developed here in the simplest setting that keeps its content, and every
closed-form claim is checked against an independent reference.

**The full write-up is [`note/main.pdf`](note/main.pdf)** — 17 pages, every
derivation, 12 figures. This README is the summary.

    python code/run_all.py            # 66 checks, exits non-zero on any failure
    cd note && python make_note_figures.py && tectonic main.tex   # rebuild the PDF

---

## Setup

Two clocks. Internal time `u = 0 … T-1` runs along the trajectory; diffusion
time `t` is the noise level. One sample is one **whole** trajectory,
`a ∈ ℝ^{2T}`, so `D = 2T`, and the channel

    x = m a + √Δ ξ,     m = e^{-t},  Δ = 1 - e^{-2t},  ξ ~ N(0, I_D)

hits all of `a` at once. That is why the joint score `s = ∇_x log P_t` is the
right object. Below, `Δ₁ = Δ` and `Δ₂ = m²σ² + Δ`.

---

## Problem 1 — rotating ring, random ω

    r₀ ~ ρ e^{-(ρ-1)²/2λ},   θ₀ ~ U[0,2π),   ψ ~ p(ψ)
    z₀ = r₀(cos θ₀, sin θ₀)
    z₁ = R_ψ z₀ + σ η,       η ~ N(0, I₂)
    a  = (z₀, z₁) ∈ ℝ⁴

Three latents `(r₀, θ₀, ψ)` in ambient dimension 4 — the board's "ρ dim 3".

**The rotation is a gauge.** `U_ψ = blockdiag(R_{-uψ})` is orthogonal, so it
commutes with the isotropic channel and turns the model into `ψ = 0`, a plain
2-D random walk started on the ring:

    P_t(x|ψ) = P_t⁰(U_ψ x)          s(x,t|ψ) = U_ψᵀ s⁰(U_ψ x, t)

At known ψ the rotation costs nothing statistically.

**The joint score is closed-form, any T.** Conditionally on `z₀` the trajectory
is Gaussian with a `z₀`-independent covariance, so `P₀` is a 2-parameter
*location* mixture of one Gaussian. With `X ∈ ℝ^{T×2}`,
`A_t = m²σ²K + ΔI_T`, `K_{uv} = min(u,v)`, `g = A_t⁻¹𝟙`, `κ = 𝟙ᵀg`,
`b = Xᵀg`, `β = ‖b‖`:

    s(X,t) = -A_t⁻¹X + Φ_t(β) · g b̂ᵀ
    Φ_t = ∂_β log 𝒵_t,   𝒵_t(β) = ∫₀^∞ ρ e^{-(ρ-1)²/2λ} e^{-m²κρ²/2} I₀(mρβ) dρ

Linear part plus a **rank-one** correction from one 1-D Bessel integral. The
correction is exactly `m E[z₀|x]`, the denoised ring anchor.

On the structure of `A_t` (R14): `K` is singular with a vanishing 0th row —
frame 0 *is* `z₀` and carries no walk noise — so `A_t` is exactly block-diagonal
and frame 0 decouples in `A_t⁻¹` at every `t`. On the remaining block the `d`-th
off-diagonal goes as `(Δ/m²σ²)^d`, so `A_t⁻¹` becomes tridiagonal as `t → 0`.
That is the band-fill law of G8 in this model. (Note `A_0⁻¹` does *not* exist;
an earlier draft asserted "`A_0⁻¹ = K⁻¹/σ²` is tridiagonal", which is
meaningless.)

**Random ω costs one more 1-D integral.** By Bayes plus the gauge,

    s(x,t) = ∫ p(ψ|x,t) U_ψᵀ s⁰(U_ψ x, t) dψ,   p(ψ|x,t) ∝ p(ψ) P_t⁰(U_ψ x)

So there is ground truth at any `T` from two nested one-dimensional quadratures.
No network, no Monte Carlo, no curse of dimension. This is the tool everything
else uses.

At **`T = 2` only** the quadratic form drops out (`A_t` is diagonal there, and
only there) leaving `p(ψ|x,t) ∝ p(ψ)𝒵_t(β(ψ))` with

    β(ψ)² = |x₀|²/Δ₁² + |x₁|²/Δ₂² + 2|x₀||x₁| cos(γ-ψ)/(Δ₁Δ₂),  γ = arg x₁ - arg x₀

so the data enter only through `(|x₀|, |x₁|, γ)`, and the posterior is a von
Mises-like bump on the observed turning angle `γ`. For two species
`p(ψ) = ½(δ_{+ω} + δ_{-ω})` the log-odds are one scalar, and for small `m`

    logit → (m²⟨ρ²⟩/Δ₁Δ₂) · |x₀||x₁| sin γ sin ω

correctly vanishing at `ω = 0` and `ω = π`, where the two species coincide.

**Marginal blindness is a theorem.** `z_u = R^u z₀ + σΣ_{k≤u}R^{u-k}η_k`, each
`R^{u-k}η_k =ᵈ η_k` by isotropy and `R^u z₀ =ᵈ z₀` by the uniform phase, so

    z_u =ᵈ z₀ + σ√u ξ        — ψ does not appear

Every single-frame marginal is ψ-free at every noise level, so the per-frame
score carries **exactly zero** information about the dynamics. Not less — zero.
This is the sharp form of the per-frame/joint error in `ch06-development.tex`
§6.4, and a better illustration than the Gaussian chain, where the marginal
score is the trivial `-x_k`: here the marginal is a visible, non-Gaussian ring
and still blind.

`outputs/fig.svg`, at `T = 12`, `D = 24`, `ψ = 30°`:

| | turning angle | concentration `R` | mean radius, frame 0 → 11 |
|---|---|---|---|
| data | 29.88° | 0.959 | 1.053 → 1.350 |
| reverse SDE, exact joint score | 30.07° | 0.942 | 1.058 → 1.348 |
| reverse SDE, per-frame marginals | — | **0.009** | 1.050 → 1.334 |

The marginal model gets every radius right to 1.3% and has no dynamics at all.
Radii are not evidence.

---

## Problem 2 — additive two-frame chain

The board's middle panel: `x₁ = x₀ + c + η`, `η ~ N(0,σ²)`,
`q(x₀,x₁) = P₀(x₀)N(x₁; x₀+c, σ²)`. Scalar, `D = 2`.

Gaussian `P₀ = N(μ₀, v₀)`: `C₀ = [[v₀,v₀],[v₀,v₀+σ²]]`, `C_t = m²C₀ + ΔI`, and
`s(x,t) = -C_t⁻¹(x - m E[a])`. The marginal object is **exactly diagonal** at
every `t`, so the whole joint/marginal difference is one number:

    (C_t⁻¹)₀₁ = -m²v₀ / (m⁴v₀σ² + 2Δm²v₀ + Δm²σ² + Δ²)

    t → 0:  -(1/σ²)[1 + Δ(1 - 2/σ² - 1/v₀)] + O(Δ²)
    t → ∞:  -e^{-2t} v₀

The `t → 0` value is the innovation precision. **The remainder prefactor is
`2/σ² + 1/v₀ - 1`**, not `2/σ² + 1/v₀` — the naive version is wrong by 4.2% at
`σ² = 0.09, v₀ = 0.7` and is explicitly rejected in the audit (same failure mode
as the band-fill factorial in G8). So temporal structure is a *low-noise*
feature of the score: the coupling dies as `e^{-2t}`.

With a mixture prior `P₀ = Σ_k w_k N(μ_k, v)` the score is exact in 2×2 algebra,
`s = Σ_k r_k(x)(-C_t⁻¹(x - mν_k))`, `ν_k = (μ_k, μ_k+c)`. Since
`ν₁ - ν₂ ∝ (1,1)`, the reverse process picks its mode from the **frame
average** — the same redundancy mechanism as Problem 1's speciation.

One caveat worth keeping: here the drift `c` *is* visible in the marginals
(frame 1 has mean `μ₀+c`). Problem 1 is the model for making the joint/marginal
point, because its symmetry hides the dynamics completely; Problem 2 is the
model to compute with.

---

## Problem 3 — terminal-frame reward

The board's `E[-∫₀¹|u|² - r(x_T)]` is the control form of tilting. Maximising
`E[r(a)] - ½E∫‖v‖²` gives the Doob `h`-transform: extra drift `g²∇log h_t`,
`h_t(x) = E[e^{r(a)}|x_t = x]`, sampling `P₀^r ∝ P₀e^r`, with control cost equal
to the KL. The board's `u_opt ≈ 2f` is this drift at `g² = 2`.

For `P₀ = N(0,C₀)`, `E` the last-frame selector, `r = -‖Ea - m*‖²/2s²`,
everything is Gaussian: with `Σ_p = (C₀⁻¹ + (m²/Δ)I)⁻¹`,
`ν_t(x) = EΣ_p(m/Δ)x`, `S_t = EΣ_pEᵀ`,

    log h_t(x)  = -½log|I + S_t/s²| - ½(ν_t - m*)ᵀ(S_t + s²I)⁻¹(ν_t - m*)
    ∇log h_t(x) = -(m/Δ) Σ_p Eᵀ (S_t + s²I)⁻¹ (ν_t - m*)

linear in `x`. The controlled reverse SDE reproduces `N(μ₀^r, C₀^r)` to 0.2%
(mean) and 1.3% (covariance).

**The board's `P(τ) ⇐ P(T)` has an exact answer.** The per-frame variance
reduction is

    C₀^{(uu)} - C₀^{r,(uu)} = (C₀^{(u,T-1)})² / (s² + C₀^{(T-1,T-1)})

so the prior's memory alone decides how far back a terminal reward reaches
(`outputs/fig_p3.svg`, `T = 16`):

- **ring-anchored walk**, `C₀^{(uv)} = ρ₀ + σ²min(u,v)`: fractional reduction
  0.22 at `u = 0` rising to 0.82 at `u = T-1`. Reaches every frame.
- **stationary AR(1)**, `C₀^{(uv)} = α^{|u-v|}`: 0.0009 at `u = 0`, decaying by
  exactly `α²` per frame going back, range `ξ = 1/log(1/α) = 4.5` frames. This
  is the `q^r` locality law (G12) in a new guise.

A factor 251 apart at the first frame. Non-stationary priors propagate a
terminal reward globally; stationary ones propagate it over a finite memory.

---

## Two claims of mine that the audit rejected

Both were asserted in the first pass of this note and are **false as stated**.
`verify_scaling.py` now tests the rejection so they cannot creep back.

**H1 — "`t_spec = ½ log T`".** Rejected. `t_spec(0.75)` measured with the exact
ψ-posterior: 0.11, 0.34, 0.59, 0.96, 1.45, 2.05 at `T = 2,4,8,16,32,64`. It
rises with `T`, but the local slope `d t_spec/d log T` drifts monotonically
0.33 → 0.36 → 0.53 → 0.72 → 0.86. There is no single power law over this range.
The asymptotic form is **open**: at large `t` the `T`-exponent of the log-odds
measures 2.7–3.2 and the `e^{-2t}` prefactor law does not hold cleanly there.
The likely cause of the drift is that `K_{uv} = min(u,v)` is non-stationary, so
neighbouring-pair terms carry weight `u` and sum to `~T²/2` rather than `~T`.

**H2 — "unstructured needs `Θ(T²)` samples, structured `O(1)`".** Rejected. The
diffusion's own noise floor `Δ_t` regularises the empirical score, so at fixed
`t` the requirement grows like `D`, not `D²`. Measured, `T = 8`, `D = 16`, for
10% relative score error:

| `t` | `Δ_t` | `N` unstructured | `N` structured | ratio | `NΔ/D` |
|---|---|---|---|---|---|
| 0.02 | 0.039 | 576 | 18 | 32.0 | 1.41 |
| 0.05 | 0.095 | 242 | 18 | 13.4 | 1.44 |
| 0.10 | 0.181 | 126 | 18 | 7.0 | 1.43 |
| 0.40 | 0.551 | 28 | 18 | 1.6 | 0.96 |
| 0.80 | 0.798 | 18 | 18 | 1.0 | 0.90 |

    N_unstructured ≈ 1.46 (±0.07) · D / Δ_t     for Δ_t ≲ 0.2,
    saturating to N ~ O(D) at large Δ_t

and the `D`-exponent at `t = 0.05` measures 0.69–0.89 across seeds — sub-linear
to linear, definitively not `D²`. The structured estimator (gauge plus two
scalars) needs `N = 18` at *every* noise level.

So the board's `N > D(D+1)/2` is a **parameter count for identifying `C₀`
itself**, which is strictly stronger than what an accurate score at `t > 0`
requires. The real content is that the structured/unstructured gap is set by the
noise level and opens only as `t → 0` — which is exactly the memorisation
regime.

---

## Ledger

Classes as in `docs/RESULT_LEDGER.md` (archived Aug 2026; see
`research/nongaussian-bp/CLAIMS_TO_UPDATE.md` for the project's current live claim ledger).

| # | Claim | Class | Evidence |
|---|-------|-------|----------|
| R1 | Rotation is an orthogonal gauge, commutes with the channel | EXACT | 1.4e-17 |
| R2 | Joint score = linear + rank-one, any `T` | EXACT | vs finite diff 4e-10 … 2e-9 |
| R2b | Rank-one term is `m E[z₀\|x]`; rank exactly 1 | EXACT | 2e-15 / 5e-17 |
| R3 | Random ψ: score = ψ-posterior average, **any `T`** | EXACT | 8.9e-10 (`T=2`), 7.7e-10 (`T=3`) |
| R3a | `p(ψ\|x,t) ∝ p(ψ)𝒵_t(β(ψ))` — **`T = 2` only** | EXACT, scope-limited | 1.8e-16 at `T=2`; fails 10–13% at `T=3,5`, both sides tested |
| R4 | `β(ψ)` identity, statistic `(\|x₀\|,\|x₁\|,γ)`, peak at γ. `T=2` | EXACT | 8.9e-16 |
| R5 | Two-species log-odds asymptote. `T=2` | EXACT (asymptotic) | 0.2% at three `t` |
| R6 | Marginals are ψ-free ⇒ marginal score carries zero dynamics | EXACT | proof + MC, quantiles ψ-invariant 3.7e-3 |
| R7 | Two-frame Gaussian score; marginal object exactly diagonal | EXACT | 2.8e-11 |
| R8 | Coupling closed form; remainder prefactor `2/σ²+1/v₀-1` | EXACT | 4.4e-4; naive prefactor rejected |
| R9 | Mixture score exact; discriminant is the frame average | EXACT | 8.6e-11 |
| R11 | Gaussian `h_t`, `∇log h_t`; controlled SDE samples `P₀^r` | EXACT + NUMERICAL | 8e-11; mean 0.2%, cov 1.3% |
| R12 | Reward reach `= (C₀^{(u,T-1)})²/(s²+C₀^{(T-1,T-1)})` | EXACT | 3.3e-16 |
| R13 | `N_uns ≈ 1.46 D/Δ_t` for `Δ_t ≲ 0.2`, saturating to `O(D)` | NUMERICAL | `c` stable to 12% over 4 decades of `Δ` |
| R14 | `A_t` block-diagonal (frame 0 decouples exactly); `d`-th off-band of `A_t^{-1}` `~ (Δ/m²σ²)^d` | EXACT | decoupling 0; band law 19% |
| ~~H1~~ | ~~`t_spec = ½ log T`~~ | **REJECTED** | slope drifts 0.33 → 0.86 |
| ~~H2~~ | ~~`Θ(T²)` vs `O(1)` samples~~ | **REJECTED** | `D`-exponent 0.69–0.89 |
| O1 | Asymptotic `t_spec(T)` law | OPEN | exponent 2.7–3.2, `e^{-2t}` law unclean |

## Files

    note/main.pdf                   the write-up: all derivations, 12 figures
    note/main.tex, make_note_figures.py
    code/run_all.py                 one command, all 66 checks
    code/verify_ring_walk.py        31 checks: gauge + closed form, T = 3, 5
    code/audit_three_problems.py    27 checks: three problems at T = 2, plus R14
    code/verify_scaling.py           8 checks: H1, H2 rejections + R13
    code/gen_figure_data.py         regenerates outputs/figdata.json (~20 min)
    code/make_svg.py, make_svg_p3.py  build the two figures
    outputs/fig.svg                 joint vs marginal score, T = 12
    outputs/fig_p3.svg              reward reach under the two priors
    requirements.txt                numpy, scipy

## Open

1. **O1**: the asymptotic `t_spec(T)` law. Ground truth exists at any `T` (R3),
   so this is a measurement, not a derivation.
2. The rest of the finite-`N` panel: the Gram-matrix conditions and the
   memorisation/collapse transition proper, now that R13 says where to look
   (small `t`).
3. `ch06-development.tex` §6.2 and §6.5 record these models as having no closed
   form in reach and being deliberately set aside. R1–R3 contradict that; the
   paragraphs need rewriting, which is a thesis-structure decision.
