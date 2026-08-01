# Layer 5 compendium — learning the denoiser by expectation maximization

Branch: `claude/em-bp-denoiser-learning-e07ike` (never pushed to `main`).
Package: `research/nongaussian-bp/`.
Date: 2026-07-31.

This is the record of what was built, what was measured, what was found to be
wrong, and what is not settled. It is meant to be readable on its own; the
theory is in `research/nongaussian-bp/report/em_bp_learning.pdf` (14 pp) and
the handoff instructions for continuing the work are in
`docs/AGENT_HANDOFF_EM_BP.md`.

---

## 1. What this layer is

Layers 1–4 established that when the clean data is a Markov chain, belief
propagation computes the diffusion score **exactly** — the posterior induced by
coordinatewise noising is again a chain, chains are trees, and sum-product BP on
a tree is exact. All of that assumed the prior was **known**.

Layer 5 removes that assumption. The transition kernel of the clean chain
becomes an unknown parameter, only *noised* sequences are observed, and the
kernel is estimated by maximum marginal likelihood using EM. This is the
suggestion from the Marc/Jérôme email: regress BP parameters instead of
training a network, and see whether it learns the denoiser more efficiently.

**Verdict: yes, by roughly an order of magnitude at fixed data, and by ≥64× in
the data needed for equal accuracy.** With the caveats in §6.

---

## 2. Theory

`research/nongaussian-bp/report/em_bp_learning.tex` → `.pdf`, 14 pp, compiles
clean (`latexmk -pdf`). Five propositions:

| # | Statement | Why it matters |
|---|---|---|
| 1 | **The E-step is exact.** Sitewise likelihood factors reweight node potentials without adding edges, so the posterior is still a chain; BP returns exact single-site *and pairwise* marginals. | In a generic latent-variable model the E-step is approximate and the monotonicity guarantee is lost with it. Here nothing is approximated but the message representation. |
| 2 | **The whole E-step is one matrix.** Everything compresses into an `M×M` matrix `Ξ` — the continuum analogue of Baum-Welch's expected transition counts. Sufficient, and independent of how `K_θ` is parameterized. | The M-step never touches the data again (cost `O(P M²)` regardless of `N`); observations at different noise levels simply *add* into one `Ξ`; swapping kernel families changes only the M-step. |
| 3 | **Fisher's identity ⇒ no autodiff.** `∇L(θ) = ⟨Ξ(θ), ∇log K_θ⟩` is the exact gradient of the marginal log-likelihood. | BP is never differentiated through — it *supplies* the expectation. Everything is pure numpy. This directly contradicts the advice that EM here "requires moving the forward-backward passes into an autodiff framework". |
| 4 | **Noising never destroys identifiability.** `φ_t,θ(ξ) = φ_θ(α_t ξ)·exp(−Δ_t‖ξ‖²/2)`; the Gaussian factor has no zeros and `α_t > 0` for all finite `t`, so `p_t,θ = p_t,θ'` ⟹ `P_θ = P_θ'`. | Makes precise the "does not require noising at all" remark. Identifiability survives any finite `t`; what degrades is *information*, which Prop. 3 makes exactly computable. Injective but ill-conditioned. |
| 5 | **Denoiser sensitivity is a posterior covariance.** `∂m_i/∂θ_p = Cov(a_i, Σ_l ∂_θp log K(a_l\|a_{l−1}))`. | One `N^{−1/2}`-consistent parameter fit gives an `N^{−1/2}`-accurate denoiser *uniformly in t*. The network gets no such statement — its error must be paid for separately in every region of `(x,t)`. |

**The structural argument in one sentence:** the learned parameters live on the
clean chain, the noise level lives in the likelihood — so one fit serves every
`t`, and the risk decomposition has no function-class approximation term (given
a prior, BP returns the *exact* Bayes denoiser).

---

## 3. Implementation

Pure numpy, no autodiff, no new dependencies.

| File | Contents |
|---|---|
| `src/em.py` | Exact BP E-step, `Ξ` accumulation, exact evidence, EM driver with monotonicity trace, and the `t→0` clean-data limit sharing the same M-step code |
| `src/kernels.py` | Four kernel rungs and their M-steps |
| `src/denoiser.py` | BP denoiser from a learned kernel + denoising-score-matching baseline, behind one evaluation interface |
| `experiments/exp_06_…` | Correctness: monotonicity, Fisher information, misspecification, rates, quantization |
| `experiments/exp_07_…` | The headline: EM-BP vs a score network |
| `experiments/exp_08_…` | The literal gradient route vs the exact M-step |
| `tests/test_em_bp.py` | 23 tests (46 total in the package, all passing), including five independent cross-checks (§5.5) |
| `hpc/`, `tools/merge_replicates.py` | Cluster templates and replicate merging |

**The four rungs**, all consuming the same `Ξ`:

1. **Gaussian AR(1)** `(ρ, q)` — closed-form M-step (belief-weighted Yule-Walker); the correctness harness, since everything is analytically available.
2. **Laplace AR(1)** `(ρ, b)` — closed form *despite* non-differentiability: profile out `b`, and maximizing over `ρ` becomes a weighted least-absolute-deviations regression whose exact solution is a weighted median of grid ratios.
3. **Mixture innovation** `(ρ, {π,μ,σ²}_c)` — the honest "we know it's Markov but not the model" estimator. Has never heard of the Laplace density it must recover. Closed-form ECM via an inner latent (the mixture label).
4. **Mixture-density network** — a small MLP emitting a state-dependent mixture. By Fisher's identity the network is evaluated and backpropagated **only at the `M` grid points**, once per EM iteration, regardless of dataset size.

---

## 4. Results

### 4.1 EM-BP vs a score network (exp_07 Part 1) — the headline

Relative score error, averaged over the noise schedule. EM-BP: **13 parameters**.
Network: **25,248 parameters**, trained on paired `(a,x)` with a fresh noise
draw every gradient step. EM sees one noisy realization per chain, at one noise
level, and never sees a clean chain.

| N chains | ε-net | x₀-net | **EM-BP** | ratio (ε/EM) |
|---|---|---|---|---|
| 32 | 0.654 | 0.768 | **0.130** | 5.0× |
| 64 | 0.588 | 0.677 | **0.034** | 17.5× |
| 128 | 0.508 | 0.577 | **0.048** | 10.6× |
| 256 | 0.403 | 0.488 | **0.031** | 13.0× |
| 512 | 0.282 | 0.372 | **0.034** | 8.2× |
| 1024 | 0.217 | 0.312 | **0.022** | 9.8× |
| 2048 | 0.179 | 0.237 | **0.016** | 10.9× |

**The sharpest reading is not the ratio at fixed N.** EM-BP on 32 chains
(0.130) already beats the network on 2048 chains (0.179) — a **≥64× gap in the
data each needs** for equal accuracy. That is a lower bound: 32 was the smallest
budget tried, and the network was still improving at 2048.

Per noise level at N=2048, EM-BP wins at every `t`, and the margin is widest at
low noise (0.041 vs 0.248 ε / 0.598 x₀) — where the score matters most for
reverse-time integration, and where Layer 2 found the Gaussian closure failing.

`identity_residual` max **5.1e-14** across all rows: the exact score/mean
relation holds to machine precision throughout.

### 4.2 The gap is not undertraining or undersizing (exp_07 Part 2)

24 configurations at N=1024: 4 architectures × 3 training budgets × 2
parameterizations.

| hidden | params | best over budgets & parameterizations |
|---|---|---|
| (32,32) | 3,264 | 0.235 |
| (128,128) | 25,248 | **0.208** |
| (256,256) | 83,232 | 0.251 |
| (512,512) | 297,504 | 0.284 |

**EM-BP (13 params): 0.0222.** Best network over *all* 24 configs: 0.2078 →
**9.4× worse**. Error does not improve with capacity; it degrades for the
largest networks (512×512 at 20k steps is the worst cell in the table, 0.373 ε /
0.408 x₀). So the gap is overfitting at this data budget, not a badly chosen
architecture — which is the control the comparison needed.

### 4.3 Transfer across the noise schedule (exp_07 Part 3) — with a correction

EM-BP wins at **all 13** probed noise levels, `t` from 0.02 to 3.2, by 4.5×–19×
against whichever parameterization is better at that level.

| t | in schedule | ε-net | x₀-net | **EM-BP** | best-net / EM |
|---|---|---|---|---|---|
| 0.02 | no | 0.766 | 3.527 | **0.170** | 4.5× |
| 0.10 | yes | 0.326 | 0.792 | **0.051** | 6.4× |
| 0.40 | yes | 0.190 | 0.216 | **0.016** | 11.9× |
| 1.60 | yes | 0.162 | 0.029 | **0.0041** | 6.9× |
| 3.20 | no | 0.247 | 0.007 | **0.0004** | 19.0× |

**The expected mechanism is not what the data shows, and this corrects the
experiment's own premise.** The part was designed around "the network must
extrapolate in `t`" — but there is *no cliff* at the schedule boundary, and the
ratio does not separate in-schedule from out-of-schedule levels (4.5, 5.3 out;
6.4 in; 7.3 out; 8.3 in; …). Time enters through smooth features and the target
moves smoothly with `t`, so the network handles that one direction adequately.
**The difficulty is in x-space, not t-space.** The extrapolation claim should
not be made.

What *is* real: EM-BP's error varies smoothly across the whole range with no
schedule dependence whatsoever, because it has no schedule to leave. And the x₀
parameterization fails outright at low noise (3.53 at t=0.02, against 0.766 for
ε and 0.170 for EM-BP) — the concrete case for training both.

### 4.4 Inference cost — the honest loss (exp_07 Part 4)

| batch | BP ms/chain | net ms/chain | slowdown |
|---|---|---|---|
| 32 | 2.01 | 0.0096 | 211× |
| 128 | 1.23 | 0.0043 | 289× |
| 512 | 1.31 | 0.0041 | 320× |

At M=401, grid BP is **211×–320× slower** per chain than a network forward pass
— larger than the ~100× seen at M=201 in smoke runs, since cost is `O(nM²)`.
Reverse diffusion calls the denoiser at *every* integration step, so this is the
weakest point of the whole story.

### 4.5 EM behaves exactly as the theory says (exp_06 Part 1)

N=1024, M=401, Laplace kernel, 6 random initializations including ρ₀ = −0.42:

- All 6 converge to the **identical** fixed point: ρ̂ = 0.8, b̂ = 0.419259, logL = −42458.1367.
- **Monotonicity violation exactly 0** in every run — the sharpest available test of the whole pipeline, since an error anywhere in the forward-backward recursion, the pairwise accumulation, or the M-step generically breaks it.
- 52–59 iterations.

⚠️ The reported `rho_err = 1e-16` is **not** perfect recovery — see §5.1. The
honest number here is `b_err = 0.005` (1.2%).

### 4.6 The price of noising, against the information budget (exp_06 Part 2)

Fisher information per noisy chain computed exactly from single-chain BP passes
via Prop. 3 — no simulation, no numerical differentiation. Realized error over
10 replicates at N=256:

| t | J[ρ,ρ] | J[q,q] | sd(ρ) | CRLB(ρ) | bias(ρ) | bias(q) |
|---|---|---|---|---|---|---|
| 0.05 | 86.8 | 66.5 | 0.0079 | 0.0069 | −0.0032 | +0.0011 |
| 0.20 | 64.1 | 30.2 | 0.0072 | 0.0092 | +0.0037 | −0.0051 |
| 0.80 | 21.3 | 4.79 | 0.0192 | 0.0231 | −0.0171 | +0.0587 |
| 1.60 | 3.33 | 0.469 | 0.0555 | 0.0806 | −0.3775 | +0.5312 |

Two readings:

1. **The channel destroys the innovation scale far faster than the correlation.** `J[q,q]` falls **142×** from t=0.05 to t=1.6 while `J[ρ,ρ]` falls **26×**. Second-order structure survives noising; the shape information carrying non-Gaussianity does not. Same asymmetry Layer 2 found for the Gaussian closure, now stated as an information budget rather than an error measurement.
2. **The estimator is close to efficient where it works** — realized sd within ~±30% of Cramér-Rao for t ≤ 0.8. At t=1.6 it stops working and the information says it must: bias dwarfs sd, and CRLB(q)=0.215 is 60% of `q_true` itself.

### 4.7 The gradient route vs the exact M-step (exp_08)

The advice was phrased as *gradient* ascent, not EM, so both were measured. They
share the E-step and `Ξ` and differ only in what they do with it. **The split is
not the one intuition suggests:**

- **Smooth (Gaussian) kernel:** gradient ascent converges to *exactly* EM's optimum (log-likelihood matching to 2e-9 nats at η=0.5, parameter errors to four digits). At η=0.1 it is merely not there yet; at η=2 it diverges (527 nats of monotonicity violation); at η=8 it leaves the admissible set. **Nothing gained over EM, and a step size must be tuned to lose nothing.**
- **Laplace kernel:** gradient ascent is genuinely **better** — higher likelihood (−8064.7 vs −8066.6) and better on both parameters (|Δρ| 0.022 vs 0.050, |Δb| 0.013 vs 0.027). Because there EM's M-step is exactly optimal *for the discretized model* and lands on a lattice point (§5.1).

**Recommendation:** exact M-step where it exists *and* the kernel is smooth;
gradient route where the M-step has no closed form (rung 4) or where its exact
solution is a discretization artifact (rung 2).

### 4.8 Recovering an unknown innovation law

The mixture kernel has never heard of the Laplace density it must recover. Fitted
to **noisy observations only** (N=1024, one noisy realization per chain, noise
levels up to t=1.6, never a clean sample):

| C | ρ̂ | innov. var | innov. excess kurtosis | innov. mean | logL |
|---|---|---|---|---|---|
| 2 | 0.806 | 0.363 | 1.95 | −0.005 | −42659.4 |
| 3 | 0.807 | 0.360 | 2.23 | −0.005 | −42658.4 |
| 5 | 0.808 | 0.360 | 2.62 | −0.005 | −42655.7 |
| 8 | 0.807 | 0.361 | **2.71** | −0.005 | −42655.5 |
| *true* | *0.800* | *0.360* | *3.00* | *0* | |

The variance is recovered essentially exactly from C=3 up; the heavy tail
climbs monotonically with C toward its true value; the likelihood increases with
C; monotonicity violation is exactly 0 throughout. The fitted innovation mean
lands at −0.005 **without being constrained to** — which is the empirical
justification for not enforcing `Σπ_cμ_c = 0` (doing so by projection broke
monotone ascent, §5.3).

Induced denoiser error against exact BP under the *true* prior is best around
C=5 (0.0117 at t=0.4, 0.0225 at t=0.2); at t=0.1 the largest mixture is
slightly worse than C=3 (0.050 vs 0.038), mild overfitting at the low-noise end.
`identity_residual` max 2.0e-14.

⚠️ **The kurtosis estimate is high-variance, and should not be presented as a
clean convergence curve.** At C=8 with one replicate per N it reads 2.91, 3.45,
3.70, 3.29, 2.64 for N = 128 … 2048 — non-monotone, and overshooting the true
3.0 in the middle. What *does* improve monotonically with N is the quantity that
matters, the denoiser error: 0.083, 0.080, 0.061, 0.060, 0.037. Any claim about
"how much data the heavy tail needs" requires the replicate runs in
`hpc/slurm_replicates.sbatch`, not this single-replicate sweep.

---

### 4.9 The N^{−1/2} rate (exp_06 Part 4, re-run at 12 replicates)

Smooth kernels only (the Laplace kernel is disqualified by §5.1). RMSE over
**12** replicates — `outputs/exp_06_rate_highrep/`:

| N | Gaussian ρ | Gaussian q | mixture ρ | mixture var |
|---|---|---|---|---|
| 64 | 0.0154 | 0.0390 | 0.0219 | 0.0399 |
| 128 | 0.0126 | 0.0225 | 0.0068 | 0.0177 |
| 256 | 0.0127 | 0.0208 | 0.0108 | 0.0119 |
| 512 | 0.0098 | 0.0107 | 0.0070 | 0.0141 |
| 1024 | 0.0052 | 0.0071 | 0.0050 | 0.0096 |

OLS log-log slopes with standard errors from the regression residuals:

| quantity | slope | distance from −0.5 |
|---|---|---|
| Gaussian ρ | −0.349 ± 0.094 | 1.6σ |
| Gaussian q | −0.600 ± 0.066 | 1.5σ |
| mixture ρ | −0.420 ± 0.179 | 0.45σ |
| mixture var | −0.445 ± 0.136 | 0.41σ |
| **combined** | **−0.500 ± 0.048** | **0.01σ** |

**The parametric rate is now supported.** The combined estimate sits on −0.500,
and −0.25 or −0.75 are excluded at 5.2σ. The four estimates are not fully
independent — the two Gaussian quantities come from one set of fits and the two
mixture quantities from another — so the honest combined error is closer to
0.048·√2 ≈ 0.068, treating the effective count as two rather than four. Even
then a rate of −0.25 or −0.75 is excluded at >3.5σ, and −0.5 is unrejected.

⚠️ **This supersedes the 4-replicate run** (`outputs/…/em_rate.csv`), from which
I reported slopes of −0.26, −0.69, −0.25, −0.41 and concluded the rate was *not*
established. That conclusion was right for that data: tripling the replicates
narrowed the spread from a range of 0.43 to 0.25 and moved every estimate toward
−0.5. The lesson is about replicate counts, not about the estimator — an RMSE
from 4 samples simply cannot resolve a slope to better than about ±0.2.

### 4.10 Lattice quantization, measured (exp_06 Part 5, re-run)

ρ̂ from the Laplace M-step, with the **data held fixed per ρ\*** so the grid is
the only thing varying down each column:

| M | ρ\*=0.7700 | ρ\*=0.7913 | ρ\*=0.8000 | ρ\*=0.8130 |
|---|---|---|---|---|
| 201 | 0.7500 | **0.8000** | 0.8000 | **0.8000** |
| 401 | 0.7778 | **0.8000** | 0.8000 | **0.8000** |
| 801 | 0.7857 | **0.8000** | 0.8000 | 0.8108 |
| 1601 | 0.7852 | **0.8000** | 0.8000 | 0.8113 |

**4/5 is an attractor of the weighted-median M-step, and refining the grid does
not escape it.** For ρ\* = 0.7913 — chosen deliberately off the simple lattice —
the estimate is pinned at exactly 0.8000 across an **8× grid refinement**, with a
constant 0.0087 bias. That is not a resolution limit that more grid points would
fix; it is a fixed bias. At M=201 three distinct true values (0.7913, 0.8000,
0.8130) all return exactly 0.8000.

The `b` error is now interpretable too, and shows the damage propagating: where
ρ is snapped away from the truth (ρ\*=0.8130) the scale error sits at ~0.027,
roughly 3–4× the ~0.008 seen where ρ is recovered. ρ error contaminates `b`,
because `b` is estimated conditional on the snapped ρ.

⚠️ **This supersedes the first run of this part, and corrects a claim I made
from it.** That run keyed its seed on the grid size, so every cell drew
different data; I reported from it that "at M=201 three distinct true values all
return exactly 0.750". With the confound removed the collapse is real but lands
on **0.800**, not 0.750 — the specific attractor in the first run was itself an
artifact of the resampling. The qualitative phenomenon survives; that particular
number did not.

---

## 5. Findings recorded rather than smoothed over

### 5.1 The Laplace M-step is exact but lattice-quantized

The minimizer of `Σ Ξ[k,j]|u_k − ρu_j|` is a *breakpoint*, i.e. one of the
ratios `u_k/u_j`. On a uniform grid through the origin these are rationals `m/l`,
and a low-denominator value like 4/5 has many aliases (8/10, 12/15, …) pooling
weight onto it. So ρ̂ snaps to simple rationals — measured in §4.10, where
ρ\*=0.7913 is pinned at exactly 0.8000 across an 8× grid refinement with a
constant 0.0087 bias, and three distinct true values collapse onto 0.8000 at
M=201. This is real for the discretized model, not an algorithmic bug, and it
is a *bias* rather than a resolution limit: more grid points do not help.
Smooth kernels (Gaussian, mixture) have M-steps that are ratios of smooth
moments and are free of it — which is why the reported rates use those.

### 5.2 The Laplace ρ-gradient loses accuracy to a sign discontinuity

`∂_ρ log K` contains `sign(e)`, and trapezoidal quadrature of a discontinuous
integrand loses the spectral accuracy the rest of the package enjoys: off by
~13% at M=201 and still ~0.1% at M=1601, decaying erratically. The `b`
direction, being smooth, agrees to machine precision — confirming Prop. 3 is
fine and the quadrature is not.

### 5.3 Three bugs found and fixed

1. **`rng_for` was not reproducible across processes.** It seeded from Python's builtin `hash`, salted per process for strings (PEP 456), and every call mixes in a string tag — so **no experiment in this package was bit-reproducible from a fresh interpreter**, contrary to the README. Verified: three runs, three different streams. Now uses a blake2b digest. Pairing *within* a run held, so exp_01–05 numbers stand as measurements; their committed outputs were not regenerated. The earlier audited package (`markov_gaussian_approx`) seeds from plain integers and is unaffected.
2. **The score-network baseline was handicapped.** It predicted the clean signal only, whereas the standard diffusion recipe predicts the noise — and the two are not equivalent in finite samples (losses differ by `α_t²/Δ_t`, and ε-prediction recovers the mean with a `√Δ_t` prefactor suppressing error at low noise). Both are now trained and reported; ε wins at small `t`, x₀ at large `t`, so the result rests on neither.
3. **The MDN M-step broke EM's guarantee.** A fixed Adam step is not an ascent step: the run raised the likelihood overall while violating monotonicity by 547 nats (η=0.02) and 1452 (η=0.05). Now backtracked until `Q` actually increases — zero violation, negligible cost (`Q = ⟨Ξ, log K⟩` is one contraction of a matrix already in hand), and diagnostic besides (at η=0.05 the run visibly stalls instead of silently descending).

### 5.4 More expressive is not better

At N=400 the neural kernel (873 params) is dominated on **both** axes by the
13-parameter mixture kernel: training log-likelihood −16781 vs −16656, held-out
score error 0.052 vs 0.026. The efficiency argument is about matching the
hypothesis class to the structure — and it applies to the structured method's
own internals, not only to the baseline it is compared against.

---

### 5.5 Independent cross-checks

The tests above mostly verify the code against itself or against finite
differences of its own output. Three checks deliberately do not, taking routes
that share no code with the E-step:

| check | route | result |
|---|---|---|
| **`Ξ` vs brute force** | Enumerate all `M^n` discretized configurations of the joint posterior and marginalize by summation — no messages anywhere | `max|Ξ_BP − Ξ_enum| / max(Ξ)` = **1.0e-14**; evidence agrees to 1.9e-15 |
| **Evidence vs Monte Carlo** | Importance sampling `p_t(x) = E_{a∼p_0}[∏ᵢ N(xᵢ; α aᵢ, Δ)]` on the **Laplace** chain, where no closed form exists | BP −6.406521 vs MC −6.406111 ± 0.000691 (4M samples) → **0.59 s.e.** |
| **EM fixed point is stationary** | Fisher's identity evaluates `∇L` without reference to how the M-step chose its update | `‖∇L‖` falls from 1.3e3 to 1.6e-4, a factor of **8.1e6** |

The first is the strongest single piece of evidence for the whole layer: it
pins the entire E-step — forward-backward recursion, pairwise accumulation, and
quadrature weights together — against explicit enumeration. The second extends
the evidence check beyond the Gaussian case, which is the only one with a
closed form. The third confirms EM lands on a stationary point of the
*marginal* likelihood rather than merely a fixed point of the M-step map.

Two further checks target the things that would invalidate the **headline
comparison** specifically, rather than the machinery:

| check | why it matters | result |
|---|---|---|
| **Reference denoiser vs Monte Carlo** | Every number in exp_07 is a deviation from `bp_posterior_mean` under the true prior. If the yardstick were wrong, all of them would be. | relative L2 difference **1.1e-3** over 6M samples, with every coordinate inside the MC standard error |
| **The baseline network can actually learn** | If the DSM implementation were subtly broken, "EM-BP wins" would be an artifact rather than a sample-efficiency effect. Trained on a *Gaussian* chain where the exact denoiser is closed-form and linear. | best mean relative error **0.099** — it learns; the implementation is sound |

The second is load-bearing and its numbers are worth quoting. Given 8000 chains
(4× exp_07's largest budget), 40 000 steps (2× exp_07's), and a much easier
problem (n=8 rather than 32, Gaussian rather than Laplace, a *linear* target),
the best the network reaches is 0.099. EM-BP reaches **0.016** on the harder
Laplace problem at n=32 with 2048 chains. So the gap in §4.1 is not an artifact
of a crippled baseline.

The same run also confirms the parameterization complementarity the write-up
claims, and shows why it is structural rather than incidental: ε-prediction
recovers the mean as `(x − √Δ·ẑ)/α`, so it amplifies network error by `√Δ/α` —
negligible at low noise, but a factor of ~4.9 at t=1.6, where its error blows up
to 1.19 against x₀'s 0.156. At t=0.1 the ordering reverses (0.038 vs 0.076).
Reporting only one parameterization would have been misleading in whichever
direction it was chosen.

All five are permanent tests, not one-off scripts.

---

## 6. Limitations

1. **Inference cost is the honest loss.** Grid BP is `O(nM²)` per chain per evaluation; a network forward pass is a few matmuls. Measured slowdown **211×–320×** at M=401 (§4.4). Since reverse diffusion evaluates the denoiser at every integration step, this is not a detail. Distilling the fitted BP denoiser into a network, or using Layer-2 Gaussian-projected BP at inference, is the obvious next step — not something claimed here.
2. **EM is slower to train here.** 443 s at N=2048 vs ~37 s for the network, and EM's cost grows linearly in `N` while the network's is fixed by its step count. The win is in *data*, not in wall-clock.
3. **Everything rests on the chain assumption.** If the prior is not Markov the estimator is misspecified in a way no amount of data fixes. Layer 4 showed the correction is exactly rank one for AR(1)+global-latent priors, suggesting a hybrid — untested here.
4. **The baseline is a vanilla MLP, as specified.** A temporal convolution or U-Net would carry a locality prior of its own and should close part of the gap. What it would not acquire is the exactness of the risk decomposition or the uniformity in `t` of Prop. 5. This is the natural next comparison and is **not settled**.
5. **One replicate per N in the headline curve.** Its non-monotonicity across `N` is run-to-run noise, not structure. `hpc/slurm_replicates.sbatch` fixes this.
6. **One chain family, n = 32.** Linear AR(1) with non-Gaussian innovations. The claim is demonstrated in this setting, not in general.

---

## 7. Status

**Complete:** theory (14 pp), implementation, 30/30 tests, cluster support, and
**all of exp_06, exp_07, exp_08**. Every experiment ran to completion and its
outputs are committed.

**Established:** the E-step exactness and the `Ξ` compression (Props. 1–2, and
monotonicity violation of exactly 0 in every EM run across all experiments);
Fisher's identity as a substitute for autodiff (Prop. 3, verified to ~1e-9);
the information budget and its asymmetry between correlation and innovation
scale (§4.6); the headline sample-efficiency advantage and its capacity control
(§4.1–4.2); recovery of an unknown innovation law from noisy data alone (§4.8).

**Also established, after re-running at 12 replicates:** the `N^{−1/2}` rate
(§4.9, combined slope −0.500 ± 0.048). The original 4-replicate run was too
noisy to support it and said so; more replicates resolved it.

**Repaired:** the exp_06 Part 5 seeding confound has been fixed and re-run
(§4.10), which both sharpened the result and overturned a number I had reported
from the confounded version.

**Not started:** the items in §6.3, §6.4, and §6.1 — hybrid non-Markov
correction, structured-architecture baseline, and distillation of the BP
denoiser for inference-time cost.
