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
| `tests/test_em_bp.py` | 23 tests, including five independent cross-checks (§5.5). Package total is now **101**, all passing (`tests/test_hierarchy.py` adds 51 for Layer 6). |
| `hpc/`, `tools/merge_replicates.py` | Cluster templates and replicate merging |

**The four rungs**, all consuming the same `Ξ`:

1. **Gaussian AR(1)** `(ρ, q)` — closed-form M-step (belief-weighted Yule-Walker); the correctness harness, since everything is analytically available.
2. **Laplace AR(1)** `(ρ, b)` — closed form *despite* non-differentiability: profile out `b`, and maximizing over `ρ` becomes a weighted least-absolute-deviations regression whose exact solution is a weighted median of grid ratios.
3. **Mixture innovation** `(ρ, {π,μ,σ²}_c)` — the honest "we know it's Markov but not the model" estimator. Has never heard of the Laplace density it must recover. Closed-form ECM via an inner latent (the mixture label).
4. **Mixture-density network** — a small MLP emitting a state-dependent mixture. By Fisher's identity the network is evaluated and backpropagated **only at the `M` grid points**, once per EM iteration, regardless of dataset size.

---

## 4. Results

### 4.1 EM-BP vs a score network — the headline, with error bars

Six independent training seeds, extended to N=4096. The test set and its
exact-BP reference are held fixed across seeds, so replicates disagree on what
they learn from and agree on what they are judged against. Relative score error
averaged over the noise schedule, ±1 standard error over seeds:

| N chains | best network | **EM-BP** | ratio |
|---|---|---|---|
| 32 | 0.6363 ± 0.0043 | **0.0651 ± 0.0031** | 9.8× |
| 128 | 0.5117 ± 0.0025 | **0.0333 ± 0.0035** | 15.4× |
| 512 | 0.2813 ± 0.0013 | **0.0211 ± 0.0022** | 13.3× |
| 1024 | 0.2154 ± 0.0009 | **0.0182 ± 0.0021** | 11.8× |
| 2048 | 0.1756 ± 0.0009 | **0.0142 ± 0.0009** | 12.4× |
| 4096 | 0.1643 ± 0.0009 | **0.0124 ± 0.0005** | 13.2× |

Both curves are now monotone; the wobble in the original single-replicate run
was seed noise, as suspected. The ratio is stable at 10–15× across seven
doublings of the data.

**The data-efficiency statement, now with non-overlapping error bars.** EM-BP on
**32** chains (0.0651 ± 0.0031) beats the network on **4096** (0.1643 ± 0.0009)
by a factor of 2.5. So the network needs **more than 128× the data** to match
EM-BP, and that remains a lower bound — 32 was the smallest budget tried and the
network never caught up. Fitted exponents are −0.35 (EM-BP) and −0.32 (network);
extrapolating the network's own scaling puts the crossing near 10⁷ chains, but
that is a four-decade extrapolation and is offered as an order of magnitude, not
a claim.

⚠️ **This margin is against a *vanilla* MLP, and that matters — see §4.11.** A
locality-respecting architecture closes most of it.

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

### 4.11 A structured architecture closes most of the gap (exp_12)

The compendium has flagged from the start that the exp_07 baseline is a fully
connected MLP carrying no locality prior, and that a reviewer would ask what
happens against an architecture that encodes sequence structure. Measured.

A weight-shared window predictor — exactly a 1-D CNN with receptive-field radius
r — trained by the same denoising score matching, at N=1024, interior sites:

| | parameters | mean relative score error |
|---|---|---|
| fully connected MLP | 25,505 | 0.1535 |
| **local CNN (r=6)** | **5,313** | **0.0595** |
| **EM-BP** | **13** | **0.0182** |

**The CNN beats the MLP by 2.5× on average with a fifth of the parameters** (up
to 3.8× at t=0.4, converging at t=1.6 where the score is nearly −x and structure
stops helping).

#### The replicated, oracle-fair version — this is the number to quote

The single-seed figures above (3.3× at fixed r=6, 2.75× with an oracle over r)
were flagged as provisional. They have been replaced. **4 seeds, oracle over
both the receptive-field radius and the parameterization for both networks**,
mean relative score error over five noise levels, ± standard error over seeds:

| N | EM-BP (13 params) | global MLP (25,505) | local CNN, oracle r (6,081) |
|---|---|---|---|
| 128 | **0.0368 ± 0.0067** | 0.3595 ± 0.0044 | 0.0774 ± 0.0024 |
| 512 | **0.0174 ± 0.0017** | 0.1794 ± 0.0035 | 0.0570 ± 0.0007 |
| 2048 | **0.0123 ± 0.0009** | 0.1246 ± 0.0027 | 0.0505 ± 0.0015 |

As ratios to EM-BP:

| N | vs global MLP | vs local CNN |
|---|---|---|
| 128 | 9.76 ± 1.79 | 2.10 ± 0.39 |
| 512 | 10.29 ± 1.02 | 3.27 ± 0.32 |
| 2048 | 10.15 ± 0.75 | **4.11 ± 0.32** |

Two things this settles, one in each direction.

**The structured architecture really does close most of the gap.** Against the
vanilla MLP the margin is ~10× and flat in `N`; against the locality-respecting
CNN it is 2–4×. Roughly 60% of the vanilla deficit was the architecture, not the
estimator, exactly as the provisional run suggested.

**But the remaining margin *grows* with data rather than closing.** 2.10 → 3.27
→ 4.11 as `N` goes 128 → 512 → 2048, a trend well outside the standard errors.
The mechanism is visible in the columns: EM-BP falls 0.0368 → 0.0123 (a factor
3.0 over 16× data) while the CNN falls 0.0774 → 0.0505 (a factor 1.5) and is
visibly flattening. The structured network converges to a floor set by its
approximation error; the estimator keeps paying down parametric error. **"More
data will close the gap" is the natural guess and it is wrong here.**

So the honest headline against a well-chosen architecture is **4.1× at
N = 2048 and rising**, not the 8.4× of the vanilla comparison and not the 2.75×
the single-seed oracle run suggested. Both earlier numbers were quoted at
N = 1024, between the second and third rows above.

### 4.12 The optimal receptive field is finite, and exceeding it hurts (exp_12)

B15 read the Gaussian locality law as "a local head is near-optimal once
r ≳ ξ log(1/ε)" — a *lower* bound on the radius. The sweep says something
sharper: the error is **U-shaped in r**, so there is an optimum and overshooting
it is costly.

Interior relative score error for the Gaussian chain at t=0.1:

| r | 1 | 2 | **3** | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| error | 0.229 | 0.100 | **0.065** | 0.087 | 0.094 | 0.125 | 0.125 |

Doubling the radius past the optimum roughly doubles the error. The mechanism is
an ordinary bias-variance trade: too small a window truncates genuine
dependence, too large a window spends the same data estimating more input
dimensions. B15's prescription is therefore only half the story.

**The optimum grows with noise level**, which the locality law predicts (the
decay rate q rises with t, so the error falls off more slowly in r):

| family | t=0.1 | t=0.2 | t=0.4 | t=0.8 | t=1.6 |
|---|---|---|---|---|---|
| gaussian | 3 | 3 | 6 | 12 | 16 |
| laplace | 3 | 4 | 8 | 12 | 16 |
| gauss_mix κ=0.9 | 3 | 4 | 8 | 12 | 12 |

And the non-Gaussian families need a *wider* field than the Gaussian at matched
(ρ, t) — 4 against 3 at t=0.2, 8 against 6 at t=0.4 — which is the direction
exp_11's locality rates predicted, at roughly the predicted 1.1–1.3× magnitude.
The sweep's granularity in r limits how sharply that can be confirmed.

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

## 4.13 Layer 6 — hierarchical priors and the two diffusion time scales

Prompted by the two papers the project was pointed at (see
`docs/PAPER_CONNECTIONS.md`, which records **up front that neither PDF could be
opened from this environment** and that nothing mathematical is quoted from
them). New code: `src/hierarchy.py`, `src/spectral.py`, `exp_13`, `exp_14`,
51 new tests (suite now 101).

### The speciation ladder (exp_13 `spectra`, `cascade`)

A hierarchical prior has one covariance eigenvalue per level, so it has a
*ladder* of speciation times rather than one. Derived in this package's
convention, the crossover along a mode of variance `Λ` is
`t_S = ½ log(1 + Λ)`. Measured on a depth-5 binary tree, ρ = 0.9, 32 leaves —
predicted against the crossing of the commitment curve through `1/√2`, along
the forward process and along the reverse SDE driven by the exact tree-BP
score:

| level | Λ | `t_S` predicted | forward | reverse SDE |
|---|---|---|---|---|
| −1 (whole tree) | 14.271 | 1.363 | 1.368 | 1.336 |
| 0 | 3.113 | 0.707 | 0.732 | 0.725 |
| 1 | 1.804 | 0.516 | 0.521 | 0.520 |
| 2 | 0.996 | 0.346 | 0.340 | 0.358 |
| 3 | 0.498 | 0.202 | 0.208 | 0.222 |
| 4 (sibling leaves) | 0.190 | 0.087 | 0.087 | 0.125 |

Six distinct transitions spanning **15.7× in time within one dataset**, each at
its predicted place: Pearson r = **0.9998** for both columns, with the forward
crossings agreeing to **≤ 3.5%** at every level.

The reverse column's deviations are −2%, +3%, +1%, +4%, +10%, **+43%** from
coarsest to finest. That the error grows monotonically toward the fine end is
the signature of the integrator rather than of the prediction: the finest level
speciates at t = 0.087, within a factor of four of the sampler's `t_min = 0.02`,
where the geometric time grid is coarsest relative to the scale that matters and
the drift stiffens as `α/Δ ~ 1/(2t)`. Four of the six levels agree to ≤ 4%.
`hpc/slurm_layer6.sbatch` runs this part at `n_steps_sde = 800` and
`t_min = 0.005`; if the finest level does not move, that is worth knowing and
the diagnosis above is wrong.

**The reverse diffusion resolves the hierarchy coarse-to-fine, one transition
per level.**

### A chain has no cascade (exp_13 `spectra`)

The same analysis says the chain this project has been studying is *outside*
that regime, which is worth knowing before generalizing from it. The AR(1)
spectrum is bounded by `(1+ρ)/(1−ρ)` at any length:

| ρ | top eigenvalue, n = 8 → 512 | limit | `t_S` saturates at |
|---|---|---|---|
| 0.50 | 2.57 → 3.00 | 3.00 | 0.693 |
| 0.85 | 5.49 → 12.32 | 12.33 | 1.295 |
| 0.95 | 7.03 → 38.52 | 39.00 | 1.844 |

64× the length moves the top mode by under 2×. A tree at ρ = 0.9 instead goes
3.12 → 23.31 over depths 2–6, with its levels spanning `t_S ∈ [0.087, 1.595]`.
**A stationary Markov chain has one saturating time scale; only the hierarchy
produces the diverging speciation time and the coarse-to-fine cascade.**

### Memorization: the axis BP does not have (exp_14 `collapse`)

The collapse mechanism is a statement about the *empirical* score, whose
sufficient statistic is the training set itself. A BP score's sufficient
statistic is the fitted kernel — two numbers, independent of `n` — reached only
through `Ξ`, which is an average. So the prediction is not that BP memorizes
less but that **the axis does not exist for it**.

Measured as nearest-training-chain distance of generated samples, divided by
the same quantity for a genuinely fresh draw from the true prior (1.0 = no
memorization, 0 = full collapse), ρ = 0.85, complete sweep:

| n | N | wall `e^{ns}` | empirical | DSM (ε) | EM-BP | true-prior BP |
|---|---|---|---|---|---|---|
| 8 | 64 | 88.8 | **0.457** | 0.800 | 1.079 | 1.076 |
| 8 | 256 | 88.8 | **0.540** | 0.929 | 1.058 | 1.044 |
| 8 | 1024 | 88.8 | **0.660** | 0.980 | 1.015 | 1.012 |
| 16 | 64 | 1.5e4 | **0.332** | 0.751 | 0.972 | 0.992 |
| 16 | 256 | 1.5e4 | **0.387** | 0.906 | 1.040 | 1.026 |
| 16 | 1024 | 1.5e4 | **0.456** | 1.066 | 1.066 | 1.072 |
| 32 | 64 | 4.3e8 | **0.268** | 0.790 | 1.021 | 1.024 |
| 32 | 256 | 4.3e8 | **0.308** | 1.053 | 1.039 | 1.024 |
| 32 | 1024 | 4.3e8 | **0.335** | 1.115 | 1.023 | 1.025 |

Both dependences come out in the predicted direction. At fixed `N` the
empirical score memorizes *more* as the chain gets longer (0.457 → 0.332 →
0.268 at N = 64), which is the curse of dimensionality; at fixed `n` it
recovers as `N` grows (0.268 → 0.335 at n = 32). **EM-BP stays in
[0.97, 1.08] in every one of the nine cells, agreeing with the true-prior BP
reference to ≤ 0.03** — its behaviour does not depend on `n` or `N` at all,
which is the claim.

**The DSM column must be read with its standard deviation, not alone.** At
n = 32 the network's sample std is 1.19–1.27, i.e. over-dispersed by 19–27%, so
its ratios above 1 there are inflated by generating too broadly rather than by
generalizing better; at n = 8 and 16, where its std is 0.90–1.15, its ratios
(0.75–1.07) sit below BP's. This is exactly why `sample_std` and `lag1_corr`
are logged in every row.

The per-site excess entropy is exact for this family, `s = −½ log(1 − ρ²)`:
**0.641 nats/site at ρ = 0.85**, so n = 33 already demands ~10⁹ chains. That is
the quantitative form of the claim that Layer 5's data advantage is structural.

### The closed-form collapse time works (exp_14 `time`)

The criterion `n·s(t_C) = log N`, with `s(t) = −(1/2n) log det(α_t²C + Δ_t I)`
and **no fitted constants**, was tested against the measured collapse of the
empirical score's own weights (the time at which their entropy falls to
½ log N), over `n ∈ {8,16,32}` × `N ∈ {32,128,512,2048}`:

- **Pearson r = +0.990** over the 9 settings where the criterion predicts a
  finite-time collapse (r = +0.995 against the same measurement made on fresh
  rather than training chains).
- The relation is **affine, not proportional**: `measured = 1.716 × predicted +
  0.276`, residual sd **0.029**. Forcing it through the origin gives a residual
  sd of 0.131, 4.5× worse.
- Independently, `dt_C/d log N` is −0.039 predicted vs −0.061 measured at
  n = 16 and −0.051 vs −0.079 at n = 32 — **the same 1.55× slope factor at both
  chain lengths**, consistent with the affine fit.

So the criterion gets the ordering, both dependences, and the `N`-scaling to
within a constant factor; it does not get the absolute time. That is what
should be expected — it is a leading-order statement, and the "half of log N"
landmark is a choice with no claim to mark the exact transition. **The offset is
reported, not absorbed into a fitted constant.**

**Where it does not apply, stated plainly.** At n = 8 with N ≥ 128 the criterion
returns *no finite-time collapse* (total excess entropy 4.487 nats against
log 128 = 4.852), yet collapse is measured at t = 0.234, 0.170, 0.122 for
N = 128, 512, 2048. There is no contradiction: the criterion is asymptotic in
`n`, and a finite system always collapses eventually as `t → 0`. "No collapse"
means "collapse only at vanishing `t`", and at n = 8 that is not a good
approximation. Those three rows are excluded from the correlation above and
listed here rather than dropped.

### Caveat carried with the collapse numbers

The ratio is only interpretable when the generated distribution is otherwise
correct, so sample standard deviation and lag-1 correlation are logged beside
it in every row. In the quick configuration the undertrained network reached
ratios of 2–3.4 with sample std 2.3–3.3 — that is over-dispersion, not
generalization, and would be a misreading if the ratio were quoted alone.

### Where each method's error lives in the hierarchy (exp_13 `levels`)

Depth-4 binary tree, ρ = 0.9, 16 leaves, N = 256 trees, both network
parameterizations trained. First, the totals — relative error of the posterior
mean against exact tree BP:

| t | EM-BP | best network | ratio |
|---|---|---|---|
| 0.1 | 0.0098 | 0.1852 (ε) | 18.9× |
| 0.2 | 0.0125 | 0.2245 (ε) | 18.0× |
| 0.4 | 0.0167 | 0.2489 (x₀) | 14.9× |
| 0.8 | 0.0334 | 0.2587 (x₀) | 7.8× |
| 1.6 | 0.0815 | 0.3332 (x₀) | 4.1× |

(`bp_exact_grid`, i.e. grid tree BP given the true kernel, is 0.0000 at every
`t` — the validation that the grid path reproduces the information form.)

Second, and more interesting: *where* the error sits. The natural null is a
method whose per-mode error is uniform, which would put a share of the total
squared error on each level proportional to its multiplicity — 1, 1, 2, 4, 8
out of 16 here. Dividing the measured share by that null gives a concentration
ratio, and its spread across levels says how much a method's error knows about
the hierarchy:

| method | spread of concentration ratio across levels (min → max over `t`) |
|---|---|
| network, ε-prediction | **1.6× – 2.2×** |
| network, x₀-prediction | 2.8× – 12.6× |
| EM-BP | **224× – 6957×** |

**The ε-network distributes its error uniformly over the hierarchy at every
noise level, as if the levels did not exist.** EM-BP's error is concentrated by
two to four orders of magnitude, and it *migrates*: at t = 0.1 it sits on the
finest levels (ratio 1.44 at level 3, 0.00 at level 0), and by t = 1.6 all of
it is on the single coarsest mode (15.8 at level −1, ≤ 0.02 everywhere else).
That is the estimator's structure showing up in its residual — at large `t`
only the top mode survives the noise, so the kernel misestimate (ρ̂ = 0.9161
against 0.9) has nowhere else to appear.

The x₀-network is genuinely intermediate (up to 12.6× at large `t`), which is
worth stating because it is the better parameterization there; the "no
structure at all" reading applies to ε-prediction, not to every network.

### Genuine symmetry breaking, and where it sits (exp_13 `speciation`)

Everything above uses a Gaussian prior, which has no class to speciate *into* —
`commitment` locates an information crossover, sharing the criterion but not the
phenomenon. Giving the root a symmetric two-component prior with the same unit
variance fixes that as a **controlled comparison**: the leaf covariance, every
eigenvalue and every predicted time are bit-identical (asserted in
`test_mixture_root_leaves_the_covariance_and_the_ladder_untouched`), and only
the modality moves. The order parameter is the exact class posterior
`P(root > 0 | x_t)`, which BP returns from the upward pass alone.

Depth-4 tree, ρ = 0.9, μ = 0.9, 256 paths, `t_S` predicted 1.136:

| root | correlation crossing | class-choice time | final magnetization |
|---|---|---|---|
| two-component | 1.155 (+1.7%) | **1.265** | 0.828 |
| Gaussian | 1.031 (−9.2%) | 0.940 | 0.647 |

Two things. First, the covariance-set crossing is where it should be for both
roots — 1.155 and 1.031 bracket the predicted 1.136, within the ±10% that the
integrator costs at this resolution (this part runs grid BP at 120 SDE steps
against the `cascade` part's 200, and the −9% is the same integrator signature
seen there). **Adding modality does not move it**, which is the control working.

Second, the class choice for the two-component root happens at **t = 1.265,
i.e. earlier in the generation than the information crossover at 1.136**, and
35% earlier than the corresponding time for the Gaussian root. That is the
expected physics — two well-separated classes can be told apart while noise
still swamps the within-class detail — and it is what makes this speciation
rather than a crossover.

**Precision, stated:** 256 paths give a binomial standard error of 0.031 on the
agreement curve, and the integrator contributes the ±10% above. The 1.265 vs
0.940 gap is far outside both; the +1.7% agreement of the bimodal correlation
crossing is *inside* them and should not be read as better than the −9.2%.

### Block-independent truncation — NOT the paper's filtering (exp_13 `block_independent`)

⚠️ **Reading arXiv:2408.15138 in full showed this part implements a different
construction from the paper's.** Their filtering draws the depth-`k` nodes
conditionally independently *given the root*, so blocks stay correlated through
it; this sweep makes the blocks outright independent. The correct construction
is now `GaussianTree(filter_level=k)` and the experiment built on it is
`exp_15`. What follows is still a valid measurement of a harsher truncation,
under its corrected name.

Making blocks independent at level `k`, at fixed sequence length (16), fixed
data budget (256 sequences), fixed network and fixed training budget:

| k | block | ρ̂ | EM-BP abs err | network abs err | advantage |
|---|---|---|---|---|---|
| 1 | 2 | 0.8796 | 0.0154 | 0.4143 | 28.7× |
| 2 | 4 | 0.8926 | 0.0085 | 0.3702 | 49.2× |
| 3 | 8 | 0.9010 | 0.0038 | 0.3640 | **99.0×** |
| 4 | 16 | 0.9041 | 0.0084 | 0.3470 | 39.4× |

**The network's error is essentially flat in `k`** (0.414 → 0.347, mildly
*decreasing*): more long-range structure in the data neither helps nor hurts
it, which is another reading of the same blindness §4.13's level analysis found.
The EM-BP advantage is 29–99× throughout and does not decay with range.

**A confound, which is the honest limit of this design.** Holding the number of
*sequences* fixed does not hold the number of *edges* fixed: a k-filtered
depth-4 tree has `32 − 32/2^k` edges per sequence, i.e. 16, 24, 28, 30 for
k = 1…4. So larger `k` hands EM more data for the same nominal budget, and ρ̂
duly improves monotonically (0.8796 → 0.9041). The sweep therefore moves two
things at once, and the rising advantage cannot be attributed to correlation
range alone. Isolating it needs a version that fixes the edge count rather than
the sequence count — not done. What the table does support without qualification
is the negative: **the advantage does not shrink as correlations reach further**,
which is what one would have guessed if the network's locality were the
binding constraint.

### The network implements the oracle matched to its training data (exp_15)

The measurement the transformer paper actually uses, transplanted: train a
denoiser on data filtered at `k_train`, test it on *unfiltered* data, and ask
which member of the exact-oracle family `BP_k` its posterior mean is closest to.

| `k_train` | argmin `k` at t = 0.1 / 0.2 / 0.4 / 0.8 | margin over runner-up |
|---|---|---|
| 0 | 0 / 0 / 0 / 0 | — |
| 1 | 0 / 0 / 0 / 0 | *exactly degenerate, see below* |
| 2 | **2 / 2 / 2 / 2** | +1.8 / +4.1 / +4.3 / +1.6 % |
| 3 | **3 / 3 / 3 / 3** | +17.8 / +29.0 / +24.3 / +5.9 % |
| 4 | **4 / 4 / 4 / 4** | +63.3 / +60.4 / +35.3 / +6.2 % |

**16 of 16 cells** pick the oracle matched to the training distribution rather
than the one matched to the test data, and the margin grows with `k_train`.

Two exclusions, both explained:

- **`k_train = 1` is exactly degenerate with `k_train = 0` in a Gaussian tree** —
  `dist_k0` and `dist_k1` agree to five decimals (gap 0.00%). This is a genuine
  limitation of the Gaussian analogue: the paper's transition *tensor* can
  correlate the two children given the parent, while a linear-Gaussian edge
  cannot, so siblings are conditionally independent by construction and
  filtering at `k = 1` changes nothing. **The one case where their filtering
  alters probabilities without altering topology is unrepresentable here.**
- **`t = 1.6` is excluded** because the network's distance to *every* oracle is
  ~1.50 against an inter-oracle spread of 0.22 — the argmin is reading network
  error, not implied correlation range. `oracle_spread` is in every row so this
  is checkable rather than asserted.

**Sequential acquisition (exp_15 `alignment`): a weak version, not a
reproduction.** Training on unfiltered data, the argmin moves in the predicted
direction at two of five noise levels — `2 → 2 → 0` at t = 0.4, `2 → 1 → 1` at
t = 0.8 — over the first ~1000 gradient steps, then saturates for the remaining
31 000. Nothing like the paper's staircase. The likely reasons are that the
Gaussian oracles are far less distinguishable than their discrete ones, that a
denoising regression loss converges far faster than MLM, and that the
architecture differs. **This setting is probably too easy to resolve the
phenomenon** — which is a different statement from the phenomenon being absent.

### A non-monotonicity that is not a bug (exp_13 `ordering`)

EM's *error against the truth* is non-monotone in the iteration count even
though its log-likelihood is monotone: total relative error 0.0729 → 0.0083 →
0.0162 at 32 → 64 → 128 iterations, with ρ̂ = 0.8708 → 0.9068 → 0.9158 against
a true 0.9. There is nothing wrong. EM converges to the **maximum likelihood
estimate**, which at N = 256 trees is ρ̂ = 0.9161, not to the true parameter;
the trajectory happens to pass close to 0.9 on the way. Reading the minimum at
64 iterations as "the right stopping point" would be fitting the answer.
`monotone_violation` is exactly 0 throughout, which is the quantity that would
actually signal a broken M-step.

### Tree EM converges, but slowly

`ρ̂ = 0.7360 → 0.7483 → 0.7487` at 50/100/150 iterations (true 0.75), **zero
monotone violation throughout**, and `q̂ = 0.4418` against 0.4375. Internal
nodes are never observed, so the missing-information fraction is far larger
than on a chain and EM's linear rate is correspondingly slower. An estimate
read at 40 iterations looks like a broken M-step and is merely unconverged.

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

**Complete:** theory (14 pp), implementation, **101/101 tests**, cluster
support, and **all of exp_06 – exp_15**. Every experiment ran to completion and its
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

**Settled since, and it moved a headline:** the structured-architecture baseline
(§6.4) is done. Against a locality-respecting CNN with an oracle over radius and
parameterization, over 4 seeds, the margin is 2.10 → 3.27 → **4.11 ± 0.32** at
N = 128 → 512 → 2048, against ~10× for the vanilla MLP. About 60% of the vanilla
deficit was the architecture — and the remainder *widens* with data rather than
closing (§4.11).

**Layer 6 (§4.13), following the two advisor papers:** a hierarchical prior has
a ladder of speciation times, one per level, measured at r = 0.9998 against
prediction; the AR(1) chain of Layers 1–5 has none, because its top eigenvalue
is bounded — which bounds how far this project generalizes; the memorization
axis does not exist for a BP score, which supplies the *mechanism* behind the
Layer-5 headline; and the ε-network's error is flat across hierarchy levels
while EM-BP's is concentrated by 2–4 orders of magnitude. `exp_13 ordering`
returned a **negative** — no sequential acquisition of levels by the network —
in a regime that is data-limited, and it is recorded as such.

**Still not started:** §6.3 (hybrid non-Markov correction) and §6.1
(distillation of the BP denoiser for inference-time cost). For Layer 6 the
ranked list is in `docs/PAPER_CONNECTIONS.md` §6, headed by reading the two
PDFs — which **could not be opened from this environment** — and by the
nonparametric collapse test that answers the obvious objection to the
memorization result.
