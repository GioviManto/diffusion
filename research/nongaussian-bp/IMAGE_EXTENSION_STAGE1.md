# Exact BP diffusion on real images: the wavelet-tree route

Giovanni Mantovani — 8 August 2026. Companion to `paper/main.pdf`,
`ANSWERS_AND_QUESTIONS_FOR_ADVISORS.md`.

This document reports a working Stage-1 substrate, the measurements that gate it,
and one measurement that **changes the plan**. It separates, as the advisor
document does, what is established, what is measured, what is interpretation,
and what is unresolved.

---

## 1. Summary

The obstacle was that the project's central claim — the posterior of a
coordinatewise-noised Markov chain is still a chain, hence a tree, hence
sum-product returns the **exact** score — dies on images if one uses the pixel
lattice, which has loops.

A multiscale wavelet decomposition restores it. What is now built and verified:

- an orthonormal 2-D Haar transform and the quadtree it induces (`src/wavelet.py`);
- exact BP on that quadtree with an observation at **every** node, a **per-scale**
  noise level, and **one kernel per scale** (`src/wavelet_bp.py`);
- the assembly into a fittable image model with an exact held-out likelihood
  (`src/wavelet_model.py`);
- a scale-mixture kernel with a parent-dependent conditional scale (`src/scale_kernel.py`);
- exact BP on a temporal chain of spatial trees (`src/video_bp.py`);
- per-depth grids, so the score is resolved down to t = 0.05 (§6.2.1);
- 69 tests, including agreement with a dense Gaussian solve to 2·10⁻¹⁵;
- the gating measurement on real CIFAR-10 (`experiments/exp_23_wavelet_statistics.py`).

**The gate passes decisively.** CIFAR-10 wavelet coefficients are strongly
non-Gaussian — the finest subbands have excess kurtosis 5.8–7.1, i.e. *heavier
tailed than the Laplace chain* (3.0) the project has been simulating.

**But one measurement redirected the modelling.** The dependence between parent
and child coefficients is largely in the **magnitude**, not the mean, beyond what
the linear effect explains, and the linear-autoregressive kernel family in
`src/kernels.py` cannot represent that (§6). `src/scale_kernel.py` now supplies a
family that can (§6.1), and generated samples confirm the diagnosis: the
linear-AR family generates essentially **zero** cross-scale magnitude dependence.

**One limit blocked half the story, and is now fixed.** Per-subband
standardisation makes the coarse subbands' likelihood narrower than a grid cell
at small *t*, so on a shared grid nothing below t ≈ 1.0 was valid — which
invalidated my own first likelihood numbers (§6.2) and blocked reverse
diffusion outright (§6.3). The fix is a **per-depth grid**, now implemented
(§6.2.1): points are spent where the likelihood is narrow, which is where there
are fewest nodes, so resolving to t = 0.05 costs *less* than the old uniform
grid that reached only t = 0.90. Measured `min_resolved_t`: **0.90 → 0.0499**.
The results in §6.2 and §6.3 were measured before this landed and are labelled
accordingly; re-running them at small *t* is the first thing to do next.

**Video is built, not just argued** (§9). The fully coupled spatio-temporal model
has 4-cycles and is *not* exact; a temporal chain of spatial trees — a
caterpillar — is, and BP on it matches a dense Gaussian solve to 10⁻⁸ using only
passes the package already had. It is fitted by EM, generates video, and
temporal coupling measurably helps.

**And the video result is a ceiling, which is the most useful thing here** (§9.2).
At most **one temporal edge per connected component** is loop-free — a second
always closes a 4-cycle — and the wavelet frame has exactly four components. So
**at most 4 of 1024 coefficients per frame can be temporally coupled in any exact
model of this kind**, carrying 47 % of the variance. The caterpillar already
couples all four and the best four, so it is optimal, and its coherence lands
~5× worse than real video. That turns "can we do video exactly?" (yes, with this
ceiling) into the sharper question of what *approximate* BP on the fully coupled
graph buys against exact BP on the best tree.

---

## 2. Why the exactness survives, and why it is not a trick

The forward process is, coordinatewise in pixel space,

    x_t = α_t a + √Δ_t z,   z ~ N(0, I).

Let `W` be the orthonormal Haar analysis operator. Then

    W x_t = α_t (W a) + √Δ_t (W z),   with  W z ~ N(0, I),

because an orthonormal map sends an isotropic Gaussian to an isotropic Gaussian.
So **the noising process in wavelet space is the same coordinatewise OU process,
with the same α_t and the same Δ_t.** Nothing is approximated by the change of
basis. The per-site Gaussian likelihood factor — the thing that makes the
posterior factor graph equal the prior graph — survives verbatim, and

    E[a | x_t] = Wᵀ E[Wa | W x_t],    score_pixel = Wᵀ score_wavelet.

This is why the transform has to be **orthonormal** rather than merely
invertible, and it is checked numerically (`W Wᵀ − I` < 10⁻¹², Parseval to
10⁻¹¹) rather than asserted. It is also what makes the held-out likelihood
comparable across models working in different bases: an orthonormal transform
has unit Jacobian determinant and contributes nothing to the log density.

---

## 3. A correction to the proposed geometry

The brief proposed "branching 4, depth 5". That is not the wavelet tree.

A depth-5 quadtree has 4⁵ = 1024 leaves — one per *pixel*. That is a quadtree of
the image, not of its wavelet coefficients.

The correct object (Crouse–Nowak–Baraniuk, 1998) is built **within an
orientation**. A 5-level decomposition of a 32×32 image gives one LL scaling
coefficient plus, at each scale, three detail subbands HL/LH/HH of sides
1, 2, 4, 8, 16. A coefficient at (m,n) has four children at (2m,2n)…(2m+1,2n+1)
in the *same orientation*. So the structure is

    three quadtrees, branching 4, **depth 4**, 341 nodes each,  plus one scalar
    3 × 341 + 1 = 1024 = 32 × 32     — every coefficient once, none twice.

Three disjoint trees plus an isolated scalar is a forest: still loop-free, still
exact. `WaveletQuadtree` refuses a partial decomposition, because leaving an
m×m coarsest band with m > 1 gives m² roots per orientation and the indexing is
no longer `TreeIndex`'s.

---

## 4. What is verified

Every number below is a test in `tests/test_wavelet*.py`, run on this machine.

| Check | Result |
|---|---|
| Haar forward→inverse, random and structured images | < 10⁻¹³ |
| Orthonormality `W Wᵀ − I`, Parseval | < 10⁻¹², 10⁻¹¹ |
| Spatial child map ≡ `TreeIndex` breadth-first rule | exact, all levels |
| **Tree BP posterior mean vs dense Gaussian solve** | **2.0 · 10⁻¹⁵** |
| Tree BP log-evidence vs closed form | rel. < 10⁻⁸ |
| Per-scale Δ spanning 0.02 … 20 | < 10⁻⁷ |
| Reduces to validated `hierarchy.tree_bp_grid` | < 10⁻¹¹ |
| Per-level Ξ mass = number of edges at that level | rel. < 10⁻⁶ |
| **Gaussian control**: per-level ρ recovered (0.75/0.60/0.45) | max error < 0.05 |
| EM monotone ascent (tree, per-level kernels) | violation 0 |

> On the innovation variance `q`, the control is looser than on ρ and the reason
> was checked rather than assumed. Refining the grid from M = 241 to M = 481
> leaves the fitted `q` **bit-identical**, so the residual (0.099 at 400 images,
> 0.077 at 1600) is not discretisation. It shrinks with data but more slowly than
> Monte Carlo alone would predict, and the fitted values sit between the truth
> and the initialisation — consistent with incomplete EM convergence at ten
> iterations, and with the project's own finding that Fisher information for the
> innovation *variance* falls 142× under the channel against 26× for the
> correlation. Not fully attributed; the tolerance reflects that.

The dense-solve check is the load-bearing one: it builds the prior covariance
from the generative recursion itself (`a = M ε` by forward substitution), so an
index or sign error in the tree bookkeeping cannot cancel against the same error
in the reference.

> **One diagnostic finding, not a bug.** An early version of the reduction test
> disagreed with `hierarchy.tree_bp_grid` by 3.8·10⁻⁷ and looked like a defect in
> one of the two. It is neither. Making internal nodes "unobserved" by setting
> Δ = 10¹² still draws x ~ 10⁶ there, and the likelihood's linear term
> 2αxu/Δ ≈ 10⁻⁶ is then a weak exponential tilt across the grid, not a flat
> factor. `hierarchy.tree_bp_grid` is exact to 10⁻¹⁵ against a dense solve when
> the model matches. Both implementations are correct.

---

## 5. The gate: CIFAR-10 coefficients are strongly non-Gaussian

10 000 training images, luminance (BT.601), globally standardised; bootstrap SE
over images. `outputs/exp_23_wavelet_statistics/subbands.csv`.

Excess kurtosis (0 = Gaussian; Laplace = 3.0), by tree depth (0 coarsest,
4 finest):

| orientation | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| HL | 2.36 | 1.62 | 3.47 | 5.31 | **7.09** |
| LH | 0.26 | 0.67 | 2.27 | 4.11 | **5.82** |
| HH | 1.96 | 1.22 | 2.69 | 4.36 | **6.42** |

Bootstrap SEs are 0.04–0.22, so the *weakest* subband sits 4.4 SE from Gaussian
and the finest sit 95–105 SE away. The LL coefficient, an average of 1024
pixels, is nearly Gaussian (0.363 ± 0.052), exactly as it should be.

**Verdict: GO.** Kurtosis rises monotonically toward fine scales, and the finest
subbands are *heavier-tailed than the Laplace chain the project simulates*. The
premise does not merely survive contact with real data; real data is a more
demanding case than the synthetic one.

Subband standard deviations span 16.47 (LL) to 0.141 (HH, finest) — a factor of
117. This is why per-subband standardisation with a per-depth
Δ_d = Δ_t / s_d² is necessary; no single grid resolves the finest while covering
the coarsest.

---

### 5.1 The pipeline runs end to end on real images

`experiments/exp_24_wavelet_fit.py`. Gaussian tree (the second-order closure,
baseline (i) of the plan), 500 training images, one noise level, 8 EM iterations,
grid M = 161, evaluated on 200 held-out test images with the normalisation and
the subband scales taken from training only:

| quantity | value |
|---|---|
| EM monotone violation | **0** |
| fitted per-level ρ (depths 0→3) | 0.209, 0.343, 0.349, 0.236 |
| held-out denoising MSE, t = 0.5 | **0.204** |
| same, raw observation x/α | 1.724 |
| held-out exact log-likelihood | −1306.3 per image |

So the exact score is being computed on real photographs and it denoises them
8.5× better than the rescaled observation. This is a *baseline*, not the claim —
it is the Gaussian closure, and the whole point of the project is what beats it.

Two honest caveats. The evidence was still climbing at iteration 8, so these
parameters are not converged. And the fitted ρ are lower than the empirical
correlations of §6 because orientations are tied by default, so one ρ is averaging
HL and LH (≈0.45) against HH (≈0.15).

---

## 6. The finding that changes the plan: the kernel family is wrong

The plan was to reuse `MixtureInnovationKernel` per scale. That family is
linear-autoregressive,

    K(a′ | a) = φ(a′ − ρ a),

so the parent's entire influence on the child is a **shift of the innovation's
location**. Measured on CIFAR
(`outputs/exp_23_wavelet_statistics/crossscale.csv`), at the finest scale
boundary (d3 → d4, 2.56 M pairs per orientation):

| orientation | linear corr | Q4/Q1 std ratio | linear-AR null | **excess** |
|---|---|---|---|---|
| HL | 0.452 | 3.65 | 1.32 | **2.33** |
| LH | 0.482 | 3.29 | 1.36 | **1.92** |
| HH | **0.148** | 2.89 | **1.03** | **1.86** |

> **The null matters, and an earlier draft of this document omitted it.** The raw
> ratio is not a pure measure of magnitude dependence: conditioning the child on
> a *set* of parent values also picks up the spread of the conditional mean ρa
> across that set, so a perfectly homoscedastic AR(1) already scores 1.31 at
> ρ = 0.45 and 1.61 at ρ = 0.60. The quantity that carries the argument is the
> **excess** over that null (`scale_kernel.linear_ar_magnitude_ratio`, verified
> against simulation to three decimals). The conclusion is unchanged and in fact
> sharper: in HH, where the linear correlation is only 0.15, essentially the
> entire ratio is excess.

Two things follow.

1. **Magnitude dependence is large, universal, and not explained by the linear
   effect.** The excess reaches 1.9–2.3 at the finest boundary in every
   orientation and grows monotonically with depth (median excess 0.82 over all
   twelve scale boundaries). A linear-AR kernel cannot express any of it: its
   conditional variance does not depend on the parent.
2. **HH has almost no linear correlation** (0.04–0.15) but full magnitude
   dependence. Fitted with a linear-AR kernel, HH would return ρ ≈ 0 and collapse
   to a *factorised* heavy-tailed model — good marginals, **zero hierarchy**. It
   would score well on per-subband statistics while capturing none of the
   cross-scale structure that is the entire point.

Controls run before claiming this: white-noise images give correlation ±0.005 and
magnitude ratio 1.000 across all levels; and recomputing the statistic by direct
spatial indexing, bypassing the Morton packing entirely, reproduces the numbers
exactly (0.3815/0.4524 and 1.822/3.647). The dependence is real.

**Consequence.** The kernel needs a parent-dependent *scale*, e.g. the
Gaussian scale-mixture
`K(a′|a) = Σ_c π_c(|a|) N(a′; ρ_c a, σ_c²)`, which is exactly the
Crouse–Nowak–Baraniuk hidden-state structure in continuous form. This is a new
M-step, not a reuse. The good news is that it changes **nothing** in the
inference layer: `wavelet_tree_bp` consumes an arbitrary (M, M) log-kernel and
the E-step returns per-level Ξ, which is all any M-step needs.

*Unresolved:* the linear correlation of 0.18–0.48 for HL/LH is larger than the
classical literature would suggest. That is likely specific to Haar, whose short
support makes edges sign-coherent across scales; a smoother orthogonal wavelet
(Daubechies-4) would probably show less. Not measured. It does not affect the
conclusion, since the magnitude effect dominates and HH has both.

---

### 6.1 The kernel that fixes it, and generation

`src/scale_kernel.py` implements the family the measurement calls for:

    K(a′|a) = Σ_c w_c(a) N(a′; ρ_c a, σ_c²),   w_c(a) = softmax_c(β_c + γ_c a²)

The quadratic logit is not an arbitrary choice: `softmax_c(β_c + γ_c a²)` with
γ_c = −1/(2τ_c²) **is** the responsibility of a zero-mean Gaussian scale mixture,
so the gate is exactly the form the Crouse–Nowak–Baraniuk story implies, at 4C
parameters per level. The M-step is ECM — closed form for ρ_c and σ_c², concave
ascent with a backtracking line search for the gate, so monotone ascent is
preserved rather than hoped for (the same ECM framing the project already uses
for `MixtureInnovationKernel`).

On synthetic data with a parent-magnitude-dependent spread it recovers a Q4/Q1
ratio of 3.33 against an empirical 3.80, and beats the linear-AR mixture by
**0.151 nats per edge**. The linear-AR mixture's conditional standard deviation
is flat to within 1.15× across the parent range, confirmed numerically rather
than asserted.

**Generation** (`exp_25`) is available two ways, deliberately:

- `sample_ancestral` — root-to-leaves from the fitted tree. Exact for the
  discretised model, 2000 images in 0.6 s.
- `sample_reverse` — reverse diffusion driven by the exact BP score, via the
  existing `src/reverse.py`.

They target the *same* distribution, so a gap between them is sampler
discretisation and not model error. That is the point of having both.

### The comparison, at size

600 training images, 12 EM iterations, **per-depth grids resolving t = 0.05**,
3 000 generated samples against 1 000 held-out real images. Monotone violation 0
for all three families. `outputs/exp_25_wavelet_generation/`.

| family | held-out loglik/image (t = 0.05) | subband kurtosis gap | **magnitude excess generated** |
|---|---|---|---|
| Gaussian tree | −517.64 | 3.120 | **−0.001** |
| mixture (linear-AR) | −565.85 | 2.751 | **0.010** |
| **scale mixture** | **−481.78** | **1.418** | **0.262** |
| *real held-out* | — | *0* | *1.030* |

Four things, in order of how much they matter.

1. **Both linear-AR families generate essentially zero cross-scale magnitude
   dependence** — −0.001 and 0.010 against a real 1.030. This is the prediction
   of §6 confirmed on real data at size, and it is a *structural* zero, not a
   training failure: their conditional variance cannot depend on the parent, so
   no amount of data or capacity moves this number. It reproduces to three
   decimals on a completely different mesh, which is about as robust as a
   negative result gets.
2. **The scale mixture recovers a quarter of it** (0.262) and halves the subband
   kurtosis gap (1.418 against 2.751).
3. **Held-out likelihood, now measured near clean data, ranks it first by a wide
   margin**: +35.9 nats/image over the Gaussian closure. On the old shared grid
   this could only be evaluated at t ≈ 1.15, where the margin was 14.4 — so
   being able to reach small *t* **more than doubles the measured advantage**.
   The ordering is the same on a metric that never looks at samples and one that
   only looks at samples, which is worth more than either alone.
4. **The two baselines swap places depending on t, and that is new.** At the
   high-noise point the mixture beat the Gaussian (−1117.9 against −1121.8); at
   t = 0.05 the Gaussian beats it by 48 nats. A flexible innovation *shape* helps
   once the channel has washed out the detail and hurts when it has not, so
   "which baseline is better" is not well posed without naming the noise level.
   Only the scale mixture wins at both.

### 6.2 A limit that per-subband standardisation creates

This one is not a detail, and it invalidated my own first likelihood numbers.

Standardising each subband turns the single pixel-space Δ_t into
Δ_d = Δ_t/s_d². So a subband with a **large** scale gets a **narrow**
likelihood — and the coarse subbands of a natural image have the largest scales
(LH depth 0 is 10.74 against 0.141 at HH depth 4, a factor of 76). On a shared
grid they are the ones that go under-resolved, and they do it **at small t**,
which is exactly where a reverse sampler spends its last steps.

Points per likelihood standard deviation (the project requires ≥ 3, audit F5):

| grid | t = 0.05 | t = 0.5 | t = 1.0 | minimum resolved t |
|---|---|---|---|---|
| M = 201 | **0.38** | **1.53** | 2.94 | **1.02** |
| M = 401 | **0.75** | 3.05 | 5.88 | 0.49 |
| M = 801 | 1.51 | 6.10 | 11.8 | 0.17 |

Meanwhile the *finest* subband sits at 650 points per standard deviation at
t = 2. The grid is being spent in precisely the wrong place.

Consequences, recorded rather than smoothed over:

- My first held-out likelihoods (−481.7 / −514.6 / −562.8 at t = 0.05, M = 161)
  were computed in an unresolved regime, so they had to be set aside. Forced up
  to the resolved t = 1.147 the numbers became −1107.4 / −1117.9 / −1121.8: same
  leader, margin 14.4 rather than 35.9 nats, and heavily smoothed (α = 0.32).
  With per-depth grids the honest small-*t* values are **−481.8 / −517.6 /
  −565.8** (§6.1).

  Worth recording that the discarded numbers turned out to be close to the
  correct ones — within 3 nats, and within 0.06 for the scale mixture. Setting
  them aside was still right: an unresolved quadrature carries no guarantee, and
  the *same* defect in `exp_26` understated an effect by 2.2× (§9.1). But the
  error here was small, and saying so is more useful than implying the caution
  was vindicated by the outcome.
- Reverse-diffusion generation was blocked outright, which is a stronger
  statement than "the samples are inaccurate" — see §6.3, and §6.2.1 for the fix
  that removed it.

`WaveletTreeModel.resolution_report` measures it; `exp_25` records the requested
and used *t* in its output so no result can be quoted from an unresolved regime
by accident.

### 6.2.1 The per-depth grid — done, and it is cheaper than what it replaced

Two constraints set the mesh at each depth, and **the binding one differs by
depth**, which is exactly why one grid cannot serve:

- *resolve the likelihood*, width `√Δ_t /(α_t s_d)` — binds at the **coarse** end,
  where the scale is large;
- *resolve the state*, unit variance after standardisation — binds at the
  **fine** end, where the likelihood is enormous.

`per_depth_grid_sizes` applies both. On CIFAR (scales 10.85 … 0.38), targeting
t = 0.05 gives **[1609, 757, 349, 149, 65]** points from coarse to fine.

The cost inverts the way one might fear. Level *d* holds 4ᵈ nodes and wants
M_d ∝ s_d, and since s_d roughly halves per level, the work per edge level
`4ᵈ·M_d·M_{d+1}` is near-constant:

| edge | nodes | matrix | ops |
|---|---|---|---|
| 0→1 | 4 | 791×1592 | 5.0 M |
| 1→2 | 16 | 349×791 | 4.4 M |
| 2→3 | 64 | 147×349 | 3.3 M |
| 3→4 | 256 | 65×147 | 2.5 M |
| **total** | | | **15.2 M** |

against **19.8 M** for the old uniform M = 241. So resolving to t = 0.05 is
**cheaper** than the grid that only reached t = 0.90 — an 18× gain in *t* reach
for a 23% saving in work. Measured on real CIFAR: `min_resolved_t` **0.90 →
0.0499**, EM monotone violation 0, 21 s per iteration on 300 images.

What it took: rectangular transition matrices `K_d[k, j]` with *k* on the child
grid and *j* on the parent grid (the existing einsums already accepted this), a
two-grid `log_transition_matrix(grid_in, grid_out)` and `m_step(stats, grid_in,
grid_out)` across every kernel in `src/kernels.py` and `src/scale_kernel.py`
(backward compatible — omitting the second grid is exactly the old behaviour),
and per-level arrays throughout `wavelet_bp` and `video_bp`.

Verified the same way as everything else: per-depth BP matches a dense Gaussian
solve to **10⁻⁸** and its log-evidence to **10⁻⁷**, agrees with a uniformly fine
grid to 10⁻⁸, and the per-level Ξ come back rectangular with the right mass. The
M-step is checked by recovering known parameters from a *population* Ξ built on
deliberately mismatched grids — and by confirming that swapping the two grids
gets the answer wrong, so the check is not vacuous.

---

### 6.3 Reverse diffusion: was blocked, now largely works

The diagnosis in this section was correct, and it is kept because it is the
reason the per-depth grid was built. Its conclusion no longer holds.

**The original finding.** On a shared grid the sampler had to stop at t ≈ 0.95,
where √Δ = 0.92 while the finest subbands have true standard deviation
0.14–0.32 — an SNR of about 0.1. The forward process had already destroyed them.
So `x(t_min)` was pure noise there (0.94 against a 0.92 floor) and the denoising
readout correctly collapsed them toward zero (0.05 against 0.32), because a
posterior mean shrinks to the prior when the likelihood carries no information.
Neither is a sample from p₀, and no better readout existed: the information was
gone. The tell was that the damage was **monotone in scale**, which is the SNR
ordering and not something a sampler discretisation could produce.

**Re-measured after the per-depth grid.** With the sampler able to reach
t = 0.05, ratio of reverse readout to ancestral standard deviation by scale
(48 samples, 80 steps, Gaussian tree fitted on 400 CIFAR images):

| subband | before (t ≈ 0.95) | **after (t = 0.05)** |
|---|---|---|
| HL depth 0 (coarsest) | 1.03 | **1.00** |
| HL depth 1 | 0.89 | **0.97** |
| HL depth 2 | 0.76 | **0.98** |
| HL depth 3 | 0.46 | **0.91** |
| HL depth 4 (finest) | **0.22** | **0.73** |
| LL | — | 0.86 |
| whole image (pixel std) | — | **0.92** |

The finest subband goes from 22 % recovered to 73 %, and the monotone decay that
identified the cause is much flatter.

**Confirmed at size, and a different weak spot emerges.** The full `exp_25`
rerun on per-depth grids (48 samples, 100 steps, three families) gives
reverse-over-ancestral standard-deviation *ratios*:

| family | detail bands d0→d4 | LL |
|---|---|---|
| Gaussian | 0.94, 0.93, 0.96, 0.93, **0.73** | 0.94 |
| mixture | 1.02, 0.89, 1.03, 0.97, **0.81** | 0.91 |
| scale mixture | 1.02, 0.97, 1.04, 0.95, **0.83** | **0.79** |

Every detail band is now within a few per cent except the finest, which sits at
0.73–0.83 against 0.22 before. **The residual error has moved to the LL band**,
and it is not resolution: at t = 0.05 LL has forward SNR ≈ 54, the best-resolved
coefficient in the model. It is the reverse *integrator*. The sampler starts at
N(0, I) in pixel space, where LL has standard deviation 1, and must inflate it
to ≈ 17 by t_min — by far the largest dynamic range any coefficient has to
traverse — and at 100 steps it undershoots. That is a step-count and schedule
question, and it is the next thing to measure.

> Read these as *ratios*, not absolute gaps. Subband standard deviations span
> 0.14 to 17, so an absolute summary is a report about LL and nothing else. I
> drew a wrong conclusion from `worst_abs_gap` once here — a family whose worst
> absolute gap grew while every detail band improved — so `profile_gap` now
> returns relative gaps too and `exp_25` prints those.

**What remains, and it is no longer discretisation.** At t = 0.05 the finest
subband still has forward-process SNR 0.44, so part of its content is genuinely
destroyed and no sampler can return it; `sample_reverse` warns when this holds.
Reaching SNR 1 there needs t ≈ 0.01 and hence a root grid of ~3700 points
against the 1609 used here — a memory question, not a structural one.

So the honest status: **reverse diffusion works**, and the missing ~25 % of the
finest scale is bounded by the forward process rather than by the mesh. That
matters for what the results mean, because ancestral sampling tests the *model*
while only reverse diffusion tests the *score*, and the score is the project's
actual subject.

---

## 7. What "exact" does and does not buy — the honest counterweight

**Look at the samples first** (`outputs/exp_25_wavelet_generation/samples.png`).
They are *texture*, not objects. Nothing in a wavelet HMT sample resembles a
frog, a truck or an aeroplane, and no amount of further training will change
that. This is not a bug and not an undertrained model — it is the model class.

The reason is exactly the structure that buys exactness. Within a subband, the
4ᵈ coefficients at depth *d* are **conditionally independent given their
parents**: there are no within-scale spatial edges, because adding them would
close a loop and destroy the tree. So the model can reproduce the *marginal* and
*cross-scale* statistics of natural images — which is what it is being asked to
do — while having no mechanism whatever for spatial coherence within a scale.
Objects live in precisely that missing structure.

Consequences to state before anyone asks:

- **FID/IS on these samples would be catastrophic and uninformative.** They would
  report "not photographs", which is visible for free. This *strengthens* the
  metric argument of §8 rather than weakening it, but it also caps the claim: the
  contribution is about exact inference and learned coefficient structure, not
  about competitive image synthesis, and it must not be written as if it were.
- The visible difference between the rows is still real and in the predicted
  direction: the scale-mixture row has visibly more large-scale blob structure
  than the Gaussian and linear-AR rows, which is the magnitude dependence of
  §6.1 showing up as coherent bright and dark regions.
- **The samples are visibly blocky, on axis-aligned squares.** That is the Haar
  basis showing through: each coefficient's support is a square block, and with
  no within-scale coupling to smooth across them, the block boundaries survive
  into the image. It is a property of the *basis*, separable from the tree
  restriction above, and it is the one artefact here with a cheap remedy — a
  smoother orthogonal wavelet (Daubechies-4) has overlapping support and would
  not produce it. That makes the Haar-vs-D4 comparison (§12, item 4) worth more
  than it looked: it addresses a visible failure, not just a statistical one.

So the inference is exact. **The model is not the truth.** The wavelet HMT
asserts that coefficients form a Markov tree across scale with no within-scale
spatial edges — and real images certainly have within-scale spatial correlation,
which this model omits by construction.

So the correct statement is: *exact inference in an approximate model of images*,
where previously the project had *exact inference in an exact model of simulated
data*. That is a real change in what the results mean, and it is the first thing
a referee will press on. It is also precisely why held-out likelihood is the
right headline metric: it measures the model, not the inference, and this method
can compute it exactly.

---

## 8. Metrics: what to lead with

**Lead with these.**

1. **Held-out exact log-likelihood** of whole images in pixel coordinates.
   `WaveletTreeModel.log_likelihood_images` returns it with the two exact
   corrections (diagonal standardisation contributes −Σ log s_v; the orthonormal
   transform contributes nothing). This is a genuine advantage: GAN-style
   baselines cannot produce this number at all, and most diffusion models can
   only bound it.
2. **Per-subband coefficient distributions** against held-out real images —
   directly the quantity §5 measures, now as a model-versus-truth comparison.
3. **Cross-scale dependency** — the magnitude ratio of §6. A factorised model
   scores 1.0 by construction; this is the metric that separates a hierarchical
   model from a heavy-tailed independent one, and it is the one the project's
   own thesis is about.

**Report FID and KID, but not first, and say why.** They rely on ImageNet-trained
InceptionV3 embeddings; FID is a biased estimator needing large samples and is
highly sensitive to resizing and compression. At this model scale it would
largely be measuring wavelet reconstruction quality rather than score quality.
Inception Score is worse: vulnerable to memorisation, and close to meaningless
without clear object categories. Reporting them buys comparability with the
field; leading with them would misdescribe what was tested.

*Practical:* `torch`/`torchvision` are absent locally and on the cluster. FID/KID
need them. That is an install, and it is on the critical path only for the
secondary metrics — every headline metric above is pure numpy.

---

## 9. Stage 2 — video: a caterpillar is exact, the coupled model is not

This is now **built and verified**, not merely articulated (`src/video_bp.py`,
`tests/test_video_bp.py`).

**What is not exact, stated first.** The model one actually wants for video —
coefficient *v* in frame *f* coupled both to its scale parent `p(v)` in frame *f*
and to itself in frame *f−1* — **has loops**:

    v_f —— p(v)_f —— p(v)_{f−1} —— v_{f−1} —— v_f

is a 4-cycle. BP on it is loopy and the exactness claim is gone. The earlier
framing in this document ("video is a product of two exact structures, neither
introduces a loop") was too generous: the *product* is precisely where the loop
comes from.

**What is exact.** A loop-free *spanning* structure. The natural one couples
frames only through the **root** of each frame's tree:

    root₁ ——— root₂ ——— root₃ ———  …        temporal chain
      │         │         │
    quadtree  quadtree  quadtree            spatial, per frame

Each root has one pendant subtree and at most two chain neighbours, and the
subtrees are disjoint, so there is no cycle anywhere. This is a **caterpillar**:
a chain backbone with trees hanging off it. BP on it is exact.

The corresponding model is a real one — frames are temporally coupled at the
coarsest scale, and fine detail is temporally independent *given* the coarse
trajectory. The restriction is the price and should be stated rather than
glossed: it cannot express a moving edge whose fine detail persists
independently of the coarse content. It is the maximal tree-structured
spatio-temporal model of this shape, not the model one would write down if
exactness were not required.

**How it is computed — with no new inference code.** BP on a caterpillar
factorises into passes the package already has:

1. run each frame's quadtree **upward**; the root's belief-up is then a single
   message summarising that whole frame;
2. run an exact **chain** BP over frames with those as unary potentials;
3. feed each root's chain context back as its incoming message and run each
   frame's quadtree **downward**.

Only step 2 is new (`chain_bp_potentials`, a forward–backward over arbitrary
unary potentials — 30 lines).

**Verified** against a dense Gaussian solve built by forward substitution over
the whole caterpillar, temporal backbone and per-frame quadtrees together:

| check | result |
|---|---|
| caterpillar posterior mean vs dense solve | < 10⁻⁸ |
| caterpillar log-evidence vs closed form | rel. < 10⁻⁷ |
| `chain_bp_potentials` alone vs dense solve | < 10⁻⁹ |
| temporal coupling is not silently dropped | ρ_time 0 vs 0.85 differ by > 0.05 |

That last one is deliberate. An implementation that ignored the chain context
entirely would still reproduce a dense solve *if the dense reference were also
wrong*; the guard checks that turning the temporal correlation off actually
changes the answer.

> One bug worth recording, because it is the kind that hides. The first version
> accumulated both forward *and* backward message scales into the chain
> evidence. The normalisation is closed by contracting at site 0, where the
> forward message is the prior itself and carries no accumulated scale, so the
> forward scales double-count. It leaves every **posterior mean** untouched and
> is invisible to every check except the evidence one.

### 9.1 The video model, fitted — and what the tree restriction costs

The inference layer above is now a working model (`src/video_model.py`,
`src/video_data.py`, `experiments/exp_26_video.py`). Data is **moving CIFAR**:
reflect-pad each frame and crop a window translating at constant integer
velocity. Real natural-image frame statistics, synthetic rigid motion, and that
is stated as a limitation rather than implied away — real video has non-rigid
motion, occlusion and lighting change, none of which this has.

Two models, identical but for one thing: `caterpillar` fits the temporal kernel
by EM through the caterpillar E-step; `frozen` holds ρ_time = 0. The LL band is
treated identically in both (exact scalar Gaussian AR(1) across frames, no grid),
so the comparison isolates temporal coupling *of the spatial trees*.

300 sequences of 6 frames, 8 EM iterations, 400 generated sequences:

| | ρ_time | held-out loglik/sequence | frame-difference energy |
|---|---|---|---|
| frozen (control) | 0 (held) | −6739.8 | 1.420 |
| **caterpillar** | **0.999** | **−6667.1** | **1.005** |
| *real video* | — | — | ***0.211*** |
| *independent frames* | — | — | *2.000* |

**Temporal coupling is worth +72.7 nats per sequence** (+12.1 per frame),
evaluated at t = 1.259 — the smallest *t* this grid resolves.

> **A correction, and it moved the answer.** An earlier version of this table
> quoted these likelihoods at t = 0.6, where the coarsest subband (scale 11.25)
> gets 0.5 points per likelihood standard deviation against the 3 required. That
> is the §6.2 defect, and I had built the guard for `exp_25` and then failed to
> carry it into `exp_26`. The unresolved evaluation gave a gap of +33.8
> nats/sequence — it **understated the benefit of temporal coupling by 2.2×**.
> Both experiments now clamp to the resolved *t* and record requested and used
> values side by side. The coherence column never involved a likelihood and is
> unchanged.
>
> The standing caveat is that t = 1.259 is a heavily smoothed regime
> (α = 0.28, Δ = 0.92). The comparison is paired and fair, but a likelihood this
> far from t = 0 is a weaker statement than a small-*t* one would be — which is,
> again, what the per-depth grid unblocks.

**Temporal coupling earns its place**: +46.9 nats per sequence on held-out
likelihood, and coherence improves from 1.538 to 1.086. EM is monotone
(violation 0) and ρ_time = 0.983 is the right answer for translating video.

**And it is nowhere near enough.** Real video sits at 0.223 while the caterpillar
reaches 1.086 — barely a third of the way from the control to the truth. This is
not undertraining, and the next subsection shows it is not even a limitation of
*this* model.

### 9.2 The ceiling is a theorem, not a tuning problem

**At most one temporal edge per connected component.** Take two coefficients
*u*, *v* in the same per-frame component, both coupled to their counterparts in
the previous frame. The component is connected, so there is a path *u_f → v_f*
inside frame *f*, and likewise inside frame *f−1*. Together with the two temporal
edges that closes

    u_f → v_f → v_{f−1} → u_{f−1} → u_f

— a cycle, always. Verified exhaustively on small trees
(`exp_26`): coupling one node per component is loop-free for every choice,
coupling two or more closes a cycle for every choice.

The wavelet frame decomposes into exactly **four** components — three orientation
trees and the isolated LL coefficient — so **at most 4 of 1024 coefficients per
frame can carry temporal dependence** in *any* loop-free model over this
coefficient set. The caterpillar already couples 4, and it couples the
largest-variance one in each component (a root has more variance than any single
coefficient below it). It is therefore **already optimal**, and the ceiling is a
property of exactness itself rather than of this construction:

| | coupled variance | floor on frame-difference energy |
|---|---|---|
| **any loop-free model** (4 of 1024 coefficients) | **46.6 %** | **1.069** naive / **1.055** exact |
| *real moving CIFAR* | — | *0.211* |
| *independent frames* | 0 % | *2.000* |

**Exactness caps temporal coherence at 5.1× worse than the data.**

**A subtlety the full-scale run exposed, and that an earlier draft of this
section got wrong twice.** The fitted caterpillar measures **1.005** at
ρ_time = 0.999 — *below* the 1.069 quoted above. That is not a violated bound;
it is a bound stated too crudely, in two separable ways, and both were measured
rather than argued away.

*First, the variance must be the model's.* Writing the energy out,

    energy = 2 · Σ_c σ_c² (1 − ρ_c) / Σ_c σ_c²,

the floor at ρ_t → 1 is `2(1 − frac)` with **frac the *model's* variance
allocation, not the data's**. Measured: the data puts 46.55 % in the coupled
components, the fitted model **49.02 %**, so the model's own naive floor is
1.020, not 1.069.

*Second, the tree leaks temporal information downward.* The "uncoupled"
coefficients are not temporally independent after all: a depth-*d* coefficient
is drawn given its parent, ultimately given the root, and the roots *are*
correlated in time. It therefore inherits an effective temporal correlation

    ρ_eff(d) = ρ_time · Π_{j≤d} ρ_j²

— attenuated by the square of the spatial correlation at every level. With the
fitted ρ_spatial ≈ 0.2–0.3 that is a factor of ~20 per level: **0.999 at the
root, 0.042 at depth 1, 0.004 at depth 2, and gone.** Real, and worth naming as
the one mechanism by which temporal structure reaches fine scales at all, but it
buys 0.014 of energy.

Putting both in: 1.0196 − 0.0142 = **1.0054 predicted against 1.0052 measured**.
The model's temporal coherence is accounted for to 0.02 %.

So the honest bound for a model that also matches the data's variance ladder is
**≈1.055**, not 1.069, and a model can only go under it by mis-allocating
variance across scales — a modelling error visible in the subband statistics of
§6.1. Coherence bought that way is paid for in marginal fidelity. Every version
of the number lands ~5× above real video (0.211), which is the conclusion that
matters and the one none of this disturbs.

Both halves are reproducible without fitting anything —
`exp_26 --only structure` writes `structure_budget.csv` (the union-find check
over every coupling choice) and `structure_floor.csv` (the variance
decomposition).

There is an exact edge budget behind this. A spanning forest on *F* frames of *n*
coefficients with *c* components per frame has `Fn − c` edges, split as `F(n−c)`
spatial and `c(F−1)` temporal. Temporal edges can only be bought by cutting
spatial ones, one for one. Coupling more scales in time therefore requires
*disconnecting* the spatial tree by the same amount — the two structures compete
for a fixed budget, and no arrangement escapes it.

**Two independent limitations, not one.** It is worth separating them, because
the obvious response to the first is closed off by the second:

* **Breadth.** At most 4 of 1024 coefficients per frame carry a temporal edge —
  the component argument above.
* **Depth.** Whatever those edges carry decays as ρ_spatial² per level going
  down the tree: 0.999 at the root, **0.042** one level below, 0.004 two levels
  below (measured, this section). So the coupling does not reach the scales
  where most of the variance lives.

The natural reply to the breadth limit is "choose a better spanning structure" —
couple a different or better-placed coefficient in each component. The depth
result says that cannot work: even a *perfectly* coupled root transmits
essentially nothing past its immediate children, so no choice of which single
coefficient to couple recovers fine-scale temporal structure. The two limits
have to be defeated together, and inside the loop-free class they cannot be.

So the honest summary: *exact BP on video is achievable, and exactness caps
temporal coherence at five times worse than the data.* That is a real result —
it quantifies the price of exactness instead of asserting it is small — and it
makes the **loopy comparison the interesting experiment** rather than a
formality. The question is no longer "can we do video exactly" (yes, and here is
the ceiling) but "what does approximate BP on the fully coupled graph buy against
exact BP on the best tree", which is now well-posed and quantitative.

**What remains**: real video data; the loopy-BP comparison above; and measuring
where on the space-time edge-budget curve the best trade-off sits.

---

## 10. Scope against 16 September

Thesis due 16 September 2026; defence in October. That is ~5.5 weeks from today,
and this is an extension competing with finishing the thesis.

**Compute budget, measured not guessed.** One BP pass costs 30 ms per
image-orientation at grid M = 241 (72 ms at M = 401) on this laptop. So one EM
iteration is:

- 2 000 images × 3 orientations × 3 noise levels ≈ **9 min** — 30 iterations ≈ 4.5 h
  single-core, and the E-step is embarrassingly parallel over images, so the
  existing array-job sharding brings it to well under an hour;
- 10 000 images at M = 401 ≈ 108 min/iteration — cluster-only.

**Achievable by mid-September**, in priority order:

1. **Per-depth grid** (§6.2). Everything downstream is capped by it: without it
   no small-*t* score is valid, so neither the headline likelihood nor the
   reverse sampler can be quoted at the noise levels that matter. Cheap in
   compute, moderate in code.
2. A converged fit on greyscale CIFAR at full size, reporting held-out
   likelihood, per-subband distributions and cross-scale dependency for all
   three kernel families. (Scaffolding done — `exp_25`.)
3. A factorised heavy-tailed baseline — same marginals, no tree edges. This is
   what makes the cross-scale metric mean something.
4. Sample figures from reverse diffusion, once (1) makes them valid.

**Not achievable, and should not be promised:** a fitted video model on real
video; CelebA at resolution; a competitive pixel-space diffusion baseline; FID
as a headline number backed by enough samples to be unbiased.

**My recommendation.** Items 1–3 are a genuine, self-contained contribution and
land inside the thesis as an "extension to real data" chapter answering the
advisor document's own "Unresolved" on question A (nothing scales to a
vector-valued state). Item 4 is the first thing to cut.

Video is now better placed than "future work": the inference layer is built and
verified (§9), so it can be stated as a *result* — the caterpillar is exact, the
fully coupled model is not — with the fitted model left as follow-on. That is a
stronger position than the earlier draft claimed and a narrower one than the
original brief hoped for.

---

## 11. Corrections to existing repo material

`paper/bibliography.bib` credited arXiv:2404.18444 to "Mei, Song and Wu, Yuchen".
Verified against the arXiv API: that paper is **"U-Nets as Belief Propagation:
Efficient Classification, Denoising, and Diffusion in Generative Hierarchical
Models" by Song Mei alone**. Yuchen Wu co-authored a *different* Mei paper.
Fixed, and Crouse–Nowak–Baraniuk (1998) added as the direct ancestor of this
construction.

The Mei paper matters more than as a citation: it shows the U-Net encoder/decoder
with skip connections *is* the up/down pass of BP on a tree-structured
hierarchical model. The architecture the field uses for image diffusion is
already approximating what this construction computes exactly — which is the
strongest available framing for why this is worth doing.

---

## 12. Immediate next steps

1. **Re-run `exp_25` and `exp_26` on the per-depth grid** and quote both
   likelihoods at a small *t* rather than the t ≈ 1.15 and t ≈ 1.26 the shared
   grid forced. The machinery is in (§6.2.1); the published numbers in §6.1 and
   §9.1 predate it and are the last things still measured in a high-noise
   regime.
3. The factorised heavy-tailed baseline (same marginals, tree edges removed) —
   the control that makes the cross-scale metric mean something.
4. Decide Haar vs Daubechies-4 by measuring §6 under both. Two reasons now: the
   linear correlation of 0.18–0.48 in HL/LH is larger than the classical
   literature suggests and is probably a Haar artefact, and Haar's square
   support is what makes the samples visibly blocky (§7).
5. **The loopy comparison** (§9.2). Now the most interesting experiment rather
   than a formality: §9.2 puts a number on the exact-tree side, so what remains
   is what approximate BP on the fully coupled spatio-temporal graph buys
   against it. Real video data belongs with this.

Files added — inference and modelling: `src/wavelet.py`, `src/wavelet_bp.py`,
`src/wavelet_model.py`, `src/scale_kernel.py`, `src/wavelet_stats.py`,
`src/video_bp.py`, `src/video_model.py`, `src/video_data.py`,
`src/image_data.py`. Experiments: `experiments/exp_23_wavelet_statistics.py`,
`exp_24_wavelet_fit.py`, `exp_25_wavelet_generation.py`, `exp_26_video.py`.
Tests: `tests/test_wavelet.py`, `test_wavelet_bp.py`, `test_wavelet_model.py`,
`test_scale_kernel.py`, `test_video_bp.py`, `test_video_model.py` — 51 tests
across the six.

Full suite: **264 passed, 12 skipped** (5 slow end-to-end fits deselected; they
pass separately and take ~14 minutes).
