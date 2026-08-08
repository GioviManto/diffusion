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
- 39 tests, including agreement with a dense Gaussian solve to 2·10⁻¹⁵;
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

**And one limit is now known that was not before.** Per-subband standardisation
makes the coarse subbands' likelihood narrower than a grid cell at small *t*, so
below t ≈ 1.02 (at M = 201) neither the score nor the evidence is valid on a
shared grid. This invalidated my own first likelihood numbers; they are corrected
in §6.2. The fix — a per-depth grid — is cheap and is the top next task.

**Video is built, not just argued** (§9). The fully coupled spatio-temporal model
has 4-cycles and is *not* exact; a temporal chain of spatial trees — a
caterpillar — is, and BP on it matches a dense Gaussian solve to 10⁻⁸ using only
passes the package already had.

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

## 6.1 The kernel that fixes it, and generation

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

At smoke scale (200 images, 4 EM iterations — **not** converged, so directional
only) the predicted failure shows up cleanly in what the models *generate*:

| family | mean magnitude excess generated | real held-out |
|---|---|---|
| mixture (linear-AR) | **0.005** | 0.920 |
| scale mixture | 0.190 | 0.920 |

The linear-AR family generates essentially **zero** cross-scale magnitude
dependence — it is structurally incapable, exactly as §6 predicts. The scale
mixture generates some but far from enough at this training budget.

---

## 6.2 A limit that per-subband standardisation creates

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
  were computed in an unresolved regime and are **not trustworthy**. Re-evaluated
  at the resolved t = 1.147: **scale mixture −1107.4, mixture −1117.9, Gaussian
  −1121.8**. The ordering survives and the scale mixture leads by 14.4 nats per
  image over the Gaussian closure, but the margins are entirely different — and
  at t = 1.147 the likelihood is heavily smoothed (α = 0.32), so this is a
  weaker statement than a small-t likelihood would be.
- The reverse-diffusion samples in the smoke run showed reverse-vs-ancestral
  standard-deviation gaps up to 8.4. That is this, not sampler noise.
  `sample_reverse` now clamps `t_min` to the resolved value.

`WaveletTreeModel.resolution_report` measures it; `exp_25` records the requested
and used *t* in its output so no result can be quoted from an unresolved regime
by accident.

**The fix is a per-depth grid, and it is cheap.** Coarse subbands have 1, 4, 16
nodes against 256 at the finest level, so refining exactly where the likelihood
is narrow costs almost nothing. It requires rectangular transition matrices
between levels (`K_d[k, j]` with *k* on the child grid and *j* on the parent
grid — the existing einsums already accept this) and an M-step taking a parent
grid and a child grid separately, which is a genuine change to
`src/kernels.py`. **Not done.** It is the single highest-value next task, because
it unlocks small-*t* scores and therefore both a meaningful held-out likelihood
and a usable reverse sampler.

---

## 7. What "exact" does and does not buy — the honest counterweight

The inference is exact. **The model is not the truth.** The wavelet HMT asserts
that coefficients form a Markov tree across scale with no within-scale spatial
edges — and real images certainly have within-scale spatial correlation, which
this model omits by construction.

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

**What remains for a video model**: real video data (none is downloaded here),
a temporal kernel fitted by EM through the caterpillar E-step, and the honest
comparison against a fully coupled loopy-BP model to measure what the tree
restriction costs. The inference layer is done; the modelling is not.

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

1. **Per-depth grid** (§6.2). Rectangular `K_d[k, j]` between levels, and an
   M-step taking parent and child grids separately. Unblocks everything at
   small *t*.
2. Re-run `exp_25` at full size once (1) lands, and quote the likelihood at a
   *small* *t* rather than at the currently-forced t ≈ 1.15.
3. The factorised heavy-tailed baseline (same marginals, tree edges removed).
4. Decide Haar vs Daubechies-4 by measuring §6 under both — the linear
   correlation of 0.18–0.48 in HL/LH is larger than the classical literature
   suggests and is probably a Haar artefact.
5. For video: real video data, a temporal kernel fitted by EM through the
   caterpillar E-step, and a loopy-BP fully coupled comparison to measure what
   the tree restriction costs.

Files added — inference and modelling: `src/wavelet.py`, `src/wavelet_bp.py`,
`src/wavelet_model.py`, `src/scale_kernel.py`, `src/wavelet_stats.py`,
`src/video_bp.py`, `src/image_data.py`. Experiments:
`experiments/exp_23_wavelet_statistics.py`, `exp_24_wavelet_fit.py`,
`exp_25_wavelet_generation.py`. Tests: `tests/test_wavelet.py`,
`test_wavelet_bp.py`, `test_wavelet_model.py`, `test_scale_kernel.py`,
`test_video_bp.py`.

Full suite: **255 passed, 12 skipped**, plus 4 in `test_wavelet_model.py` run
separately (~6 minutes).
