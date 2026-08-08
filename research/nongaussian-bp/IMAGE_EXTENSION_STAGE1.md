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
- 37 tests, including agreement with a dense Gaussian solve to 2·10⁻¹⁵;
- the gating measurement on real CIFAR-10 (`experiments/exp_23_wavelet_statistics.py`).

**The gate passes decisively.** CIFAR-10 wavelet coefficients are strongly
non-Gaussian — the finest subbands have excess kurtosis 5.8–7.1, i.e. *heavier
tailed than the Laplace chain* (3.0) the project has been simulating.

**But one measurement redirects the modelling.** The dependence between parent
and child coefficients is largely in the **magnitude**, not the mean, and the
linear-autoregressive kernel family in `src/kernels.py` cannot represent that.
See §6. This is the single most important result in this document.

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

| orientation | linear corr | child std given \|parent\| in Q4 vs Q1 |
|---|---|---|
| HL | 0.452 | **3.65×** |
| LH | 0.482 | **3.29×** |
| HH | **0.148** | **2.89×** |

Two things follow.

1. **Magnitude dependence is large and universal.** A top-quartile-magnitude
   parent has children roughly 3× more spread than a bottom-quartile one, in
   *every* orientation, growing with depth (1.34× at the coarsest boundary).
   A linear-AR kernel cannot express this at all: its conditional variance does
   not depend on the parent.
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

## 9. Stage 2 — video (articulated, not promised)

Video is structurally the better fit, and the reason is precise: its dependence
graph is a **product** of two structures that are each already exact here.

- a **spatial tree** per frame — Stage 1;
- a **temporal chain** across frames — the entire existing project.

Neither introduces a loop. A tree × chain is a tree-structured graph in the
combined index only if the coupling is restricted; the honest version of the
claim is that **exact BP holds on the temporal chain of any fixed
low-dimensional per-frame representation**, and separately on the spatial tree
within a frame. The fully coupled spatio-temporal model — every coefficient
linked to its parent in scale *and* its predecessor in time — has loops (a
coefficient, its parent, the parent's predecessor and its own predecessor close
a 4-cycle). That must be said plainly rather than glossed.

**What is deliverable as proof of concept** is therefore the temporal half: exact
BP over time on a low-dimensional per-frame representation (e.g. the LL band, or
the coarsest subbands), where the chain machinery already in the repo applies
with no approximation and no new inference code. A full video model is a
follow-on project.

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

1. Scale-mixture kernel + M-step, with the Gaussian control carried over
   (on Gaussian tree data it must reduce to the second-order closure). — *the
   one genuinely new piece of modelling*
2. Fit on greyscale CIFAR; report held-out likelihood, per-subband distributions,
   cross-scale dependency, against (i) the Gaussian tree closure and (ii) a
   factorised heavy-tailed model.
3. Generation by reverse diffusion on tree coefficients + inverse transform;
   sample figures.

**Not achievable, and should not be promised:** a full video model; CelebA at
resolution; a competitive pixel-space diffusion baseline; FID as a headline
number backed by enough samples to be unbiased.

**My recommendation.** Items 1–2 are a genuine, self-contained contribution and
land inside the thesis as an "extension to real data" chapter answering the
advisor document's own "Unresolved" on question A (nothing scales to a
vector-valued state). Item 3 is worth attempting but is the first thing to cut.
Video belongs in the "future work" section as §9 states it — including the
honest note about the 4-cycle.

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

1. Implement the scale-mixture kernel and its M-step; verify monotone ascent and
   the Gaussian reduction.
2. Fit on 2 000 CIFAR images at M = 241 locally end to end; then scale on the
   cluster.
3. Build the factorised-heavy-tailed baseline (same marginals, no tree edges) —
   this is the baseline that makes the cross-scale metric meaningful.
4. Decide Haar vs Daubechies-4 by measuring §6 under both.

A first indication that §6 is right, though at smoke scale and not to be quoted:
in `exp_24 --quick` (300 images, 6 iterations) the 4-component
`MixtureInnovationKernel` beats the Gaussian tree by 0.09 nats per image
(−1037.71 vs −1037.80) and ties it on denoising MSE. A flexible innovation
*shape* on a linear-AR kernel buys essentially nothing, which is exactly what
the magnitude-dependence measurement predicts. A converged run at full size is
needed before that is evidence rather than a hint.

Files added: `src/wavelet.py`, `src/wavelet_bp.py`, `src/wavelet_model.py`,
`src/image_data.py`, `experiments/exp_23_wavelet_statistics.py`,
`experiments/exp_24_wavelet_fit.py`, `tests/test_wavelet.py`,
`tests/test_wavelet_bp.py`, `tests/test_wavelet_model.py`.

Full suite after these additions: **242 passed, 12 skipped**, plus 4 in
`test_wavelet_model.py` run separately (they take ~6 minutes).
