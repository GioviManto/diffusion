# Open questions posed by this project — and where each one now stands

Last updated 2026-08-01. Branch `claude/em-bp-denoiser-learning-e07ike`.

This document collects **every question the project has posed to itself**, from
three sources, and answers each with evidence or says plainly that it is still
open.

Sources of the questions:

- **[R]** `research/gaussian-ar1-bp/markov_gaussian_approx/report/bp_markov_diffusion_gaussian_approx.pdf`, §"Interpretation and next steps" — the five numbered next steps of the report we discussed.
- **[F]** `docs/RESULT_LEDGER.md`, §"Explicitly future" — F1–F5.
- **[E]** The Marc/Jérôme email of 2026-07-30 — the two suggestions.
- **[L]** Questions that arose *during* the Layer-5 work itself.

Status vocabulary, used strictly:

| label | meaning |
|---|---|
| **ANSWERED** | Measured or proved, with the evidence named. |
| **PARTIAL** | Something real is established, but the question as posed is not closed. The gap is stated. |
| **OPEN** | Not started, or started and inconclusive. |

---

## [R] From the report we discussed

### R1. "Increase trials and grid size to reduce Monte Carlo variability." — **ANSWERED**

Both axes were pushed and, more usefully, *quantified*.

- **Grid.** Convergence of the BP posterior mean measured directly against an M=1201 reference: relative error 1.9e-5 (M=201), 7.7e-6 (M=301), **4.1e-6 (M=401)**, 1.5e-6 (M=601) at t=0.05, with the same ordering at every t tested. M=401, A=8 is the working default and its error sits four orders of magnitude below any learning error in the project.
- **Trials.** This turned out to matter more than expected and produced a lesson worth recording. The `N^{−1/2}` rate study at **4** replicates gave log-log slopes of −0.26, −0.69, −0.25, −0.41 against a predicted −0.50 and did **not** support the rate. Re-run at **12** replicates it gives a combined slope of **−0.500 ± 0.048**, excluding −0.25 and −0.75 at 5.2σ. An RMSE over 4 samples cannot resolve a slope better than about ±0.2.

> **Rule adopted from this:** any rate claim needs replicates in the tens, not the units. It is now in the agent handoff so it is not relearned.

*Evidence:* compendium §4.9; `outputs/exp_06_rate_highrep/`; `docs/AGENT_HANDOFF_EM_BP.md` §7.

### R2. "Explore different ρ — stronger correlations make messages more informative and potentially more non-Gaussian." — **ANSWERED, and the sign is the opposite of the conjecture**

Swept ρ ∈ {0.5, 0.85, 0.95} across six innovation families. The Gaussian-closure
error **decreases** with stronger correlation, uniformly:

| family | ρ=0.5 | ρ=0.85 | ρ=0.95 |
|---|---|---|---|
| Laplace | 0.520 | 0.354 | 0.270 |
| Student-t(5) | 0.317 | 0.241 | 0.135 |
| mixture κ=0.9 | 0.939 | 0.714 | 0.368 |

(median relative score error at small t.) The conjecture in the report was that
stronger correlation would make messages *more* non-Gaussian and so harder. The
measurement says the reverse: a more informative message is closer to Gaussian,
because it concentrates. Stronger correlation *helps* the Gaussian closure.

*Evidence:* `outputs/exp_03_nongaussian_innovation_sweep/SUMMARY_TABLE.md`; ledger B13.

### R3. "Compare Laplace, Student, Gaussian-mixture, and bounded-support innovations." — **PARTIAL**

Three of the four families are done and swept (Laplace; Student-t at ν=5,8;
symmetric Gaussian mixture at κ=0.3,0.6,0.9), giving excess kurtosis from −1.62
to +6.00. The organizing finding is that **|excess kurtosis| predicts the
closure error and its sign does not matter much** — bimodal (negative) innovations
are the *worst* case at κ=0.9, not the heavy tails.

**Gap:** bounded-support innovations (e.g. uniform, triangular) were never
implemented. They are the interesting missing case because the posterior then has
hard edges that no Gaussian message can represent, and they would test whether
excess kurtosis remains the right one-number summary when the failure is a
support mismatch rather than a tail mismatch.

*Evidence:* `src/priors.py`; exp_03 outputs. **To close:** add a `UniformAR1`
prior and re-run exp_03.

### R4. "Add Gaussian-mixture *message* approximations — how many components to beat single-Gaussian closure?" — **OPEN** (and not to be confused with what Layer 5 did)

This is the same question as **F1** and it is genuinely untouched. It is worth
being precise, because Layer 5 built something that sounds identical and is not:

| | object approximated | purpose |
|---|---|---|
| **R4 / F1 (open)** | the BP **messages** `L_i(a)`, `R_i(a)` | a *representation* closure — make continuous BP finite-dimensional without a grid |
| **Layer 5 (done)** | the transition **kernel** `K_θ(a'\|a)` | a *model* family — learn an unknown prior |

Layer 5's `MixtureInnovationKernel` fits a mixture to the innovation law of the
chain. R4 asks to propagate mixture-parameterized *messages* through the
forward-backward recursion, with a component-collapsing step to stop the count
exploding (a mixture of C components through a mixture kernel of C' components
gives C·C'). Nothing in the current code does that.

**To close:** implement mixture-message BP with collapsing, and measure the error
against the grid reference as a function of component count — the answer to
"how many components" is the deliverable.

### R5. "Do neural message approximators help only when they preserve the local BP update structure?" — **PARTIAL, with evidence on both sides**

Two experiments bear on this and they point the same way.

- **Layer 4 (exp_04):** a residual MLP on top of a *frozen* Markov-BP score beats a direct score MLP by ~20× in sample and parameter efficiency, on identical exact-score supervision. Structure preserved → large gain.
- **Layer 5 (exp_07):** EM-learned BP beats a vanilla score network by ~10× at fixed data and ≥64× in data required. Structure preserved → large gain.
- **Layer 5, the other direction (rung 4):** a mixture-density network placed *inside* the BP recursion — maximal structure preservation — is **dominated** by a 13-parameter mixture kernel: training log-likelihood −16781 vs −16656, held-out score error 0.052 vs 0.026.

So "preserving BP structure helps" is well supported, but the sharper reading is
that **structure preservation is necessary, not sufficient**: rung 4 preserves it
perfectly and still loses, because on this data the extra flexibility buys
nothing and costs optimization difficulty. The honest statement is that the
network should be inserted where the structure is genuinely unknown, not
wherever it can be inserted.

**Gap:** all three data points use chain priors where a low-dimensional family is
adequate. The question is not settled for a prior complex enough that no small
parametric family fits.

*Evidence:* compendium §4.1–4.2, Remark "More expressive is not better here";
ledger item 4.

---

## [F] From the result ledger's "explicitly future" list

### F1. Mixture-of-Gaussians message closure — **OPEN**

Identical to R4 above. See there for why it is not what Layer 5 did.

### F2. Discrete-alphabet chain (exact vector messages, no closure) — **OPEN**

Untouched. Worth noting *why* it is attractive: on a finite alphabet the messages
are vectors, BP is exact with no representation error at all, and every
discretization caveat this project has accumulated — the grid, the trapezoidal
quadrature, the ratio-lattice quantization of the Laplace M-step (§5.1) — simply
disappears. It is the cleanest possible setting in which to state the EM result,
and would make the Baum-Welch correspondence exact rather than analogical.

**To close:** a categorical chain prior + categorical EM. The `Ξ` machinery
already works unchanged, since `Ξ` is a matrix of expected transition counts and
that is *literally* what Baum-Welch uses.

### F3. Approximate Markovianity: chain + global latent; hybrid BP + learned residual — **PARTIAL**

Layer 4 established the operator-level result: for AR(1)-plus-global-latent
priors the non-Markov score correction is **exactly rank one** (Woodbury), and a
residual MLP on a frozen Markov-BP score is ~20× more efficient than a direct
score net.

**Gap:** that used the *true* chain prior. The natural product of Layer 4 and
Layer 5 — a *learned* chain kernel carrying a learned low-rank correction — is
untouched. This is the single most natural next piece of research the repository
contains.

### F4. Reverse-SDE dynamics under exact/closed/truncated scores — **PARTIAL**

exp_05 ran it: the Gaussian score reproduces second-order statistics of the
Laplace chain but washes out heavy-tailed innovations (excess kurtosis 0.12
against a true 2.7–2.9); the score error is dynamically stable in L2 (trajectory
divergence 0.11 despite pointwise deviation 0.49 at small t) yet distributionally
decisive in higher moments.

**Gap:** never run with a *learned* score. Now that Layer 5 produces one, the
obvious experiment is reverse dynamics under the EM-learned kernel versus the
network, which is also where BP's 211×–320× inference-cost penalty (§4.4) would
actually bite, since the denoiser is called at every integration step.

### F5. Non-Gaussian locality laws (basis-independent statement) — **OPEN**

Untouched. The Gaussian locality law (`q^r` decay of the radius-r estimator
error, ledger G12) has no non-Gaussian analogue yet.

---

## [E] From the Marc/Jérôme email

### E1. "Introduce learning — regress BP parameters by EM rather than train a neural network." — **ANSWERED**

This is the whole of Layer 5. The theory is in
`research/nongaussian-bp/report/em_bp_learning.pdf` (14 pp, five propositions);
the implementation is `src/em.py`, `src/kernels.py`, `src/denoiser.py`; the
results are compendium §4.

Four structural points, none of which was obvious at the outset:

1. **The E-step is exact**, not variational — the tree structure that made the score exact makes the posterior expectations exact.
2. **The entire E-step compresses into one `M×M` matrix `Ξ`**, the continuum analogue of Baum-Welch's expected transition counts, sufficient and independent of the parameterization.
3. **No autodiff is needed anywhere** — see E5.
4. **One fit serves every noise level**, because the learned object lives on `R×R` and carries no `t`.

### E2. "This does not require noising at all, I think." — **ANSWERED, and the intuition is right for a sharper reason than stated**

Two separate things were conflated in the remark, and separating them is the
useful answer.

- **Identifiability is never destroyed by noising.** `φ_{t,θ}(ξ) = φ_θ(α_t ξ)·exp(−Δ_t‖ξ‖²/2)`. The Gaussian factor has no zeros and `α_t = e^{−t} > 0` for every finite `t`, so `p_{t,θ} = p_{t,θ'}` forces `P_θ = P_θ'`. The model is identifiable at *any* noise level exactly when it is identifiable from clean data. (Proposition 4.)
- **Information is destroyed, and quickly.** Injective does not mean stable — this is Gaussian deconvolution, and the inverse map is unbounded. Measured per chain: `J[q,q]` falls **142×** from t=0.05 to t=1.6 while `J[ρ,ρ]` falls **26×**.

So the remark is correct — clean data is the `t→0` limit and the most informative
case, and if you have it you should use it (`fit_clean` does exactly that, no BP
required during fitting). But the reason is not that noise breaks identification;
it is that noise costs *information*, at a rate that hits the innovation shape
about 5× harder than the correlation. **Non-Gaussianity is the first thing the
channel destroys**, which is precisely the property this project cares about.

*Evidence:* theory §3, Prop. 4; compendium §4.6.

### E3. "If we show very efficient learning of the denoiser by EM relative to a vanilla neural network, we have something publishable." — **ANSWERED**

Relative score error, averaged over the noise schedule. EM-BP: **13 parameters**.
Network: **25,248**, and given every advantage (paired clean/noisy data, a fresh
noise draw each gradient step, both standard parameterizations, a swept
architecture):

| N chains | best network | **EM-BP** | ratio |
|---|---|---|---|
| 32 | 0.654 | **0.130** | 5.0× |
| 128 | 0.508 | **0.048** | 10.6× |
| 512 | 0.282 | **0.034** | 8.2× |
| 2048 | 0.179 | **0.016** | 10.9× |

**EM-BP on 32 chains beats the network on 2048** — a ≥64× gap in the data each
needs, and a lower bound since 32 was the smallest budget tried.

Three controls make this more than a favourable setup:

- **Capacity is not the explanation.** 24 configurations from 3.2k to 297k parameters; the best network reaches 0.208 against EM-BP's 0.022, and error *degrades* with capacity.
- **The yardstick is verified.** The reference denoiser everything is scored against agrees with 6M-sample importance sampling to 1.1e-3 relative.
- **The baseline is verified to work.** Trained on a Gaussian chain where the exact denoiser is closed-form and linear, it reaches 0.099 — so the gap is a sample-efficiency effect, not a broken competitor.

**Two things stated against the claim.** BP inference is **211×–320× slower** per
evaluation, which matters because reverse diffusion calls the denoiser at every
step; and the baseline is a vanilla MLP as specified, so a temporal CNN or U-Net
would likely close part of the gap.

### E4. "Start an Overleaf write-up in paper format." — **PARTIAL**

`report/em_bp_learning.tex` was written paper-shaped on purpose: context,
numbered propositions with proofs, detailed calculation pushed into remarks and
appendices, limitations as a first-class section. It compiles clean to 14 pp.

**Gap:** it has no results section — the numbers live in the compendium and the
CSVs — and no Overleaf project exists. Converting it is mechanical; §4 of the
compendium is the raw material.

### E5. (implicit, from the follow-up chat) "Does this require moving BP into an autodiff framework?" — **ANSWERED: no**

The advice received elsewhere was that EM here "will require moving the grid BP
forward-backward passes into an automatic differentiation framework". That is
wrong, and it is the single most useful practical finding of the layer.

Fisher's identity gives `∇_θ L(θ) = ⟨Ξ(θ), ∇_θ log K_θ⟩` — the gradient of the
*exact* marginal log-likelihood from one BP pass, with nothing differentiated
through the recursion. BP is not a computation graph to backpropagate; it is the
oracle that supplies the expectation. Verified against finite differences of the
exact evidence to ~1e-9. The entire implementation is pure numpy.

The point generalizes past this project: whenever an E-step is available exactly,
differentiating through the inference algorithm is wasted work.

---

## [L] Questions that arose during Layer 5

### L1. Does the *gradient* route work, as literally described in the email? — **ANSWERED, and the answer is not one-sided**

Both routes share the E-step and `Ξ` and differ only in what they do with it.

- **Smooth (Gaussian) kernel:** gradient ascent converges to *exactly* EM's optimum (log-likelihood matching to 2e-9 nats at η=0.5). It gains nothing and requires a tuned step size — at η=2 it diverges by 527 nats of monotonicity violation.
- **Laplace kernel:** gradient ascent is **better** — higher likelihood (−8064.7 vs −8066.6) and better on both parameters. Because there the exact M-step is the quantized one (L3).

**Recommendation:** exact M-step where it exists *and* the kernel is smooth;
gradient where the M-step has no closed form or where its exact solution is a
discretization artifact.

### L2. Is the observed EM behaviour trustworthy? — **ANSWERED**

Five independent cross-checks, by routes sharing no code with what they test:

| check | result |
|---|---|
| `Ξ` vs brute-force enumeration of all `M^n` configurations | **1.0e-14** |
| Evidence vs Monte Carlo on the Laplace chain (no closed form exists) | **0.59 s.e.** |
| EM fixed point is stationary for the *marginal* likelihood | `‖∇L‖` ÷ **8.1e6** |
| Reference denoiser vs importance sampling | **1.1e-3** |
| Baseline network can learn a closed-form linear denoiser | **0.099** |

Plus monotonicity violation of exactly 0 in every EM run across every
experiment — the sharpest single test, since an error anywhere in the recursion,
the accumulation, or the M-step generically breaks it.

### L3. Are there discretization artifacts, and do they matter? — **ANSWERED: two, both confined to the non-smooth kernel**

- **Lattice quantization.** The Laplace M-step minimizes `Σ Ξ|u_k − ρu_j|`, whose minimizer is a breakpoint — one of the ratios `u_k/u_j`, i.e. a rational. **4/5 is an attractor.** For ρ\*=0.7913, chosen off the simple lattice, ρ̂ is pinned at exactly 0.8000 across an **8× grid refinement**, holding a constant 0.0087 bias. This is a bias, not a resolution limit; more grid points do not help. It also contaminates `b` by 3–4× where ρ is snapped.
- **Gradient quadrature.** `∂_ρ log K` carries `sign(e)`; trapezoidal quadrature of a discontinuous integrand loses the spectral accuracy the rest of the package enjoys — off ~13% at M=201, still ~0.1% at M=1601. The `b` direction, being smooth, is exact to machine precision.

Both are absent for the smooth kernels, whose M-steps are ratios of smooth
moments. **Consequence adopted:** every rate and accuracy claim in the project
uses the Gaussian or mixture kernel, never the Laplace one.

### L4. Can an unknown innovation law be recovered from noisy data alone? — **ANSWERED: yes**

A mixture kernel that has never heard of the Laplace density, fitted to noisy
observations only (N=1024, one noisy realization per chain, up to t=1.6, never a
clean sample):

| C | ρ̂ | innov. var | excess kurtosis |
|---|---|---|---|
| 3 | 0.807 | 0.360 | 2.23 |
| 8 | 0.807 | 0.361 | **2.71** |
| *true* | *0.800* | *0.360* | *3.00* |

Variance exact from C=3; the heavy tail climbs monotonically with C toward its
true value; monotonicity violation exactly 0.

**Caveat, recorded because I first got it wrong:** the *kurtosis* estimate is
high-variance. At one replicate per N it reads 2.91, 3.45, 3.70, 3.29, 2.64 for
N=128…2048 — non-monotone and overshooting. The denoiser error, which is what
matters, improves monotonically (0.083 → 0.037). Any "how much data does the
heavy tail need" claim needs the replicate machinery.

---

## Where this leaves the project

**Closed:** R1, R2, E1, E2, E3, E5, L1–L4.
**Partial, gap stated:** R3 (bounded support), R5 (needs a hard prior), F3 (learned + low-rank), F4 (learned score), E4 (Overleaf, results section).
**Open:** R4 = F1 (mixture *message* closure), F2 (discrete alphabet), F5 (non-Gaussian locality).

The two most valuable open items, in order:

1. **F3 — learned chain kernel + learned rank-one correction.** The natural product of Layers 4 and 5, and the one that turns "the prior is Markov" from an assumption into an approximation with a measured correction.
2. **F2 — discrete alphabet.** Cheap, and it removes every discretization caveat the project has accumulated at once, including L3.

The most important *unasked* question, which a reviewer will ask: **does a
structured architecture (temporal CNN / U-Net) close the gap in E3?** The email
specified a vanilla network and that is what was built, but the comparison is not
complete until an architecture with its own locality prior has been tried.
