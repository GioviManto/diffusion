# Open questions posed by this project — and where each one now stands

Last updated 2026-08-01. Branch `claude/em-bp-denoiser-learning-e07ike`.

This document collects **every question the project has posed to itself**, from
five sources, and answers each with evidence or says plainly that it is still
open.

Sources of the questions:

- **[R]** `research/gaussian-ar1-bp/markov_gaussian_approx/report/bp_markov_diffusion_gaussian_approx.pdf`, §"Interpretation and next steps" — the five numbered next steps of the report we discussed.
- **[F]** `docs/RESULT_LEDGER.md`, §"Explicitly future" — F1–F5.
- **[E]** The Marc/Jérôme email of 2026-07-30 — the two suggestions.
- **[C]** The 2026-07-29 call with Jérôme, preceding the email — concerns raised about the grid BP and the Gaussian BP.
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

### R3. "Compare Laplace, Student, Gaussian-mixture, and bounded-support innovations." — **ANSWERED**

All four families are now covered. Laplace, Student-t (ν=5,8) and the symmetric
Gaussian mixture (κ=0.3,0.6,0.9) were swept in exp_03, spanning excess kurtosis
−1.62 to +6.00. The missing bounded-support case is now `UniformAR1`
(excess kurtosis −1.20, verified to −1.200 on samples) and was run in exp_09.

The bounded case was the interesting one because it fails a Gaussian closure
through **support** rather than shape: the true posterior is exactly zero
outside a finite interval and no Gaussian message can represent that edge. The
question was whether |excess kurtosis| survives as the one-number summary when
the failure mode changes in kind.

**It does.** Fitting `log₁₀(error) = 0.801·|kurt| − 2.234` on the three mixtures
alone and extrapolating to the uniform prior predicts 5.33e-2 at t=0.05; the
measured value is **6.07e-2**, a factor of 1.14. So the support mismatch adds
essentially nothing beyond what the kurtosis already accounts for — at least for
the posterior mean, which is what the score depends on.

That is a genuine negative result and worth stating as one: a qualitatively
different failure mode did *not* produce a qualitatively different error.

*Evidence:* `outputs/exp_09_mixture_message_closure/bounded_support.csv`; exp_03.

### R4. "Add Gaussian-mixture *message* approximations — how many components to beat single-Gaussian closure?" — **ANSWERED** (= F1)

Implemented in `src/bp_mixture.py` and measured in exp_09. Messages are Gaussian
mixtures; multiply, forward-push and backward-push are exact and closed; the
component count multiplies at every step, so Runnalls' KL-based pairwise merge
collapses back to C. **That merge is the only approximation in the recursion.**

Note this is *not* Layer 5's `MixtureInnovationKernel`, which is a mixture
**transition kernel** (a model family). Here the mixture is the **message**
representation. The distinction is the whole point, per audit F2.

**The direct answer.** With a kernel that is exactly a two-component mixture, so
that the model is exactly representable and every error is pure representation
error:

| prior | C for 10× over single-Gaussian | C for 100× |
|---|---|---|
| κ=0.3 (kurt −0.18) | 2 | 4 |
| κ=0.6 (kurt −0.72) | 2 | 4 |
| κ=0.9 (kurt −1.62) | 3 | 6 |

(at t=0.05; one to two more components at t=0.2.) **Two to three components buy
an order of magnitude; four to twelve buy two.** The error decays roughly
geometrically in C — at κ=0.3 it falls from 7.0e-3 (C=1) to 7.5e-10 (C=16).

**And the separation the project has wanted since audit F2.** On a Laplace chain,
which the mixture family *cannot* represent exactly, the curve flattens onto a
floor. That floor is model error; everything above it is representation error:

| kernel fitted with | C=1 | floor reached |
|---|---|---|
| 2 components | 5.28e-2 | 9.25e-3 |
| 4 components | 5.44e-2 | 3.33e-3 |
| 8 components | 5.76e-2 | 1.68e-3 |

The floor roughly halves per doubling of kernel richness, while the decay down to
it is bought with message components. This is the first place in the project
where the two error sources are visible as separate quantities rather than the
single conflated number audit F2 identified.

One caveat worth carrying: with a richer kernel the true message is itself more
structured, so C must grow with kernel richness to reach the floor. At 8 kernel
components, C=2 barely improves on C=1 (4.59e-2 vs 5.76e-2) — the message budget
has to keep up with the model.

*Evidence:* `outputs/exp_09_mixture_message_closure/`; `tests/test_bp_mixture.py`.

### R5. "Do neural message approximators help only when they preserve the local BP update structure?" — **PARTIAL, with evidence on three sides**

*(Updated by exp_12: a network that preserves* locality *but not the BP update
itself — a 1-D CNN — recovers 60% of the gap to EM-BP. So "preserving structure"
admits degrees, and even a weak structural prior buys most of what is available.
See N2.)*


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

## [C] From the call: "the grid BP and the Gaussian BP had problems"

The concern, as stated: the Gaussian BP was *"not updating just the mean and
variance, or information and precision, but doing something strange"*, there were
problems in the definition of Gaussian BP as well, and both the grid BP and the
Gaussian case should be tried again.

**This was correct, it was diagnosed independently in the Layer-1 audit
(`research/nongaussian-bp/audit/AUDIT_NOTE.md`), and it is fixed.** Both halves
have now been re-run and re-verified.

### C-i. "The Gaussian BP was doing something strange, not just mean/variance." — **ANSWERED: correct, and fixed**

The old routine (`grid_projected_gaussian_bp`, audit finding **F1**) evaluated
each Gaussian message *on the grid*, pushed it through the exact grid update,
then moment-matched the result. At weakly informative `t` the outgoing message is
a near-flat ramp whose maximizer lies outside `[−A, A]`; moment-matching that
**truncated** function drags the mean toward the grid boundary, and the next
update pushes it further — positive feedback, ending with the message pinned at
the edge.

The replacement (`gaussian_chain_bp`) does exactly what the call asked for:
**analytic information form**, carrying only precision `λ` and information `h`,
with closed-form updates and no grid anywhere. A flat message is `λ = 0` exactly,
so nothing is ever truncated.

Verified side by side on a Gaussian AR(1) prior, where Gaussian closure is exact
so the true answer is known:

| t | information form | legacy grid projection |
|---|---|---|
| 0.2 | 1.2e-15 | 3.9e-06 |
| 0.6 | 1.0e-15 | 9.4e-03 |
| 1.0 | 4.4e-16 | 1.1e-01 |
| 1.3 | 6.7e-16 | 4.9e-01 |
| 1.8 | 4.4e-16 | **9.1e-01** |

(max absolute error in the posterior mean). The analytic form is **exact to
machine precision at every noise level**; the legacy form is up to six orders of
magnitude worse, and worsens precisely as the likelihood stops being informative
— the signature of the mechanism above.

The consequence in the experiment itself: the old package's large-`t` rows were
contaminated (posterior-mean MSE 2.22 ± 7.47 at t=1.3). Re-run now, the Gaussian
baseline is clean and monotone — median relative score error 0.370 at t=0.02
falling smoothly to 7.6e-5 at t=2.4, with **no large-`t` blowup at all**.

*Pinned by* `tests/test_gaussian_bp_equivalence.py::test_information_form_is_exact_where_grid_projection_collapses`, which asserts both halves: the analytic form stays exact, and the legacy form keeps failing. If someone repairs the legacy routine, that test fails and says to delete the routine and the assertion together.

### C-ii. "There were problems in the definition of Gaussian BP as well." — **ANSWERED: correct, and it changes an interpretation**

Audit finding **F2**. For *linear-transition* chains `a_i = ρ a_{i−1} + ε_i` with
Gaussian OU likelihoods, moment-matched single-Gaussian BP is **mathematically
identical** to exact Gaussian BP on the covariance-matched Gaussian AR(1) model.
The reason: a Gaussian message times a Gaussian likelihood is exactly Gaussian,
and the transition step maps `N(m, v)` to a density with first two moments
`(ρm, ρ²v + q)` *regardless of the innovation shape*. The moment projection
therefore discards every property of the innovation beyond its variance.

So "Gaussian **message** approximation error" and "Gaussian **model**
approximation error" are the same object at the single-Gaussian level. What the
old package reported as a message-representation error was a model error. The
values were right; the label was not.

The distinction becomes real only for richer message families or nonlinear
transitions — which is exactly why **R4/F1 (mixture-message closure) is the
open question worth doing**, and why it must not be confused with Layer 5's
mixture *kernel*.

*Pinned by* the equivalence tests in `tests/test_gaussian_bp_equivalence.py`.

### C-iii. "We should try it again — the grid BP and the Gaussian case." — **ANSWERED: both re-run**

Re-run under the corrected code *and* the fixed seeding (their committed outputs
predated the `rng_for` fix, so they had never been bit-reproducible).

- **Grid BP** is spectrally accurate at the working default: worst relative error **9.2e-15** at M=401, A=8. The failure modes are visible and understood rather than hidden — M=101 with A=8 gives 1.6e-2 (too few points per unit length), while A=4 gives 1.1e-7 for *every* M (truncation-limited, so refining M cannot help).
- **Gaussian BP** as above: clean, monotone, artifact-free, with the score/mean identity holding to ≤2.9e-12 on every row.

Two further audit findings were fixed in the same package and are worth naming
because both recurred later in this project: **F3** (grid sizes compared on
different random trials — the identical confound I reintroduced in exp_06 Part 5
and had to fix, see L3) and **F4** (linear-domain likelihood rows underflowing to
all-zero, fixed by working in the log domain with per-row max subtraction).

### C-iv. "Then do the learning Marc was asking, and address all the questions." — **ANSWERED**

The learning is Layer 5 — see **E1**, **E3**. The questions are this document.

---

## [F] From the result ledger's "explicitly future" list

### F1. Mixture-of-Gaussians message closure — **ANSWERED**

Identical to R4 above; see there for the numbers and for why it is not what
Layer 5 did.

### F2. Discrete-alphabet chain (exact vector messages, no closure) — **ANSWERED**

Implemented in `src/discrete.py`, measured in exp_10. The clean chain takes
values in a finite set of levels; the noising is still the continuous OU channel,
so the observations are real-valued, but the latent alphabet is finite and the
messages are S-vectors. **BP is exact up to roundoff** — verified against
explicit enumeration of all `S^n` configurations: Ξ to 4.4e-15, posterior means
to 2.9e-15, log-evidence to *exactly* zero difference.

Every discretization caveat the project accumulated disappears simultaneously:
no quadrature, no truncation, no resolution condition at small t, and in
particular no ratio-lattice quantization (L3), which was a *bias* that no amount
of grid refinement could remove.

Two things this buys beyond cleanliness.

**The Baum-Welch analogy becomes an identity.** In the continuous case Ξ is "the
continuum analogue of the expected transition-count matrix". Here it *is* the
expected transition-count matrix, and the M-step is exactly Baum-Welch's:
normalize the counts. Where the continuous case needed a separately derived
M-step per kernel family — weighted Yule-Walker, a weighted median, an inner ECM
— the transition matrix is its own parameterization and the maximizer is one
line. Monotonicity violation is **exactly 0.0**, not "below 1e-8".

**A confound-free retest of the headline.** With the reference denoiser exact
rather than a fine-grid approximation (S=5, 20 learned parameters against the
network's 25,248):

| N | best network | **EM-BP** | ratio |
|---|---|---|---|
| 32 | 0.761 | **0.184** | 4.1× |
| 128 | 0.641 | **0.085** | 7.5× |
| 512 | 0.398 | **0.052** | 7.7× |
| 2048 | 0.263 | **0.032** | 8.3× |

EM-BP on 32 chains (0.184) again beats the network on 2048 (0.263), reproducing
the ≥64× data-efficiency gap with **zero grid error**. Monotonicity violation
was exactly 0.0 in every run; identity residual ≤1.4e-13.

**Two honest findings from the alphabet sweep**, both of which qualify the
headline rather than support it.

*The advantage erodes as the model grows.* At N=512:

| S | EM params | ratio over best network |
|---|---|---|
| 3 | 6 | 15.4× |
| 4 | 12 | 17.7× |
| 6 | 30 | 8.0× |
| 8 | 56 | 5.7× |
| 12 | 132 | 4.4× |

Still 4.4× ahead at 132 parameters, but the trend is unambiguous: the
sample-efficiency advantage is tied to the parametric dimension being small,
exactly as Prop. 5 would predict. Nothing here says a structured estimator wins
once its parameter count approaches the network's.

*The recovery rate degrades with S, and it is not a metric artifact.* Fitted
slopes go −0.55 (S=3), −0.44 (S=4), −0.37 (S=6), −0.18 (S=8). Suspecting the
max-norm over `S²` entries, I re-ran with a per-entry metric: at S=8 the slopes
are −0.123 (max) and −0.107 (mean), so the degradation is real.

The mechanism is **level crowding**. At fixed unit marginal variance the S
levels are spaced ~1/S apart, while the OU channel resolves differences only
down to `sqrt(Δ_t)/α_t`. The ratio spacing/resolution:

| S | t=0.1 | t=0.2 | t=0.4 | t=0.8 |
|---|---|---|---|---|
| 4 | 1.90 | 1.28 | 0.81 | 0.45 |
| 8 | **0.93** | 0.62 | 0.39 | 0.22 |
| 12 | **0.62** | 0.41 | 0.26 | 0.15 |

Below 1 the adjacent levels are not distinguishable through the channel at all.
At S=8 they are already unresolvable at the *lowest* training noise level, so
most of the schedule contributes almost no information about which of two
neighbouring levels was visited. This is the same phenomenon as §4.6's
information collapse, in a setting where it can be computed exactly rather than
measured: it is an identifiability limit imposed by the channel, not a failure
of the estimator.

*Evidence:* `outputs/exp_10_discrete_alphabet/`; `tests/test_discrete.py`.

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

### F5. Non-Gaussian locality laws (basis-independent statement) — **ANSWERED**

Measured in exp_11. The sharp version of "basis-independent" is: *is the
locality decay rate a property of the second-order structure alone?*

**The exponential law survives for every innovation family, but the rate is not
second-order universal.** Six families at matched (ρ, q), so they differ only
beyond second moments. Ratio of fitted rate to the Gaussian rate, averaged over
families:

| ρ | t=0.05 | t=0.10 | t=0.20 | t=0.40 | t=0.80 |
|---|---|---|---|---|---|
| 0.50 | **2.08** | 1.65 | 1.19 | 1.05 | 0.99 |
| 0.85 | **1.47** | 1.19 | 1.06 | 1.02 | 1.00 |
| 0.95 | 1.10 | 1.01 | 1.01 | 1.00 | 1.00 |

Non-Gaussian chains are **less local** than their covariance-matched Gaussian,
and the excess is governed by two knobs, vanishing in both limits: it is largest
at **low noise and weak correlation**, and is gone by t ≳ 0.4 or ρ ≳ 0.95. The
ρ-dependence is the same mechanism R2 found for the closure error — strong
correlation concentrates the messages and makes them look Gaussian.

**The architectural consequence, which is what B15 cares about.** Since
r ~ log(1/ε)/log(1/q), a receptive field sized on Gaussian intuition is too
small by:

| family | worst case | factor |
|---|---|---|
| gauss_mix κ=0.9 | ρ=0.85, t=0.05 | **2.45×** |
| gauss_mix κ=0.9 | ρ=0.50, t=0.05 | 2.32× |
| uniform | ρ=0.10, t=0.10 | 1.27× |
| laplace | ρ=0.50, t=0.05 | 1.21× |

So B15's prescription is a *Gaussian* statement with a correction factor up to
~2.5× for bimodal data at low noise. exp_12 tests whether a trained local head
actually behaves this way.

**And the governing scalar is not the one that governs closure.** The closure
error tracked |excess kurtosis| so well it predicted the bounded-support case to
within 14% (R3). The locality excess does not: Student-t with kurtosis **+6.0**
sits at 1.09–1.25, while bimodal κ=0.9 with |kurtosis| **1.62** reaches 4.94 in
rate (2.45× in radius). Bimodality specifically — not tail weight — is what
destroys locality. Identifying the right scalar is open.

*Evidence:* `outputs/exp_11_nongaussian_locality/`.

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

**Closed:** R1, R2, **R3**, **R4 = F1**, **F2**, **F5**, E1, E2, E3, E5,
C-i–C-iv, L1–L4. Every question the project originally posed to itself now has
an answer or a measured bound.
**Partial, gap stated:** R5 (needs a hard prior), F3 (learned + low-rank),
F4 (learned score), E4 (Overleaf, results section).
**Open:** none of the original questions. Two *new* ones were opened by the
answers, below.

The most valuable remaining item is **F3 — a learned chain kernel carrying a
learned rank-one correction.** It is the natural product of Layers 4 and 5, and
the one that turns "the prior is Markov" from an assumption into an
approximation with a measured correction. F4 (reverse dynamics under a *learned*
score) is now unblocked too, since Layer 5 produces such a score and exp_10
provides a setting where the reference is exact.

### New questions raised by the answers

**N1. What scalar governs locality?** — **ANSWERED: negentropy, not kurtosis.**

The candidate screened against the measured locality excess: |excess kurtosis|,
negentropy (`KL(f ‖ N(0,q))`, the KL divergence from the variance-matched
Gaussian), L¹ and L^∞ distance to that Gaussian, and mode count. Negentropy wins
decisively, and *within the cells where a locality effect exists at all*:

| cell | negentropy | \|excess kurtosis\| |
|---|---|---|
| ρ=0.50, t=0.05 | **+0.962** | −0.055 |
| ρ=0.50, t=0.10 | **+0.994** | −0.093 |
| ρ=0.85, t=0.05 | **+0.980** | −0.067 |
| ρ=0.85, t=0.10 | **+0.975** | −0.014 |
| pooled (30 pts) | **+0.738** (Spearman +0.842) | −0.043 (Spearman +0.483) |

(Pearson correlation with `rate_over_gaussian`.) Over the 60 points where the
effect has already vanished — large t, or ρ=0.95 — negentropy correlates at
−0.078, i.e. nothing, which is the correct behaviour when there is no signal to
predict.

**Why this resolves the puzzle.** Negentropy is scale-invariant, so it is a pure
*shape* statistic, one number per family:

| family | negentropy | \|excess kurtosis\| | locality excess (ρ=0.5, t=0.05) |
|---|---|---|---|
| gauss_mix κ=0.6 | 0.026 | 0.72 | 1.12 |
| student-t ν=5 | 0.045 | **6.00** | 1.25 |
| laplace | 0.072 | 3.00 | 1.62 |
| uniform | 0.177 | 1.20 | 1.46 |
| gauss_mix κ=0.9 | **0.462** | 1.62 | **4.94** |

Student-t has the largest kurtosis of any family and the second-*smallest*
negentropy: heavy tails carry huge fourth moments but very little probability
mass, so they sit close to a Gaussian in KL. Bimodality moves mass, so κ=0.9 is
10× further in KL than Student-t while having a quarter of its kurtosis. Ordering
the families by negentropy orders them by locality damage; ordering by kurtosis
does not.

That locality — an information-flow property — is governed by an
information-theoretic distance rather than a moment is the natural answer in
hindsight, and it is a *different* scalar from the one that governs closure
error, where kurtosis works well enough to extrapolate (R3). Two distinct
mechanisms, two distinct summaries.

**Scope, stated plainly:** six families, one chain model, correlation not a
derivation. No functional form is claimed — the relationship is monotone and
strongly correlated, not fitted.

**N2. Does a structured architecture close the EM-BP gap?** — **ANSWERED: most
of it, but the remainder grows with data rather than closing.** Replicated and
oracle-fair: 4 seeds, oracle over both receptive-field radius and
parameterization for both networks, mean relative score error over five noise
levels (± standard error over seeds).

| N | EM-BP (13 params) | global MLP (25,505) | local CNN, oracle r (6,081) |
|---|---|---|---|
| 128 | **0.0368 ± 0.0067** | 0.3595 ± 0.0044 | 0.0774 ± 0.0024 |
| 512 | **0.0174 ± 0.0017** | 0.1794 ± 0.0035 | 0.0570 ± 0.0007 |
| 2048 | **0.0123 ± 0.0009** | 0.1246 ± 0.0027 | 0.0505 ± 0.0015 |

Against the vanilla MLP the margin is ~10× and flat in N. Against the
locality-respecting CNN it is 2.10 ± 0.39 → 3.27 ± 0.32 → **4.11 ± 0.32** as N
goes 128 → 512 → 2048. So roughly 60% of the vanilla deficit was the
architecture rather than the estimator — but **the remaining margin widens with
data**, because EM-BP keeps paying down parametric error (0.0368 → 0.0123 over
16× data) while the CNN flattens toward an approximation floor (0.0774 →
0.0505). "More data will close the gap" is the natural guess and it is wrong
here.

This supersedes the single-seed figures previously recorded here (3.3× at fixed
r = 6, 2.75× with an oracle over r, both at N = 1024, which lies between the
second and third rows). It was the most important unasked question in the
project; the answer qualifies the headline and then partly un-qualifies it.

The most important *unasked* question, which a reviewer will ask: **does a
structured architecture (temporal CNN / U-Net) close the gap in E3?** The email
specified a vanilla network and that is what was built, but the comparison is not
complete until an architecture with its own locality prior has been tried.

---

## [P] Questions raised by the two papers (Layer 6)

Context and the access caveat are in `docs/PAPER_CONNECTIONS.md`: **neither PDF
could be opened from this environment** (gateway 403 on arxiv, nature, PMC, HAL,
PMLR and every mirror tried), so the papers were identified and their content
established through web search alone. Nothing mathematical below is quoted from
them — every formula is derived in this package's conventions and verified in
`tests/test_hierarchy.py`.

### P1. Does a hierarchical prior show a *ladder* of speciation times, one per level? — **ANSWERED: yes, six of them, each at its predicted place**

Neither paper asks this: one supplies the hierarchical data model, the other the
speciation time scale, and the product is untouched. The balanced-tree prior has
an ultrametric leaf covariance with exactly `L + 1` distinct eigenvalues (closed
form in `GaussianTree.level_eigenvalues`, matched to `eigh` at 1e-10), so it has
`L + 1` speciation times.

Measured on a depth-5 binary tree, ρ = 0.9 (exp_13 `cascade`): predicted
`t_S = ½ log(1 + Λ)` versus the measured crossing of the commitment curve
through `1/√2` — **1.363/1.368, 0.707/0.732, 0.516/0.521, 0.346/0.340,
0.202/0.208, 0.087/0.087** along the forward process, agreeing to ≤ 3.5% at
every level. The reverse SDE driven by exact tree BP reproduces the same ladder
and therefore **resolves the hierarchy coarse-to-fine**.

Reported rather than trimmed: the reverse crossing at the finest level lands at
0.125 against 0.087 predicted — the Euler–Maruyama step and the `t_min = 0.02`
floor, which bite hardest exactly there.

### P2. Does the chain this project has been studying sit in that regime? — **ANSWERED: no, and this bounds what Layers 1–5 can be generalized to**

The AR(1) top eigenvalue is bounded by `(1 + ρ)/(1 − ρ)` at any length: 64× the
chain length moves it under 2× (0.85: 5.49 → 12.32 against a limit of 12.33). So
its speciation time **saturates**, and a stationary Markov chain has no diverging
speciation time and no cascade. Only the hierarchy produces them. Any claim of
this project transferred to image-like data has to cross that gap explicitly.

### P3. Does the collapse/memorization mechanism apply to a BP score? — **ANSWERED: the axis does not exist for it**

Not a matter of degree. The empirical score's sufficient statistic is the
training set — `N × n` numbers — which is why it must collapse onto that set
unless `N` is exponential in `n`. A BP score's sufficient statistic is the
fitted kernel, reached only through `Ξ`, which is an average; its size does not
depend on `n`, and there is no training point for a trajectory to be captured
by.

Measured as nearest-training-chain distance over the same quantity for a fresh
true draw (1.0 = no memorization), ρ = 0.85, exp_14 `collapse`: the empirical
score sits at **0.33–0.66** and recovers monotonically as `N` crosses its
entropic wall, while EM-BP sits at **0.97–1.08** at every `n` and `N` measured,
tracking the true-prior BP reference to within 0.03. The network is
consistently slightly below 1.

**This is the mechanism behind Layer 5's headline.** The ≥64× data advantage and
the `N^{−1/2}` rate were reported in Layer 5 as empirical facts; this is why
they hold. The per-site excess entropy is exact for this family,
`s = −½ log(1 − ρ²) = 0.641` nats at ρ = 0.85, so n = 33 already demands ~10⁹
chains for a memorizing score.

**Caveat carried with the ratio:** it only means anything when the generated
distribution is otherwise correct, so sample std and lag-1 correlation are
logged in every row. An undertrained network reached ratios of 2–3.4 with std
2.3–3.3 — over-dispersion, not generalization.

### P4. Can EM through exact BP be run on a tree at all? — **ANSWERED: yes, and the M-step is literally unchanged**

`tree_e_step` produces the **same `Ξ`** as the chain E-step. That is where the
topology stops being visible, so every kernel in `src/kernels.py` consumes it
without modification — a chain-trained M-step and a tree-trained M-step are the
same code. Evidence verified against the closed-form Gaussian marginal
likelihood to relative 1e-6, which is what keeps the monotone-ascent check
available.

**But the convergence rate is different, and this matters practically.**
`ρ̂ = 0.7360 → 0.7483 → 0.7487` at 50/100/150 iterations (true 0.75), zero
monotone violation throughout. Internal nodes are never observed, so the missing
information fraction is far larger than on a chain. An estimate read at 40
iterations looks like a broken M-step and is merely unconverged — this cost real
time here.

### P5. Does the network acquire hierarchy levels in an order, as it does in the transformer paper? — **ANSWERED: no, and the more interesting statement is about *where* its error sits**

**No sequential acquisition.** The MLP denoiser's per-level error is flat across
levels *and* across training budget (250 → 16 000 gradient steps), and its total
plateaus by roughly 2 000 steps at N = 256 trees. The reading is that this is a
data-limited regime, not evidence against sequential acquisition — the network
never gets far enough for an ordering to become visible — and it is a different
architecture, task and loss from the paper's. The cluster script scales this up.

**But the level-resolved error is decisive about something else.** Against the
right null — a method with uniform per-mode error, whose share of total squared
error would be proportional to each level's multiplicity — the concentration
ratio spreads across levels by **1.6–2.2× for the ε-network** (flat: as if the
hierarchy did not exist), 2.8–12.6× for the x₀-network, and **224–6957× for
EM-BP**, migrating from the finest levels at small `t` to the single coarsest
mode at large `t`. That is a quantitative sense in which BP inherits the
hierarchy from the graph and the network does not.

EM *does* show an ordering, with a transparent mechanism that is worth not
over-reading: ρ̂ starts far too small, which destroys long-range structure
first, so the coarsest level carries nearly all the error early (1.29 against
0.02 at level 1 after one iteration) and the levels equalize as ρ̂ climbs. That
is the parameterization, not an inductive bias.

One methodological note already established from the smoke runs: **the
per-level error must be reported absolutely, not only relatively.** A relative
error divides by the reference magnitude in that subspace, which shrinks with
the level's eigenvalue, so fine levels look worse for every method — including
one with uniform absolute error. The first version of this measurement showed a
clean coarse-to-fine gradient that was entirely the normalization. All three
forms (relative, absolute, share of total squared error) are now written to the
CSV.

### P6. Does the EM-BP advantage grow with the range of correlations in the data? — **PARTIAL: it does not shrink, and the design cannot cleanly say more**

Filtering a depth-4 tree at level `k` leaves correlations only inside blocks of
`2^k` leaves. At fixed sequence length, data budget, network and training
budget:

| k | block | ρ̂ | EM-BP abs err | network abs err | advantage |
|---|---|---|---|---|---|
| 1 | 2 | 0.8796 | 0.0154 | 0.4143 | 28.7× |
| 2 | 4 | 0.8926 | 0.0085 | 0.3702 | 49.2× |
| 3 | 8 | 0.9010 | 0.0038 | 0.3640 | 99.0× |
| 4 | 16 | 0.9041 | 0.0084 | 0.3470 | 39.4× |

The network's error is essentially flat in `k` (0.414 → 0.347, mildly
*decreasing*): more long-range structure neither helps nor hurts it, which is
the same blindness P5 quantifies. The advantage is 29–99× throughout.

**The confound, which is why this is PARTIAL.** Fixing the number of *sequences*
does not fix the number of *edges*: a k-filtered depth-4 tree has `32 − 32/2^k`
edges, i.e. 16, 24, 28, 30 for k = 1…4. Larger `k` therefore hands EM more data
at the same nominal budget, and ρ̂ improves monotonically as a result. The sweep
moves two things at once, so the *rising* advantage cannot be attributed to
correlation range. What the table supports without qualification is the
negative: the advantage does not shrink as correlations reach further, which is
what one would expect if the network's locality were the binding constraint.
Isolating the effect needs a version that fixes the edge count — not done.
