# Answers and questions for the advisors

Giovanni Mantovani — 7 August 2026. Companion to `paper/main.pdf` and `compendium/main.pdf`.

Every answer below separates four things explicitly: what is an **established result**, what is
**numerical evidence**, what is **current interpretation**, and what is **unresolved**. Where
the honest answer is "partially", it says so.

---

## Part I — Questions posed in the meetings

### A. Can belief propagation for continuous non-Gaussian variables be made operational?

**Established result.** Yes, at the level of functional messages. Coordinatewise OU noising
contributes one unary factor per site, so the posterior factor graph of a Markov chain is again
that chain (proved in `compendium` Ch. 6). A chain is a tree, so sum-product returns the exact
single-site and pairwise marginals. Nothing here is approximate.

**Numerical evidence.** Making it *computable* requires representing the messages. A truncated
uniform grid with $N_g$ points and trapezoidal weights turns each message update into one
$N_g \times N_g$ matrix–vector product, so a sequence costs $O(n N_g^2)$ against
$O(N_g^{\,n})$ for naive marginalisation — at $n=32$, $N_g=401$, about $5\times 10^6$
operations. Batching over sequences makes each update a single GEMM, which is why the same code
runs on a GPU without restructuring.

Accuracy then depends on truncation and quadrature, and these are now measured separately
(Fig. 2 of the note). Truncation responds to the domain $A$ and not to $N_g$: worst boundary mass
over 32 sites falls $3.4\times10^{-4} \to 1.4\times10^{-8} \to 1.4\times10^{-10}$ at
$A = 4, 8, 10$. Quadrature responds to the spacing: the interior column-mass residual follows
$O(h^2)$ exactly, a factor of four per halving. Validation against the closed-form Gaussian
posterior mean gives $9.2\times10^{-15}$; against brute-force enumeration of the discretised
posterior, $1.6\times10^{-14}$.

**Interpretation.** "Operational" holds for a scalar state. The $O(N_g^2)$ per update is a dense
matrix product; for a translation-invariant kernel $K(a'|a) = \varphi(a'-\rho a)$ a convolutional
structure is available and unexploited, which would bring it to $O(N_g\log N_g)$.

**Unresolved.** Nothing here scales to a vector-valued state without a further approximation:
the grid cost is exponential in the per-site dimension.

---

### B. Is a Gaussian-message approximation justified, and how inaccurate is it?

**Established result.** On this model, single-Gaussian message closure computes exactly the
**LMMSE estimator of the covariance-matched Gaussian model**
(Proposition 1 of the note; derivation in `compendium` Ch. 8). Both moment maps read only the
mean and variance of the innovation, so the recursion returns the same function of $x$ for every
innovation law with moments $(0,q)$ — in particular the same as on the Gaussian chain, where it
is exact. It is therefore exact for the Gaussian state-space model and, for every other member
of the family, exactly the best *linear* estimator.

**A terminological point we would like to confirm.** We deliberately do **not** call this AMP. A
degree-two chain supplies no large-degree central-limit justification for that terminology, so
the result is proved directly rather than argued asymptotically. We now use one name throughout,
**Gaussian second-order baseline**, and note once that on this linear model message closure and
outright model replacement coincide (they separate for richer message families).

**Numerical evidence.** The gap is largest at *small* $t$ and decays monotonically. On a Laplace
chain at $\rho=0.85$ the median relative score error falls from $0.175$ at $t=0.08$ to
$7.6\times10^{-5}$ at $t=2.4$ (Fig. 3, from
`outputs/exp_02_laplace_gaussian_message_error/laplace_summary.csv`). The ordering across
families follows innovation excess kurtosis. Large $t$ Gaussianises the posterior, so the shape
of $\varphi$ survives only where the likelihood is weak.

> One correction: an earlier draft quoted $0.198 \to 9.9\times10^{-5}$ for this pair. Those
> numbers do not appear in any committed output and were presumably from a superseded run. The
> committed medians are $0.175 \to 7.6\times10^{-5}$.

**Unresolved.** The zero-mean hypothesis in Proposition 1 is load-bearing (with $\mathbb{E}[a]=1.3$
the estimator departs by 0.65 in sup-norm), and the *fitted* mixture is not constrained to satisfy
it, so the proposition applies to the true kernel and only approximately to the estimated one.

---

### C. Does knowing the clean distribution is Markov reduce the score-learning problem?

**Established result.** Structurally, yes: the posterior chain is preserved under coordinatewise
noising, so an unrestricted $n$-dimensional regression is replaced by (i) learning a *local*
transition kernel and (ii) two sweeps of inference. The parameters live on the clean chain and
the noise level lives only in the likelihood, so one fitted kernel is a denoiser at **every**
noise level with no further fitting.

**Numerical evidence.** Twelve free parameters against 25,248, at a relative denoising error 9–14×
lower, across seven doublings of the data. Against a locality-respecting convolution the margin
is 2–4. Parameter recovery is clean: $\rho$ from $0.3$ to $[0.8497, 0.8507]$, RMSE falling at
roughly $M^{-1/2}$.

**Interpretation.** The gain depends on correct specification and on the representational
capacity of $K_\theta$. Both caveats are measurable and we measured them: capacity is the subject
of question F; misspecification is not addressed at all (see "Unresolved").

**Unresolved.** Every result assumes the Markov structure is *correct*. We have no measurement of
graceful degradation under a non-Markov prior, and that is the single largest gap in the story.

---

### D. Can the unknown Markov model be learned efficiently through EM-style methods?

**Established result.** Yes, and the structure is unusually favourable. The complete-data
log-likelihood is a sum over edges, so its $\theta$-dependent part depends on the data only
through the expected transition mass
$\Xi_{kl} = \sum_\mu \sum_i P(a_{i-1}=u_k, a_i=u_l \mid x^\mu)$ — the continuum analogue of
Baum–Welch counts. The M-step never revisits the data. Fisher's identity gives the exact gradient
of the *marginal* log-likelihood from one sum-product pass, so BP is never differentiated
through; verified against finite differences to $\sim 10^{-9}$.

**Precision about the algorithm, which the previous draft got wrong.** The E-step is exact **for
the discretised model**, not for the continuous one. The M-step for the mixture kernel is **not**
an exact maximisation: it runs four inner conditional-maximisation sweeps, so the algorithm is
**generalised EM of ECM type**. Monotone ascent is preserved; the exact-maximiser property is not.
Monotonicity is observed: the largest decrease across iterations is zero in all eight recorded
runs.

**Numerical evidence.** Both $\rho$ and the innovation density are estimated. The **initial law
$\mu$ is fixed** at $\mathcal{N}(0,1)$ — which is the true one, so nothing is misspecified, but it
is not recovered either. Fisher information per sequence for the innovation variance falls 142×
between $t=0.05$ and $t=1.6$; for the correlation, 26×. Higher-order structure is far more
fragile under the channel than second-order structure.

**Unresolved.** Identifiability is assumed, not proved (Assumption 1 of the note). Injectivity of
Gaussian deconvolution gives $P_t \Rightarrow P_0$; kernel identifiability additionally needs a
fixed or separately identifiable $\mu$ with full support, identifiability of the parametric family
(which for finite mixtures holds only on the interior of the parameter set), and identification
modulo label permutation.

---

### E. How local is the score computation? Can the required inference be localised?

**This is the question we answer least well, and we want to be blunt about it.**

**Established result.** The computation is *composed* of local updates — each message update
touches one edge — but the marginal at site $i$ depends on the whole sequence: information
propagates from both ends. There is no finite exact receptive field in general.

**Numerical evidence.** `exp_11` and `exp_12` measure how the error of a *local predictor* falls
with its window radius, and the fitted slopes match a closed-form $\log q$ prediction to 0.8–8%.
That is evidence about how much context a *denoiser* needs to reach a given accuracy.

**Interpretation.** It is a proxy, not a measurement of the score's locality. The two differ:
the first is about approximating the score with a restricted architecture, the second is about
the score's own dependence structure.

**Unresolved.** We have no direct statement. The two experiments that would give one — windowed
inference with the truncation error measured, or the Jacobian of the score with respect to
distant observations — have not been run. And `exp_12`'s receptive field was selected on the
evaluation set, so even the proxy is currently exploratory.

---

### F. Does better pointwise denoising imply better generation?

**Established result.** No, not automatically. Reverse integration accumulates errors unevenly
across the schedule, and the marginal likelihood and the squared-error loss are both dominated by
the bulk of the distribution while shape statistics are driven by the tail.

**Numerical evidence, in two parts.**

*The disagreement is real and general.* At $C=4$ the estimator is an order of magnitude better
pointwise and three times better on second-order structure, and yet the convolution reproduces
the innovation law more faithfully (kurtosis 1.357 vs 0.897, target 1.910). This holds across all
four non-Gaussian families at matched covariance, at $n \in \{32,64,128\}$, and identically to
three decimals under $N_g \in \{401, 801, 1601\}$. On the **Gaussian** chain it vanishes
(disagreement $-0.028$, within noise of zero) — the negative control the capacity account
predicts and an information account does not.

*The capacity sweep now covers both axes.* This was the open item; the run has returned.

| $C$ | MSE vs Bayes denoiser | generated excess kurtosis |
|---|---|---|
| 2 | 0.002281 | −0.034 |
| 4 | 0.000510 | 0.812 |
| 8 | 0.000276 | 1.319 |
| 12 | 0.000249 | 1.363 |
| 16 | **0.000234** | **1.487** |

Pointwise error improves 9.7× while generated kurtosis climbs monotonically. **The two axes do
not trade off** anywhere in the range tested; at $C=16$ the estimator leads the CNN on both
simultaneously.

**Interpretation.** The disagreement at small $C$ is a property of a too-small innovation model
rather than of the information surviving the channel, since enlarging the model removes it at no
cost on the other axis.

**Unresolved.** This does not show that information is never binding: the kurtosis curve has not
saturated at $C=16$. And the joint experiment is complete in three coordinates of five — marginal
likelihood versus $C$ and runtime versus $C$ are still missing.

---

## Part II — Our questions for Jérôme

1. **Framing.** $\rho$ *is* estimated (initialised at 0.3, recovered to 0.850), so we kept
   "transition-kernel estimation" in the title rather than downgrading to "innovation law".
   But the kernel is constrained to the linear form $K(a'|a) = \varphi(a'-\rho a)$. Is the
   stronger title defensible, or would you rather the restriction were in the title?

2. **Should $\rho$ and $\mu$ both be learned next?** The E-step already computes the site-one
   statistic and discards it, so learning $\mu$ is a small change. It would also cost us the
   sufficiency argument for $\Xi$ as currently stated. Worth it?

3. **Strict stationarity.** Should we replace $a_1 \sim \mathcal{N}(0,1)$ with each family's
   invariant law? It needs a fixed-point computation per family. We expect it to change nothing,
   but right now that is a prediction rather than a measurement.

4. **Where does the generation result belong?** It is currently a full section of the main text.
   Now that the capacity sweep resolves it on both axes, is it a headline result or an appendix
   ablation?

5. **Which experiment first?** Our ranking is (a) capacity versus pointwise error — *done*;
   (b) validation-based model selection for the neural baselines; (c) a direct locality
   experiment; (d) strict-stationarity rerun; (e) richer neural comparison. Would you reorder?

6. **Is the local-CNN comparison enough?** Should we add a structured autoregressive or
   mixture-density baseline — something that is *given* the sequential structure but has to learn
   the transition, which is the fairest comparison we can think of?

7. **Which generative metric should lead?** We currently emphasise innovation excess kurtosis
   because it is a scalar non-Gaussianity knob with a closed-form target at $t_{\min}$. KL and
   the covariance-lag error are also reported. Is there one you would rather see first?

8. **Diffusion contribution or structured-inference contribution?** The result is arguably more
   about what structure buys in inference than about diffusion specifically. How should this be
   framed for the thesis?

9. **How central should the locality/receptive-field direction stay?** It is the question we
   answer least well (Part I.E) and the one with the most room.

10. **What would make this publishable rather than internal?** Specifically: how much of the
    identifiability question needs proving, and how much non-Markov robustness needs measuring?

---

## Part III — Proposed weekend work plan

**Priority 1 — done in this revision.**
Notation ($K$ for the kernel, $N_g$ for the grid); stationarity corrected to covariance
stationarity; the four exactness tiers separated and applied; the reverse-SDE convention boxed
and used without variation; the $t\to0$ claim corrected; private lecture notes removed
repo-wide and replaced with public sources; numbered references; appendices A–J; ten figures
generated from committed data.

**Priority 2 — measurement gaps.**
Capacity versus pointwise error (**done**, resolved on both axes); domain/grid convergence
(**done**, truncation and quadrature separated); reverse-sampler convergence (**done** over
100–800 steps). Remaining: validation-based neural model selection; marginal likelihood and
runtime versus $C$.

**Priority 3 — presentation.** All figures regenerated, both PDFs compiled, advisor package
prepared. Done.

**Priority 4 — whatever you recommend after reading.** Held open deliberately.

---

## Part IV — Evidence map

| answer | note § | compendium | figure / table | script | output |
|---|---|---|---|---|---|
| A: BP operational | §3, App. C | Ch. 6–7 | Fig. 1 | `src/bp_grid.py` | — |
| A: grid cost & accuracy | §4.1, App. E | Ch. 9 | Fig. 2 | `exp_01`, `exp_18 --parts boundary` | `exp_01_.../grid_heatmap.csv`, `exp_18/boundary.csv` |
| B: LMMSE closure | §4.3, App. D | Ch. 8 | Prop. 1 | `src/bp_gaussian.py` | — |
| B: closure error vs $t$ | §4.3 | Ch. 8 | Fig. 3, Fig. 10 | `exp_02`, `exp_03` | `exp_02_.../laplace_summary.csv`, `exp_03_.../innovation_sweep.csv` |
| C: sample efficiency | §7 | Ch. 11 | Tab. 1, Fig. 5 | `exp_07` | `replicates/merged_summary.csv` |
| C: cost of the structure | §9 | Ch. 11 | Fig. 9 | `exp_07` | `exp_07_.../{inference,training}_cost.csv` |
| D: EM/GEM, monotonicity | §5, App. F | Ch. 10 | Fig. 4 | `exp_18 --parts emtrace` | `exp_18/em_trace.csv` |
| D: parameter recovery | §5 | Ch. 10 | Fig. 4 | `exp_06` | `exp_06_.../em_rate.csv`, `monotonicity.csv` |
| D: Fisher identity | §5 | Ch. 10 | — | `exp_08` | `exp_08_.../gradient_vs_exact.csv` |
| D: information loss 142/26 | §5 | Ch. 10 | — | `exp_06` | `exp_06_.../price_of_noising.csv` |
| E: locality proxy | §9 | Ch. 11 | — | `exp_11`, `exp_12` | `exp_12_.../efficiency_three_way.csv` |
| F: disagreement | §8 | Ch. 11 | Tab. 2 | `exp_16` | `exp_16/family_*/generation.csv` |
| F: capacity, both axes | §8 | Ch. 11 §7 | Tab. 3, Fig. 7, Fig. 6 | `exp_16` | `exp_16/{cpoint,components}_C*/…` |
| F: sampler convergence | §8 | Ch. 9 | Fig. 6 | `exp_16 MODE=calibrate` | `exp_16/calibrate_steps*/steps.csv` |
| learned innovation law | §7 | Ch. 10 | Fig. 5 | `exp_18 --parts density` | `exp_18/innovation_density.csv` |
