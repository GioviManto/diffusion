# Revision audit

Claim-level audit of `paper/main.tex` (the internal note) and `compendium/main.tex`
against the implementation and the committed outputs. Written before any rewriting.

Every row records: the claim as it currently stands, where it is made, what actually
supports it, an evidence tier, and the correction required.

**Evidence tiers.**

| tier | meaning |
|---|---|
| **T1** | Exact analytical statement (theorem or identity, proved here or standard). |
| **T2** | Exact inference on a tree at the level of functional messages. |
| **T3** | Exact computation for the finite discretised state-space model, up to floating point. |
| **T4** | Numerical approximation to the continuous model; error measured, not assumed. |
| **T5** | Monte Carlo / finite-sample empirical estimate. |
| **T6** | Interpretation, hypothesis, or open question. |

A claim is mis-tiered whenever the prose asserts a stronger tier than the evidence supports.
That is the single most common defect found below.

---

## A. Findings that change what the documents may claim

### A1. The chain is covariance-stationary, not strictly stationary — **CORRECTION REQUIRED**

*Claim.* `paper/main.tex:165` — "Choosing $q = 1-\rho^2$ makes the chain stationary with unit
marginal variance and $\mathrm{Cov}(a_i,a_j)=\rho^{|i-j|}$ *independently of the shape of
$\varphi$*." The docstring of `src/priors.py:38` likewise says "A **stationary**
linear-AR(1)-type Markov chain".

*What the code does.* Every family in `src/priors.py` initialises

```python
a[0] = rng.standard_normal()          # GaussianAR1.sample, LaplaceAR1.sample,
                                      # StudentTAR1.sample, UniformAR1.sample,
                                      # GaussianMixtureAR1.sample — all identical
```

i.e. $a_1 \sim \mathcal N(0,1)$, followed by non-Gaussian innovations. The invariant law of
$a_i = \rho a_{i-1} + \varepsilon_i$ for non-Gaussian $\varepsilon$ is *not* Gaussian (it is
the law of $\sum_{k\ge0}\rho^k\varepsilon_{-k}$), so $a_1$ is not drawn from it.

*What is true.* Second moments propagate exactly: $\mathrm{Var}(a_i)=1$ for all $i$ and
$\mathrm{Cov}(a_i,a_j)=\rho^{|i-j|}$, for every variance-matched innovation law. This is
**T1** and is the property the controlled comparison actually needs. Strict stationarity in
distribution is **false** for the non-Gaussian families: the marginal law of $a_i$ drifts from
Gaussian towards the invariant law over a burn-in of order $1/\log(1/\rho) \approx 6$ sites at
$\rho=0.85$.

*Correction.* Say **covariance-stationary** (equivalently second-order stationary, or
covariance-matched) everywhere. Add an explicit sentence stating that the construction is not
strictly stationary for non-Gaussian innovations and why that does not affect the comparison
(all families share the same $a_1$ law and the same covariance, so they still differ only
beyond second moments). Fix the `src/priors.py` docstring too. Raise the alternative — sampling
the invariant law — as an advisor question, since it is a real experiment, not a wording choice.

*Collateral.* The E-step defaults `log_mu` to $\mathcal N(0,1)$ (`src/em.py:167`), which is
*exactly* the law the priors sample from. So the fixed initial law is **correct**, not
misspecified. That is worth stating: there is no $\mu$-misspecification anywhere in the results.

---

### A2. $\rho$ *is* estimated — the transition-kernel framing survives — **CLAIM CONFIRMED**

*Claim under test.* Whether the title's "estimation of the transition kernel" overstates what
is learned, and whether the honest framing is "estimation of the innovation law at known $\rho$".

*What the code does.* `src/kernels.py`, `MixtureInnovationKernel.m_step` updates $\rho$ in a
dedicated block inside each inner sweep:

```python
num = float((wr * inv_s2 * grid[None,None,:]
             * (grid[None,:,None] - current.mu[:,None,None])).sum())
den = float((wr * inv_s2 * (grid**2)[None,None,:]).sum())
current = replace(current, rho=num / den)
```

and every experiment initialises it **away from the truth**:

| experiment | init | truth |
|---|---|---|
| `exp_07_em_vs_score_network.py:165,218,285` | `rho=0.3` | 0.85 |
| `exp_16_sampling_validation.py:302` | `rho=0.3` | 0.85 |
| `exp_06_em_parameter_recovery` (Laplace) | random inits incl. $-0.42$ | 0.80 |

`outputs/exp_06_em_parameter_recovery/monotonicity.csv` shows recovery to
`rho_err = 1.1e-16` from initialisations spanning $[-0.42, 0.55]$.

*Verdict.* $\rho$, the mixture weights, locations and scales are **all** estimated jointly from
noised sequences. The autoregressive coefficient is not supplied. **Tier T5** (finite-sample
estimation) with **T1** identities behind the M-step. The stronger title is *earned* and should
be kept.

*Correction.* State the parameterisation completely and say explicitly that $\rho$ is estimated
and initialised at 0.3 against a truth of 0.85 — currently the reader cannot tell. Say equally
explicitly that the **initial law $\mu$ is held fixed** at the (correct) $\mathcal N(0,1)$ and
is not estimated, and that the kernel is assumed **linear** ($K(a'|a)=\varphi(a'-\rho a)$), so
what is free is the innovation *density* and the single autoregressive coefficient — not an
arbitrary bivariate transition density.

*Free-parameter count.* For $C$ components: $\rho$ (1) $+$ weights ($C$, one simplex constraint,
so $C-1$) $+$ locations ($C$) $+$ scales ($C$) $= 3C$. At $C=4$ that is **12**, which matches
the number in Table 1 and `N_COMPONENTS = 4` at `exp_07_em_vs_score_network.py:80`. **Confirmed
correct**; the derivation should be shown rather than asserted.

---

### A3. "Exact E-step" is exact for the *discretised* model — **CORRECTION REQUIRED**

*Claim.* `paper/main.tex:434` "Exact expectation step"; `src/em.py` module docstring "Exact EM
… the E-step is exact"; several compendium passages.

*What is true, in tiers.*

| statement | tier |
|---|---|
| The posterior factor graph of a coordinatewise-noised chain is a tree | **T1** (proved) |
| Sum–product on a tree returns the exact functional marginals | **T2** |
| The grid recursion returns those marginals for the finite-state model it defines | **T3** |
| That finite-state model approximates the continuous one | **T4**, error measured |

The E-step is **T3**: exact for the discretised latent model, not for the continuous one.
Remark 1 of the note already draws these distinctions correctly — but the rest of the note and
the compendium then use the bare word "exact" anyway, which undoes it.

*Correction.* Purge bare "exact" from claim positions. Use "exact for the discretised model",
"exact tree inference", or "high-resolution grid-BP reference" as appropriate. In the empirical
tables, rename the `exact` column: it is a **reference arm computed by grid BP under the true
kernel**, and where it reports a sample statistic it is itself a **T5** estimate. Table
`tab:families` currently labels a Monte Carlo median as `exact`, which is the worst instance.

---

### A4. The score does **not** generically diverge as $t\to0$ — **CORRECTION REQUIRED**

*Claim.* `paper/main.tex:577` "Integration must stop at $t_{\min}>0$, since $S$ diverges as
$\Delta_t\to0$." Also `compendium/chapters/ch03-diffusion.tex:147` and
`ch09-numerics.tex:97`.

*Why it is wrong as stated.* $S(x,t) = -(x - e^{-t}\langle a\rangle_{x,t})/\Delta_t$. As
$t\to0$ the numerator vanishes at the same order: for a $P_0$ with a $C^1$ density,
$S(x,t)\to\nabla\log P_0(x)$, which is finite. The $1/\Delta_t$ is not by itself a singularity.

*What is actually going on, separated.*

1. **Genuine singularity, family-dependent.** For `UniformAR1` the clean density has compact
   support, so $\nabla\log P_0$ is genuinely unbounded at the support boundary. For `LaplaceAR1`
   the density has a kink, so the score has a jump discontinuity, not a divergence. For the
   Gaussian and mixture chains $\nabla\log P_0$ is bounded on compacta. **T1**, per family.
2. **Numerical cancellation.** The implementation forms a difference of two nearly equal
   quantities and divides by $\Delta_t\approx 2t$; relative error is amplified by $O(1/t)$.
   **T4**.
3. **Narrowing likelihood factors.** $\ell_t(x_i|a_i)$ has width $\sqrt{\Delta_t}$; once that is
   comparable with the grid spacing $h$ the quadrature resolves it with $O(1)$ points. At
   $A=8$, $N_g=401$, $h=0.04$, so this bites below $t\approx 10^{-3}$. **T4**.
4. **Integrator stiffness.** The reverse drift's Jacobian scales like $1/\Delta_t$, so the
   Euler–Maruyama step must shrink accordingly. `src/reverse.py:44` already uses a geometric
   time grid for exactly this reason. **T4**.

*Correction.* Replace the divergence claim with the resolution/conditioning statement, and note
the one family (uniform) where a genuine singularity does exist. Keep $t_{\min}>0$ — it is
still required — but for the right reason.

---

### A5. Gaussian closure: keep the LMMSE result, drop nothing, add no AMP — **MOSTLY CLEAN**

*Checked.* `grep -rn "AMP\|approximate message passing"` over `paper/` and
`compendium/chapters/` returns **no matches**. The documents never invoke AMP, so there is no
unjustified large-degree central-limit appeal to remove. Good.

*What Proposition 2 says and what it needs.* Moment-projected sum–product with linear
transitions, Gaussian unary likelihoods and $\mathbb E[a]=0$ returns
$e^{-t}\Sigma_0(e^{-2t}\Sigma_0+\Delta_t I)^{-1}x$, the LMMSE estimator of the
covariance-matched Gaussian model, for every innovation law with moments $(0,q)$. The proof in
the note is correct; its last sentence is garbled ("exact inference returns the posterior mean,
which for jointly Gaussian zero-mean $(a,x)$ the posterior mean is linear…").

*Correction.* Fix the garbled sentence. Adopt one stable name — **Gaussian second-order
baseline** — and use it everywhere instead of alternating between "message closure", "moment
projection" and "Gaussian model replacement"; state once, in the remark, that on *this* linear
model the three coincide, and that they separate for richer message families. Say explicitly
that the degree-2 chain admits no large-degree CLT justification, which is why the result is
proved directly rather than argued asymptotically.

*Zero-mean hypothesis.* Load-bearing and already quantified in the note (a prior mean of 1.3
displaces the estimator by 0.65 in sup-norm). Keep.

---

### A6. Innovation centering is **not** enforced — **ALREADY HONEST IN CODE, MUST REACH THE PAPER**

*What the code does.* `src/kernels.py`, `MixtureInnovationKernel` docstring:

> The component means are left unconstrained. Imposing the natural-looking zero-mean condition
> $\sum_c \pi_c \mu_c = 0$ by recentering after each update is not harmless: the projection does
> not maximize $Q$ and it broke monotone ascent by ~1e-2 nats in testing.

So the fitted family does **not** satisfy the hypothesis of Proposition 2. The note says this
at `main.tex:492`, which is correct and should be kept and made more prominent — it currently
sits inside the identifiability paragraph where it reads as an aside.

*Correction.* Promote to its own statement: the fitted innovation mean is a **diagnostic**
(observed at the $10^{-3}$ level), not a constraint; Proposition 2 therefore applies to the true
kernel and only approximately to the estimated one. Report the fitted mean as a number in the
appendix. Do not describe the observed near-zero value as if it were imposed.

---

### A7. Identifiability is assumed, not proved — **ALREADY QUALIFIED, KEEP AND SHARPEN**

The note (`main.tex:482–496`) already separates injectivity of Gaussian deconvolution (which is
**T1**: the characteristic function acquires a non-vanishing factor $e^{-\Delta_t\|\xi\|^2/2}$)
from *kernel* identifiability (which does **not** follow), and lists the extra conditions:
initial law fixed, family identifiable, mixture modulo label permutation, interior of the
parameter set, full-support $\mu$. It says "We assume these rather than prove them."

*Correction.* This is the right treatment. Move it to a clearly-labelled **Assumption** block so
it cannot be skimmed as prose, and cross-reference it from the limitations section. No
substantive change needed. **T6.**

---

### A8. Neural comparison — information asymmetry and one test-set selection — **CORRECTION REQUIRED**

*Facts from the code.*

- `exp_07_em_vs_score_network.py:84` builds one test bundle (`N_TEST = 256` held-out chains,
  seed `exp07-test`), used by every arm. Training seeds are mixed with the replicate index, so
  replicates vary both the training data and the initialisation.
- There is **no validation split**. Model selection that happens, happens on the test set.
- `exp_12_receptive_field.py:154`: `best_r = cfg["compare_radius"]` — the CNN receptive field is
  a fixed configuration value read off the same sweep that is evaluated on the test set.
- Table 1's network column reports, at each noise level, the better of the $\varepsilon$- and
  $a$-parameterisations. That is **oracle post-selection over two models per noise level**,
  favouring the baseline.

*Correction.* (i) Never write that BP or EM "beats neural networks"; write the scoped sentence:
*under the correctly specified Markov model and the tested data budgets, the structured
estimator is more sample-efficient in pointwise denoising than the selected neural baselines*.
(ii) State the information asymmetry as a first-class caveat: the estimator is given the
factorisation, the linear-autoregression form, and homogeneity across sites; the networks are
given none of these. (iii) Label the per-noise-level "better of two parameterisations" as oracle
post-selection *in the caption*, and note it favours the baseline. (iv) Keep the existing
`[pending]` on the receptive-field rerun and mark those numbers exploratory — the note already
does this at `main.tex:562`; it must survive the rewrite. (v) Appendix G must carry parameter
counts, optimiser, steps, batch sizes, seeds and hardware.

---

### A9. Pointwise-vs-generation: the **[pending]** is now resolved — **NEW RESULT**

*Prior state.* The note (`main.tex:689`) says: "What this does not yet establish is whether the
richer kernel keeps its pointwise advantage… If pointwise error degrades as $C$ grows, the two
axes trade against each other and the reading changes. That run is in progress."

*The run has returned.* SLURM job `617485_[0-4]` (`cpoint_C{2,4,8,12,16}`, all `COMPLETED`),
outputs now committed at `outputs/exp_16/cpoint_C*/pointwise.csv` (45 rows each = 3 arms × 5
noise levels × 3 seeds).

Median MSE against the Bayes denoiser, Laplace chain, $M=2048$:

| $C$ | EM–BP | local CNN | global MLP |
|---|---|---|---|
| 2 | 0.002281 | 0.005161 | 0.012854 |
| 4 | 0.000510 | 0.005161 | 0.012854 |
| 8 | 0.000276 | 0.005161 | 0.012854 |
| 12 | 0.000249 | 0.005161 | 0.012854 |
| 16 | **0.000234** | 0.005161 | 0.012854 |

Pointwise error **improves monotonically in $C$** — a factor of 9.7 from $C=2$ to $C=16$ — while
generated excess kurtosis also climbs monotonically ($-0.034 \to 1.487$ against a target
$1.910$). The network arms are $C$-independent by construction and are identical down the
column, which is the internal consistency check.

*Consequence.* **The two axes do not trade off.** There is no capacity level in the tested range
at which buying generative fidelity costs pointwise accuracy. At $C=16$ the estimator is ahead
of the CNN on *both* axes simultaneously (pointwise $0.000234$ vs $0.005161$; kurtosis deficit
$|1.487-1.921| = 0.434$ vs $|1.273-1.921| = 0.648$).

*Correction.* Remove the `[pending]`. State the joint result. Keep the causal language
disciplined: the capacity sweep now covers pointwise error, generated statistics and (via
`components_C*/generation.csv`) the generative metrics jointly, so "consistent with a capacity
bottleneck" can be strengthened — but the sweep does **not** include marginal likelihood versus
$C$ or runtime versus $C$, so the joint experiment the reviewer specified is complete in three
of five coordinates. Report it that way, and add likelihood/runtime as a stated gap.

---

### A10. Grid **and domain** convergence data already exist — **UNDER-REPORTED**

*Claim.* The note reports only a grid-resolution sweep, $N_g\in\{401,801,1601\}$ at fixed
$A$, and concedes at `main.tex:292` that the boundary-mass diagnostic is computed but not
persisted (`[pending]`).

*What exists.* `outputs/exp_01_grid_validation/grid_heatmap.csv` has columns

```
t, grid_size, half_width, score_rel_error_mean, score_rel_error_std,
mean_rel_error_mean, identity_residual_max, resolution_ok
```

i.e. a **joint sweep over $N_g$ and $A$ across $t$** — precisely the fixed-$A$/increasing-$N_g$
versus fixed-$h$/increasing-$A$ decomposition the revision asks for. Plus
`grid_error_vs_A.png`, `grid_error_vs_M.png`, `grid_error_vs_t.png`.

*Correction.* Promote this to a real figure and an appendix subsection. It removes the
"fourfold refinement only" objection: the domain axis is swept independently. Separately, the
boundary-mass diagnostic genuinely is not persisted — either compute and commit it (cheap: one
pass over the grid recursion recording the mass in the edge cells) or keep the `[pending]` and
say exactly what is missing. **Chosen: compute and commit it.**

---

### A11. Reverse-sampler convergence data already exist — **UNDER-REPORTED**

*What exists.* `outputs/exp_16/calibrate_steps{100,200,400,800}/steps.csv`, each with all three
arms and columns `n_steps, arm, innov_kurtosis, innov_kurtosis_se, innov_kl,
cov_worst_lag_abs`. The reference arm's kurtosis over an eightfold range of step counts:

| steps | reference arm kurtosis $\pm$ s.e. |
|---|---|
| 100 | $1.989 \pm 0.127$ |
| 200 | $1.817 \pm 0.096$ |
| 400 | $1.913 \pm 0.108$ |
| 800 | $1.936 \pm 0.099$ |

All four are within one standard error of the closed-form target $1.910$, and the arm ordering
is unchanged across the sweep. `calibrate_steps1600` and `calibrate_steps3200` are **empty
directories** — those tasks did not produce output and must not be cited.

*Correction.* Report the sweep as a convergence study with a figure, stating the range actually
covered (100–800) and that 1600/3200 were launched but produced no output. Do not claim a range
that is not there.

---

## B. Numerical claims traced to evidence

| # | claim | location | evidence | tier | action |
|---|---|---|---|---|---|
| 1 | $9.2\times10^{-15}$ rel. against closed-form Gaussian mean | `main.tex:300,342` | `tests/test_grid_convergence.py` | T3/T4 | keep; say "at $N_g=401$, $A=8$" |
| 2 | $1.6\times10^{-14}$ mean, $1.8\times10^{-15}$ log-evidence vs brute-force enumeration | `main.tex:343` | `tests/test_em_bp.py` | T3 | keep; state it shares no code with the recursion |
| 3 | $3.9\times10^{-16}$ pairwise statistic | `ch07-bp.tex:188` | `tests/test_em_bp.py` | T3 | keep |
| 4 | importance sampling: evidence within 5 s.e. at $10^6$; mean $10^{-2}$ rel. at $2\times10^6$ | `main.tex:304` | test assertions, tolerances set by MC error | T5 | keep, already correctly attributed |
| 5 | closure rel. score error $0.198$ at $t{=}0.08 \to 9.9\times10^{-5}$ at $t{=}2.4$ | `main.tex:411` | **does not match any committed output** — see B* below | — | **replace with the committed numbers** |
| 6 | Fisher identity vs finite differences to $\sim10^{-9}$ | `main.tex:468` | `outputs/exp_08_.../gradient_vs_exact.csv` | T3 | keep |
| 7 | gradient ascent reaches EM optimum to $2\times10^{-9}$ nats (Gaussian); higher likelihood on Laplace | `main.tex:472` | `outputs/exp_08_.../gradient_vs_exact.csv` | T5 | keep |
| 8 | Fisher information falls $142\times$ (scale) vs $26\times$ (correlation), $t{=}0.05\to1.6$ | `main.tex:499` | `outputs/exp_06_.../` | T5 | verify file before quoting |
| 9 | Table 1 ratios $9.3$–$14.1$ over six replicates | `main.tex:527` | `outputs/replicates/merged_summary.csv` | T5 | **plot it**; caption must name the oracle post-selection |
| 10 | capacity control: 24 configs, 3.2k–297k params, best $0.208$ vs $0.022$ | `main.tex:547` | `outputs/exp_07_.../capacity.csv` | T5 | keep; **plot it** |
| 11 | competence threshold on Gaussian chain — "achieved value not persisted" | `main.tex:551` | test assertion only | T5 | keep the honest non-quotation |
| 12 | CNN margin $2.31\pm0.39$, $3.36\pm0.30$, $4.16\pm0.24$ | `main.tex:559` | `outputs/exp_12_.../efficiency_three_way.csv` | T5 | keep + exploratory label |
| 13 | reference sampler reproduces target within 1.4%: $1.897,1.883,1.914,1.908$ vs $1.9098$ | `main.tex:584` | `outputs/exp_16/gen_n*_seed*/generation.csv` | T5 | keep |
| 14 | $(q/(q+v))^2$ kurtosis dilution at $t_{\min}$ | `main.tex:580` | `exp_16_sampling_validation.py:188` | T1 | keep; derivation to appendix |
| 15 | dissociation identical to 3 d.p. under $N_g\in\{401,801,1601\}$ | `main.tex:652` | `outputs/exp_16/grid_M*/generation.csv` | T5 | keep |
| 16 | inference 200–300$\times$ slower than a network | `main.tex:707` | `outputs/exp_07_.../inference_cost.csv` | T5 | keep; **plot it** |
| 17 | discrete-alphabet advantage $15 \to 3.6$–$6.1$ at 132 params | `main.tex:713` | `outputs/exp_10_.../discrete_*.csv` | T5 | keep |
| 18 | EM monotone in every run | `ch10-estimation.tex` | `outputs/exp_06_.../monotonicity.csv`, `monotone_violation = 0.0` for all inits | T5 | **state it as measured, with the column name** |
| 19 | $\rho$ recovered to $1.1\times10^{-16}$, $b$ to $5\times10^{-3}$ | not currently in the note | `outputs/exp_06_.../monotonicity.csv` | T5 | **add** — it is the direct evidence that $\rho$ is learned |

### B\*. One quoted pair is not traceable — **CORRECTION REQUIRED**

`main.tex:411` states the Gaussian-closure relative score error on a Laplace chain at
$\rho=0.85$ falls "from $0.198$ at $t=0.08$ to $9.9\times10^{-5}$ at $t=2.4$". Neither number
appears in any committed file.

- `outputs/exp_02_.../laplace_summary.csv` (the only run with $t=0.08$; $\rho=0.85$, $n=40$,
  60 trials, `ref_grid` $N_g=801$, $A=8$) gives **$0.1752$** at $t=0.08$ and
  **$7.63\times10^{-5}$** at $t=2.4$, as medians over trials.
- `outputs/exp_03_.../innovation_sweep.csv` at `laplace, rho=0.85` has no $t=0.08$ row; its
  neighbouring values are $0.2657$ at $t=0.05$ and $0.1284$ at $t=0.1$, and
  $8.27\times10^{-5}$ at $t=2.4$.

The quoted pair is presumably from a superseded run. **Action:** quote the `exp_02` medians
$0.175 \to 7.6\times10^{-5}$, name the file, the aggregation (median over 60 trials) and the
configuration, and plot the whole curve rather than its endpoints.

Two numbers previously flagged as unsupported were re-checked and remain correctly handled: the
five-family LMMSE check (`main.tex:394`) and the boundary mass (`main.tex:292`) both carry
`[pending]` and quote nothing. The first stays pending; the second is now being computed.

---

## C. Presentation defects

| defect | where | action |
|---|---|---|
| Results carried entirely by tables; no result plots | both documents | generate 10 figures from committed CSVs |
| Author-voice section titles ("What we have bought") | `ch06-graphical.tex:164`, elsewhere | rewrite as descriptive headings |
| Box density: `intuition` / `pitfall` / `codebox` on nearly every page | compendium | retain a box only for a precise assumption, warning or implementation fact |
| Repeated conclusion in abstract, intro, section open, section close, conclusion | paper | state once, in the results section |
| Author-numbered citations (`plainnat`) | both | switch to numbered style |
| Compendium has almost no references | compendium | add public sources for conditional expectation/LMMSE, BP on trees, forward–backward, Kalman/RTS, EM and GEM, Fisher's identity, score matching and diffusion SDEs |
| Private lecture notes cited | `bibliography.bib:17`, `main.tex:83`, `ch03-diffusion.tex:12`, `notation.tex:4`; also `research/unified-note/`, `research/gaussian-bp/` | remove entirely, replace with public sources |
| No appendix | paper | add A–J |
| $K$ means grid size; $W$ means kernel | `notation.tex:41,43` | $K$/$K_\theta$ = kernel, $N_g$ = grid size |
| Hyperlink colours | both | subtle/black |

---

## D. Items that remain open after this revision

1. **Marginal likelihood and runtime versus $C$** are not measured; the capacity sweep covers
   pointwise error and generative statistics only.
2. **Receptive-field selection** still lacks a validation split; those numbers stay exploratory.
3. **Five-family LMMSE equivalence check** is asserted in tests but not persisted as an output.
4. **Strict stationarity** — no run yet initialises from the invariant non-Gaussian law.
5. **Locality of the score** — `exp_11`/`exp_12` measure a receptive-field proxy; no direct
   statement about the exact receptive field of the score as a function of the observations.
6. **Identifiability** is assumed, not proved.

These are the six items the advisor documents must carry forward, and they are the basis for the
questions in `ANSWERS_AND_QUESTIONS_FOR_ADVISORS.md`.
