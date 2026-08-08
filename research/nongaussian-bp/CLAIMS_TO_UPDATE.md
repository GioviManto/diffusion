# Claims the overnight runs have changed

Prepared for review, **not applied**. Each row gives the claim as it currently stands, what
the evidence now supports, and where that evidence lives. Rewriting the argument is a
judgement call, so the wording below is a proposal rather than an edit.

Sections 1–9 carry complete data. **Section 10 is partial** — the generation rerun (`618378`)
has returned only its smallest data budget, and its finding is the one that could still move the
paper's headline, so it is explicitly marked not-yet-writable and should not be acted on until
the remaining budgets land.

---

## 1. Locality — claim survives, evidence upgrades

**Currently:** the non-Gaussian rate ratio of 1.12–1.46× is quoted from `exp_11`, whose window
BP was given `N(0,1)` as the initial law at every window's left endpoint.

**Concern raised, then refuted.** That default is the correct marginal only for the Gaussian
chain, so the measurement was exact for the family the others are normalised *against* and
approximate for the families whose departure is the result. Under strictly stationary
initialisation the median `n01/invariant` ratio is **1.0038**, with 89 of 90 cells inside 10%
and one beyond 20%. In the headline regime the committed protocol spans 1.163–4.618 and the
exact one 1.133–4.666.

**Proposed wording.** Keep the number. Add that the windowed estimator is now the *exact*
conditional expectation `E[a_C | x_{C-r..C+r}]` under strictly stationary initialisation — a
T3 quantity rather than a proxy — and that the initial-law mismatch was measured and found not
to account for the effect. Report the one exception: uniform innovations at ρ=0.5, t=0.05,
where the exact endpoint law moves the ratio 1.927 → 1.163.

*Evidence:* `outputs/exp_11_nongaussian_locality/stationary_rho*/locality_stationary.csv`
(three arms: `n01`, `n01_exact`, `invariant`).

---

## 2. Capacity — the sweep is now complete in five coordinates, and it saturates

**Currently:** pointwise error and generated tail fidelity both improve monotonically in C, with
likelihood and runtime listed as missing.

**Now measured.** Held-out evidence per edge rises but **saturates**: gains of 1.75e-3, 1.6e-4,
9.5e-5, 5e-6 across C = 2→4→8→12→16. Nothing meaningful is bought past C ≈ 8. Runtime is
essentially flat, 13.50 → 14.28 s/iter while free parameters go 6 → 48, confirming the E-step's
`O(N n N_g²)` dominates and only the M-step scales with C. `monotone_violation` is exactly 0 in
all 15 runs.

**Now settled properly, on a paired design.** The audit's objection was that separate medians
over three seeds cannot resolve the small C = 12→16 difference, and it was right. Fitting every
capacity on the *same* training set within a cell and scoring on the *same* held-out bundle
turns the comparison into a within-dataset contrast, whose standard error is an order of
magnitude below the effect. At N = 128 over six paired seeds, held-out log-evidence per edge
against C = 1:

| C | paired difference | SE | beyond 2 SE? | s_min/h | effective C |
|---|---|---|---|---|---|
| 2 | +0.000013 | 0.000090 | **no** | 12.1 | 2.00 |
| 4 | +0.001286 | 0.000107 | yes | 9.5 | 3.82 |
| 8 | +0.001735 | 0.000136 | yes | 9.8 | 7.70 |
| 16 | +0.001740 | 0.000152 | yes | 9.5 | 15.50 |

Three things follow. **C = 2 buys nothing over a single Gaussian innovation.** Capacity begins
to pay at C = 4. And **C = 8 and C = 16 differ by 5e-6, far inside one standard error** — so the
saturation is now measured, not inferred from the shape of a median curve.

It is also not a resolution artefact: the effective component count tracks the nominal one
(15.50 at C = 16) and the narrowest fitted component stays ≥ 9× the grid spacing throughout, so
the mixture is genuinely using its capacity and every component is resolved.

**Proposed wording.** State the saturation with the paired interval, and say plainly that
capacity beyond C ≈ 8 is not supported by the data. One caveat still travels with it: the
likelihood is a **composite** objective, not the marginal likelihood of the generated dataset
(see §5).

*Evidence:* `outputs/exp_16/cpoint2_C*/heldout_evidence.csv`, `em_trace.csv`.

---

## 3. Boundary diagnostic — one number was never a bound

**Currently:** the truncation diagnostic quotes a per-configuration "boundary mass" and the
prose treats it as bounding the truncation error.

**What is true.** It was computed from a *single* sampled chain, seeded with Python's
per-process-salted `hash`, and it is a maximum over sites of one trajectory — so neither a
bound nor a sample. Recomputed over 256 chains per cell with deterministic seeding, the
single-chain value sits near the **median** over chains while the maximum is **152× higher**
(range 2.9× to 8369×).

**What did *not* change.** The kernel column-mass residuals reproduce to **0.000e+00 relative
difference** across all 84 cells, because they are pure functions of `(prior, A, N_g)` with no
sample in them. The truncation-versus-quadrature separation — the actual scientific content of
that section — is untouched.

**Proposed wording.** Call the edge mass an *empirical diagnostic*, report max and upper
quantiles over chains, and say plainly that it is not a bound. Leave the truncation/quadrature
result as it stands, noting it is sample-free.

*Evidence:* regenerated `outputs/exp_18/boundary.csv`; determinism verified across
`PYTHONHASHSEED` 1 and 999.

---

## 4. Higher-order information — the claim was unsupported, and is now stronger

**Currently:** "information about innovation variance falls 142× while information about
correlation falls 26×, so higher-order structure is more fragile."

**Why that did not follow.** Both parameters are second order — `q` is a scale, not a shape —
and the comparison used raw diagonal entries, ignoring nuisance coupling.

**Now measured** on a generalised-Gaussian family where `β` moves the fourth moment at
*exactly* fixed variance, reporting efficient information after projecting out `(ρ, q)`:

| true β | ρ | q | **β (shape)** | β/ρ |
|---|---|---|---|---|
| 1.0 | 78× | 235× | **8,738×** | 112× |
| 1.5 | 83× | 459× | **64,227×** | 776× |
| 3.0 | 56× | 522× | **68,460×** | 1,222× |

**Proposed wording.** Replace the sentence entirely. The corrected claim is far stronger than
the original: shape information decays **two to three orders of magnitude** faster than
correlation information. Note that the nuisance correction matters and is asymmetric — at
t = 1.6, ρ and q become 90% correlated and each loses a factor ~5.5, while β is nearly
orthogonal and loses ~1.0.

*Evidence:* `outputs/exp_22/shape_information.csv`.

---

## 5. "Innovation law" at t_min — a misnomer with a measurable size

**Currently:** the generated-sample comparison reports "innovation excess kurtosis" and the
headline says the CNN "reproduces the innovation law more faithfully".

**What is true.** `r_i = x_i − ρx_{i−1}` on a sample of `p_{t_min}` gives
`r_i = αε_i + √Δ(z_i − ρz_{i−1})`, so `Cov(r_i, r_{i+1}) = −ρΔ ≠ 0`. Predicted lag-1
correlation is **−0.0997** at t_min = 0.02, rising to −0.195 at 0.05 and −0.286 at 0.1. The
clean-chain control confirms the filter does recover innovations there (|ACF| < 0.02).

**Proposed wording.** Rename throughout to **AR-filtered residual excess kurtosis**. The
marginal shape comparison remains valid and remains the sharpest discriminator between families
at matched covariance; what must go is any claim that it measures recovery of the *clean*
transition innovation law.

*Evidence:* `src/sample_metrics.predicted_residual_autocorr`, checked against measurement.

---

## 6. Two statements about exactness that need qualifying

- **EM monotonicity.** `fit_em` returned a kernel one M-step beyond the last logged evidence, on
  both exit paths, so the reported trace never covered the final update and the returned model's
  likelihood was never evaluated. Fixed; the returned kernel's evidence is now
  `trace.log_evidence[-1]` by construction, with a test asserting it.
- **Grid semantics.** Parametric kernels evaluate continuous densities on the grid without
  renormalising columns; the MDN normalises. Both cannot be "the exact finite-state
  likelihood". Recommended reading: BP is an exact tree recursion and the implementation
  evaluates a **truncated quadrature approximation** to the continuous evidence, with EM and
  Fisher gradients exact *for that quadrature objective*.

---

## 7. Sample-size language

Every likelihood reported from `exp_16` to date is a **composite** objective: one clean batch
noised at five levels, the groups passed to `fit_em` as if independent. Each marginal is
correctly specified so the estimator is legitimate, but the effective sample size is the number
of clean chains, not five times it. `src/protocols.py` now provides `one_view` (exact marginal
likelihood) and `multi_view` (correct joint), and every output records the sample budget.

---

## Pending — do not move any claim on these yet

- **Generation rerun under continuous-`t`** (`618378`). The current dissociation result was
  produced with networks trained on five discrete levels and evaluated across [0.02, 3.0], with
  a parameterisation switch mid-integration. Until the rerun lands, the size of that artefact is
  unknown, and it bears directly on the paper's headline.
- **Non-Markov robustness** (`618380`, `618390`). First-ever runs. The quick configuration hinted
  at a CNN crossover at β = 0.5, but its Markov control mis-fits ρ as 0.777 against a true 0.85,
  so that is under-convergence and not a result.

---

## 8. Non-Markov robustness — the inversion happens, and which violation matters

**Currently:** untested. The report called this the largest gap in the story.

**Now measured** (`exp_21`, Gaussian non-Markov priors with an exact linear-algebra reference).
Two mechanisms behave completely differently, and the difference is principled rather than
incidental:

| violation | strength | cnn / em-bp across t | ρ − ρ_ChowLiu | reading |
|---|---|---|---|---|
| global latent | β = 0 | 11.4 → 5.4 | −0.0009 | Markov control |
| global latent | β = 0.25 | 10.9 → 6.3 | +0.0032 | degraded |
| global latent | β = 0.5 | 4.1 → 3.2 | +0.0091 | degraded, still winning |
| global latent | **β = 1.0** | **2.1 → 3.1** | +0.0128 | **still winning, never inverts** |
| long-range | γ = 0 | 25.4 → 10.5 | +0.0014 | Markov control |
| long-range | γ = 0.05 | 1.04 → 1.46 | +0.0352 | advantage gone |
| long-range | γ = 0.10 | **0.77** → 1.32 | +0.0574 | **inverts at low t** |
| long-range | γ = 0.20 | **0.56** → 1.13 | +0.0845 | **inverts across most of the schedule** |
| long-range | γ = 0.40 | **0.47** → 1.03 | +0.1010 | **inverts at 4 of 5 noise levels** |

Sweep complete (10/10 tasks). The single sentence the data supports: **the minimum ratio over
the schedule never falls below 2.08 for the rank-one violation, at any strength up to β = 1.0,
and falls to 0.47 for the long-range one.** At γ = 0.40 the global MLP also beats EM-BP (0.67),
so this is a failure of the chain assumption rather than something specific to the local CNN,
and EM-BP's relative error there reaches 57% at high noise.

**Why the two differ, and it is not incidental.** `exp_04` already established that a global
latent makes the score residual *exactly* rank one (Woodbury), while long-range coupling is
only approximately low-rank. A chain family absorbs the first and cannot represent the second.
At β = 1.0 half the marginal variance is a shared global constant and the estimator *still*
wins by 2–3×; at γ = 0.10 — a far milder-looking perturbation — it has already lost. So the
honest claim is not "the advantage degrades under misspecification" but that it survives
rank-one contamination essentially intact and collapses under genuine long-range structure.

The Chow-Liu deviation corroborates this independently: +0.013 at β = 1.0 against +0.085 at
γ = 0.20, i.e. the fitted chain remains close to the best-Markov projection under the rank-one
violation and departs from it under the long-range one, exactly where the chain family stops
being able to represent the truth.

**EM converges to the best-Markov projection**, as predicted, to within 0.0009–0.0128 across
the whole β sweep.

**Proposed wording.** State the inversion and the mechanism distinction plainly. The honest
claim is that the structural prior pays under correct specification and under rank-one
contamination, and stops paying under long-range coupling at γ ≳ 0.05–0.10. Do not generalise
the advantage beyond that.

**Caveat that must travel with these numbers.** The sweep is *not paired across strengths*:
`exp_21` seeds its data on a tag containing the violation strength, so each β and γ draws
independent data. The trend is far larger than seed noise (25× → 0.56×), but individual
strengths are not directly comparable to one another. A paired rerun is queued.

*Evidence:* `outputs/exp_21/gauss_beta*/`, `outputs/exp_21/gauss_gamma*/`.

---

## 9. Validation split — objection correct, effect nil

**Now measured.** Choosing the network parameterisation on validation rather than on the test
bundle agrees with the test-set oracle in **33 of 35 cells**; the two disagreements move the
ratio by 0.5% and 0.9%. Mean advantage **11.66×** validation-selected against 11.65×
test-oracle — the oracle was worth **0.999×**.

**Proposed wording.** Adopt the validation protocol and say the oracle was measured and found
worth nothing, rather than merely removing it. The eps/x0 choice is stable (eps at low t, x0 at
high t, in every cell across seven data budgets), which is *why* there was nothing to exploit.
The reported 9–14× advantage is unchanged.

**The harder oracle costs nothing either.** `exp_12` selected over *two* axes at once — three
receptive-field radii and two parameterisations, six configurations — which is where an oracle
had the most room to flatter the baseline. Moving that choice to validation agrees with the
test-set argmin in **54 of 60 cells**, and the mean CNN/EM ratio is 3.862 validation-selected
against 3.846 test-oracle: the two-axis oracle was worth **0.9959×**. That ratio is also
consistent with the 2–4× the paper reports against the locality-respecting convolution.

Both defects were therefore genuine methodological errors with no empirical consequence. Report
them as measured and corrected, not as corrections whose effect is unknown.

*Evidence:* `outputs/exp_07_em_vs_score_network/sample_efficiency_val.csv`,
`outputs/exp_12_receptive_field/efficiency_val.csv`.

---

## 10. Generation under the corrected protocol — the dissociation survives and sharpens

**The audit's most serious item.** Networks were trained on five discrete noise levels and then
called across [0.02, 3.0] by the integrator, so the generated-sample comparison mixed estimator
error with interpolation and extrapolation. Rerun with continuous log-uniform training and one
parameterisation chosen per arm.

**Internal validity check passes exactly.** `exact` and `em_bp` involve no networks, so they must
be unchanged; paired seed-by-seed they are **bit-identical** (max |Δ| = 0.000000). The protocol
change is isolated to the network arms, as intended.

**Paired result at N = 32** (AR-filtered residual excess kurtosis, target 1.9098):

| arm | seed 0 | seed 1 | seed 3 |
|---|---|---|---|
| em_bp | 0.8516 → 0.8516 | 0.6143 → 0.6143 | 0.3804 → 0.3804 |
| cnn | 0.8323 → **1.1140** | 0.5662 → **0.7300** | 0.5539 → **0.6226** |
| mlp | 1.3003 → 1.7108 | 0.8710 → **44.2892** | 0.8298 → **12.4258** |

Two effects, in opposite directions.

**The CNN improves consistently**, moving toward the target in all three seeds. The old protocol
was *understating* its generative quality, so the dissociation the paper reports — the arm that
loses pointwise winning generatively — survives the fix and is sharper: at seed 0 the CNN
reaches 1.11 against EM-BP's 0.85.

**The MLP becomes unstable**, blowing up in two of three seeds, with covariance error rising
0.37 → 1.47. The plausible mechanism is that training only at t ≥ 0.1 left it *extrapolating*
below that during integration, whereas continuous training makes it fit small t, where the drift
is stiff (α/Δ ≈ 25 at t = 0.02) and an imperfect score is amplified by the reverse SDE.
Extrapolation was evidently more benign than a badly-fitted stiff score.

**The MLP failure is bimodal, which matters for reading it.** With N = 32 now complete across
all four seeds, two are catastrophic (44.3, 12.4) and two are unremarkable (1.71, 0.79) — not a
uniform degradation but an instability threshold that some fits cross and others do not. The CNN
improves in three of four seeds. Both network-free arms stay bit-identical in every seed.

**N = 128 settles it: the instability was small-data over-parameterisation, not the protocol.**
At N = 128 the MLP is pathological in **0 of 3 seeds** and improves substantially
(0.59 → 2.23, 0.67 → 1.26, 0.39 → 1.29). A 128×128 network fitted on 32 chains was simply far
too large; given 128 it is stable and better. The N = 32 blow-up should be reported as a
small-data artefact, not as a finding about continuous-time training.

**The result at N = 128**, against the target 1.9098:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| em_bp | 0.857 | 0.862 | 0.745 |
| cnn | **1.364** | **1.261** | **1.467** |
| mlp | **2.225** | **1.262** | **1.290** |

**Proposed wording.** The audit's most serious objection was correct as a criticism of the
protocol and does **not** overturn the conclusion it threatened. EM-BP wins pointwise by roughly
an order of magnitude and loses the generated residual law to *both* network arms — a sharper
dissociation than the flawed protocol showed, because the fix improved the networks while
leaving EM-BP bit-identical. State the correction, state that it strengthened rather than
weakened the finding, and drop any suggestion that the earlier five-level numbers were adequate
for a reverse-SDE comparison.

Larger budgets (512, 2048) are still running; nothing above depends on them, but they should be
folded in before the figure is regenerated.

### 10b. N = 512: the dissociation is clean, and there is a new within-arm version of it

Generated AR-filtered residual excess kurtosis, mean over four seeds (target 1.9098):

| N | em_bp | cnn | mlp |
|---|---|---|---|
| 32 | 0.491 | 0.728 | 14.803 (2/4 unstable) |
| 128 | 0.814 | 1.293 | 1.441 |
| 512 | 0.897 | **1.601** | **0.167** |

At N = 512 the ranking is unambiguous: generatively **cnn > em_bp > mlp**, pointwise
**em_bp ≫ cnn > mlp**. That is the dissociation, and it strengthens with data rather than
washing out.

**A new observation worth following up.** The MLP's four N = 512 seeds are 0.162, 0.170, 0.222,
0.115 — tightly clustered, with *small* covariance error (0.08–0.13). So these are not unstable
samples; they are systematically near-Gaussian residuals. The MLP's generated non-Gaussianity
therefore *decreases* as training data increases, even though its pointwise error decreases too.
That is a dissociation appearing **within a single arm as a function of data**, which is a
sharper statement of the paper's central phenomenon than the between-arm comparison it currently
rests on. It is one experiment away from being a result: fit the MLP at several budgets and plot
pointwise error and generated kurtosis against N on the same axis.

Not claimed yet: N = 2048 is still running, and the mechanism (a better L2 fit to the posterior
mean being smoother, hence more Gaussian in the generated law) is a hypothesis, not a
measurement.

### 2b. Paired capacity contrasts on the primary pointwise metric, both budgets

Held-out denoising MSE against the Bayes denoiser, paired within (budget, seed) cells, six seeds
each, differences against C = 1 (negative = better):

| N | C=2 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| 128 | +1.09e-5 (ns) | −1.287e-3 | **−1.699e-3** | −1.683e-3 |
| 512 | **+3.79e-5 (significant, worse)** | −1.412e-3 | **−1.741e-3** | −1.760e-3 |

**C = 8 is the optimum, and this is now measurable rather than eyeballed.** C = 16 differs from
C = 8 by 1.6e-5 (N=128) and 1.9e-5 (N=512), inside one standard error in both cases, and at
N = 128 the larger model is marginally *worse*. **C = 2 is not an improvement on a single
Gaussian innovation** and at N = 512 is significantly worse than it — the extra parameters cost
more than they buy at that capacity.

Every fit is resolved: the narrowest component is at least 8.8 grid spacings wide in all 60
cells, so none of this is a discretisation artefact. That was the specific doubt the audit
raised about interpreting the C = 12→16 region, and it is now excluded by measurement.

*Evidence:* `outputs/exp_16/paired_local/paired_contrasts.csv`, `paired_capacity.csv`.

### 8b. The non-Gaussian, non-Markov corner confirms the dichotomy

`exp_21 laplace` puts a global latent on a **Laplace** chain, measured against the exact
g-marginalised reference in `src/nonmarkov.py`. Complete sweep, paired seeds:

| β | min cnn/em (Laplace) | min cnn/em (Gaussian) | fitted innovation kurtosis | fitted ρ |
|---|---|---|---|---|
| 0.00 | 3.34 | 5.38 | 2.927 (true 3.0) | 0.8478 |
| 0.25 | 4.28 | 5.66 | 2.706 | 0.8608 |
| 0.50 | 2.59 | 2.67 | 2.470 | 0.8849 |
| 1.00 | **1.45** | **2.08** | 0.988 | 0.9387 |

The rank-one violation never inverts the advantage for non-Gaussian innovations either. The
mechanism is visible in the fitted parameters: as β grows the estimator absorbs the latent's
contribution into the chain, raising ρ from 0.848 to 0.939 and flattening the fitted innovation
kurtosis from 2.93 to 0.99. It is fitting the best chain approximation, and paying for it in
exactly the coordinate the method exists to estimate.

**Combined statement.** The structural prior survives rank-one contamination — Gaussian or
non-Gaussian innovations, up to β = 1.0 where half the marginal variance is a shared constant —
and fails under long-range coupling by γ ≈ 0.10. Do not generalise the advantage past that.

### 10c. All four budgets — final generation result

AR-filtered residual excess kurtosis of generated samples, four seeds each, target **1.9098**:

| N | em_bp | cnn | mlp |
|---|---|---|---|
| 32 | 0.491 | 0.728 | 14.803 (2/4 unstable, over-parameterised) |
| 128 | 0.814 | 1.293 | 1.441 |
| 512 | 0.897 | **1.601** | 0.167 |
| 2048 | 0.902 | **1.572** | **0.046** |

**The dissociation is confirmed at every budget and is not a small-sample effect.** EM-BP wins
pointwise by roughly an order of magnitude and never exceeds 0.90 generatively; the local CNN
reaches 1.57–1.60 and is closest to the target at every budget from 128 up.

**A second, sharper result the corrected protocol exposed.** The global MLP's generated
residuals become *exactly Gaussian* as data grows: per-seed values at N = 2048 are 0.069, 0.019,
−0.002, 0.097 — four seeds, all indistinguishable from zero — while its pointwise error is
improving over the same range. So within a single arm, more data buys pointwise accuracy and
*destroys* generative fidelity. That is the paper's central phenomenon in its strongest form,
and the old five-level protocol could not have shown it: at N = 32 the same arm was blowing up
instead.

**Proposed framing.** Lead with the within-arm version. "Pointwise accuracy does not determine
generative fidelity" is much better evidenced by one estimator getting monotonically better at
one and monotonically worse at the other than by a ranking disagreement between three arms.

*Evidence:* `outputs/exp_16/gen2_n*/generation.csv`, against committed `gen_n*/` for the
protocol contrast.
