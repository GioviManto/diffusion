# Learning the BP transition kernel: what we implemented and what it measures

Answering the proposal in Jérôme's email — learn the kernel `K` instead of supplying it,
by likelihood-based learning rather than a denoising network — and Marc's observation
that this needs no noising when the clean chains are available.

Every number below is reproducible from a committed output by a named script. Numbers we
could not reproduce have been withdrawn rather than restated; §4 lists them.

---

## 1. The answer in one paragraph

Marc is right, and the distinction is sharp enough to be worth stating as two regimes.

**Regime A — clean chains observed.** There is no latent chain, so no noising and no BP
are needed to estimate the transition law. It is ordinary maximum likelihood. For a
Gaussian transition it is exact least squares in closed form. For a Gaussian-mixture
innovation only the component *labels* stay latent, so an inner EM over labels is still
useful — but that EM is over labels, not over the chain, and no message passing occurs.

**Regime B — only OU-noised chains observed.** The clean chain is latent. Coordinatewise
noise adds unary factors and leaves the chain factor graph intact, so sum-product still
applies and supplies exactly the statistic EM needs: the posterior pairwise transition
mass `Ξ`. The M-step then never revisits the data.

The payoff is that the learned kernel is a property of the clean chain, **not of a noise
level**. One fit gives a posterior mean and a score at every `t` through BP and Tweedie's
identity, with no per-`t` retraining — which is the structural difference from a denoising
network trained at a schedule.

One wording correction worth making explicitly: EM here is not "following the gradient."
The literal likelihood gradient is available from Fisher's identity and we measure that
route too (§3.4), but the algorithm that runs uses closed-form or conditional-maximisation
M-steps. For the mixture it is generalised EM of ECM type — four inner conditional
maximisations, each increasing `Q` without globally maximising it.

---

## 2. What is exact, and what only looks exact

We now keep four levels apart, because collapsing them is how "exact BP" turns into a
claim we cannot support:

1. the noised posterior graph **is** a chain — exact structural statement;
2. sum-product on a chain returns exact marginals **if the integrals are exact**;
3. the implemented recursion is exact to floating point for the **finite quadrature
   factor graph** defined by the grid and trapezoidal weights;
4. that quadrature graph **approximates** the continuous model, with separate truncation
   and quadrature error.

The defensible phrase is "exact tree inference for the implemented finite quadrature
factorisation." Not "the continuous E-step is exact."

Two related scoping points. The non-Gaussian chains are **covariance-stationary, not
strictly stationary** — `a_1` is drawn Gaussian while the innovations are not, so
`Var(a_i)=1` and `Cov(a_i,a_j)=ρ^|i-j|` hold by construction but the chain is not started
from its invariant law. And the fitted mixture's innovation mean is **reported, not
constrained**: recentring after each update is not a conditional maximiser and broke
monotone ascent, so it is a diagnostic.

---

## 3. Measured results

### 3.1 ρ is estimated, not supplied

From initialisations at `ρ⁽⁰⁾ ∈ {0, 0.3, 0.6, −0.4}` against a truth of **0.85**, the
fitted value lands in **[0.8517, 0.8520]** at `C=4` (and [0.8517, 0.8529] across the eight
`C ∈ {4,8}` runs). Largest likelihood decrease across iterations: zero in all eight.

> `outputs/exp_18/em_trace.csv` — `exp_18_revision_diagnostics.py --parts emtrace`

Note the truth differs by experiment: **0.85** in exp_02/16/18/22, **0.8** in exp_06/07/08.
No table pools the two, and each now states its own.

### 3.2 The mixture learns a heavy tail without being told it is Laplace

Against Laplace innovations (true excess kurtosis 3.0), a `C`-component Gaussian mixture
recovers the shape. Across eight independent draws at 200 iterations (`ρ=0.85, N=512, C=8`)
the fitted excess kurtosis is **2.972 ± 0.074** (clean) and **3.024 ± 0.204** (through the
channel) — unbiased in both arms, with a paired clean − noised difference of
**−0.052 ± 0.217**: no resolved channel penalty on shape *at convergence*. Single draws
span 2.40 to 3.99, so no single-dataset shape number is meaningful at this budget.

This retired an earlier claim of ours that shape was "recovered to about 72%", which
rested on one draw at an iteration count it had not converged at.

> `audit/audit_seed_spread.py`, written up in `audit/AUDIT_NOTE.md` §3

### 3.3 Sample efficiency against neural denoisers

Six training seeds, budgets 32→4096, paired on a common held-out test set scored against
grid BP under the true kernel. Ratio of aggregate mean relative score error,
network / EM-BP:

| N | selected network | EM-BP | ratio |
|---:|---:|---:|---:|
| 32 | 0.607007 ± 0.003985 | 0.065072 ± 0.003083 | 9.33 |
| 64 | 0.556940 ± 0.004800 | 0.051515 ± 0.007161 | 10.81 |
| 128 | 0.469970 ± 0.002303 | 0.033271 ± 0.003498 | 14.13 |
| 256 | 0.338483 ± 0.003962 | 0.028799 ± 0.002106 | 11.75 |
| 512 | 0.236145 ± 0.001574 | 0.021109 ± 0.002158 | 11.19 |
| 1024 | 0.173611 ± 0.000919 | 0.018213 ± 0.002096 | 9.53 |
| 2048 | 0.137181 ± 0.001000 | 0.014213 ± 0.000875 | 9.65 |
| 4096 | 0.123994 ± 0.000339 | 0.012410 ± 0.000477 | 9.99 |

Uncertainty is over the six independent training seeds (`ddof=1`, `SE = SD/√6`);
the displayed ratio is the ratio of aggregate means, not the mean of seed-level ratios.

> `outputs/replicates/merged_raw.csv` — recomputed from raw rows, reproduces to 6 d.p.

**On the selection objection.** The table above lets the network keep the better of the
`eps`/`x0` parameterisations per noise level, which is oracle post-selection on the
evaluation set and favours the *baseline*. Choosing on a disjoint validation bundle
instead: the two protocols **agree in 33 of 35 cells**, and the mean cell ratio is
**11.659 selected vs 11.653 oracle**. The oracle is worth essentially nothing here.

> `outputs/exp_07_em_vs_score_network/sample_efficiency_val.csv`

This is one seed and excludes `N=4096` — it bounds the selection bias, it does not
replace the six-seed table.

**The honest scope.** EM-BP is given the correct graph, a homogeneous linear-autoregressive
transition form, and a low-dimensional innovation family. The networks are not. The result
measures the value of correct structure in this controlled model — not superiority over
neural denoisers in general.

**And it is slower at inference.** Per chain, grid BP runs **211–320× slower** than a
network forward pass (2.01 ms vs 0.0096 ms at batch 32; 1.31 ms vs 0.0041 ms at batch 512).
Statistical efficiency and inference cost point in opposite directions and both belong in
any summary.

> `outputs/exp_07_em_vs_score_network/inference_cost.csv`

### 3.4 The literal gradient route, measured

Fisher's identity gives the exact marginal-likelihood gradient from one sum-product pass.
On the smooth Gaussian kernel, gradient ascent reaches EM's optimum to
**1.6 × 10⁻⁹ nats** but needs a tuned step size; on the Laplace kernel it attains a
likelihood **1.9 nats higher**, because there EM's exact maximiser is a lattice artefact
(§3.7). Larger steps diverge — at `η=8` the Laplace run reaches the constraint boundary
`b=10⁻³`, where the density spikes and the reported likelihood is degenerate, not better.

> `outputs/exp_08_gradient_vs_exact_mstep/gradient_vs_exact.csv` (regenerated, §4.2)

### 3.5 Stop on the shape, not on ρ

The coordinates do not converge at the same rate. At `ρ=0.85, N=512, C=8`, ρ settles
within **10⁻²** of its final value by iteration **27–30**, but only by **51–54** at a
tolerance of **10⁻³** — "flat" is a statement about a tolerance, and quoting it without one
is how a coordinate gets certified by eye. Meanwhile the fitted excess kurtosis reads
**0.84 at 30 iterations, 1.85 at 60, 2.30 at 120, 2.29 at 400**.

Monitoring ρ — the natural thing to plot, and a trace that looks textbook-converged —
certifies a kernel still missing a third of the higher-order structure.

Importantly, the shape is slow **even on clean data** — tens of iterations against one for
ρ. So that slowness is the mixture's own inner latent label, not the observation channel.
The channel then slows everything further on top of it.

### 3.6 What the channel actually costs: rate, not accuracy

Same chains, same initialisation, same budget; one arm sees them clean, the other once each
through the OU channel:

| iteration | clean ρ | clean kurt | noised ρ | noised kurt |
|---:|---:|---:|---:|---:|
| 1 | 0.8473 | 1.470 | 0.4276 | 0.112 |
| 30 | 0.8467 | 2.858 | 0.8366 | 1.527 |
| 120 | 0.8473 | 3.014 | 0.8464 | 3.164 |
| 600 | 0.8475 | 3.132 | 0.8466 | 3.096 |

(truth: ρ = 0.85, excess kurtosis = 3.0)

**Both arms reach the truth.** On clean data ρ is converged after a *single* M-step — there
is no missing information at the chain level — while through the channel it takes 30–60.
This is the Dempster–Laird–Rubin missing-information mechanism showing up in the
*convergence rate*, which is exactly where the theory puts it. Combined with §3.2's null
paired shape difference at convergence, the channel's cost in this regime is iterations,
not asymptotic accuracy.

> `audit/audit_rate_vs_bias.py`, written up in `audit/AUDIT_NOTE.md` §2

### 3.7 A caveat on the Laplace arm specifically

The Laplace M-step maximises `Q` exactly, but its ρ is the minimiser of a weighted mean
absolute residual, hence a **breakpoint** — one of the ratios `u_k/u_j`. On a uniform grid
through the origin those ratios are rationals `a/b` with `|a|,|b| ≤ (M−1)/2`, so the
estimator is **lattice-valued**, and low-denominator values pool the weight of all their
aliases. Every fitted value we observed is exactly such a rational: 3/4, 7/9, 11/14,
106/135, 30/37, 43/53, 4/5.

The consequence is a bias, not a resolution limit: at a truth of **0.7913** the estimate is
pinned at exactly **0.8** across an eightfold refinement (`M = 201 → 1601`), holding a
constant 0.0087 error. This also explains why exp_06's Laplace arm "recovers" ρ to
1.1 × 10⁻¹⁶ — its truth is 0.8 = 4/5, the strongest attractor in the neighbourhood. That
figure measures the lattice, not the estimator.

The mixture kernel's ρ block is a continuous weighted least-squares solve and is **not**
affected, which is why the headline recovery numbers of §3.1 are clear of this.

> `outputs/exp_06_em_parameter_recovery/laplace_quantization.csv`;
> guarded by `tests/test_laplace_rho_lattice.py`

---

## 4. Corrections made this pass

### 4.1 A sign error in the mixture ρ gradient

`MixtureInnovationKernel.grad_log_transition_matrix` returned the **negative** of the true
derivative. With `e = u_k − ρ u_j` we have `∂e/∂ρ = −u_j`, so the two minus signs cancel and
the derivative is `+u_j Σ_c r_c (e − μ_c)/s²_c`. Against central differences the analytic
value gave a ratio of **−1.0000000000239**, with `analytic + finite_difference` vanishing to
2 × 10⁻¹⁰.

This never touched any recovery result: the ECM M-step solves its ρ block through the
normal equations directly and had the correct sign. It corrupts only the routes that
consume the analytic derivative — Fisher-gradient ascent, score tests, and the observed
information estimates of exp_22.

Fixed, with two regression tests: a finite-difference check, and a check that the
one-component zero-mean mixture reproduces `GaussianAR1Kernel` exactly — the invariant the
bug violated.

### 4.2 exp_08 reported two different models in one row

`gradient_ascent` logged `(θ, logL)` at the top of the loop, then stepped, then returned the
stepped parameter. So every result row mixed iterates: `param_err` described the returned
parameter while `logL_final`, `logL_gap_to_em` and `monotone_violation` described its
predecessor — and the final update was never scored at all, so a step that diverged on
precisely the last iteration was reported as monotone.

Fixed to evaluate every state it returns (`n_updates + 1` evaluated states, with
`n_updates` stored rather than inferred from the trace length). exp_08 was regenerated;
the Gaussian gap moved from 1.93 × 10⁻⁹ to 1.57 × 10⁻⁹ and both paper claims survive.
Four regression tests, including one that pins a descent on the *final* update — the case
the old alignment hid.

`fit_em` in `src/em.py` had the same bug and had already been repaired; the experiment side
now matches.

### 4.3 The ρ = 0.85 provenance conflict

The paper asserted that "every experiment initialises ρ at 0.3 against a truth of 0.85"
while exp_06/07/08 use **0.8**, and exp_18 sweeps four initialisations rather than one.
Both halves were wrong. Separately, `exp_16` claimed its `ρ=0.85` "matches exp_07 so the
pointwise numbers line up" — exp_07 is at 0.8, so they do not.

Resolved by scoping each claim to its own truth rather than by editing a constant: the
truths are genuinely different and legitimately so.

### 4.4 A recovery interval that reproduced from nothing

`[0.8497, 0.8507]` appeared in the paper, the compendium and two summary documents. The
committed exp_18 trace gives **[0.8517, 0.8520]** at `C=4`. Corrected everywhere; the
audit script now fails if the withdrawn value reappears.

### 4.5 The "discretisation floor" is withdrawn

We had reported that the clean arm's innovation-variance error was flat in `N` and
attributed the flatness to a discretisation floor. Both halves were wrong.

The flatness was **replication noise**. At three replicates an RMSE is estimated from three
numbers, and its own sampling error is comparable to the effect being read off it. At
sixteen the curve is not flat:

| N | clean, grid-binned | clean, no grid at all | N^(−1/2) reference |
|---:|---:|---:|---:|
| 64 | 0.01357 | 0.01348 | 0.01348 |
| 128 | 0.00731 | 0.00715 | 0.00953 |
| 256 | 0.00565 | 0.00565 | 0.00674 |
| 512 | 0.00422 | 0.00428 | 0.00477 |
| 1024 | 0.00318 | 0.00320 | 0.00337 |

The `N=64` entry alone moves by a factor of three between the two replication levels.

And the grid was never a plausible culprit. Removing it entirely — exact clean-data OLS and
raw-transition ECM on the raw pairs, implemented in `src/clean_mle.py` — reproduces the
binned curve, with a paired grid-minus-raw discrepancy that is a **constant 4.4 × 10⁻⁴**
across every budget. That is an order of magnitude below the estimation error at `N=64`.
A quantity that small cannot produce a floor at 4 × 10⁻³.

> `outputs/exp_06_em_parameter_recovery/clean_raw_mle.csv` — `exp_06 --only clean_raw_mle`;
> estimators guarded by `tests/test_clean_mle.py`

This also gives Regime A a proper grid-free reference, which it previously lacked: the
clean arm was the MLE of a *binned* objective, so any error it showed was open to being
blamed on the grid. It no longer is.

### 4.6 Stale test count

The documented expectation was `135 passed, 12 skipped`. Actual on this environment:
**236 passed, 12 skipped** (Python 3.12.5, NumPy 2.5.2, SciPy 1.18.0), including the 16
regression tests added here.

---

## 5. What we are *not* claiming

- not "the continuous E-step is exact" — see §2;
- not that the four-sweep mixture update is the exact global M-step — it is GEM/ECM;
- not that EM has converged because ρ is flat — see §3.5;
- not that EM beats neural networks in general — see the scope note in §3.3;
- not that the non-Gaussian chain is strictly stationary under the current initialisation;
- not that the fitted mixture is constrained to zero innovation mean;
- not that the shape-information decay ratio is known precisely — the estimated
  information matrices are ill-conditioned and were inverted without Monte Carlo
  uncertainty.

---

## 6. Reproducing this

```bash
cd research/nongaussian-bp
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
OMP_NUM_THREADS=4 python3 -m pytest tests/ -q          # 236 passed, 12 skipped
python3 ../../tools/audit_em_bp_provenance.py          # provenance gate, exit 0
```

Every experiment writes a `params*.json` beside its output recording the git commit,
branch, dirty state, seeds, grid, truth, and library versions.
