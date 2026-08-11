# Executive summary

**Diffusion scores for Markov data: exact chain inference and estimation of the transition
kernel.** Giovanni Mantovani, Department of Computing Sciences, Bocconi University.
Internal note, work in progress, 7 August 2026.

---

## The question

A diffusion model is trained by denoising regression, and what that regression targets is the
posterior mean of the clean signal given its noised version. For realistic data this target is
unknown, so the error a trained network makes cannot be separated from the difficulty of the
object it is approximating. We work in a family where it can: a first-order Markov chain on
$\mathbb{R}^n$ with continuous, non-Gaussian transitions.

The question the work is organised around: **how much does known Markov structure reduce the
statistical and computational burden of learning a diffusion score, and what is lost when the
transition law or the continuous messages must be approximated?**

## Model and assumptions

Clean data $a = (a_1,\dots,a_n)$ with $a_1 \sim \mathcal{N}(0,1)$ and
$a_i = \rho a_{i-1} + \varepsilon_i$, innovations i.i.d. with mean zero and variance
$q = 1-\rho^2$. Forward noising is coordinatewise Ornstein–Uhlenbeck,
$x = e^{-t}a + \sqrt{\Delta_t}\,z$ with $\Delta_t = 1-e^{-2t}$.

Two assumptions are load-bearing and both are stated rather than tested:

- **Exact first-order Markov structure.** Under a non-Markov prior the estimator is
  misspecified in a way more data does not repair.
- **Coordinatewise noise.** If the forward noise were correlated across coordinates the
  likelihood would not factorise and the whole construction collapses.

One correction from the previous draft matters here: choosing $q = 1-\rho^2$ makes the chain
**covariance-stationary**, not strictly stationary. Second moments propagate exactly
($\mathrm{Var}(a_i)=1$, $\mathrm{Cov}(a_i,a_j)=\rho^{|i-j|}$ for every innovation law), but for
non-Gaussian innovations the invariant law is not Gaussian, so a Gaussian $a_1$ starts the chain
off it. Nothing in the comparisons depends on this — all families share the same $a_1$ and the
same covariance — but the earlier claim of stationarity was too strong.

## What is exact

1. Coordinatewise noising contributes one **unary** factor per site. It reweights node
   potentials without creating edges among latent variables, so the posterior factor graph is
   the prior's chain. This is a theorem.
2. A chain is a tree, so sum-product returns the exact single-site **and pairwise** marginals
   at the level of functional messages. Also a theorem.
3. Consequently the E-step of an EM procedure needs no variational or sampled approximation.

## What is numerically approximated

Messages are functions on $\mathbb{R}$, so an implementation must represent them. We use a
truncated uniform grid with trapezoidal weights, which defines a **finite-state model** for
which the recursion is exact up to floating point, and which **approximates** the continuous
model. The two error sources are now measured separately:

| source | diagnostic | behaviour |
|---|---|---|
| truncation | worst boundary mass over 32 sites | $3.4\times10^{-4}$ at $A{=}4 \to 1.4\times10^{-8}$ at $A{=}8$; independent of $N_g$ |
| quadrature | interior column-mass residual | $O(h^2)$, factor 4 per halving of the spacing |

At the working configuration $N_g = 401$, $A = 8$ both sit far below any reported effect. The
discretisation reproduces the closed-form Gaussian posterior mean to $9.2\times10^{-15}$ and
agrees with brute-force enumeration — which shares no code with the recursion — to
$1.6\times10^{-14}$ on means and $1.8\times10^{-15}$ on the log-evidence.

## What is learned

The kernel is estimated inside the linear-autoregressive family
$K_\theta(a'\mid a) = \sum_c \pi_c\,\mathcal{N}(a' - \rho a;\ \nu_c, s_c^2)$, by maximum
marginal likelihood. **Both** the autoregressive coefficient $\rho$ and the innovation density
are estimated: $\rho$ is initialised away from its truth and recovered. In `exp_18` (truth
$0.85$) it lands in $[0.8517, 0.8520]$ at $C=4$ from initialisations spanning $[-0.4, 0.6]$.
The recovery and sample-efficiency experiments (`exp_06`, `exp_07`, `exp_08`) use a truth of
$0.8$ instead; no table pools the two. At $C=4$ there are twelve free
parameters. The **initial law is held fixed** at $\mathcal{N}(0,1)$, which is the correct one,
so there is no misspecification in it — but neither is there evidence it could be recovered.
The linear form of the kernel is assumed, not learned.

Two algorithmic points. The gradient of the *marginal* log-likelihood comes from Fisher's
identity, $\nabla_\theta \mathcal{L} = \langle \Xi(\theta),\nabla_\theta \log K_\theta\rangle$,
so belief propagation is never differentiated through — it supplies the conditional expectation.
And the M-step for the mixture is **not** an exact maximisation: it runs four inner
conditional-maximisation sweeps, so the algorithm is **generalised EM / ECM**. Monotone ascent is
preserved in theory and observed in every run (largest decrease across iterations: zero, in all
eight runs recorded).

## Main pointwise result

Under the correctly specified Markov model and at the data budgets tested, the structured
estimator attains lower relative denoising error than the neural baselines we trained, by a
factor of **9–14** across seven doublings of the data, with no trend in the budget. Against a
locality-respecting convolution the margin narrows to **2–4**, so more than half of the fully
connected network's deficit was its architecture rather than the estimator.

The estimator is given the Markov factorisation, the linear form of the kernel, and homogeneity
across sites. The networks are given none of these. The comparison measures the value of that
structure; it is not a claim about neural denoisers in general.

## Main generative result

Errors accumulate through reverse integration, so the pointwise ranking need not survive. At a
deliberately small innovation model ($C=4$) it does not: the estimator is an order of magnitude
better pointwise and three times better on second-order structure, and yet a convolutional
network reproduces the innovation law more faithfully. The effect holds across four non-Gaussian
families at matched covariance, at three chain lengths, and is unchanged under fourfold grid
refinement. On the **Gaussian** chain it disappears — the negative control the capacity account
predicts and an information account does not, since a four-component mixture represents a
Gaussian innovation exactly.

Enlarging the innovation model resolves it, on both axes at once:

| $C$ | MSE vs Bayes denoiser | generated excess kurtosis (target 1.910) |
|---|---|---|
| 2 | 0.002281 | −0.034 |
| 4 | 0.000510 | 0.812 |
| 8 | 0.000276 | 1.319 |
| 12 | 0.000249 | 1.363 |
| 16 | **0.000234** | **1.487** |

(Local CNN: 0.005161 and 1.273 throughout, being $C$-independent.) Pointwise error improves by
9.7× while generated kurtosis climbs monotonically, so the two axes **do not trade off** anywhere
in the tested range; at $C=16$ the estimator leads on both simultaneously.

## Limitations

- Exact Markov structure and coordinatewise noise are assumed, not tested.
- The initial law is fixed; strict stationarity has never been run.
- Identifiability of the kernel is assumed, not proved.
- Inference costs $O(nN_g^2)$ per sequence — measured at 210–289× a network forward pass. Since
  reverse diffusion calls the denoiser at every step, this is the practical weak point.
- No validation split. Two selections happen on the evaluation set, both favouring the
  baselines; the receptive-field numbers are labelled exploratory because of it.
- The capacity sweep covers pointwise error and generated statistics, but not marginal
  likelihood or runtime against $C$.
- Locality of the score is probed only through a receptive-field proxy.

## Immediate next experiments

1. Rerun the receptive-field comparison with a genuine validation split; replace the
   per-noise-level "better of two parameterisations" with validation-based selection.
2. Add marginal likelihood and runtime to the capacity sweep, completing the joint experiment.
3. Sample $a_1$ from each family's invariant law and confirm that nothing changes.
4. Learn $\rho$ *and* $\mu$ jointly, using the site-one statistic the E-step already computes but
   currently discards.
5. A direct locality measurement — windowed inference, or the Jacobian of the score with respect
   to distant observations — rather than the current proxy.

---

*Full derivations: `compendium/main.pdf` (43 pp). Results and protocol: `paper/main.pdf`.
Claim-level audit against the implementation: `REVISION_AUDIT.md`. Commands and environment:
`REPRODUCIBILITY.md`.*
