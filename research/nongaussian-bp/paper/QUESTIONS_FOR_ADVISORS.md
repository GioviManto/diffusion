# Questions for Marc and Jérôme

Drafted 6 August 2026, alongside the internal note. These are the decisions I could not
resolve from the code or from the two guidance documents, ordered by how much they change
the write-up. Each states what I would do absent an answer, so nothing is blocked.

---

## 1. The headline no longer survives the sampling test — how should the note be framed?

This is the substantive one, and it follows directly from Jérôme's point in the 29 July call
that score error and generated-data quality are different quantities.

We now have that comparison. Laplace chain, `N = 2048` clean sequences, four score functions
inside the same reverse integrator with common random numbers:

| | EM–BP (12 free params) | local CNN (6k) | global MLP (25k) | target |
|---|---|---|---|---|
| MSE against the Bayes denoiser | **0.00052** | 0.00525 | 0.01330 | 0 |
| generated innovation kurtosis | 0.897 | **1.357** | 0.206 | 1.910 |
| KL(true ‖ generated), innovations | 0.0055 | **0.0028** | 0.0165 | 0 |
| worst covariance-lag error | **0.0292** | 0.0973 | 0.0518 | 0 |

The validation passes first: run with the *true* kernel, the sampler reproduces the
closed-form target for `P_t` at `t_min` at every budget (1.897–1.914 against 1.9098).

So EM–BP is ten times better pointwise, three times better on second-order structure, and
**worse than the CNN on the innovation law**. The MLP is the sharpest case: its pointwise
error improves twelvefold with data while its generated kurtosis degrades from 0.850 to
0.206 — it fits the bulk and loses the tail.

Three framings, and we would like your view:

**(a) Sample efficiency as the headline**, with the generation result as a stated limitation.
Closest to the original suggestion, but it foregrounds the metric that the experiment just
showed is not the one that matters.

**(b) The dissociation itself as the result** — pointwise score accuracy and distributional
fidelity rank these estimators differently, and we can say why. This is the more honest
reading of what we measured, and arguably more interesting, but it is a weaker claim about
EM–BP specifically.

**(c) Wait.** A sweep over the mixture component count `C ∈ {2,4,8,12,16}` is running and
decides whether EM–BP's under-dispersed tail is a *capacity* limit (fixable — more
components) or an *information* limit (not fixable — the channel destroys innovation-shape
information far faster than correlation, measured at 142× against 26× across the schedule).
If it is capacity, framing (a) survives with a fix. If it is information, (b) is the result.

*Default if no answer:* write (b), and add the `C` sweep when it lands.

---

## 2. Should the note claim anything about generation at all?

Related but separable. The reverse-diffusion comparison is complete for one chain family
(Laplace) and running for the other four. If you would rather the note stay strictly on
denoising and defer all generation claims to the thesis, say so — it would shorten the note
considerably and remove its most contestable section.

*Default:* include it, clearly marked as one family with four more in flight.

---

## 3. `exp_12` selected the CNN receptive field on the test set

The oracle over radius and parameterisation takes a minimum over the same held-out set the
number is then reported on. The bias favours the **baselines**, so it makes our margin look
*smaller* than a clean protocol would — but it is still improper.

Options: rerun with a proper validation split (about half a day of compute), or keep the
number and label it exploratory with the direction of the bias stated.

*Default:* rerun. It is cheap and the current number cannot go in a paper as-is.

---

## 4. Which estimator is the headline — EM, generalised EM, or Fisher-gradient ascent?

All three are implemented and compared (`exp_08`). Marc's email describes *gradient* ascent
on the likelihood; what we mainly report is exact-M-step EM. They are not the same algorithm
and the comparison is not one-sided: on the smooth Gaussian kernel gradient ascent converges
to exactly EM's optimum and gains nothing while requiring a tuned step size; on the Laplace
kernel it is genuinely better, because there EM's exact M-step lands on a lattice artefact.

*Default:* lead with EM, present the gradient route as the alternative Marc described, and
state the split.

---

## 5. Is the initial law learned?

`μ(a₁)` is currently fixed to a standard normal and is **not** counted among the estimated
parameters. Is that the right scope, or should it be estimated too? It matters for the
identifiability statement — kernel identifiability needs the initial law fixed or separately
identifiable, and we would rather state the assumption than quietly rely on it.

*Default:* keep it fixed, state the assumption explicitly.

---

## 6. Parameter count

The mixture-innovation kernel at `C = 4` has 1 (ρ) + 4 (π) + 4 (μ) + 4 (σ²) = 13 raw
parameters, but π is simplex-constrained, so there are **12 free**. The earlier draft's title
used 13. We will write "twelve free parameters" and note the redundancy, unless you prefer
the raw count for comparability with the network's parameter count.

---

## 7. Are the neural baselines varied over datasets or only over initialisations?

At present the training set is fixed per `N` and only the initialisation seed varies, so the
error bars measure optimisation variance, not sampling variance. The honest fix is a fresh
draw per seed. This is a moderate rerun.

*Default:* rerun with fresh draws before any number is called a standard error.

---

## 8. Scope of the note

Layer 6 (hierarchical priors, speciation ladder, memorisation) is parked, on the grounds that
it is not in either guidance document and depends on a paper we were unable to obtain. Please
confirm that is right, or say if you would rather it appear as a section.

---

## A note on what is *not* in question

These are settled and we are not asking about them:

* The posterior of a coordinatewise-noised Markov chain is a chain, so sum-product is exact
  at the level of functional messages. Verified against brute-force enumeration to 1.0e-14.
* The single-Gaussian moment closure returns exactly the LMMSE estimator of the
  covariance-matched Gaussian model, for any innovation law with the same first two moments.
  Verified to 6e-16 across five families.
* Fisher's identity gives the exact gradient of the marginal likelihood from one BP pass, so
  no automatic differentiation through the recursion is needed. Verified against finite
  differences to ~1e-9.
