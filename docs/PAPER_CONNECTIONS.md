# Layer 6 — what the two papers imply for this project

Two papers were handed to the project as things to study and take inspiration
from. Both turn out to be by the advisors themselves, and together they define
almost exactly the two axes this package has been measuring blind.

- **P1** — Jérôme Garnier-Brun, Marc Mézard, Emanuele Moscato, Luca Saglietti,
  *How transformers learn structured data: insights from hierarchical
  filtering*, arXiv:2408.15138; ICML 2025 (PMLR 267:18831–18847).
- **P2** — Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, Marc Mézard,
  *Dynamical regimes of diffusion models*, Nature Communications **15**, 9957
  (2024), doi:10.1038/s41467-024-54281-3; arXiv:2402.18491.

---

## 0. How these papers were read, and what that means for what follows

**This has to be stated first, because it bounds everything below.** The
execution environment blocks outbound HTTP at the gateway: `arxiv.org`,
`www.nature.com`, `pmc.ncbi.nlm.nih.gov`, `hal.science`, `proceedings.mlr.press`
and every mirror tried all return `403` on `CONNECT`, from both the fetch tool
and `curl`. **Neither PDF could be opened.** What is recorded below came from
web *search* (which routes server-side and does work), which returned titles,
authors, abstracts and summaries of the papers' content.

The consequences, stated plainly:

- **Titles, authors, venues and the substance of the abstracts are reliable** —
  they were returned consistently across independent queries and cross-checked
  against multiple indexes.
- **No equation, constant, notation or figure below is quoted from either
  paper.** Everything mathematical here is *derived from scratch in this
  package's own conventions* and verified numerically in
  `tests/test_hierarchy.py`. Where a result of theirs is described in words,
  it is described as the mechanism, not as a formula.
- One search summary rendered the speciation condition as
  `t_S = ½ log(1/Λ)`. That is almost certainly a transcription slip: the same
  summary states that `t_S` *diverges* when `Λ` grows with the dimension,
  which requires `t_S = ½ log Λ`. **Rather than adopt either, §2 derives the
  crossover independently** and checks it against the sampled process
  (`test_commitment_matches_the_forward_process_empirically`). The derived
  form is `t_S = ½ log(1 + Λ)`, which reduces to `½ log Λ` for large `Λ`.
- **Anyone continuing this work should read both PDFs directly** and reconcile
  notation. Nothing here depends on their notation — the code and the tests are
  self-contained — but a write-up that cites them must use their symbols.

---

## 1. What the two papers say

### P1 — hierarchical filtering

Sequences are generated on a balanced tree: a root symbol, then children drawn
from transition tensors, level after level, with the leaves forming the observed
sequence. Correlations between two leaves are then controlled by their
*hierarchical* distance — how far up you must go to find a common ancestor —
rather than by their position in the sequence. A **filtering parameter `k`**
truncates the hierarchy, so the range of positional correlations in the data can
be dialled from none to the full depth.

The model is exactly solvable at every filtering level: the graph is a tree, so
**belief propagation gives exact inference in linear time**, and BP is therefore
the reference against which a trained network is measured.

Findings: vanilla encoder-only transformers, trained on root classification and
masked-token prediction, **approximate the exact inference algorithm**; and
**correlations at increasing distance are acquired sequentially during
training**, level by level up the hierarchy. Attention maps and layer-wise
probes show a reconstruction of correlations on successive length scales.

### P2 — dynamical regimes

Diffusion models are analysed with statistical-physics methods in the regime of
large dimension and large sample count, with an optimally trained score. The
backward dynamics passes through two cross-overs:

- **Speciation** — the coarse structure of the sample is decided, by a mechanism
  analogous to symmetry breaking. Its time scale is set by the **top eigenvalue
  of the data covariance**; when that eigenvalue grows with dimension, the
  speciation time diverges logarithmically.
- **Collapse** — trajectories are captured by a single training point, by a
  mechanism analogous to glassy condensation. Its time scale is set by an
  **excess entropy** of the data, and the consequence is a curse of
  dimensionality: memorization is avoidable at finite times **only if the
  dataset is exponentially large in the dimension**.

The collapse statement is about the **empirical score** — the exact score of
the measure putting mass `1/N` on each training point, which is what an
unconstrained model with a perfectly minimized training loss converges to.

---

## 2. Why this is directly load-bearing here, not just adjacent

Three things line up, and one of them is a genuine gap in this project's story
that P2 fills.

**(a) P1's reference algorithm is this project's method.** P1 measures a
transformer against exact BP on a tree. Layers 1–5 here compute a diffusion
score by exact BP on a chain and measure a network against it. The difference is
the task, not the epistemics; the "structure first, then compute" stance is
identical. What P1 adds is the *hierarchy*: on a chain there is one length scale
and "which correlations has the model learned" has no graded answer. On a tree
it does. That is why `src/hierarchy.py` exists.

**(b) P2 explains the sample-efficiency result that Layer 5 measured but did
not explain.** Layer 5's headline is that EM-BP matches a score network with
≳64× less data, and that its error falls at the parametric `N^{−1/2}` rate
(measured slope −0.500 ± 0.048 at 12 replicates). That was reported as an
empirical fact. P2 supplies the mechanism, and it is sharper than "BP has an
inductive bias":

> The empirical score's sufficient statistic **is the training set** — `N × n`
> numbers, growing in both dataset size and dimension, which is why it must
> collapse onto that set unless `N` is exponential in `n`. A BP score's
> sufficient statistic is the fitted kernel: two numbers for a Gaussian AR(1),
> `M × M` for a nonparametric one, and **independent of `n`**. The data reaches
> it only through `Ξ`, which is an average. There is no training point for a
> trajectory to be captured by.

So the prediction is not "BP memorizes less by some margin". It is that **the
memorization axis does not exist for a BP score**, at any `n` and any `N`, while
for the empirical score it is governed by an exactly computable entropy. That
is falsifiable, and `exp_14` measures it.

**(c) The two papers compose into a question neither asks.** P1's data model has
a ladder of length scales; P2's speciation time is set by a covariance
eigenvalue. A hierarchical prior has **one eigenvalue per level**, so it should
show not one speciation time but `L + 1` of them, and the reverse diffusion
should resolve the hierarchy **coarse-to-fine**, one transition per level. This
appears to be untouched by either paper and is the subject of `exp_13`.

---

## 3. What was derived, in this package's conventions

All of it verified in `tests/test_hierarchy.py` (25 tests) rather than asserted.

### 3.1 The speciation crossover

Along an eigendirection of the clean covariance with eigenvalue `Λ`, the
variance-preserving forward process splits into signal and noise,

    Var(⟨v, x_t⟩) = α_t² Λ + Δ_t,     α_t² = e^{−2t},  Δ_t = 1 − e^{−2t}.

The mode is decided while signal dominates, undecided once noise does, so

    **t_S(Λ) = ½ log(1 + Λ)**,

and the measurable statement is that the correlation between the projection at
time `t` and at time `0`,

    commitment(t, Λ) = α_t √Λ / √(α_t² Λ + Δ_t),

passes through `1/√2` exactly at `t_S`. Checked analytically
(`test_commitment_crosses_one_over_sqrt_two_at_the_speciation_time`) and against
40 000 sampled trajectories of the forward process, agreeing to < 0.02
(`test_commitment_matches_the_forward_process_empirically`).

### 3.2 A chain has no speciation cascade; a tree does

For a stationary AR(1) chain with `Cov = ρ^{|i−j|}`, the covariance spectrum is
bounded for every length by the spectral density at zero frequency,

    **Λ_max → (1 + ρ)/(1 − ρ)**,

so `t_S` *saturates*: 64× the chain length changes the top eigenvalue by under
2× (`test_chain_top_eigenvalue_saturates_at_the_limit`). **A stationary Markov
chain therefore has no diverging speciation time and no coarse-to-fine cascade
to resolve** — it sits outside the regime P2 describes for image data, and that
is a property of the data model, not of the method.

For the balanced-tree prior (`z_child = ρ z_parent + innovation`, leaves
observed) the leaf covariance is ultrametric, `Cov = ρ^{2(L − d_LCA)}`, and it
has exactly `L + 1` distinct eigenvalues, derived in closed form in
`GaussianTree.level_eigenvalues`:

    Λ_d = S_{d+1} − b^{L−d−1} ρ^{2(L−d)},   multiplicity (b−1) b^d,
    S_d = 1 + (b−1) Σ_{m=1}^{L−d} b^{m−1} ρ^{2m},

with the uniform mode at `Λ = S_0`. Verified against `eigh` of the dense
covariance to 1e-10 for three (branching, depth, ρ) combinations, with
multiplicities summing to the number of leaves and the trace to it as well.
When `bρ² > 1` the top eigenvalue grows geometrically in depth, hence
**logarithmically in dimension** — which is the regime P2's analysis describes,
now with an exactly known spectrum rather than an empirical one.

### 3.3 A closed-form collapse criterion for the chain

For a Gaussian AR(1) chain the differential entropy is exactly extensive, so the
per-site excess entropy relative to the terminal measure `N(0,1)` is

    **s = −½ log(1 − ρ²)**,

exactly, with no fitting — and the dataset size below which memorization is
forced is `exp(n s)`. At `ρ = 0.85` that is **0.641 nats per site**, so 33 sites
already demand ~10⁹ chains. Checked against `−½ log det C`
(`test_excess_entropy_matches_the_gaussian_determinant`).

At finite `t` the noised law is `N(0, α_t² C + Δ_t I)`, so

    s(t) = −(1/2n) log det(α_t² C + Δ_t I)

is available in closed form too, and `n s(t_C) = log N` gives a **predicted
collapse time with no fitted constants**. `exp_14` part `time` tests it.

### 3.4 Exact BP on a tree, two ways

- `tree_bp_gaussian` — information form `(h, λ)`, exact, `O(n)`. The upward and
  downward updates touch only those two numbers per message: the audit's
  finding F2 (that the previous Gaussian BP was doing "something strange"
  instead of updating mean and variance) applies here too, and the code is
  written so that there is nowhere for a projection step to hide. Matches a
  dense linear solve to **< 1e-10** at four noise levels and for branching 2
  and 3.
- `tree_bp_grid` — grid messages, any innovation law. Matches the information
  form to **< 2e-6**, and returns finite, visibly different answers on a
  Laplace-innovation tree where no closed form exists. The leave-one-out
  sibling product is computed by prefix/suffix scans rather than by dividing
  out a message, since message entries are legitimately ~1e-16 in the tails.

### 3.5 EM on a tree

`tree_e_step` produces the **same `Ξ` matrix** as the chain E-step — the
topology stops being visible at exactly that point, so **every kernel in
`src/kernels.py` consumes it unmodified**. A chain-trained M-step and a
tree-trained M-step are the same code. The evidence is recovered by accumulating
the log-scales discarded at each renormalization, and is verified against the
closed-form Gaussian marginal likelihood to a **relative 1e-6**, which is what
makes the monotone-ascent check available on trees as well.

**Measured, and worth knowing before running anything:** tree EM converges much
more slowly than chain EM. At depth 3 with 512 trees the estimate is
`ρ̂ = 0.7360` at 50 iterations, `0.7483` at 100 and `0.7487` (converged, true
value 0.75) at 150, with **zero monotone violation throughout**. On a chain
every site is observed; on a tree the internal nodes never are, the missing
information fraction is far larger, and EM's linear rate is correspondingly
slower. An estimate read off at 40 iterations looks like a broken M-step and is
merely unconverged — this cost real time to diagnose here and is now asserted
as a *rate* property in the tests rather than as a single-budget accuracy claim.

---

## 4. What was built

| Added | Purpose |
|---|---|
| `src/hierarchy.py` | Tree prior, analytic ultrametric spectrum, exact tree BP (information form + grid), tree E-step, `fit_em_tree` |
| `src/spectral.py` | Speciation time, commitment, chain spectrum and its limit, excess entropy, collapse dataset size |
| `experiments/exp_13_speciation_cascade.py` | `spectra`, `cascade`, `levels`, `ordering` |
| `experiments/exp_14_memorization_collapse.py` | `budget`, `collapse`, `time` |
| `tests/test_hierarchy.py` | 25 tests |

---

## 5. Status of the experimental results

`exp_13` and `exp_14` were launched at full settings; §6 records what has
already been measured and what is still running. **Numbers below the line in
each section are from completed runs only**; anything provisional says so.
