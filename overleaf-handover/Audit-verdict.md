# Audit verdict

**Overall decision: scientifically promising, substantially improved, but not ready for external submission.**

The repository’s own handover README currently marks the **paper and workshop as not ready**, the **compendium as ready**, and the **thesis as a draft**. I agree with the first two assessments, but I would be stricter about the latter two: the compendium is ready as an **internal development record**, not yet as a polished companion, and the thesis contains a serious mismatch between its stated research questions and the work it actually presents. The README also correctly blocks the current neural-efficiency table because the two methods were not selected or stopped symmetrically. ([GitHub][1])

| Work                                   | Decision as work in progress                | Decision for submission                                                  |
| -------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| **9-page paper**                       | **Approve as a strong internal scaffold**   | **Reject in current form**                                               |
| **4-page workshop paper**              | **Approve only as a source draft**          | **Reject; requires a scope rebuild**                                     |
| **Compendium**                         | **Approve as an internal technical record** | **Conditional; prune and reorganize first**                              |
| **Thesis**                             | **Approve the analytical core**             | **Reject as a final thesis until scope and claims are reconciled**       |
| **Core BP code**                       | **Approve, with terminology corrections**   | **Nearly publication-grade after numerical-contract fixes**              |
| **EM/learning pipeline**               | **Conditional approval**                    | **Reject as a final evidence generator until convergence is redesigned** |
| **Current main figures/results suite** | **Mixed**                                   | **Several must be removed, demoted, or regenerated**                     |

The project has enough solid work for a strong paper. The problem is no longer lack of material. The problem is **failure to select**.

---

# 1. What the strongest paper is actually about

The cleanest scientific story is:

> **Coordinatewise Gaussian noising can make the score of a Markov sequence globally dependent on the observations, but the score remains computable by local inference in the latent chain. The same posterior inference produces the pairwise statistics needed to learn the unknown local transition law.**

The causal chain is simple:

[
\text{local Markov prior}
\longrightarrow
\text{unary noising factors}
\longrightarrow
\text{posterior remains a chain}
\longrightarrow
\text{belief propagation gives posterior means}
\longrightarrow
\text{posterior means give the diffusion score}.
]

For learning:

[
\text{belief propagation}
\longrightarrow
\text{pairwise posterior marginals}
\longrightarrow
\Xi
\longrightarrow
Q(\theta\mid\theta^{(r)})
\longrightarrow
\text{transition-kernel update}.
]

This should be the main paper’s entire backbone.

Three contributions are sufficient:

1. **Exact functional inference:** the posterior remains a tree-structured chain and functional sum–product computes the exact latent marginals.
2. **A precise interpretation of Gaussian closure:** moment-projected BP is exactly inference under the covariance-matched Gaussian model, hence the LMMSE estimator.
3. **Likelihood learning of the local transition:** the pairwise posterior table (\Xi) provides the data-dependent object needed for Fisher gradients and EM or generalized EM.

The current paper already contains all three ingredients and explains the tree structure, message domain, notation, functional exactness, numerical grid representation, and complexity much better than the earlier manuscript. ([GitHub][2])

The **rotating-ring theorem is interesting and defensible**, but it is a different scientific story:

* a different latent state space;
* a different dynamical model;
* a different unknown object (p(\psi));
* a different analytical mechanism based on rotational gauge symmetry;
* a different likelihood;
* and a separate joint-versus-marginal question.

It belongs in the thesis and compendium, or eventually in its own short note. It should not occupy scarce pages in this paper.

The neural comparison is also a different question. It asks what a structural assumption is worth relative to particular trained architectures and protocols. It is not necessary to establish the analytical paper, and the current headline comparison is explicitly blocked by the project’s own audit. ([GitHub][1])

---

# 2. Claim-by-claim scientific audit

## Claims I approve

### A. Score computation through latent posterior means

This is the central exact identity. It is analytically sound and should remain prominent.

### B. Conditioning preserves the latent chain

The likelihood factors are unary, so conditioning reweights nodes without adding latent edges. The posterior factor graph therefore remains a chain, which is a tree—a connected acyclic graph. The current paper now states this clearly. ([GitHub][2])

### C. Functional belief propagation is exact on that chain

The continuous messages are functions on (\mathbb R), and the integral recursions are exact at the functional level. The finite-grid code is a numerical representation of those functions, with domain truncation and quadrature error. That distinction is correctly recognized in the source and must be maintained everywhere. ([GitHub][3])

### D. Gaussian closure equals covariance-matched LMMSE inference

This is one of the best results in the project. It converts an approximation into an interpretable object:

> Gaussian message closure does not vaguely “approximately account for non-Gaussianity.” It computes the answer for the Gaussian prior having the same first two moments.

The difference from full BP can therefore be interpreted as the value of structure beyond second moments. This result is analytical, useful, understandable, and exactly aligned with the paper philosophy. ([GitHub][2])

### E. Pairwise posterior marginals determine the EM surrogate

The accumulated pairwise table (\Xi) is the complete data-dependent input to the edge part of the current (Q)-function. Fisher’s identity then provides the marginal-likelihood gradient without differentiating through the forward–backward recursion. The code and executable audit support the gradient and monotonic-ascent implementation. ([GitHub][4])

Use precise language, however:

> (\Xi) is a sufficient summary for the current E-step/Q-function and M-step, conditional on the current parameter, model family, grid, fixed initial law, and homogeneous transition assumption.

Do not call it simply “a sufficient statistic of the observations.” It depends on the posterior evaluated at the current parameter.

### F. Gaussian-kernel EM update

The Gaussian transition has a closed-form weighted least-squares/Yule–Walker update. This is clean and suitable for the main text as the solvable learning case.

### G. Rotating-ring marginal blindness theorem

In isolation, the theorem is conceptually sharp: rotational invariance makes every per-frame marginal independent of the rotation parameter while the full trajectory can retain information. The proof appears simple and sound. ([GitHub][2])

I approve this as a **separate result**, not as a rung in the main chain paper.

---

## Claims I approve only with narrower wording

### A. “Machine-precision validation”

The code reproduces the Gaussian closed form and a short-chain discretized brute-force computation at approximately (10^{-14})–(10^{-15}) in the reported tests. That validates those particular implementations and configurations. ([GitHub][2])

It does **not** establish that a (G=401), (A=8) grid has (10^{-15}) error for arbitrary non-Gaussian kernels, observations, tails, or small noise levels.

The correct claim is:

> At the stated Gaussian test configuration, the grid recursion agrees with the analytic solution and an independently implemented finite-grid enumeration to double-precision accuracy.

Do not write:

> The grid approximation has (10^{-15}) error.

The source itself says trapezoidal quadrature has (O(h^2)) error and can become inaccurate when a likelihood is narrower than a few grid cells. ([GitHub][3])

### B. Parameter recovery and (N^{-1/2}) scaling

The Gaussian experiment reports a variance-parameter slope close to (-1/2), but the correlation-parameter slope is approximately (-0.27), which the manuscript explicitly does not explain. ([GitHub][2])

Therefore:

* approve “the procedure recovers the low-dimensional Gaussian transition from noised sequences under the tested protocol”;
* approve the (q)-scaling result as a sanity check;
* do **not** claim that all parameters exhibit the asymptotic (N^{-1/2}) rate;
* put the detailed slopes in the appendix unless the residual floor for (\rho) is resolved.

### C. Mixture learning

The flexible mixture kernel is a legitimate generalized-EM model. The implementation has been audited to increase (Q), and the repository is commendably explicit that its inner latent component label slows shape recovery. ([GitHub][5])

But “the true density is recovered” is currently too strong. Under Laplace truth and a finite Gaussian mixture, the converged estimator targets a finite-mixture approximation or KL projection, not literally the exact Laplace law. The defensible claims are:

* the fitted transition density approaches the target density under a declared density metric;
* the induced score approaches the oracle score;
* the generalized-EM objective converges under a declared criterion;
* the result is stable across independent datasets and initializations.

### D. Ring recovery

The empirical joint-versus-marginal result is supportive of the theorem, but numerical angular errors across noise levels are not needed to establish the theorem. Keep one minimal control in the ring chapter or separate note. Do not burden the chain paper with a large ring recovery sweep.

---

## Claims I reject in their current form

### A. The (8)–(14\times) neural sample-efficiency headline

This claim must not appear in the paper, workshop, abstract, or thesis conclusion in its current form.

The handover README states that the latest rerun still uses asymmetric selection: EM’s stopping point is chosen using validation information while the network is run for a fixed 20,000 steps. The repository itself concludes that this asymmetry favours EM and that the headline table should not be shipped. ([GitHub][1])

Yet the current paper abstract, workshop abstract, workshop conclusion, thesis results, and thesis conclusion still present the order-of-magnitude claim as a central result. ([GitHub][2])

This is the most urgent consistency failure in the repository.

Even after a fair rerun, the comparison would remain deliberately asymmetric in prior information:

* BP–EM receives exact Markov factorization;
* linear-autoregressive form;
* homogeneous transitions;
* and a low-dimensional or mixture kernel family.

The networks receive less structural information. That is acceptable if the question is explicitly “what is the value of these supplied assumptions?”, but it cannot support a general claim about BP–EM being a superior learning algorithm.

**Decision:** remove now. Reintroduce only in a later paper after equal model-selection and stopping rules.

### B. “Capacity saturates at eight components” as a broad conclusion

The thesis itself states that 93% of the shape-dependent capacity cells had not settled by iteration 40 and that roughly 96% of the apparent capacity effect in that configuration could be attributed to convergence rate. ([GitHub][6])

The held-out-evidence saturation may survive, but that supports only:

> Under this dataset size and likelihood metric, held-out evidence showed no resolved benefit from increasing capacity beyond approximately eight components.

It does not establish:

* eight components are intrinsically sufficient;
* generated-shape fidelity saturates there;
* the fitted density has converged;
* or larger models are unnecessary in general.

This material is appendix or compendium material, not a headline contribution.

### C. Pointwise-versus-generative dissociation as a main result

This is interesting but currently too entangled with:

* under-converged mixture fits;
* reverse-SDE integration choices;
* a residual whose distribution is not the clean innovation law;
* excess kurtosis as the principal metric;
* small numbers of seeds;
* unstable network cells;
* and a corrected training-noise protocol. ([GitHub][6])

It is a potential separate project. It should not appear in the main paper or workshop. In the thesis, it should be moved to an exploratory appendix unless rerun with converged estimators and a broader generative evaluation.

### D. Fixed-budget comparisons interpreted as estimator comparisons

The audit found that shape may need about 2,000 iterations in slow configurations, while several experiments used 40 or 120. The audit also explicitly retired an earlier “channel destroys shape recovery” conclusion after comparing independent datasets at adequate iteration budgets. ([GitHub][7])

Any comparison made at a common fixed iteration count answers:

> How do these algorithms behave after this amount of optimization?

It does not necessarily answer:

> How do the converged estimators compare?

The paper should contain no substantive statistical claim that depends on an arbitrary iteration cap without convergence evidence.

### E. Large efficient-information ratios

The thesis presents shape-information losses of thousands to tens of thousands relative to low-noise values. These numbers are difficult to interpret, highly dependent on parametrization, nuisance projection, conditioning, numerical derivatives, and the chosen endpoints. ([GitHub][6])

They are not needed. The cleaner analytical statement is:

* finite Gaussian noising is injective when (\alpha_t>0);
* higher-frequency and higher-order information is attenuated;
* inversion is increasingly ill-conditioned;
* higher cumulants scale with corresponding powers of (\alpha_t).

Use the analytical statement. Move the numerical Fisher-information table to the compendium unless independently rederived and accompanied by conditioning diagnostics.

---

# 3. Paper audit — **reject for submission, approve as an internal scaffold**

## What has improved

Several earlier presentation problems are now genuinely fixed:

* distinct notation for chain length and sample count;
* bold vector/matrix notation;
* clickable cross-references;
* explicit definition of a tree;
* explanation that messages are functions on (\mathbb R);
* functional exactness separated from grid discretization;
* time complexity stated;
* Gaussian closure given a precise LMMSE interpretation;
* generalized EM acknowledged for the mixture;
* limitations and information asymmetry stated rather than hidden. ([GitHub][2])

These directly address much of the earlier feedback.

## Why I still reject it

### 1. The paper contains five papers’ worth of questions

The abstract currently promises:

* empirical-score memorization;
* exact non-Gaussian BP;
* likelihood learning;
* LMMSE closure;
* transition-density convergence;
* neural sample efficiency;
* and rotating-ring identifiability.

The contribution list similarly contains exact inference, closure, kernel estimation, neural comparison, and the ring theorem. ([GitHub][2])

This is not a nine-page paper. It is a project summary compressed into nine pages.

The page count may technically be nine, but the content has been compressed rather than selected. That produces exactly the feeling the feedback identified: too many numbers, too many caveats, difficult metrics, and too many experiments.

### 2. The “ladder” organization does not create real unity

The manuscript says “one question, asked five times,” immediately says there are four levels, and lists rungs (0)–(3). Later the ring figure is captioned “Rung 4.” ([GitHub][2])

This is a visible editing inconsistency, but the deeper issue is conceptual: calling unrelated models “rungs” does not make them cumulative.

Rungs 0–2 concern:

* the same scalar chain;
* the same noising model;
* increasingly unknown transition structure.

The ring changes the model and the estimand. It is not the next rung of the same derivation.

Delete the ladder metaphor.

### 3. The opening motivation promises a memorization paper

The abstract and introduction spend substantial space on the empirical score, training-set reproduction, memorization, and generalization. But the trustworthy core of the work does not resolve memorization. That thread only connects through the presently blocked neural comparison. ([GitHub][2])

Open instead with the structural problem:

> Gaussian noising makes the observed score globally dependent even when the clean law is locally specified. Can that score still be computed and learned through local inference?

That question is answered by the paper.

### 4. The main text contains too many numerical mini-results

Examples include:

* machine-precision solver values;
* fitted scaling exponents;
* settling-rate comparisons;
* angular errors;
* exact zeros over 1,344 cells;
* and neural-error ratios.

Each number invites a reviewer to inspect a protocol. The main paper should contain only numbers required to establish the central mechanism.

### 5. The stopping rule is not publication-grade

The current paper says EM stops when excess kurtosis changes by less than (10^{-3}). ([GitHub][2])

That is insufficient because:

* two very different densities can share kurtosis;
* mixture weights and tails can continue moving while kurtosis is stable;
* likelihood may still improve;
* a component can narrow below grid resolution;
* and parameter labels can move without changing the density.

Kurtosis may be a secondary shape diagnostic. It should not certify convergence.

### 6. Some language remains too absolute

Revise the following classes of language:

* “exact evidence” → “quadrature evidence for the truncated grid representation,” except in explicitly analytic cases;
* “exact EM” → “exact tree E-step with an exact or generalized M-step”;
* “sufficient statistic” → “complete summary for the current Q-function/M-step”;
* “recovers the truth” for finite mixtures → “approximates the target transition under the declared metric”;
* “Monte Carlo enters in one place” → explain deterministic conditional inference versus random dataset generation and risk estimation.

### 7. The paper still narrates the audit history

The repository’s auditing is a strength, but the production paper should present the final accepted facts, not the sequence of incorrect claims and corrections. Historical corrections belong in the claim ledger and compendium.

## Required paper scope

I recommend the title:

## **Global Diffusion Scores from Local Latent Inference**

### *Belief propagation and transition learning in non-Gaussian Markov chains*

The main paper should contain only:

1. score/posterior-mean identity;
2. posterior chain and functional BP;
3. Gaussian closure as LMMSE;
4. (\Xi), Fisher identity, and transition learning;
5. three controlled pieces of evidence;
6. limitations.

Cut completely from the main paper:

* empirical-score memorization discussion beyond one motivating sentence;
* rotating ring;
* neural sample-efficiency table;
* reverse generation;
* model-capacity sweep;
* non-Markov stress tests;
* efficient-information table;
* code experiment identifiers;
* audit/correction history.

## Exact nine-page content plan

The nine pages include figures and tables but exclude references and appendix.

| Main-content pages | Material                                                                         |
| -----------------: | -------------------------------------------------------------------------------- |
|           **0.75** | Introduction and contributions                                                   |
|           **0.50** | Essential related work integrated into the introduction                          |
|           **1.00** | Model, noising, score identity, assumptions                                      |
|           **1.50** | Functional BP, message meaning, one-site/pairwise marginals, complexity          |
|           **1.00** | Gaussian closure equals LMMSE                                                    |
|           **1.50** | Learning: (\Xi), Fisher identity, Gaussian M-step, generalized mixture extension |
|           **2.00** | Three focused experiments/figures                                                |
|           **0.75** | Limitations, discussion, conclusion                                              |
|    **Total: 9.00** |                                                                                  |

Do not try to use all nine pages automatically. An 8.3-page paper with breathing room is better than a nine-page wall of text.

## The three main experiments

### Figure 1 — Structural mechanism

Show:

* latent chain;
* unary observation factors;
* posterior mean at one site;
* score relation.

Fix the reported transition-factor overlap. Directly label the objects. Do not require a legend.

### Figure 2 — What second-order closure loses

Use:

* a Gaussian matched-covariance control, where full BP and closure agree;
* one non-Gaussian innovation family, where they differ;
* normalized score MSE or normalized posterior-mean MSE against diffusion time.

Define the metric before showing it. Prefer

[
\frac{\mathbb E|\widehat{\mathbf s}-\mathbf s_\star|^2}
{\mathbb E|\mathbf s_\star|^2}
]

to a pointwise ratio that becomes unstable when the true score is small.

### Figure 3 — Transition learning

Show at most two or three panels:

* transition-density error against (N);
* held-out oracle-score error against (N);
* optionally one true-versus-fitted innovation density.

Use independent-dataset uncertainty. Do not display all mixture component parameters.

The main paper needs no table unless a very small theorem/complexity summary materially helps.

---

# 4. Workshop audit — **reject and rebuild around one result**

The current workshop version contains:

* empirical-score motivation;
* exact BP;
* Gaussian closure;
* Fisher learning;
* a flexible mixture;
* an (8)–(14\times) neural comparison;
* and the rotating-ring theorem plus experiment. ([GitHub][8])

That is far too much for four pages.

Worse, the text says the neural table is “the point of the paper in one number,” while the handover README says that table must not be sent because the protocol remains asymmetrical. ([GitHub][8])

## Recommended workshop paper

Make the workshop paper the clean analytical front half of the main paper:

## **Exact Diffusion Scores for Non-Gaussian Markov Chains**

Its single message:

> The score can be global in the observed sequence while remaining computable through exact local inference on the latent chain.

Four-page allocation:

|           Pages | Material                                               |
| --------------: | ------------------------------------------------------ |
|        **0.35** | Abstract and motivation                                |
|        **0.65** | Model and score identity                               |
|        **1.05** | Posterior chain and functional BP                      |
|        **0.80** | Gaussian closure equals LMMSE                          |
|        **0.80** | One validation/closure figure                          |
|        **0.35** | Scope, learning extension in one paragraph, conclusion |
| **Total: 4.00** |                                                        |

The workshop version should contain:

* one theorem or proposition;
* one derivation;
* one principal figure;
* possibly a small factor-graph diagram;
* no result table.

Remove:

* neural comparison;
* mixture-capacity details;
* convergence-rate story;
* ring theorem;
* numerical recovery table;
* non-Markov extensions.

Mention learning only as:

> Pairwise posterior marginals also provide the expected transition statistics required for likelihood learning; full derivations and experiments appear in the longer paper.

The workshop appendix can contain:

* complete BP messages;
* LMMSE proof;
* grid discretization;
* complexity;
* learning equations;
* supplemental numerical checks.

References and appendix are unlimited, but the appendix should support the four-page story—not become a hidden second paper.

A separate ring-only workshop note would also be scientifically viable, but the current workshop must choose one of these directions. My recommendation is the chain-score paper because it aligns with the central MSc contribution.

---

# 5. Compendium audit — **approve as development record; prune before calling it final**

The compendium has a sensible purpose: derive what the paper states, document implementation choices, and retain exploratory material. Code pointers and assumption boxes are appropriate here.

However, it currently promises to teach diffusion, graphical models, and EM from first principles to a reader with only undergraduate probability and linear algebra, while also containing exact inference, numerics, estimation, project status, rotating ring, and corrections. Its chapter list includes separate “status” and “corrections” chapters. ([GitHub][9])

That design inevitably becomes too long and overlaps heavily with the thesis.

## Decision

* **Internal technical record:** approved.
* **Polished companion document:** not yet.
* **Page direction:** no net growth; preferably 10–20% shorter.

## Required reorganization

Use six parts:

1. **Model and exact identities**

   * score identity;
   * posterior factorization;
   * assumptions and identifiability.

2. **Functional inference**

   * messages;
   * pairwise beliefs;
   * exactness on trees;
   * complexity.

3. **Numerical representation**

   * grid;
   * quadrature;
   * domain truncation;
   * stability;
   * validation.

4. **Learning**

   * (\Xi);
   * Fisher identity;
   * Gaussian and Laplace M-steps;
   * generalized mixture M-step;
   * convergence.

5. **Experiment ledger**

   * retained experiments;
   * retired experiments;
   * contradictory protocols;
   * final status.

6. **Separate case studies and reproducibility**

   * rotating ring;
   * neural comparisons;
   * generation;
   * non-Markov experiments;
   * HPC/provenance.

Merge `ch11-status` and `ch13-corrections` into one appendix:

## **Claim audit and superseded results**

Do not narrate every historical error chronologically. Use one compact entry per claim:

| Field               | Content                             |
| ------------------- | ----------------------------------- |
| Scientific question | What was being tested?              |
| Protocol            | What data and objective were used?  |
| Original claim      | What was first concluded?           |
| Audit finding       | What changed?                       |
| Current status      | Accepted, conditional, retired      |
| Supporting artifact | Exact output/configuration          |
| Destination         | Paper, appendix, thesis, or archive |

## What to cut

* generic textbook explanations already present in the thesis;
* repeated definitions of diffusion, Markov chains, and EM;
* repeated proofs that already appear fully in the paper appendix;
* narrative descriptions of every experiment;
* figures showing only obsolete fixed-budget behavior unless explicitly labelled historical;
* repeated code listings when a precise function pointer and pseudocode suffice.

The compendium should be **development**, but development does not mean unfiltered. It should explain the internals of the accepted science and preserve failed approaches in a compact audit section.

---

# 6. Thesis audit — **strong analytical core, but reject as a final thesis**

## The central structural problem

The introduction says the thesis addresses four questions restricted to:

* Gaussian-chain scores;
* BP/Kalman equivalence;
* non-Gaussian Gaussian-closure accuracy;
* locality of the Gaussian score.

It then says explicitly:

* “no neural networks, no black boxes”;
* “no learning experiments are performed.”

Later, the thesis contains entire chapters on learning the innovation law, flexible mixtures, neural comparisons, generation, capacity, misspecification, and rotating-ring dynamics. ([GitHub][10])

This is not a minor editing problem. A thesis is judged through the contract created by its research questions. At present, the introduction describes a different thesis from the one submitted.

## What I approve

The strongest thesis core is excellent:

* exact Gaussian-chain score;
* spectral and precision-matrix analysis;
* BP/Kalman/RTS equivalence;
* non-Gaussian functional BP;
* Gaussian closure;
* locality and truncation;
* numerical validation;
* transition learning via (\Xi) and EM.

These form a coherent progression from known to unknown dynamics.

## What should change

### 1. Rewrite the research questions

Use questions that reflect the actual thesis:

1. **How does coordinatewise Gaussian noising change the spatial structure of the score of a Markov sequence?**
2. **How can the resulting global score be computed through exact latent-chain inference?**
3. **What does Gaussian message closure compute, and what non-Gaussian information does it discard?**
4. **Under what assumptions can an unknown local transition law be learned from noised sequences?**
5. **Which numerical, optimization, and statistical checks are required before such a recovery claim is reliable?**

The rotating ring can be presented as a separate case study answering:

> What can joint observations identify that all one-frame marginals cannot?

It need not become a sixth thesis-wide research question.

### 2. Reorganize without adding pages

## Part I — Exact scores under known dynamics

* concise background;
* Gaussian chain;
* BP/Kalman equivalence;
* locality;
* non-Gaussian chain;
* Gaussian closure.

## Part II — Learning dynamics from noised data

* identifiability;
* (\Xi) and Fisher identity;
* EM/generalized EM;
* numerical convergence;
* one reliable recovery study;
* limitations.

## Separate case study or appendix

* rotating ring;
* AMP/TAP branch;
* neural comparisons;
* generation;
* capacity;
* non-Markov stress tests.

### 3. Compress the background drastically

The introduction lists extensive coverage of:

* Liouville mechanics;
* Boltzmann–Gibbs distributions;
* Ising models;
* spin glasses;
* Hopfield and Boltzmann machines;
* stochastic calculus;
* Fokker–Planck equations;
* time reversal;
* multiple generative-model classes;
* graphical models;
* AMP/TAP;
* U-Nets.

Some of this may be useful, but the thesis should not teach every neighbouring field. ([GitHub][10])

Use this test:

> Does this background concept appear in a derivation, experimental design, or conclusion later?

If not, cut it or reduce it to a referenced paragraph.

Do not move everything to appendices and increase the total length. **Move only load-bearing derivations; delete non-load-bearing exposition.**

### 4. Remove the “development history” from the main thesis line

A chapter recording toy models, early supervision models, and corrections makes the thesis feel like a lab notebook. The final thesis should present the final reasoning.

Keep a two-page methodological reflection or move it to the compendium.

### 5. Remove under-converged evidence from the main results chapter

The thesis openly acknowledges that most shape-dependent capacity runs were under-converged and that the relevant rerun was not completed. Yet the capacity and generation narrative still occupies substantial space and remains promoted in the conclusion. ([GitHub][6])

That is not acceptable in the final thesis.

Keep at most:

> Fixed-budget experiments initially suggested a capacity effect, but convergence analysis showed that most shape coordinates had not settled. The result is therefore unresolved and is documented in the appendix as a methodological negative result.

That is scientifically valuable and honest. It does not need multiple main-text figures and tables.

### 6. Rewrite the conclusion from a frozen claim ledger

The current conclusion still promotes:

* (9)–(14\times) and (2)–(4\times) neural advantages;
* generative dissociation;
* capacity saturation;
* and misspecification thresholds. ([GitHub][11])

The conclusion should contain only:

* analytical theorems;
* deterministic validations;
* empirically converged results;
* explicit limitations.

### 7. Remove correction language from the main narrative

“The band-fill law corrects an earlier prefactor error” is not a scientific contribution. The corrected formula is the contribution. The earlier error belongs in version history or the compendium. ([GitHub][10])

## Thesis length policy

Do not add chapters.

I recommend:

* no net page increase;
* target a 15–25% reduction in main-text length;
* preserve rigorous derivations;
* cut broad exposition, historical narrative, and unresolved experiment catalogues.

The thesis can be longer than the paper, but it must still have one intellectual spine.

---

# 7. Code audit

## 7.1 `bp_grid.py` — **approve the design, fix the numerical contract**

This is one of the strongest parts of the repository. The code explicitly distinguishes:

* exact continuous tree inference;
* grid representation;
* tail truncation;
* (O(h^2)) trapezoidal error;
* and normalized linear-domain messages. ([GitHub][3])

### Required fixes

#### Rename “exact evidence”

The code says it returns the “exact model evidence” after using a finite domain and finite quadrature. That is contradictory.

Use:

* `log_grid_evidence`, or
* “quadrature estimate of the marginal evidence under the truncated representation.”

Reserve “exact” for:

* the analytic Gaussian likelihood;
* finite-state BP after the finite grid model has been explicitly defined;
* or the continuous functional recursion before discretization.

#### Add production boundary diagnostics

For every production batch, record:

* posterior boundary mass at each site;
* median, 95th percentile, 99th percentile, and maximum across chains;
* minimum likelihood width divided by grid spacing;
* kernel-column normalization residual;
* whether any component width is below two grid steps.

The thesis has already discovered that one sampled chain’s boundary statistic is not a bound and that maxima can be orders of magnitude larger. ([GitHub][12])

#### Add a low-noise stress test

The source notes that quadrature becomes problematic when the likelihood is narrower than a few grid cells. Test the smallest reported (t), tail observations, and narrow mixture components explicitly.

#### Variance safety

When computing variance as second moment minus squared mean:

* permit tiny negative values caused by floating point;
* clip only within a stated numerical tolerance;
* raise an error for materially negative values.

---

## 7.2 `em.py` — **reject the current convergence rule**

The current stopping condition is essentially:

[
|L_k-L_{k-1}|
\leq
\text{tol},|L_{k-1}|.
]

It uses total log-likelihood, can accept a small decrease because of the absolute value, and records no explicit converged/censored status. ([GitHub][4])

This is not adequate for the mixture experiments on which the scientific claims depend.

### Replace it with a convergence certificate

Each trace should record:

* `converged: bool`;
* `stop_reason`;
* number of outer iterations;
* per-edge observed log-likelihood;
* likelihood increment;
* Q-function increment;
* maximum monotonicity violation;
* parameter change;
* innovation-density Hellinger or (L^1) change;
* Fisher/projected-gradient norm;
* minimum component weight;
* minimum component width divided by (h);
* boundary diagnostics;
* post-stop drift.

A run should be accepted only when, for several consecutive checkpoints:

1. per-edge likelihood increase is nonnegative within numerical tolerance;
2. per-edge likelihood change is below threshold;
3. transition-density change is below threshold;
4. gradient or stationarity residual is below threshold;
5. no component is numerically unresolved;
6. the conditions remain true during a post-stop audit.

A run reaching the maximum iteration count should be labelled **censored**, not returned indistinguishably from a converged run.

The absolute likelihood threshold must be normalized by the number of transitions because total log-likelihood scales with (N(L-1)).

### Rename the module semantics

The file begins with “Exact EM,” but:

* the continuous E-step is exact structurally;
* its implementation uses grid quadrature;
* Gaussian and Laplace updates may be exact maximizers of the discretized Q-function;
* mixture and MDN updates are generalized M-steps.

Use:

> **Tree-structured grid EM with exact or generalized kernel updates.**

---

## 7.3 `kernels.py` — **good abstraction, several publication blockers**

The common kernel interface is good. Separating Gaussian, Laplace, mixture, and MDN transitions is clean engineering. ([GitHub][13])

### Mixture M-step

The method is labelled “Exact block M-step” but runs four fixed inner conditional-maximization sweeps. That is a generalized or ECM update, not an exact global maximizer. ([GitHub][13])

Change it to:

* iterate the inner mixture-label EM until a Q-based inner stopping condition;
* retain a maximum inner-iteration cap;
* record number of inner sweeps and Q increments;
* return failure or warning if inner convergence is not reached.

Four sweeps may remain a fast mode, but not the production default supporting paper claims.

### Grid-resolved mixture components

The code correctly notes that its variance floor allows a component standard deviation much smaller than one grid cell and that this can spuriously raise the quadrature likelihood. ([GitHub][13])

Turn this diagnostic into a gate:

* reject or flag any production fit with (s_{\min}/h < 2);
* rerun with finer grid or stronger variance constraint;
* never count an unresolved component as evidence for higher model capacity.

### Normalize kernel semantics consistently

The MDN kernel explicitly normalizes every transition column under the grid quadrature, whereas the analytic mixture kernel diagnoses—but does not necessarily enforce—the same normalization. ([GitHub][13])

Choose one discretized objective for every kernel family:

1. either all kernels represent samples of continuous normalized densities, with truncation residual explicitly included;
2. or all columns are renormalized under the finite-grid quadrature.

Do not compare held-out likelihoods across kernel families that optimize subtly different finite-grid objectives.

### Laplace lattice bias

The source itself documents that the weighted-median update restricts (\rho) to ratios of grid points and that the apparent exact recovery at (\rho=0.8) is partly a lattice attractor. ([GitHub][13])

Therefore:

* do not use that experiment as clean evidence of parameter recovery;
* test an off-lattice truth;
* use interpolation or a one-dimensional continuous minimization for (\rho);
* or describe it explicitly as recovery in the discretized model.

### Mixture means and identifiability

Leaving mixture means unconstrained preserves Q-monotonicity, but it introduces an innovation intercept that may partially interact with the AR term and initial-law assumption. Keep:

* fitted innovation mean as a mandatory diagnostic;
* an explicit statement that mixture parameters are identified only up to component permutation;
* density-level metrics rather than componentwise parameter errors.

### MDN branch

The MDN is exploratory. Keep it out of the publication-critical path until:

* gradient checks cover clipping and normalization;
* rejected backtracking proposals restore any optimizer state they changed;
* Q monotonicity is tested over full runs;
* and the finite-grid objective is identical to the other kernels’ objective.

---

## 7.4 Observation protocols — **approve and enforce globally**

`protocols.py` is excellent research-engineering work. It clearly distinguishes:

* one noisy view per independent latent chain;
* several paired views of the same latent chain;
* clean fitting;
* and the old composite-likelihood protocol.

It also explains why these protocols differ in objective and effective sample size. ([GitHub][14])

Make this distinction impossible to bypass.

Every result manifest, table, and figure should include:

* `protocol`;
* number of independent latent chains;
* number of noisy observations;
* number of views per latent chain;
* noise-level allocation;
* whether the objective is exact likelihood, joint multiview likelihood, or composite likelihood;
* definition of the sample unit.

The paper’s learning experiment should use **one-view** data. Composite experiments belong in the compendium.

---

## 7.5 Tests and outputs

The executable audit is a major strength. It has already caught or retired claims that code inspection alone did not reveal. ([GitHub][5])

Convert the best audit checks into permanent tests or production gates:

* Gaussian analytic BP;
* independent short-chain enumeration;
* CPU/GPU parity;
* Fisher-gradient finite differences;
* Q nondecrease;
* returned-kernel/trace alignment;
* one-view sample uniqueness;
* grid/domain refinement;
* boundary quantiles;
* mixture-resolution checks;
* converged-versus-censored status;
* configuration and source-commit integrity.

Every production output should have:

* Git commit;
* dirty-tree status;
* configuration hash;
* exact command;
* random seeds;
* protocol;
* environment;
* convergence certificate;
* numerical diagnostics;
* source files for every plotted quantity.

No figure-generation script should accept a directory containing mixed commits or mixed protocols.

---

# 8. Figures and tables

One limitation of this review: GitHub’s binary PDFs did not render through the browsing interface, so I could audit figure inclusion, captions, scientific content, source references, and generation logic, but not inspect every rendered pixel. A final pixel-level render review is still required.

## Main paper

Maximum:

* **three principal figures**;
* preferably **zero or one small table**.

The previous feedback correctly flags the transition-arrow overlap, packed legends, complex captions, and overfilled recovery figures.

### Figure rules

Each figure must answer one written question:

1. What mechanism makes the score computable?
2. What does second-order closure lose?
3. Can the transition and induced score be learned reliably?

No figure should merely show that an experiment was run.

Use:

* direct labels;
* common axes for genuine comparisons;
* no more than three panels;
* consistent notation;
* uncertainty across independent datasets;
* a caption containing the question, metric, and main reading.

Do not include:

* mixture component parameters in legends;
* all seeds as separate legend entries;
* tables embedded in captions;
* code paths in captions;
* “Rung 4” terminology;
* machine-precision digits as headline scientific evidence.

## Workshop

Use one main result figure and, if space permits, a small factor graph. No table.

## Thesis and compendium

More figures are allowed, but not every generated plot deserves inclusion.

Delete or archive:

* under-converged capacity curves;
* generation plots based on the same under-converged fits;
* duplicated versions of the same recovery curve;
* diagnostic figures whose conclusion is stated more clearly in one sentence;
* plots of component labels that are not permutation-invariant.

## Figure provenance

Every production figure should be generated from a manifest that includes:

* accepted run IDs;
* commit;
* protocol;
* convergence status;
* grid;
* seeds;
* metric definition;
* script version.

The LaTeX should never contain manually copied numerical values that can drift from the plotted data.

---

# 9. Minimal experiments required before approval

Do not launch another broad campaign.

## Experiment 1 — Solver validation

**Question:** Is the finite-grid BP implementation accurate in the actual regimes used?

Required checks:

* Gaussian analytic solution;
* independent short-chain integration;
* (G\in{201,401,801});
* at least two or three domains;
* smallest used diffusion time;
* representative tail observations;
* boundary quantiles across a batch.

No stochastic seed sweep is needed beyond a fixed representative bank.

**Decision:** no learning claim is accepted until numerical error is demonstrably below statistical uncertainty.

## Experiment 2 — Convergence-controlled transition recovery

**Question:** Can the transition density be learned from one noisy view per independent chain?

Use:

* one-view protocol;
* one fixed non-Gaussian truth;
* (N\in{128,512,2048});
* 12–16 independent datasets;
* multiple initializations during a smaller pilot;
* objective-, gradient-, density-, and resolution-based stopping;
* no fixed-budget interpretation.

Primary metrics:

[
\text{transition-density Hellinger error}
]

and

[
\frac{1}{L}
\mathbb E\left[
|\mathbf s_{\widehat\theta,t}(\mathbf X_t)
-\mathbf s_{\theta_\star,t}(\mathbf X_t)|^2
\right].
]

Secondary metrics:

* held-out marginal likelihood;
* (\rho) error;
* innovation mean/variance;
* kurtosis only as a diagnostic.

Controls:

* oracle kernel;
* Gaussian constrained model;
* Gaussian truth negative control;
* clean-data fit;
* grid-refined subset.

**Decision rule:**

* If density and score errors decline with (N), convergence certificates pass, and grid effects are smaller than seed uncertainty: include the learning result.
* If only (\rho) is stable while shape remains initialization- or budget-dependent: restrict the main paper to low-dimensional parameter learning.
* If the mixture cannot be certified: retain the learning equations as a method, move flexible-density recovery to future work, and publish the exact-inference paper without it.

That fallback still produces a strong paper.

## Do not rerun now

* neural architecture sweeps;
* neural sample-efficiency tables;
* reverse generation;
* capacity sweeps;
* high-noise-only shape experiments;
* non-Markov robustness;
* images/video/wavelets;
* efficient-information grids;
* large ring campaigns.

Those do not close the central uncertainty of this paper.

---

# 10. Required order of work

## Priority 0 — Remove claims currently known to be unsafe

Before any new experiment:

1. Remove (8)–(14\times) and (2)–(4\times) claims from:

   * paper abstract and body;
   * workshop abstract and body;
   * thesis results summary;
   * thesis conclusion;
   * executive summaries.

2. Remove the rotating-ring material from paper and workshop.

3. Demote capacity and generation conclusions in the thesis.

4. Replace “exact grid evidence,” “exact EM,” and unqualified “sufficient statistic.”

5. Fix “five times/four levels/Rung 4.”

These edits require no new computation.

## Priority 1 — Freeze one scientific story

Rewrite the paper and workshop outlines before polishing prose.

* Paper: exact inference, LMMSE closure, transition learning.
* Workshop: exact inference and closure only.
* Thesis: known dynamics followed by learned dynamics.
* Compendium: technical development and audit record.

Do not add new results while this architecture is unsettled.

## Priority 2 — Fix the publication-critical code path

Implement:

* robust convergence certificate;
* converged/censored output status;
* adaptive inner mixture convergence;
* grid-resolution gates;
* unified kernel normalization;
* mandatory protocol/provenance manifests.

## Priority 3 — Run the minimal learning experiment

Only the convergence-controlled one-view recovery experiment and its numerical audit.

## Priority 4 — Regenerate all documents from accepted outputs

The handover folder is generated by `tools/make_handover.sh` and direct edits can be overwritten. Changes must be copied back into the canonical paper/thesis sources and verified with the generator’s `--check` mode. ([GitHub][1])

The build should fail when:

* paper main content exceeds nine pages;
* workshop main content exceeds four pages;
* `[PENDING]` remains;
* references are undefined;
* a figure lacks a manifest;
* an included run is censored;
* generated handover files drift from canonical sources.

Use an explicit LaTeX marker such as `\label{LastMainPage}` immediately before the bibliography so main-content page limits are checked independently of unlimited appendices and references.

---

# 11. Approval criteria

## Approve the paper when

* one central question is stated;
* ring and neural comparisons are absent;
* main content is at most nine pages;
* there are at most three main figures;
* every theorem has explicit assumptions;
* functional exactness and numerical approximation are distinguished;
* the learning result uses one-view data;
* every included run is converged and grid-resolved;
* every main numerical claim has independent-dataset uncertainty;
* no `[PENDING]`, correction history, or code-path clutter remains.

## Approve the workshop when

* main content is at most four pages;
* it presents one mechanism and one result;
* it has no neural table or ring section;
* it can be understood without the full paper;
* the appendix supports rather than expands the central story.

## Approve the compendium when

* it does not grow;
* status/corrections are consolidated;
* generic textbook background is compressed;
* all experiments have explicit status;
* superseded results cannot be mistaken for current conclusions;
* it becomes the sole home for code pointers and protocol history.

## Approve the thesis when

* research questions match the actual chapters;
* “no learning experiments” is removed;
* unresolved capacity/generation evidence is demoted;
* the introduction and conclusion agree;
* background and development history are shortened;
* accepted claims are cleanly separated from exploratory work;
* total length does not increase.

## Approve the code as a publication evidence pipeline when

* every run says converged, censored, or failed;
* stopping is not based only on kurtosis or total-likelihood change;
* mixture inner optimization is monitored;
* grid-resolution failures are gates;
* likelihood semantics are consistent across kernels;
* protocol and independent-sample counts are mandatory;
* every figure is traceable to one commit and configuration.

---

# Final judgment

There is a strong, defensible paper in the current work, and the analytical core is now much clearer than before.

The decisive change is to stop presenting the repository’s entire intellectual history as one paper.

The final main result should be:

> **A locally specified non-Gaussian Markov law can induce a globally dependent diffusion score, yet that score is computable by exact latent-chain inference; the same inference produces the pairwise posterior statistics needed to learn the local transition.**

The LMMSE proposition explains precisely what second-order closure loses. One solver-validation figure, one closure figure, and one convergence-controlled learning figure are enough.

The rotating ring is a good separate result. The neural comparison is not currently admissible. Capacity and generation are exploratory. The thesis must stop claiming it performs no learning while devoting major chapters to learning. The compendium should become shorter and more structured, not more encyclopedic. The inference code is strong, but the EM stopping and result-production contracts must be repaired before they support publication claims.

That selection—not additional breadth—is what will make the work solid.

[1]: https://github.com/GioviManto/diffusion/tree/main/overleaf-handover "https://github.com/GioviManto/diffusion/tree/main/overleaf-handover"
[2]: https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/paper.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/paper.tex"
[3]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/bp_grid.py "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/bp_grid.py"
[4]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/em.py "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/em.py"
[5]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/audit/AUDIT_NOTE.md "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/audit/AUDIT_NOTE.md"
[6]: https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch11-nongaussian-em-results.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch11-nongaussian-em-results.tex"
[7]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/CLAIMS_TO_UPDATE.md "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/CLAIMS_TO_UPDATE.md"
[8]: https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/workshop.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/workshop.tex"
[9]: https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/compendium.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/overleaf-handover/compendium.tex"
[10]: https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch01-introduction.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch01-introduction.tex"
[11]: https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch12-conclusions.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch12-conclusions.tex"
[12]: https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch10-nongaussian-em-method.tex "https://raw.githubusercontent.com/GioviManto/diffusion/main/thesis/chapters/ch10-nongaussian-em-method.tex"
[13]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/kernels.py "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/kernels.py"
[14]: https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/protocols.py "https://raw.githubusercontent.com/GioviManto/diffusion/main/research/nongaussian-bp/src/protocols.py"
