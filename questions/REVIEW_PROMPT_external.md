# External review prompt

Paste everything below the line into ChatGPT Pro (or any strong model), with the
repository attached or the relevant files pasted. Written to be self-contained:
it assumes the reviewer has never seen this project.

---

You are reviewing an MSc thesis project and its associated paper. Adopt **two
roles simultaneously** and keep them clearly separated in your output.

**Role A — Senior NeurIPS area chair / reviewer.** You have reviewed for NeurIPS,
ICML and ICLR for a decade. You work on diffusion models and on the statistical
mechanics of learning. You are the reviewer who reads the appendix, checks whether
the proposition actually says what the abstract claims it says, and notices when
an experiment answers a slightly different question from the one posed. You are
not hostile, but you are not generous either: you assume every unqualified claim
is overclaimed until the evidence is shown, and you are specifically alert to
comparisons where the authors' own method has been tuned more carefully than the
baseline.

**Role B — Senior professor at Bocconi supervising this thesis.** You care about
different things: whether the document teaches, whether the mathematical
development is self-contained and correct, whether the student demonstrates
mastery rather than assembly, whether the scope is defensible at viva, and
whether an examiner outside the immediate subfield could follow the argument. You
also care about what is examinable: a thesis that hides its weaknesses fails
worse at viva than one that states them.

Both roles must be **specific and falsifiable**. "The evaluation could be
stronger" is worthless. "Table 1 selects the network's training length on
validation but the architecture is fixed at (128,128) throughout, so the network
arm is tuned on one axis and the estimator on two" is useful. Quote file, line or
section for every point.

---

## What the work is

The motivating puzzle: the global minimiser of the denoising objective that
trains a diffusion model is **not** the score of the data distribution; it is the
score of the *empirical* distribution smoothed by the forward noise. A model
holding that score exactly would reproduce its training set and generate nothing
new. Diffusion models nevertheless generalise. Deciding what a trained network
has actually learned requires comparing it against the score it *should* have
learned — and for essentially every distribution of interest that target is
unavailable, so a network's error cannot be separated from the difficulty of its
target. The analytical literature therefore leans on Gaussians and Gaussian
mixtures, where the score is linear and every question about how *structure* is
exploited has a degenerate answer.

This work studies a family where the exact score is computable and the data is
genuinely structured: a **first-order Markov chain on R^d with continuous,
non-Gaussian transitions**, variance-matched so that innovation families (Gaussian,
Laplace, Student, uniform, bimodal) differ *only* beyond second moments.

The enabling fact: coordinatewise Ornstein–Uhlenbeck noising multiplies the prior
by **unary** factors. It reweights node potentials and creates no edges among the
latent variables, so the posterior factor graph is still the prior's chain — a
tree, on which sum-product is exact. The exact score is therefore available for a
correlated, non-Gaussian model. The obstruction is representational: messages are
functions on R and must be discretised on a grid.

The results, as claimed:

- **R1.** Grid BP gives the exact score for this family; the Gaussian case
  validates the discretisation to ~1e-14 against the closed form.
- **R2.** Single-Gaussian message closure computes *exactly* the LMMSE estimator
  of the covariance-matched Gaussian model, so the gap between it and full
  sum-product is precisely the price of discarding everything beyond second
  moments. Claimed as a proved proposition, not an observation.
- **R3.** The transition kernel is learnable from noised sequences alone. Fisher's
  identity supplies the exact gradient of the marginal log-likelihood from one
  sum-product pass, with no differentiation through the recursion, so both
  gradient ascent and EM are available; for Gaussian innovations the M-step is
  closed form and the two routes reach the same optimum.
- **R4 (headline).** Against networks trained by denoising score matching on
  identical data, with the optimisation budget of *both* arms selected on a
  held-out bundle, the structured estimator attains roughly 7x to 18x lower
  pointwise denoising error across eight training-set sizes (32 to 8192), and the
  network at the largest budget does not reach what the estimator attains at the
  smallest.
- **R5.** A "rotating ring" model whose structure is invisible to every marginal
  (proved) and still recoverable from the joint.

The comparison in R4 is **deliberately asymmetric in prior information**: the
estimator is handed the Markov factorisation, the autoregressive form and a
low-dimensional innovation family; the networks are handed none of them. The
authors claim this measures the worth of that structure on this family, not a
general ranking of methods.

## Repository map

```
overleaf/
  shared/    notation.tex, references.bib, sections/, figures/   (shared by all roots)
  paper/     main.tex, appendix.tex          9-page NeurIPS-format body
  workshop/  main.tex, appendix.tex          4-page version
  compendium/chapters/                       ~63pp, unconstrained development record,
                                             including a per-claim audit chapter
  thesis/    chapters/, preamble.tex         ~169pp, target ~100pp
research/nongaussian-bp/
  src/         bp_grid.py (grid BP), em.py (EM E-step), kernels.py (M-step),
               priors.py, metrics.py, backend.py (CPU/GPU dispatch), ring.py,
               denoiser.py (DSM baselines), wavelet_*.py (image/video extension)
  experiments/ exp_01..exp_30, frozen_config.py (single source of run parameters)
  tests/       incl. test_backend_parity.py, test_backend_coverage.py
  tools/       generators that emit LaTeX tables/macros from frozen outputs;
               check_paper.sh (honesty gate)
  hpc/         Slurm scripts
  outputs/frozen/  committed CSVs with params.json provenance
```

Three conventions you should judge:

1. **Every number the prose quotes is generated, not typed.** `tools/` scripts read
   `outputs/frozen/` and emit `.tex` macros; the prose cites `\ratiolo` etc. and
   cannot hold a stale value. Judge whether this is genuinely load-bearing or
   ceremony.
2. **`check_paper.sh` is an "honesty gate"**: no code references in the paper, no
   withdrawal language, no unfilled placeholders, page limit, and a check that
   every efficiency figure comes from a generated macro. Judge whether it checks
   the things that actually matter.
3. **The compendium carries a per-claim ledger** marking each claim settled /
   conditional / withdrawn, including three claims the authors withdrew after
   finding their own errors.

## What has already gone wrong (check these patterns recur)

State these plainly rather than being tactful; the authors have been finding
these themselves and want more found.

- A capacity claim ("mixture capacity saturates near eight components") was
  withdrawn: every run behind it used `em_iters=40`, and EM on this kernel
  converges on rho by ~25 iterations but on the innovation *shape* only by ~229.
  Enlarging the mixture at a fixed budget bought convergence **rate**, not
  representational capacity.
- A "flat curve" was read as a discretisation floor when it was three-replicate
  sampling noise.
- A claim was made from **one seed** and had to be corrected at sixteen.
- A cell-level split (48 vs 144 cells) was really a seed-level split (4 vs 12
  seeds), because the EM fit is shared across noise levels within a seed.
- A GPU gate passed while every parity test **skipped** (pytest exits 0 on an
  all-skipped run), so a job reported success on a device it never used.
- `BP_DEVICE=gpu` did not reach `src/em.py` at all for a month: only some modules
  imported the backend, so EM-heavy "GPU" jobs were CPU-bound in the part that
  dominates their cost.

The common thread is **reading a mechanism into a pattern that was noise, or into
an artefact of a budget**. Look hard for further instances.

## The specific things to attack

**Analytical (Role A leads, Role B checks the exposition):**

1. Is the R2 proposition (moment-projected sum-product = LMMSE of the
   covariance-matched Gaussian) actually *proved*, and is it stated with the right
   hypotheses? Does it need the grid at all, or is it a statement about exact
   messages that the discretisation then approximates? Is "exactly" doing honest
   work?
2. The chain-is-a-tree argument makes sum-product exact **functionally**. The
   implementation is a grid quadrature. Are "exact" and "grid-exact" kept
   distinct everywhere, or does the paper slide between them?
3. The marginal-blindness theorem for the ring: is the gauge argument complete? Is
   the numerical "machine-precision zero" a real control or a tautology?
4. Fisher's identity gives the gradient from one sum-product pass. Is the
   interchange of differentiation and integration justified on the grid, and is
   the claimed 1e-9 finite-difference agreement a meaningful check or a check of
   the same code against itself?

**Empirical (Role A leads):**

5. **Attack R4 hardest.** The baselines are MLPs and a weight-shared 1-D
   convolution, two parameterisations (eps and x0), architecture fixed at
   (128,128) for the headline table with a separate capacity sweep. Is that a
   competent baseline in 2026? What would a reviewer demand — a transformer, a
   U-Net, wider/deeper nets, better optimiser schedules, more training? The
   authors report a Gaussian-chain competence check where the optimal denoiser is
   linear and closed form. Is that sufficient to establish the baseline is not
   undertrained, which is the one direction of error that would invalidate the
   headline?
6. The budget question: at large sample sizes both arms often select their largest
   allowed budget. The authors tested this by tripling **both** caps at nseq=2048
   on the same seeds and bundles, finding the ratio moves by -0.16 +/- 0.18 and
   the network's error by 0.3%. Is that calibration sound, and does it license the
   nseq=8192 row being run at a different budget from the rest of the table?
7. A "resolution gate" drops cells where the narrowest fitted mixture component is
   under two grid cells wide (16 of 1536), with a robustness check that excluding
   them moves no ratio by more than 0.08. Is drop-and-disclose defensible here, or
   is it a garden of forking paths?
8. The density-level metric is a Hellinger distance between transition kernels
   with a ~4e-8 resolution floor. Is that the right metric, and is the floor
   correctly characterised?
9. Aggregation is per-seed then across seeds, because the twelve noise levels
   within a seed share one training set and one set of fitted models. Check this
   is done consistently everywhere, including in any figure.

**Implementation (both roles):**

10. Read `src/em.py`, `src/bp_grid.py`, `src/kernels.py`. Is the forward-backward
    recursion numerically sound — log-domain where it needs to be, normalisers
    guarded, no silent underflow at large `t` or small `delta`?
11. `src/em.py` duplicates the recursion that `src/bp_grid.py` implements. Is that
    duplication justified (the E-step also accumulates pairwise statistics) or is
    it a correctness hazard — two implementations that can drift apart?
12. The GPU port keeps `einsum` on numpy and multiply-and-sum on cupy because they
    differ at ~1e-15 and published validation figures were measured on the einsum
    path. Is that the right call or is it superstition?
13. Are the tests testing behaviour or implementation? Are there tests that cannot
    fail? Check `tests/` for vacuous assertions.
14. Is `frozen_config.py` genuinely a single source of truth, or do experiments
    still carry hard-coded values that contradict it? (At least one such defect —
    a literal `n_iters=120` in four places while the config said 400 and nothing
    read it — has been found before.)

**Thesis-specific (Role B leads):**

15. The thesis is ~169pp against a ~100pp target. What should go? Judge the arc:
    Lagrangian/stationary action -> Hamiltonian -> statistical mechanics -> the
    history into AI -> stochastic calculus -> diffusion literature -> toy models
    -> Gaussian AR(1), Laplace, mixtures -> precision-matrix structure and its
    Fourier reading -> BP, why not AMP, chain => tree => exactness -> discretising
    messages -> the learning phase. Is that a defensible arc or two theses stapled
    together?
16. Is the research question stated once and answered, or does it drift between
    chapters?
17. What will an examiner attack at viva, and is the document ready for it? Give
    the three questions you would ask.
18. Is the related work adequate? It should cover: diffusion and score matching
    (Sohl-Dickstein, Ho, Song; Hyvarinen, Vincent; Tweedie/Robbins-Miyasawa-Efron);
    the statistical mechanics of memorisation vs generalisation (Biroli-Mezard,
    Biroli-Bonnaire-De Bortoli-Mezard, Achilli et al.); tractable structured data
    and architectures that implement BP (Sclocchi et al., Garnier-Brun et al. on
    transformers implementing BP on trees, Mei et al. on U-Net as BP); and
    graphical models (Pearl, Kschischang, Mezard-Montanari; Baum-Welch; Kalman/RTS;
    Dempster EM). What is missing, and what is cited but not engaged with?

## What NOT to spend time on

- Typography, LaTeX style, figure aesthetics, bibliography formatting.
- Suggesting the authors "try more datasets" — the entire point is a family with a
  computable exact score. If you think a second such family exists, name it.
- Restating the summary above back at me.
- Praise. If something is genuinely right, one line is enough; spend your budget
  on what is wrong.

## Output format

1. **Verdict as NeurIPS reviewer**: score, confidence, and the two sentences you
   would put in the meta-review. Would you accept the 9-page paper as it stands?
2. **Verdict as Bocconi supervisor**: is this thesis ready to submit on 16
   September and defend in October? What is the single biggest risk at viva?
3. **Blocking issues** — things that are *wrong*, not merely improvable. For each:
   file/section, what is wrong, why it matters, and what would fix it. Ordered by
   severity.
4. **Claims you believe are overclaimed**, with the wording you would accept
   instead.
5. **Experiments you would demand** before acceptance, ranked by cost/benefit,
   with an explicit note on which are feasible in three weeks on a university
   cluster with ~8 H200 nodes shared with other users.
6. **Code defects**, separately, with severity.
7. **What to cut from the thesis** to reach ~100pp, as a concrete list.

Be direct. If the headline result does not survive your scrutiny, say so plainly
and explain exactly which measurement would change your mind.
