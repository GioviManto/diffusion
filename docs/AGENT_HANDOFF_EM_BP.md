# Agent handoff — Layers 5–6 (EM + BP denoiser learning; hierarchical priors)

You are picking up work on `GioviManto/diffusion`, branch
**`claude/em-bp-denoiser-learning-e07ike`**. Read this before touching anything;
several of the traps below cost real time to find and are easy to reintroduce.

For *what was done and what it showed*, read `docs/EM_BP_LEARNING_COMPENDIUM.md`
first — this file is only about how to continue safely. Layer 6 (trees,
speciation, memorization) has its own study note in `docs/PAPER_CONNECTIONS.md`;
read its §0 before citing anything: the transformer paper has been read in full,
the Nature Communications one has **not**, and §0 records two substantive things
that reading the first one corrected.

---

## 0. Ground rules

- **Never push to `main`.** All work goes to the branch above. `main` is at `cca0018` and must stay there unless the user says otherwise.
- The thesis lives in `thesis/`. Layer-5 work is confined to `research/nongaussian-bp/` plus these two `docs/` files. Don't edit `thesis/` unless asked.
- This is a **research** repo. A number that is wrong is worse than a number that is missing. Every experiment writes `params_*.json` + CSV + PNG; keep that contract.

## 1. Environment

```bash
cd /path/to/diffusion
python3 -m venv .venv && .venv/bin/pip install -r research/nongaussian-bp/requirements.txt
cd research/nongaussian-bp
../../.venv/bin/python -m pytest tests/ -q          # expect 101 passed, ~4 min
```

`.venv/` is gitignored — recreate it, don't look for it. Pure numpy/scipy; **no
torch, no jax, and none is needed** (see trap 1). LaTeX for the report needs
`texlive-latex-recommended texlive-fonts-recommended texlive-science lmodern`.

Experiments run from the package root and import via `common.py`, which inserts
the package on `sys.path`. `conftest.py` does the same for pytest.

## 2. Orientation, in reading order

1. `docs/EM_BP_LEARNING_COMPENDIUM.md` — results and findings (§4.13 is Layer 6).
2. `research/nongaussian-bp/report/em_bp_learning.pdf` — the theory (14 pp, 5 propositions). §2 is the algorithm, §4 the kernel families, §5 the efficiency argument, §6 the limitations.
3. `research/nongaussian-bp/src/em.py` — the module docstring explains the `Ξ` compression in full; that is the one idea the whole layer rests on.
4. `docs/PAPER_CONNECTIONS.md` — Layer 6: what the two advisor papers imply here, what was derived from them, and (§0) which of them has actually been read.
5. `research/nongaussian-bp/src/hierarchy.py` / `src/spectral.py` — the tree prior, exact tree BP, the tree E-step, and the two time scales.
6. `research/nongaussian-bp/README.md` — package conventions, all layers.

## 3. Conventions you must not break

- **Forward process:** `dX = −X dt + √2 dW`, so `α_t = e^{−t}`, `Δ_t = 1 − e^{−2t}`, and `x_t = α_t a + √Δ_t z`.
- **Score identity:** `s(x,t) = −(x − α_t·E[a|x])/Δ_t`. Every estimator goes through this, which is why `identity_residual` appears in every results CSV and must sit at machine precision (~1e-14). **If it ever exceeds ~1e-10, stop and find the bug** — it means an estimator bypassed the identity.
- **Seeding:** all randomness flows through `src.utils.rng_for(*keys)`. Same keys ⇒ same stream, across processes. Compared methods must share keys (common random numbers), or comparisons pick up Monte Carlo noise.
- **`Ξ` conventions:** `K[out, in]`, i.e. `log_K[k, j] = log K(u_k | u_j)`. `Ξ[k,j]` is posterior mass on the transition `u_j → u_k` and **sums to the edge count**. Both the BP `Ξ` (belief mass, weights absorbed) and the clean-data `Ξ` (a counting measure, no weights) satisfy this, which is what makes them interchangeable M-step inputs.

## 4. Traps — each of these was a real bug here

1. **Do not backpropagate through BP.** The instinct (and some LLM advice) is to port the forward-backward recursion to torch/jax and differentiate the evidence. Fisher's identity (Prop. 3) makes that unnecessary: `∇L(θ) = ⟨Ξ(θ), ∇log K_θ⟩` from one BP pass. Verified against finite differences to ~1e-9. Porting to autodiff would be strictly more code, strictly slower, and no more accurate.
2. **`rng_for` used to be non-reproducible across processes** (builtin `hash` is salted per process for strings). Fixed via blake2b. If you add seeding anywhere, do not reintroduce `hash()` on a string.
3. **Every M-step must be justified as an ascent step.** Adding a reasonable-looking projection breaks EM's one guarantee. Concretely: recentering the mixture component means to enforce `Σπ_cμ_c = 0` looks harmless and violated monotonicity by ~1e-2 nats. The MDN's fixed Adam step violated it by up to 1452 nats. **Always check `trace.monotone_violation`** — it is the sharpest test of the entire pipeline.
4. **The score-network baseline must train both parameterizations.** ε-prediction and x₀-prediction share a minimizer but not a finite-sample loss (they differ by `α_t²/Δ_t`). ε wins at small `t`, x₀ at large `t`. Reporting only one is cherry-picking, in whichever direction.
5. **Do not reseed the test set when adding replicates.** `exp_07` mixes `SEED_TAG` into training-data and model-init streams only; `make_test_set` deliberately uses a plain `rng_for("exp07-test")`. Replicates must differ in what they learn from and agree on what they are judged against. There is a check for this in §6.
6. **The Laplace kernel has two discretization artifacts** (compendium §5.1, §5.2): its exact M-step is quantized onto the grid's ratio lattice, and its ρ-gradient loses accuracy to a `sign` discontinuity. **Never quote a Laplace-kernel ρ recovery as evidence of accuracy.** Measured (compendium §4.10): 4/5 is an *attractor* of the weighted-median M-step. For ρ\* = 0.7913, chosen deliberately off the simple lattice, the estimate is pinned at exactly 0.8000 across an 8× grid refinement (M = 201 → 1601) with a constant 0.0087 bias — refining the grid does not help, because this is a bias and not a resolution limit. At M=201 three distinct true values collapse onto 0.8000. And it contaminates `b`: where ρ is snapped away from the truth the scale error is 3–4× larger. Use the smooth kernels (Gaussian, mixture) for any rate or accuracy claim.
7. **Tree EM converges much more slowly than chain EM, and it looks like a bug.** On a chain every site is observed; on a tree the internal nodes never are, so the missing-information fraction is far larger and EM's linear rate is correspondingly slower. Measured at depth 3 with 512 trees: `ρ̂ = 0.7360 / 0.7483 / 0.7487` at 50 / 100 / 150 iterations against a true 0.75, **with zero monotone violation throughout**. An estimate read at 40 iterations is off by 0.06 and reads exactly like a broken M-step. **Check `dL` at the last iteration before concluding anything about accuracy.** `tests/test_hierarchy.py::test_em_on_a_tree_ascends_and_converges_to_the_truth` asserts this as a rate property, at two budgets, for that reason.
8. **A per-level error normalized within the level measures its own denominator.** Fine levels of a hierarchy have small eigenvalues, hence small reference magnitude, so a *relative* per-level error makes them look worse for every method — including a method whose absolute error is uniform. The first version of the exp_13 `levels` measurement showed a clean coarse-to-fine gradient that was entirely this effect. All three forms (relative, absolute, share of total squared error) are now written to the CSV; **quote the absolute or the share when comparing across levels**, the relative only within a level.
9. **Hierarchical filtering is not "independent blocks", and getting it wrong is easy.** arXiv:2408.15138 §2.2 draws the depth-`k` nodes conditionally independently *given the root*, so blocks stay correlated through it at `ρ^{2L}`. The intuitive reading — chop the tree into independent subtrees — puts a zero there instead, and I shipped that version before reading the paper. `GaussianTree(filter_level=k)` is now the paper's construction; the old one survives as exp_13 `block_independent` under a name that says what it is. Related: in a **Gaussian** tree `k=0` and `k=1` are *exactly* the same model, because a linear-Gaussian edge makes siblings conditionally independent given the parent while their transition tensor need not. Do not read a `k=0`/`k=1` null result as a failure.
10. **Grid adequacy.** `M=401, A=8` is the validated default. Grid error on the posterior mean is ~4e-6 there, far below any learning error. Below `t ≈ 0.05` the likelihood gets narrow relative to `dx` — check `noising.likelihood_resolution_ok` before trusting small-`t` results.

## 5. Running things

```bash
# smoke everything in minutes
python experiments/exp_06_em_parameter_recovery.py --quick
python experiments/exp_07_em_vs_score_network.py --quick

# one part, scaled up
python experiments/exp_06_em_parameter_recovery.py --list-parts
python experiments/exp_06_em_parameter_recovery.py --only rate \
    --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,4096)'
```

`--set` accepts any key in `params_*.json`; **unknown keys are rejected** so a
typo fails immediately rather than silently running defaults.

**On a cluster:** `hpc/` has SLURM array templates, one task per part (parts are
independent and write disjoint CSVs — no merge step). Lines marked `# SITE` need
the account/partition/module names. `hpc/README.md` ranks what is worth the
compute. Set `OMP_NUM_THREADS` from the allocation or array tasks will
oversubscribe each other.

## 6. Useful invariants to assert when you change things

```python
stats.xi.sum() == stats.n_edges                    # Ξ conserves mass exactly
trace.monotone_violation < 1e-8                    # EM ascends
row["identity_residual"] < 1e-10                   # score/mean identity holds
```

For tree code (`src/hierarchy.py`) the equivalents, all asserted in
`tests/test_hierarchy.py`:

```python
tree_bp_gaussian(...)   vs tree_posterior_mean_dense(...)      # < 1e-10
tree_bp_grid(...)       vs tree_bp_gaussian(...)               # < 2e-6
tree_e_step(...).log_evidence vs log N(x; 0, α²C + ΔI)         # rel < 1e-6
level_eigenvalues()     vs eigh(leaf_covariance())             # < 1e-10
spectral.commitment(t_S, Λ) == 1/sqrt(2)                       # exactly
```

The evidence check is the load-bearing one: it is what makes
`monotone_violation` meaningful on a tree, since the messages are renormalized
at every node and `log p(x)` is reassembled from the discarded log-scales.

Test-set invariance across replicate seeds (trap 5) — reproduce with:

```python
import exp_07_em_vs_score_network as E
E.SEED_TAG = 0; A0, b0 = E.make_test_set(prior, grid, w, (0.2,), 64)
E.SEED_TAG = 7; A7, b7 = E.make_test_set(prior, grid, w, (0.2,), 64)
assert np.array_equal(A0, A7) and np.array_equal(b0[0.2][1], b7[0.2][1])
```

## 7. What to do next

In rough priority order. Items 1–2 strengthen the headline; the rest are new
work. Two items previously listed here are done: the exp_06 Part 5 seeding
confound is fixed and re-run, and the `N^{−1/2}` rate is established at 12
replicates (combined slope −0.500 ± 0.048).

Note the pattern behind both, since it will recur: a 4-replicate RMSE cannot
resolve a log-log slope to better than about ±0.2, so any rate claim needs
replicates in the tens, not the units.

1. **Error bars on the headline curve.** `hpc/slurm_replicates.sbatch` + `tools/merge_replicates.py`. The current curve has one replicate per `N`, which is why the EM line wobbles. This is the single highest-value cluster run.
2. **Extend the sample-efficiency sweep upward.** The network was still improving at `N = 2048`, so "EM-BP needs ≥64× less data" is a lower bound read off a truncated curve. Follow it until the network catches up or visibly plateaus — either outcome is publishable, and the honest version needs it.
3. **A structured-architecture baseline** (temporal CNN / small U-Net). This is the most important open question for the paper's credibility: the current baseline is a vanilla MLP, as the email specified, and a reviewer will ask. Expect it to close part of the gap; it will not acquire the exactness or the uniformity in `t`.
4. **Inference-time cost.** BP is 211×–320× slower per evaluation than a forward pass, and reverse diffusion calls the denoiser at every step. Either distil the fitted BP denoiser into a network, or use Layer-2 Gaussian-projected BP at inference and measure what the closure costs. This is the weakest point of the whole story.
5. **Hybrid non-Markov correction.** Layer 4 showed the score correction is exactly rank one for AR(1)+global-latent priors. Combining that with a *learned* chain kernel is the natural Layer 4 × Layer 5 product and is untouched.
6. **The write-up (advice 2 in the original email).** `report/em_bp_learning.tex` is deliberately paper-shaped — context, propositions, detail in remarks — and is the natural seed for the shared Overleaf document. It has no results section yet; §4 of the compendium is the raw material.

### Layer 6 specifically

7. **Read the two PDFs.** `docs/PAPER_CONNECTIONS.md` §0 explains why they could not be read here. Everything in Layer 6 is self-contained and independently verified, so nothing needs revising on that account — but a write-up that *cites* them must use their notation, and the correspondence between their symbols and `t_S`, `Λ_d`, `s` here has not been checked against the source.
8. **A non-Gaussian tree.** `tree_bp_grid` already handles any innovation law and `fit_em_tree` already feeds every kernel in `src/kernels.py`, so the Layer-3 question (recover an unknown innovation law from noisy data alone) transfers to trees with no new machinery. Untried.
9. **A discrete-alphabet tree.** The closest thing to the actual data model of the hierarchical-filtering paper: symbols on a tree with transition tensors, exact vector messages, no closure. `src/discrete.py` does this for chains and `src/hierarchy.py` does trees for continuous variables; the two have not been crossed.
10. **The cascade under a learned score.** exp_13 `cascade` runs the exact tree-BP score. Running the same measurement under the EM-BP and network scores would say *which levels of the ladder each method gets right dynamically*, rather than statically as `levels` does.

## 8. Things deliberately not done

- Layers 1–4 outputs were **not** regenerated after the seeding fix. They were paired correctly within each run, so they stand as measurements; they are simply not bit-reproducible. Regenerating them is a defensible task but was out of scope.
- No PR was opened (none was requested).
- `report/em_bp_learning.tex` has no results section — the numbers live in the compendium and the CSVs.
