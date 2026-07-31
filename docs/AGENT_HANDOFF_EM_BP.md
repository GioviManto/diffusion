# Agent handoff — Layer 5 (EM + BP denoiser learning)

You are picking up work on `GioviManto/diffusion`, branch
**`claude/em-bp-denoiser-learning-e07ike`**. Read this before touching anything;
several of the traps below cost real time to find and are easy to reintroduce.

For *what was done and what it showed*, read `docs/EM_BP_LEARNING_COMPENDIUM.md`
first — this file is only about how to continue safely.

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
../../.venv/bin/python -m pytest tests/ -q          # expect 30 passed, ~70 s
```

`.venv/` is gitignored — recreate it, don't look for it. Pure numpy/scipy; **no
torch, no jax, and none is needed** (see trap 1). LaTeX for the report needs
`texlive-latex-recommended texlive-fonts-recommended texlive-science lmodern`.

Experiments run from the package root and import via `common.py`, which inserts
the package on `sys.path`. `conftest.py` does the same for pytest.

## 2. Orientation, in reading order

1. `docs/EM_BP_LEARNING_COMPENDIUM.md` — results and findings.
2. `research/nongaussian-bp/report/em_bp_learning.pdf` — the theory (14 pp, 5 propositions). §2 is the algorithm, §4 the kernel families, §5 the efficiency argument, §6 the limitations.
3. `research/nongaussian-bp/src/em.py` — the module docstring explains the `Ξ` compression in full; that is the one idea the whole layer rests on.
4. `research/nongaussian-bp/README.md` — package conventions, all layers.

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
6. **The Laplace kernel has two discretization artifacts** (compendium §5.1, §5.2): its exact M-step is quantized onto the grid's ratio lattice, and its ρ-gradient loses accuracy to a `sign` discontinuity. **Never quote a Laplace-kernel ρ recovery as evidence of accuracy** — ρ\* = 0.8 sits exactly on the lattice, so the error reads as 1e-16 and means nothing. Use the smooth kernels (Gaussian, mixture) for any rate or accuracy claim.
7. **Grid adequacy.** `M=401, A=8` is the validated default. Grid error on the posterior mean is ~4e-6 there, far below any learning error. Below `t ≈ 0.05` the likelihood gets narrow relative to `dx` — check `noising.likelihood_resolution_ok` before trusting small-`t` results.

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

Test-set invariance across replicate seeds (trap 5) — reproduce with:

```python
import exp_07_em_vs_score_network as E
E.SEED_TAG = 0; A0, b0 = E.make_test_set(prior, grid, w, (0.2,), 64)
E.SEED_TAG = 7; A7, b7 = E.make_test_set(prior, grid, w, (0.2,), 64)
assert np.array_equal(A0, A7) and np.array_equal(b0[0.2][1], b7[0.2][1])
```

## 7. What to do next

In rough priority order. The first three are finishing what is started; the rest
are new.

1. **Finish the runs.** At handoff, `exp_06` Parts 3–5 and `exp_07` Parts 3–4 were still executing; partial outputs are committed as WIP. Re-run with `--only` if they were interrupted, then update the compendium's §4 and §7 with real numbers.
2. **Error bars on the headline curve.** `hpc/slurm_replicates.sbatch` + `tools/merge_replicates.py`. The current curve has one replicate per `N`, which is why the EM line wobbles. This is the single highest-value cluster run.
3. **Extend the sample-efficiency sweep upward.** The network was still improving at `N = 2048`, so "EM-BP needs ≥64× less data" is a lower bound read off a truncated curve. Follow it until the network catches up or visibly plateaus — either outcome is publishable, and the honest version needs it.
4. **A structured-architecture baseline** (temporal CNN / small U-Net). This is the most important open question for the paper's credibility: the current baseline is a vanilla MLP, as the email specified, and a reviewer will ask. Expect it to close part of the gap; it will not acquire the exactness or the uniformity in `t`.
5. **Inference-time cost.** BP is ~100× slower per evaluation than a forward pass, and reverse diffusion calls the denoiser at every step. Either distil the fitted BP denoiser into a network, or use Layer-2 Gaussian-projected BP at inference and measure what the closure costs. This is the weakest point of the whole story.
6. **Hybrid non-Markov correction.** Layer 4 showed the score correction is exactly rank one for AR(1)+global-latent priors. Combining that with a *learned* chain kernel is the natural Layer 4 × Layer 5 product and is untouched.
7. **The write-up (advice 2 in the original email).** `report/em_bp_learning.tex` is deliberately paper-shaped — context, propositions, detail in remarks — and is the natural seed for the shared Overleaf document. It has no results section yet; §4 of the compendium is the raw material.

## 8. Things deliberately not done

- Layers 1–4 outputs were **not** regenerated after the seeding fix. They were paired correctly within each run, so they stand as measurements; they are simply not bit-reproducible. Regenerating them is a defensible task but was out of scope.
- No PR was opened (none was requested).
- `report/em_bp_learning.tex` has no results section — the numbers live in the compendium and the CSVs.
