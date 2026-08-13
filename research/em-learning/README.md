# EM learning of the denoiser for Markov-chain diffusion data

Follow-up to `../gaussian-ar1-bp/markov_gaussian_approx/`, prompted by Jérôme's message of
30 July 2026 relaying Marc's suggestions.

## The claim we are building

For a Markov prior observed through OU noise, BP gives the **exact** score in `O(n)`.
A neural denoiser must learn a function of `(x, t)` across the whole diffusion-time range.
EM instead learns the handful of parameters of the *generative* model, and BP then produces
the exact score at **every** `t` for free. The headline experiment is sample efficiency:
score error as a function of the number of training chains, EM versus a vanilla MLP.

## Layout

```
code/
  chain_models.py     data models: 5 innovation families, all with Var(eps) = q exactly,
                      so every family shares the covariance rho^{|i-j|}
  bp_core.py          grid BP, Gaussian closure (lambda,h), pairwise marginals, exact
                      Gaussian reference, boundary diagnostics
  test_bp_core.py     validation of the above  -- run this before trusting anything
outputs/              generated CSV + figures
compendium/           the detailed self-contained PDF
paper/                4-page anonymised NeurIPS-format write-up
```

## Status

| component | state |
|---|---|
| `chain_models.py` | done, validated |
| `bp_core.py` | done, validated |
| `test_bp_core.py` | 40+ checks, all passing |
| EM rung 1 (parametric `rho,q`) | probe done, needs productionising |
| EM rung 2 (innovation shape) | not started |
| EM rung 3 (nonparametric kernel) | not started |
| MLP baseline | not started |
| reverse-diffusion sampling | not started |
| compendium / paper | not started |

## Two facts worth knowing before using this code

**Gaussian closure is exactly the linear denoiser.** Moment matching uses only the first two
moments of the innovation, so closure returns `alpha Sigma_0 Sigma_t^{-1} x` for *every*
innovation family. Verified to `1e-16` across all five. Consequently the "Gaussian message
error" is the excess error of the best linear estimator, not a numerical artefact — and it is
the exact quantity a learned score must beat.

**Grid-BP accuracy is governed by the smoothness of the innovation density.** Measured
quadrature convergence of `innovation_pdf`:

| family | rate | cause |
|---|---|---|
| gaussian, mixture | machine precision | smooth |
| laplace | `O(h^2)` | cusp at 0 |
| uniform | `O(h)` | jump discontinuities at `±h` |
| student | `h`-independent floor `~2e-9` | tail truncated at `±40` |

So a bounded-support innovation cannot be resolved as accurately as a smooth one at fixed
grid resolution. Budget for this when interpreting the `uniform` results in the family sweep,
and widen the grid rather than refine it for `student`.

## Running

```bash
python code/test_bp_core.py
```

Requires only numpy and matplotlib.
