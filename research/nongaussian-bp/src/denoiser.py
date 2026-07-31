"""The two denoisers being compared, behind one interface.

Both estimate the same object -- the posterior mean m_i(x, t) = E[a_i | x],
which determines the score through the exact OU identity
s = -(x - alpha_t m) / Delta_t -- but they get there in opposite ways.

BP denoiser (structure first)
    Learn the transition kernel K_theta of the clean chain, then *compute* the
    posterior mean by exact belief propagation. The learned object lives on
    R x R and does not depend on t at all; the noise level enters only through
    the likelihood factors inside BP. One fit therefore serves every noise
    level, and the estimated quantity is a low-dimensional parameter, so the
    statistical rate is parametric.

DSM network (function first)
    Learn the map (x, t) -> m directly by denoising score matching, the vanilla
    diffusion-model recipe. The learned object is a function on R^n x R_+, it
    must be fitted across the whole noise schedule at once, and nothing in the
    architecture knows the chain is Markov.

The comparison is deliberately generous to the network: it trains on *paired*
(clean, noisy) data with a fresh noise draw at every gradient step, i.e. it may
consume unlimited noise realizations of the same clean chains, while EM sees a
single noisy realization per chain and never sees the clean chain at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bp_grid import grid_bp_batch
from src.nnet import MLP, time_features
from src.noising import alpha_delta


# ----------------------------------------------------------------------------
# BP denoiser from a (learned or true) kernel
# ----------------------------------------------------------------------------

def bp_posterior_mean(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    t: float,
    log_mu: np.ndarray | None = None,
) -> np.ndarray:
    """E[a | x] under the chain prior defined by `kernel`, by exact grid BP.

    `kernel` is anything with `log_transition_matrix(grid)`, which covers both
    the ground-truth priors in `priors` and the learned ones in `kernels`.
    """
    alpha, delta = alpha_delta(t)
    log_k = kernel.log_transition_matrix(grid)
    means, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta, log_mu)
    return means


def score_from_mean(
    X: np.ndarray, means: np.ndarray, t: float
) -> np.ndarray:
    """Exact OU identity, applied to a batch."""
    alpha, delta = alpha_delta(t)
    return -(X - alpha * means) / delta


# ----------------------------------------------------------------------------
# Denoising-score-matching baseline
# ----------------------------------------------------------------------------

@dataclass
class DSMResult:
    net: MLP
    loss_history: list[float]
    seconds: float
    n_params: int
    n_grad_steps: int


def train_dsm_denoiser(
    A_train: np.ndarray,
    t_values,
    rng: np.random.Generator,
    hidden: tuple[int, ...] = (128, 128),
    n_steps: int = 6000,
    batch_size: int = 128,
    lr: float = 2e-3,
    log_every: int = 200,
) -> DSMResult:
    """Vanilla diffusion training: predict the clean chain from the noisy one.

    Minimizes  E_{a, t, z} || m_phi(x, t) - a ||^2  with x = alpha_t a +
    sqrt(Delta_t) z, whose minimizer is exactly E[a | x, t]. This is the same
    target BP computes, so the two methods are directly comparable and the
    network is not being handicapped by a different objective.

    A_train : (N, n) clean chains -- the data budget.
    t_values: noise levels sampled uniformly at each step (the training
              schedule); the same levels are used for evaluation.
    """
    import time

    n_data, n_sites = A_train.shape
    sizes = (n_sites + 3,) + tuple(hidden) + (n_sites,)
    net = MLP.init(sizes, rng)
    t_arr = np.asarray(t_values, dtype=float)

    m_state = [np.zeros_like(p) for p in net.params]
    v_state = [np.zeros_like(p) for p in net.params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_data, size=min(batch_size, n_data))
        a = A_train[idx]
        t = t_arr[rng.integers(0, len(t_arr), size=len(idx))]
        alpha = np.exp(-t)[:, None]
        delta = (1.0 - np.exp(-2.0 * t))[:, None]
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

        feats = np.concatenate([x, time_features(t)], axis=1)
        out, cache = net.forward(feats)
        diff = out - a
        grad_out = 2.0 * diff / len(idx)
        grads = net.backward(cache, grad_out)
        for j, g in enumerate(grads):
            m_state[j] = beta1 * m_state[j] + (1 - beta1) * g
            v_state[j] = beta2 * v_state[j] + (1 - beta2) * g**2
            m_hat = m_state[j] / (1 - beta1**step)
            v_hat = v_state[j] / (1 - beta2**step)
            net.params[j] = net.params[j] - lr * m_hat / (np.sqrt(v_hat) + eps)
        if step % log_every == 0 or step == 1:
            history.append(float(np.mean(diff**2)))
    seconds = time.perf_counter() - t0

    return DSMResult(
        net=net,
        loss_history=history,
        seconds=seconds,
        n_params=net.n_params,
        n_grad_steps=n_steps,
    )


def dsm_posterior_mean(net: MLP, X: np.ndarray, t: float) -> np.ndarray:
    """Network estimate of E[a | x] at a single noise level."""
    t_arr = np.full(X.shape[0], float(t))
    feats = np.concatenate([X, time_features(t_arr)], axis=1)
    out, _ = net.forward(feats)
    return out


# ----------------------------------------------------------------------------
# Shared evaluation
# ----------------------------------------------------------------------------

def evaluate_denoiser(
    means_hat: np.ndarray, means_ref: np.ndarray, X: np.ndarray, t: float
) -> dict:
    """Relative L2 errors on the posterior mean and on the induced score.

    The two are linked by the exact identity  s_hat - s_ref =
    (alpha/Delta)(m_hat - m_ref), so `identity_residual` must sit at machine
    precision; it is the same guard used by every other experiment here.
    """
    alpha, delta = alpha_delta(t)
    s_hat = score_from_mean(X, means_hat, t)
    s_ref = score_from_mean(X, means_ref, t)

    dm = means_hat - means_ref
    ds = s_hat - s_ref
    mean_rel = float(np.linalg.norm(dm) / np.linalg.norm(means_ref))
    score_rel = float(np.linalg.norm(ds) / np.linalg.norm(s_ref))
    resid = float(
        np.linalg.norm(ds - (alpha / delta) * dm) / (np.linalg.norm(ds) + 1e-300)
    )
    return {
        "mean_rel_l2": mean_rel,
        "score_rel_l2": score_rel,
        "mean_mse": float(np.mean(dm**2)),
        "identity_residual": resid,
    }
