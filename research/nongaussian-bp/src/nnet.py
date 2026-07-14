"""Minimal pure-numpy MLP + Adam for supervised score / residual regression.

Deliberately dependency-free and deterministic: initialization and minibatch
order are driven by an explicit `numpy.random.Generator`. The network maps
(x, time features) -> R^n and is trained on exact-score targets, so the
comparison "direct score net vs BP baseline vs BP + residual net" is a clean
supervised regression problem with a known ground truth (no denoising-score-
matching variance confound).

Time features: (t, alpha_t, sqrt(Delta_t)) -- a smooth 3-dim embedding that
lets one network serve all diffusion times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def time_features(t: np.ndarray | float) -> np.ndarray:
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    alpha = np.exp(-t_arr)
    delta = 1.0 - np.exp(-2.0 * t_arr)
    return np.stack([t_arr, alpha, np.sqrt(delta)], axis=-1)


@dataclass
class MLP:
    """Fully connected tanh MLP; parameters stored as a flat list of arrays."""

    sizes: tuple[int, ...]
    params: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def init(cls, sizes: tuple[int, ...], rng: np.random.Generator) -> "MLP":
        params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            w = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            params.extend([w, b])
        return cls(sizes=sizes, params=params)

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return output and cached activations for backprop."""
        cache = [X]
        h = X
        n_layers = len(self.params) // 2
        for k in range(n_layers):
            w, b = self.params[2 * k], self.params[2 * k + 1]
            z = h @ w + b
            h = np.tanh(z) if k < n_layers - 1 else z
            cache.append(h)
        return h, cache

    def backward(
        self, cache: list[np.ndarray], grad_out: np.ndarray
    ) -> list[np.ndarray]:
        """Gradients of a scalar loss with upstream gradient `grad_out`."""
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        n_layers = len(self.params) // 2
        g = grad_out
        for k in reversed(range(n_layers)):
            h_in, h_out = cache[k], cache[k + 1]
            if k < n_layers - 1:  # undo tanh
                g = g * (1.0 - h_out**2)
            grads[2 * k] = h_in.T @ g
            grads[2 * k + 1] = g.sum(axis=0)
            g = g @ self.params[2 * k].T
        return grads


def train_score_net(
    net: MLP,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    n_steps: int = 3000,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> MLP:
    """Adam training of net((x, timefeat)) -> Y with MSE loss. Returns the net.

    X: (N, n) observations; T: (N,) times; Y: (N, n) targets (scores or residuals).
    """
    feats = np.concatenate([X, time_features(T)], axis=1)
    n_data = X.shape[0]
    m_state = [np.zeros_like(p) for p in net.params]
    v_state = [np.zeros_like(p) for p in net.params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_data, size=min(batch_size, n_data))
        xb, yb = feats[idx], Y[idx]
        out, cache = net.forward(xb)
        grad_out = 2.0 * (out - yb) / xb.shape[0]
        grads = net.backward(cache, grad_out)
        for j, g in enumerate(grads):
            m_state[j] = beta1 * m_state[j] + (1 - beta1) * g
            v_state[j] = beta2 * v_state[j] + (1 - beta2) * g**2
            m_hat = m_state[j] / (1 - beta1**step)
            v_hat = v_state[j] / (1 - beta2**step)
            net.params[j] = net.params[j] - lr * m_hat / (np.sqrt(v_hat) + eps)
    return net


def predict_score(net: MLP, X: np.ndarray, T: np.ndarray) -> np.ndarray:
    feats = np.concatenate([X, time_features(T)], axis=1)
    out, _ = net.forward(feats)
    return out
