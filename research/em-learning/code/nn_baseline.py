"""Neural denoiser baseline: the thing EM has to be compared against.

What is being compared, and what would be unfair
-----------------------------------------------
The diffusion training objective is denoising regression.  With ``x = alpha_t a + sqrt(Delta_t) z``,

    L(theta) = E_{a,t,z} || f_theta(x, t) - a ||^2

is minimised by ``f*(x,t) = E[a | x, t]`` -- exactly the posterior mean that BP computes
exactly.  So the network and BP are estimating the *same* function, which makes the
comparison meaningful rather than apples-to-oranges.

The honest framing of the comparison matters as much as the numbers.  EM is handed the
model *class* (an additive first-order Markov chain with known OU noise) and estimates two
scalars inside it.  The network is handed nothing and must discover the structure from
data.  So this is **not** evidence that "EM beats deep learning"; it is evidence about how
much a correct structural prior is worth on this problem.  Any statement of the result
should say which of the two is being credited.

Two further asymmetries we do not paper over:

* The network is trained on ``t`` sampled across a range and can be evaluated anywhere in
  it.  A common but incorrect claim is that a network needs retraining to move to a new
  ``t``; a ``(x,t)``-conditioned denoiser does not.  The real asymmetry is in *training
  supervision*: EM here is fitted from a single low-noise ``t`` and extrapolates, whereas
  the network needs supervision spanning the range it will be used on.
* The comparison is at matched training-set size, not matched wall-clock or matched
  hyperparameter tuning effort.  The network is small and lightly tuned; a much larger
  search might close some of the gap.  We report the architecture and budget so the claim
  can be checked.

Implementation
--------------
Plain NumPy with explicit backprop and Adam, because the model is tiny and this keeps the
environment to numpy alone.  ``test_nn_baseline.py`` gradient-checks every layer against
finite differences, which is the only way to be confident hand-written backprop is right.

Time conditioning uses random Fourier features of ``t`` rather than raw ``t``, which is
standard for diffusion models and matters here because the posterior mean depends on ``t``
through ``alpha_t/Delta_t``, a strongly non-linear function near ``t = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chain_models import ChainConfig, ou_coefficients, sample_chains


# -----------------------------------------------------------------------------

def fourier_time_features(t: np.ndarray, n_features: int = 16) -> np.ndarray:
    """Sinusoidal features of the diffusion time.

    ``t`` has shape ``(batch,)``; returns ``(batch, n_features)``.  Frequencies are
    geometrically spaced so that both the sharp behaviour near ``t=0`` and the slow
    behaviour at large ``t`` are representable.
    """
    half = n_features // 2
    freqs = np.exp(np.linspace(np.log(1.0), np.log(200.0), half))
    ang = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


@dataclass
class MLP:
    """Fully connected denoiser with tanh activations.

    ``tanh`` rather than ReLU: the target is a smooth function of the input, and with a
    small network tanh trains more stably here without needing normalisation layers.
    """

    layer_sizes: list[int]
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def init(cls, layer_sizes: list[int], seed: int = 0) -> "MLP":
        rng = np.random.default_rng(seed)
        w, b = [], []
        for a, c in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot scaling, appropriate for tanh.
            w.append(rng.normal(0.0, np.sqrt(2.0 / (a + c)), (a, c)))
            b.append(np.zeros(c))
        return cls(layer_sizes, w, b)

    @property
    def n_params(self) -> int:
        return sum(m.size for m in self.weights) + sum(v.size for v in self.biases)

    def forward(self, h: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return the output and the per-layer activations needed for backprop."""
        acts = [h]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            h = np.tanh(z) if i < len(self.weights) - 1 else z   # linear output layer
            acts.append(h)
        return h, acts

    def backward(
        self, acts: list[np.ndarray], grad_out: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Gradients of the loss w.r.t. weights and biases, given dL/d(output)."""
        gw = [np.zeros_like(w) for w in self.weights]
        gb = [np.zeros_like(b) for b in self.biases]
        delta = grad_out
        for i in range(len(self.weights) - 1, -1, -1):
            gw[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                # d/dz tanh(z) = 1 - tanh(z)^2, and acts[i] is already tanh(z).
                delta = (delta @ self.weights[i].T) * (1.0 - acts[i] ** 2)
        return gw, gb


# -----------------------------------------------------------------------------

@dataclass
class TrainResult:
    model: MLP
    loss_trace: list[float]
    n_params: int
    n_train_chains: int
    epochs: int


def make_batch(
    rng: np.random.Generator,
    a: np.ndarray,
    t_lo: float,
    t_hi: float,
    n_time_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise a batch of clean chains at random diffusion times.

    Fresh noise is drawn every epoch, which is the standard diffusion training protocol:
    the finite resource being varied in the sample-efficiency study is the number of *clean
    chains*, not the number of noise realisations.
    """
    m = a.shape[0]
    t = rng.uniform(t_lo, t_hi, m)
    alpha = np.exp(-t)[:, None]
    delta = (1.0 - np.exp(-2.0 * t))[:, None]
    x = alpha * a + np.sqrt(delta) * rng.normal(size=a.shape)
    feats = np.concatenate([x, fourier_time_features(t, n_time_features)], axis=1)
    return feats, a


def train_denoiser(
    a_train: np.ndarray,
    *,
    hidden: tuple[int, ...] = (256, 256),
    n_time_features: int = 16,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 3e-3,
    t_lo: float = 0.05,
    t_hi: float = 2.5,
    seed: int = 0,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> TrainResult:
    """Train ``f(x,t) -> E[a|x,t]`` by denoising regression, with Adam."""
    rng = np.random.default_rng(seed)
    n_chains, n = a_train.shape
    sizes = [n + n_time_features, *hidden, n]
    model = MLP.init(sizes, seed=seed)

    mw = [np.zeros_like(w) for w in model.weights]
    vw = [np.zeros_like(w) for w in model.weights]
    mb = [np.zeros_like(b) for b in model.biases]
    vb = [np.zeros_like(b) for b in model.biases]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    trace: list[float] = []

    for ep in range(epochs):
        feats_all, targ_all = make_batch(rng, a_train, t_lo, t_hi, n_time_features)
        perm = rng.permutation(n_chains)
        ep_loss, nb = 0.0, 0
        for s in range(0, n_chains, batch_size):
            idx = perm[s : s + batch_size]
            feats, targ = feats_all[idx], targ_all[idx]
            pred, acts = model.forward(feats)
            diff = pred - targ
            loss = float(np.mean(np.sum(diff ** 2, axis=1)))
            grad_out = 2.0 * diff / feats.shape[0]
            gw, gb = model.backward(acts, grad_out)

            step += 1
            for i in range(len(model.weights)):
                if weight_decay:
                    gw[i] = gw[i] + weight_decay * model.weights[i]
                mw[i] = b1 * mw[i] + (1 - b1) * gw[i]
                vw[i] = b2 * vw[i] + (1 - b2) * gw[i] ** 2
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                mhw = mw[i] / (1 - b1 ** step)
                vhw = vw[i] / (1 - b2 ** step)
                mhb = mb[i] / (1 - b1 ** step)
                vhb = vb[i] / (1 - b2 ** step)
                model.weights[i] -= lr * mhw / (np.sqrt(vhw) + eps)
                model.biases[i] -= lr * mhb / (np.sqrt(vhb) + eps)
            ep_loss += loss
            nb += 1
        trace.append(ep_loss / max(nb, 1))
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep + 1:4d}  loss {trace[-1]:.5f}", flush=True)

    return TrainResult(model, trace, model.n_params, n_chains, epochs)


def predict_posterior_mean(
    model: MLP, x: np.ndarray, t: float, n_time_features: int = 16
) -> np.ndarray:
    """Evaluate the trained denoiser at a single diffusion time."""
    tv = np.full(x.shape[0], float(t))
    feats = np.concatenate([x, fourier_time_features(tv, n_time_features)], axis=1)
    out, _ = model.forward(feats)
    return out
