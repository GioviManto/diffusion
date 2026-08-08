"""Exact BP on video: a temporal chain of spatial trees.

The structural question, stated precisely
-----------------------------------------
Video has two dependence structures, and the project already computes exactly on
each of them separately: a **spatial tree** per frame (`src/wavelet_bp.py`) and a
**temporal chain** across frames (everything else in the package). The tempting
claim is that the combination is therefore also exact. It is not, and the reason
is worth being exact about.

Take the *fully coupled* model: coefficient v in frame f depends on its scale
parent `p(v)` in frame f, and on itself in frame f-1. Then

    v_f -- p(v)_f -- p(v)_{f-1} -- v_{f-1} -- v_f

is a 4-cycle. The graph has loops, BP is loopy, and the exactness claim is gone.
Coupling every coefficient in both directions is exactly what one would want for
video, and it is exactly what cannot be done here.

What *is* loop-free is a spanning structure. The natural one couples the frames
only through the **root** of each frame's tree:

    root_1 --- root_2 --- root_3 --- ...        (temporal chain)
      |          |          |
    quadtree   quadtree   quadtree              (spatial, per frame)

Each root has one pendant subtree and at most two chain neighbours; the subtrees
are disjoint; there is no cycle anywhere. This is a "caterpillar": a chain
backbone with trees hanging off it. BP on it is exact, and the model it
corresponds to is a real one -- frames are temporally coupled at the coarsest
scale, and fine detail is temporally independent *given* the coarse trajectory.

That restriction is the price, and it should be stated rather than glossed:
this model cannot express a moving edge whose fine-scale detail persists
independently of the coarse content. It is the maximal tree-structured
spatio-temporal model of this shape, not the model one would write down if
exactness were not required.

How it is computed
------------------
Not with new inference code. BP on a caterpillar factorises into the two passes
the package already has:

1. run each frame's quadtree **upward**; the root's belief-up is then a single
   message summarising that entire frame;
2. run an exact **chain** BP over the frames with those messages as unary
   potentials;
3. feed each root's chain context back in as its incoming message and run each
   frame's quadtree **downward**.

Step 2 needs a chain BP over *arbitrary* unary potentials rather than Gaussian
likelihoods built from observations, which is the one genuinely new routine here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wavelet_bp import wavelet_tree_bp


@dataclass(frozen=True)
class ChainBPResult:
    """Forward-backward on a chain with supplied unary potentials.

    context : (B, F, M) product of the messages from the left and right
              neighbours, *excluding* the site's own potential -- which is
              exactly what a site must pass on to a subtree hanging below it.
    log_z   : (B,) log normalisation of the chain, given the potentials as
              supplied (so the caller's own normalisation must be added back).
    """

    context: np.ndarray
    log_z: np.ndarray


def chain_bp_potentials(
    pot: np.ndarray,
    k_time: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
) -> ChainBPResult:
    """Exact forward-backward on a chain of arbitrary unary potentials.

    `pot` is (B, F, M), `k_time[k, j] = K(a_{f+1} = u_k | a_f = u_j)` -- the same
    `K[out, in]` convention as everywhere else -- and `mu` is the (M,) initial
    law. Messages are renormalised at each step and the discarded scales are
    accumulated, so `log_z` is usable at any chain length.
    """
    pot = np.asarray(pot, dtype=float)
    b, f_len, m = pot.shape
    if k_time.shape != (m, m):
        raise ValueError(f"k_time must be ({m}, {m}), got {k_time.shape}")

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / np.maximum(s, 1e-300), np.log(np.maximum(s, 1e-300))[..., 0]

    fwd = np.empty((b, f_len, m))
    fwd[:, 0] = mu[None, :]
    for i in range(1, f_len):
        src = weights[None, :] * pot[:, i - 1] * fwd[:, i - 1]
        fwd[:, i], _ = norm(src @ k_time.T)

    # Only the *backward* scales enter the evidence. The normalisation is
    # closed by contracting at site 0, where the forward message is `mu` itself
    # and carries no accumulated scale; adding the forward scales as well would
    # count the same normalisation twice, which is a mistake that leaves the
    # posterior means untouched and so survives every check but this one.
    log_z = np.zeros(b)
    bwd = np.empty((b, f_len, m))
    bwd[:, -1] = 1.0
    for i in range(f_len - 2, -1, -1):
        src = weights[None, :] * pot[:, i + 1] * bwd[:, i + 1]
        msg, ls = norm(src @ k_time)
        bwd[:, i] = msg
        log_z += ls

    context = fwd * bwd
    total = (weights[None, :] * pot[:, 0] * context[:, 0]).sum(axis=1)
    log_z = log_z + np.log(np.maximum(total, 1e-300))
    return ChainBPResult(context, log_z)


@dataclass(frozen=True)
class VideoBPResult:
    posterior_mean: np.ndarray   # (B, F, n_nodes)
    log_evidence: float


def caterpillar_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_k_space: list[np.ndarray],
    k_time: np.ndarray,
    log_mu: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta_by_depth,
    branching: int,
    depth: int,
    chunk: int = 32,
) -> VideoBPResult:
    """Exact BP on a temporal chain of spatial quadtrees.

    `x` is (B, F, n_nodes) standardised coefficients: B videos, F frames, one
    quadtree per frame in breadth-first order.

    Each frame's tree is traversed twice -- once to obtain its root message, once
    to push the chain context back down. That is a factor of two against a
    single-pass implementation and is deliberate: it reuses `wavelet_tree_bp`
    unchanged rather than duplicating the upward and downward passes, and a
    duplicated BP implementation that silently drifts from the tested one is a
    worse problem than a factor of two.
    """
    x = np.asarray(x, dtype=float)
    b, f_len, n_nodes = x.shape
    m = grid.size
    flat = x.reshape(b * f_len, n_nodes)

    # -- 1. upward pass per frame -----------------------------------------
    # The root prior is *not* applied here: the chain supplies what the root
    # knows from outside, and applying a prior as well would double-count it.
    uniform = np.zeros(m)
    up = wavelet_tree_bp(
        grid, weights, log_k_space, uniform, flat, alpha, delta_by_depth,
        branching, depth, chunk=chunk, root_message=np.ones((b * f_len, m)),
    )
    pot = up.root_belief_up.reshape(b, f_len, m)
    tree_scale = up.log_scale.reshape(b, f_len).sum(axis=1)

    # -- 2. exact chain over the frames -----------------------------------
    chain = chain_bp_potentials(pot, k_time, weights, np.exp(log_mu))

    # -- 3. downward pass per frame, with the chain context as the root message
    context = chain.context.reshape(b * f_len, m)
    down = wavelet_tree_bp(
        grid, weights, log_k_space, uniform, flat, alpha, delta_by_depth,
        branching, depth, chunk=chunk, root_message=context,
    )

    log_evidence = float(np.sum(tree_scale + chain.log_z))
    return VideoBPResult(down.posterior_mean.reshape(b, f_len, n_nodes), log_evidence)
