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
    xi      : (M, M) expected transition mass on the temporal edges, summed over
              every edge of every sequence, or None if not requested. Same
              `Xi[k, j]` convention as everywhere else, so the temporal kernel's
              M-step is the existing M-step with no changes.
    """

    context: np.ndarray
    log_z: np.ndarray
    xi: np.ndarray | None = None


def chain_bp_potentials(
    pot: np.ndarray,
    k_time: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    want_stats: bool = False,
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

    xi = None
    if want_stats:
        # Pairwise belief on each temporal edge:
        #   b(a_f = u_j, a_{f+1} = u_k)  ~  K[k, j] left_f[j] right_{f+1}[k],
        # with `left` carrying everything known at f from the left and its own
        # potential, and `right` the same from the right. Normalised per edge, so
        # Xi sums to the number of temporal edges -- the property every M-step in
        # `src/kernels.py` relies on.
        xi = np.zeros((m, m))
        for i in range(f_len - 1):
            left = weights[None, :] * pot[:, i] * fwd[:, i]
            right = weights[None, :] * pot[:, i + 1] * bwd[:, i + 1]
            partition = np.einsum("bk,bk->b", right, left @ k_time.T)
            c_mat = (right / np.maximum(partition, 1e-300)[:, None]).T @ left
            xi += c_mat * k_time

    return ChainBPResult(context, log_z, xi)


@dataclass(frozen=True)
class VideoBPResult:
    posterior_mean: np.ndarray                  # (B, F, n_nodes)
    log_evidence: float
    xi_space: list[np.ndarray] | None = None    # per spatial level
    xi_time: np.ndarray | None = None           # temporal edges


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
    want_stats: bool = False,
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
    chain = chain_bp_potentials(
        pot, k_time, weights, np.exp(log_mu), want_stats=want_stats
    )

    # -- 3. downward pass per frame, with the chain context as the root message
    # Once the root holds its correct incoming message, every belief the downward
    # pass computes is the full-model posterior, not the per-frame one. So the
    # spatial Xi taken from *this* call is the caterpillar's, which is the whole
    # reason the statistics are collected here and not in step 1.
    context = chain.context.reshape(b * f_len, m)
    down = wavelet_tree_bp(
        grid, weights, log_k_space, uniform, flat, alpha, delta_by_depth,
        branching, depth, chunk=chunk, root_message=context,
        want_stats=want_stats,
    )

    log_evidence = float(np.sum(tree_scale + chain.log_z))
    return VideoBPResult(
        down.posterior_mean.reshape(b, f_len, n_nodes), log_evidence,
        down.xi_by_level, chain.xi,
    )
