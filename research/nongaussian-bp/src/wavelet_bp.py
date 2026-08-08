"""Exact BP on a wavelet quadtree, where *every* node is observed.

Three things separate this from `hierarchy.tree_bp_grid`, and each is forced by
the data rather than chosen:

1. **Evidence at every node, not only at the leaves.** In `hierarchy` the
   internal nodes are latent and only the leaves are observed. A wavelet
   decomposition is invertible: every coefficient, at every scale, is part of
   the data, and the forward process noises all of them. So every node carries a
   likelihood factor. This changes nothing structurally -- one unary factor per
   site is exactly the situation under which the posterior factor graph equals
   the prior graph -- so BP is still exact. It only changes where `ell` is
   applied, and it means the downward message must exclude the child's upward
   message while *including* the parent's own evidence.

2. **A per-scale noise level.** Subband coefficient variances in natural images
   span orders of magnitude across scales, and one grid cannot resolve the
   finest while covering the coarsest. Each subband is therefore standardised by
   its own training scale s_d. That is a diagonal reparametrisation, so the
   likelihood stays an exact per-site Gaussian: from

       x_v = alpha a_v + sqrt(Delta) z_v,   xt_v = x_v / s_d,   at_v = a_v / s_d,

   one gets xt_v = alpha at_v + sqrt(Delta / s_d^2) z_v, i.e. the same alpha and a
   per-depth Delta_d = Delta / s_d^2. Nothing is approximated; the noise level is
   simply not the same number at every scale once the coordinates are rescaled.

3. **One kernel per scale.** Wavelet coefficients are not scale-stationary, so a
   single shared transition kernel is misspecified by construction. `log_k[d]`
   governs the edge from depth d to depth d+1, and the E-step accumulates a
   separate Xi per level. Because every kernel in `src/kernels.py` consumes only
   `stats.xi`, the per-level M-step is the existing M-step called once per level
   -- no kernel code changes.

The upward and downward passes are written once and serve both the score and the
E-step, which is why `want_stats` is a flag on the same call rather than a second
implementation to keep in sync.

Convention, inherited unchanged from `src/bp_grid.py`: `log_k[d][k, j]` is
log K_d(u_k | u_j) with k the *child* value and j the *parent* value, so the
upward message is `K.T @ f` and the downward message is `K @ g`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .em import ExpectedStatistics
from .hierarchy import TreeIndex, _leave_one_out_product


@dataclass(frozen=True)
class WaveletBPResult:
    """Posterior summaries on the tree.

    posterior_mean : (B, n_nodes) E[a_v | x], exact up to quadrature.
    log_evidence   : summed over the batch, log p_t(x), all constants included.
    xi_by_level    : per-level expected transition mass, or None if not asked for.
                     `xi_by_level[d]` belongs to the edge depth d -> depth d+1.
    """

    posterior_mean: np.ndarray
    log_evidence: float
    xi_by_level: list[np.ndarray] | None


def node_delta(ti: TreeIndex, delta_by_depth) -> np.ndarray:
    """(n_nodes,) noise variance for each node, from its depth."""
    d_arr = np.asarray(delta_by_depth, dtype=float)
    return np.concatenate([
        np.full(ti.branching**d, d_arr[d]) for d in range(ti.depth + 1)
    ])


def wavelet_tree_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_k: list[np.ndarray],
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta_by_depth,
    branching: int,
    depth: int,
    want_stats: bool = False,
    chunk: int = 64,
) -> WaveletBPResult:
    """Exact sum-product on the quadtree with observations at every node.

    `x` is (B, n_nodes) *standardised* coefficients in breadth-first order.
    `log_k` has length `depth`; `log_k[d]` is the (M, M) kernel for the edge
    from depth d to depth d+1. `delta_by_depth` has length `depth + 1`.
    """
    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != ti.n_nodes:
        raise ValueError(f"x has {x.shape[1]} nodes, tree has {ti.n_nodes}")
    if len(log_k) != depth:
        raise ValueError(f"need {depth} kernels, got {len(log_k)}")
    if len(np.asarray(delta_by_depth)) != depth + 1:
        raise ValueError(f"need {depth + 1} deltas, got {len(np.asarray(delta_by_depth))}")

    m = grid.size
    k_mats = [np.exp(lk) for lk in log_k]
    root = np.exp(log_root)
    nd = node_delta(ti, delta_by_depth)

    means = np.empty_like(x)
    log_evidence = 0.0
    xi_total = [np.zeros((m, m)) for _ in range(depth)] if want_stats else None

    for start in range(0, x.shape[0], chunk):
        part = _bp_chunk(
            ti, grid, weights, k_mats, root, x[start : start + chunk],
            alpha, nd, want_stats,
        )
        means[start : start + chunk] = part.posterior_mean
        log_evidence += part.log_evidence
        if xi_total is not None and part.xi_by_level is not None:
            for d in range(depth):
                xi_total[d] += part.xi_by_level[d]

    return WaveletBPResult(means, log_evidence, xi_total)


def _bp_chunk(
    ti: TreeIndex,
    grid: np.ndarray,
    weights: np.ndarray,
    k_mats: list[np.ndarray],
    root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    nd: np.ndarray,
    want_stats: bool,
) -> WaveletBPResult:
    b, n_nodes = x.shape
    depth, branching = ti.depth, ti.branching
    m = grid.size

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / np.maximum(s, 1e-300), np.log(np.maximum(s, 1e-300))

    # -- evidence at every node -------------------------------------------
    # Row-shifted so exp() cannot underflow a whole row; the discarded shift and
    # the Gaussian normaliser are both restored in `log_scale`.
    z = x[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / nd[None, :, None]
    row_max = log_ell.max(axis=2)
    ev = np.exp(log_ell - row_max[:, :, None])            # (B, n_nodes, M)
    log_scale = row_max.sum(axis=1) - 0.5 * np.sum(np.log(2.0 * np.pi * nd))

    # -- upward pass -------------------------------------------------------
    # bu[v] = evidence(v) * prod over children c of up[c]; up[v] is v's message
    # to its parent. `ev` is kept intact because the downward pass needs the
    # parent's own evidence separately from its children's contributions.
    bu = ev.copy()
    up = np.ones((b, n_nodes, m))
    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        msg, ls = norm(np.einsum("cnk,kj->cnj", weights * bu[:, nodes], k_mats[d - 1]))
        up[:, nodes] = msg
        log_scale += ls.sum(axis=(1, 2))
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(b, -1, branching, m).prod(axis=2)
        bu[:, parents], ls = norm(bu[:, parents] * prod)
        log_scale += ls.sum(axis=(1, 2))

    log_evidence = float(np.sum(
        log_scale + np.log(np.maximum((weights * bu[:, 0] * root).sum(1), 1e-300))
    ))

    # -- downward pass -----------------------------------------------------
    down = np.ones((b, n_nodes, m))
    down[:, 0] = root
    xi_by_level = [np.zeros((m, m)) for _ in range(depth)] if want_stats else None

    for d in range(depth):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        # Leave-one-out over siblings by prefix/suffix scan rather than division:
        # sibling messages are legitimately ~1e-16 in the tails, and dividing by
        # them is where a tree BP implementation quietly loses digits.
        loo = _leave_one_out_product(up[:, kids].reshape(b, -1, branching, m))
        # What the parent knows *apart from* this child: its own evidence, the
        # message from above, and its other children.
        extra = ev[:, parents] * down[:, parents]
        excl, _ = norm(loo * extra[:, :, None, :])
        excl = excl.reshape(b, -1, m)

        if want_stats:
            f_all = (weights * excl).reshape(-1, m)             # parent side
            g_all = (weights * bu[:, kids]).reshape(-1, m)      # child side
            partition = np.einsum("ek,ek->e", g_all, f_all @ k_mats[d].T)
            c_mat = (g_all / np.maximum(partition, 1e-300)[:, None]).T @ f_all
            xi_by_level[d] += c_mat * k_mats[d]

        down[:, kids], _ = norm(np.einsum("cnj,kj->cnk", weights * excl, k_mats[d]))

    belief, _ = norm(bu * down)
    return WaveletBPResult(belief @ grid, log_evidence, xi_by_level)


def stats_by_level(
    xi_by_level: list[np.ndarray],
    log_evidence: float,
    n_images: int,
    branching: int,
) -> list[ExpectedStatistics]:
    """Wrap per-level Xi as `ExpectedStatistics`, one per edge level.

    The evidence is attached to level 0 only, so that summing the list does not
    count it `depth` times; the M-step ignores it in any case.
    """
    return [
        ExpectedStatistics(
            xi=xi,
            site1=np.zeros(xi.shape[0]),
            log_evidence=log_evidence if d == 0 else 0.0,
            n_edges=n_images * branching ** (d + 1),
            n_chains=n_images,
        )
        for d, xi in enumerate(xi_by_level)
    ]
