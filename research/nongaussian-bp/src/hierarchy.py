"""Hierarchical (tree) priors and exact BP on them.

Why this module exists
----------------------
Everything else in this package lives on a *chain*. A chain is a tree, so BP is
exact there, but a chain has only one length scale: correlations decay as
rho^{|i-j|} with a single correlation length. Garnier-Brun, Mezard, Moscato and
Saglietti (arXiv:2408.15138) study data generated on a *balanced tree*, where
correlations live on a discrete ladder of length scales, one per level of the
hierarchy, and can be switched off level by level ("hierarchical filtering").
That is the natural setting in which "which correlations has the model learned"
becomes a question with a graded answer rather than a yes/no.

This module provides the continuous analogue of that data model:

    z_root ~ N(0, 1)
    z_child = rho z_parent + sqrt(1 - rho^2) eps,     eps ~ p_innov (unit var)
    leaves are observed; internal nodes are latent.

Two leaves whose lowest common ancestor sits at depth d are separated by
2(L - d) edges, so for Gaussian innovations

    Cov(leaf_i, leaf_j) = rho^{2(L - d(i,j))},

an ultrametric covariance with exactly L + 1 distinct eigenvalues. Those
eigenvalues are what `src/spectral.py` turns into a ladder of speciation times.

Two BP implementations, mirroring Layers 1-2 of the package:

- `tree_bp_gaussian` -- information form (h, lambda), exact for Gaussian
  innovations, O(n) and free of any discretization. This is the reference.
- `tree_bp_grid` -- grid messages, exact up to quadrature for *any* innovation
  law. Same two primitives as the chain (`K.T @ f` upward, `K @ g` downward),
  so the conventions of `src/bp_grid.py` carry over unchanged: K[out, in].

`tree_bp_grid` agrees with `tree_bp_gaussian` to ~1e-10 on Gaussian innovations
(tests/test_hierarchy.py), which is the same cross-check that validated the
chain code.

Node indexing
-------------
Nodes are numbered breadth-first, root = 0. A node at depth d has index in
[offset(d), offset(d) + b^d). Children of node i at depth d are
offset(d+1) + b * (i - offset(d)) + k for k in 0..b-1. Leaves are the last
b^L entries, in left-to-right order, so leaf j of the sequence is node
offset(L) + j.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Tree bookkeeping
# ----------------------------------------------------------------------------

def level_offset(depth: int, branching: int) -> int:
    """Index of the first node at `depth` in the breadth-first numbering."""
    if branching == 1:
        return depth
    return (branching**depth - 1) // (branching - 1)


@dataclass(frozen=True)
class TreeIndex:
    """Breadth-first index of a balanced `branching`-ary tree of depth `depth`."""

    depth: int
    branching: int

    @property
    def n_nodes(self) -> int:
        return level_offset(self.depth + 1, self.branching)

    @property
    def n_leaves(self) -> int:
        return self.branching**self.depth

    def nodes_at(self, d: int) -> np.ndarray:
        start = level_offset(d, self.branching)
        return np.arange(start, start + self.branching**d)

    def children(self, node: int, d: int) -> np.ndarray:
        pos = node - level_offset(d, self.branching)
        start = level_offset(d + 1, self.branching) + self.branching * pos
        return np.arange(start, start + self.branching)

    def leaf_nodes(self) -> np.ndarray:
        return self.nodes_at(self.depth)

    def lca_depth_matrix(self) -> np.ndarray:
        """`D[i, j]` = depth of the lowest common ancestor of leaves i, j.

        Computed from the base-`b` digit expansion of the leaf positions: the
        LCA depth is the length of the common prefix.
        """
        b, L = self.branching, self.depth
        n = self.n_leaves
        idx = np.arange(n)
        digits = np.stack([(idx // b**(L - 1 - k)) % b for k in range(L)], axis=1)
        same = digits[:, None, :] == digits[None, :, :]
        # Common prefix length = number of leading True values.
        prefix = np.cumprod(same, axis=2).sum(axis=2)
        return prefix.astype(int)


# ----------------------------------------------------------------------------
# The prior
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianTree:
    """Balanced-tree Gaussian prior on `branching**depth` observed leaves.

    `rho` is the per-edge correlation; innovations have variance 1 - rho^2 so
    every node, including the leaves, has unit marginal variance -- the same
    normalization the chain priors use, which is what makes the two families
    directly comparable at fixed noise schedule.
    """

    depth: int
    branching: int = 2
    rho: float = 0.9

    @property
    def name(self) -> str:
        return f"tree(L={self.depth},b={self.branching},rho={self.rho})"

    @property
    def index(self) -> TreeIndex:
        return TreeIndex(self.depth, self.branching)

    @property
    def n_leaves(self) -> int:
        return self.index.n_leaves

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    def sample_nodes(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample `n` full trees; returns `(n, n_nodes)` including latents."""
        ti = self.index
        z = np.empty((n, ti.n_nodes))
        z[:, 0] = rng.standard_normal(n)
        scale = np.sqrt(self.q)
        for d in range(self.depth):
            parents = ti.nodes_at(d)
            kids = ti.nodes_at(d + 1)
            # Children of parents[p] are kids[b*p : b*p+b]; repeat parent values.
            z[:, kids] = (
                self.rho * np.repeat(z[:, parents], self.branching, axis=1)
                + scale * rng.standard_normal((n, kids.size))
            )
        return z

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample `n` leaf sequences; returns `(n, n_leaves)`."""
        return self.sample_nodes(rng, n)[:, self.index.leaf_nodes()]

    def leaf_covariance(self) -> np.ndarray:
        """Dense `(n_leaves, n_leaves)` covariance -- reference, O(n^2) memory."""
        d = self.index.lca_depth_matrix()
        return self.rho ** (2.0 * (self.depth - d))

    # -- spectrum -----------------------------------------------------------

    def subtree_row_sum(self, d: int) -> float:
        """Sum of Cov(i, j) over leaves j inside a depth-`d` subtree, i fixed in it.

        With m = L - d levels below, the subtree holds b^m leaves; of these,
        (b-1) b^{k-1} sit at LCA distance k for k = 1..m, contributing
        rho^{2k} each, plus the leaf itself.
        """
        b, m = self.branching, self.depth - d
        k = np.arange(1, m + 1)
        return float(1.0 + (b - 1) * np.sum(b ** (k - 1.0) * self.rho ** (2.0 * k)))

    def level_eigenvalues(self) -> list[tuple[int, float, int]]:
        """Exact spectrum as `[(level, eigenvalue, multiplicity), ...]`.

        The ultrametric covariance is diagonalized by the uniform vector plus,
        at each depth d = 0..L-1, the contrasts between the b children of a
        depth-d node (constant on each child subtree, summing to zero). For such
        a contrast, cross-subtree terms collapse because the weights sum to zero:

            Lambda_d = S_{d+1} - b^{L-d-1} rho^{2(L-d)},

        with multiplicity (b-1) b^d; the uniform vector has Lambda = S_0 with
        multiplicity 1. Multiplicities sum to b^L and the trace to b^L, both
        asserted in the tests.

        Levels are labelled by the depth `d` of the node whose children the
        contrast separates, with `-1` reserved for the uniform mode. So the
        label increases as the mode gets *finer*: level -1 is the mean of the
        whole tree, level 0 splits it in half, level L-1 separates sibling
        leaves. Eigenvalues decrease monotonically in that order, which is what
        makes the speciation ladder well ordered.
        """
        b, L = self.branching, self.depth
        out: list[tuple[int, float, int]] = [(-1, self.subtree_row_sum(0), 1)]
        for d in range(L):
            lam = self.subtree_row_sum(d + 1) - b ** (L - d - 1) * self.rho ** (
                2.0 * (L - d)
            )
            out.append((d, float(lam), (b - 1) * b**d))
        return out

    def level_projector_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Orthonormal eigenbasis `(V, levels)` with `V[:, k]` an eigenvector.

        Built explicitly (rather than by `eigh`) so that each column carries the
        hierarchy level it belongs to; `eigh` would return an arbitrary rotation
        inside each degenerate eigenspace, which destroys exactly the labelling
        this module is for.
        """
        b, L = self.branching, self.depth
        n = self.n_leaves
        cols: list[np.ndarray] = []
        levels: list[int] = []

        uniform = np.full(n, 1.0 / np.sqrt(n))
        cols.append(uniform)
        levels.append(-1)

        for d in range(L):
            block = b ** (L - d)          # leaves under a depth-d node
            child = b ** (L - d - 1)      # leaves under each of its children
            for node in range(b**d):
                base = node * block
                # (b-1) orthonormal contrasts among the b child blocks: use the
                # Helmert basis, which is orthonormal and sums to zero.
                for r in range(1, b):
                    v = np.zeros(n)
                    for k in range(r):
                        v[base + k * child: base + (k + 1) * child] = 1.0
                    v[base + r * child: base + (r + 1) * child] = -float(r)
                    v /= np.linalg.norm(v)
                    cols.append(v)
                    levels.append(d)
        return np.stack(cols, axis=1), np.asarray(levels)


# ----------------------------------------------------------------------------
# Exact BP on the tree -- information form (Gaussian)
# ----------------------------------------------------------------------------

def tree_bp_gaussian(
    tree: GaussianTree, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact posterior mean of the leaves given noisy leaves, by BP.

    `x` is `(n_chains, n_leaves)` of observations y_i = alpha a_i + sqrt(delta) z.
    Returns `(n_chains, n_leaves)` posterior means E[a | y].

    Messages are carried in information form (h, lam) = (Sigma^{-1} mu,
    Sigma^{-1}), the parameterization the audit identified as the correct one:
    the update touches *only* these two numbers per message, with no projection
    or moment-matching step anywhere. Upward and downward passes use

        up:    lam' = rho^2 lam / (1 + q lam),      h' = rho h / (1 + q lam)
        down:  lam' = 1 / (rho^2 / lam + q),        h' = lam' rho h / lam

    which are the marginalize-then-transport and transport-then-condition forms
    of the same linear-Gaussian edge.
    """
    ti = tree.index
    b, L, q, rho = tree.branching, tree.depth, tree.q, tree.rho
    x = np.atleast_2d(x)
    n_chains = x.shape[0]
    if x.shape[1] != tree.n_leaves:
        raise ValueError(f"x has {x.shape[1]} leaves, tree has {tree.n_leaves}")

    n_nodes = ti.n_nodes
    # Evidence: only leaves are observed.
    lam_obs = np.zeros((n_chains, n_nodes))
    h_obs = np.zeros((n_chains, n_nodes))
    leaves = ti.leaf_nodes()
    lam_obs[:, leaves] = alpha**2 / delta
    h_obs[:, leaves] = alpha * x / delta

    # Upward pass: message from each node to its parent.
    lam_up = np.zeros((n_chains, n_nodes))
    h_up = np.zeros((n_chains, n_nodes))
    lam_tot = lam_obs.copy()   # evidence + all upward messages received
    h_tot = h_obs.copy()
    for d in range(L, 0, -1):
        nodes = ti.nodes_at(d)
        lam, h = lam_tot[:, nodes], h_tot[:, nodes]
        denom = 1.0 + q * lam
        lam_up[:, nodes] = rho**2 * lam / denom
        h_up[:, nodes] = rho * h / denom
        # Accumulate into parents: children are contiguous in blocks of b.
        parents = ti.nodes_at(d - 1)
        lam_tot[:, parents] += lam_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)
        h_tot[:, parents] += h_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)

    # Root prior N(0, 1): precision 1, information 0.
    lam_tot[:, 0] += 1.0

    # Downward pass.
    lam_down = np.zeros((n_chains, n_nodes))
    h_down = np.zeros((n_chains, n_nodes))
    for d in range(L):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        lam_p = lam_tot[:, parents] + lam_down[:, parents]
        h_p = h_tot[:, parents] + h_down[:, parents]
        # Exclude each child's own upward message (repeat parent along children).
        lam_excl = np.repeat(lam_p, b, axis=1) - lam_up[:, kids]
        h_excl = np.repeat(h_p, b, axis=1) - h_up[:, kids]
        lam_new = 1.0 / (rho**2 / lam_excl + q)
        lam_down[:, kids] = lam_new
        h_down[:, kids] = lam_new * rho * h_excl / lam_excl

    lam_leaf = lam_tot[:, leaves] + lam_down[:, leaves]
    h_leaf = h_tot[:, leaves] + h_down[:, leaves]
    return h_leaf / lam_leaf


def tree_posterior_mean_dense(
    tree: GaussianTree, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Same quantity by dense linear algebra -- O(n^3), used only to check BP."""
    c = tree.leaf_covariance()
    n = c.shape[0]
    # Posterior of a given y = alpha a + sqrt(delta) z:  mean = C A^T (A C A^T + D)^-1 y
    gram = alpha**2 * c + delta * np.eye(n)
    return np.atleast_2d(x) @ np.linalg.solve(gram, alpha * c).T


def tree_score_gaussian(
    tree: GaussianTree, x: np.ndarray, t: float
) -> np.ndarray:
    """Exact score via the identity s = -(x - alpha m) / Delta."""
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    m = tree_bp_gaussian(tree, x, alpha, delta)
    return -(np.atleast_2d(x) - alpha * m) / delta


# ----------------------------------------------------------------------------
# Exact BP on the tree -- grid messages (any innovation law)
# ----------------------------------------------------------------------------

def tree_bp_grid(
    log_k: np.ndarray,
    grid: np.ndarray,
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    branching: int,
    depth: int,
) -> np.ndarray:
    """Posterior mean of the leaves by grid BP on the tree.

    `log_k[k, j] = log p(u_k | u_j)` -- the same `K[out, in]` convention as
    `src/bp_grid.py`, so the upward message is `K.T @ f` (marginalize the child)
    and the downward message is `K @ g` (transport the parent). `log_root` is
    the root's log prior on the grid.

    Works for any innovation law, which is the whole point: for a non-Gaussian
    tree there is no information-form shortcut, and the dense reference does not
    exist either.
    """
    from .noising import likelihood_matrix

    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(x)
    n_chains, n_leaves = x.shape
    m = grid.size
    if n_leaves != ti.n_leaves:
        raise ValueError(f"x has {n_leaves} leaves, tree has {ti.n_leaves}")

    k_mat = np.exp(log_k - log_k.max())
    dx = float(grid[1] - grid[0])
    root_prior = np.exp(log_root - log_root.max())

    def norm(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(v.sum(axis=-1, keepdims=True), 1e-300)

    n_nodes = ti.n_nodes
    up = np.ones((n_chains, n_nodes, m))         # message node -> parent
    belief_up = np.ones((n_chains, n_nodes, m))  # evidence * all children

    leaves = ti.leaf_nodes()
    for c in range(n_chains):
        belief_up[c, leaves] = likelihood_matrix(grid, x[c], alpha, delta)

    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        f = norm(belief_up[:, nodes])
        msg = norm(np.einsum("cnk,kj->cnj", f, k_mat) * dx)
        up[:, nodes] = msg
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(n_chains, -1, branching, m).prod(axis=2)
        belief_up[:, parents] = norm(belief_up[:, parents] * prod)

    down = np.ones((n_chains, n_nodes, m))       # message parent -> node
    for d in range(depth):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        # Leave-one-out product over siblings, by prefix/suffix scans. Doing it
        # as (product / own message) instead would divide by message entries
        # that are legitimately ~1e-16 in the tails, which is where a tree BP
        # implementation usually starts quietly losing digits.
        msgs = up[:, kids].reshape(n_chains, -1, branching, m)
        loo = _leave_one_out_product(msgs)
        extra = down[:, parents]
        if d == 0:
            extra = extra * root_prior
        excl = norm(loo * extra[:, :, None, :]).reshape(n_chains, -1, m)
        down[:, kids] = norm(np.einsum("cnj,kj->cnk", excl, k_mat) * dx)

    belief = norm(belief_up[:, leaves] * down[:, leaves])
    return belief @ grid


def _leave_one_out_product(msgs: np.ndarray) -> np.ndarray:
    """`out[..., i, :] = prod_{j != i} msgs[..., j, :]`, without dividing."""
    b = msgs.shape[-2]
    prefix = np.ones_like(msgs)
    suffix = np.ones_like(msgs)
    for i in range(1, b):
        prefix[..., i, :] = prefix[..., i - 1, :] * msgs[..., i - 1, :]
    for i in range(b - 2, -1, -1):
        suffix[..., i, :] = suffix[..., i + 1, :] * msgs[..., i + 1, :]
    return prefix * suffix


# ----------------------------------------------------------------------------
# EM on a tree: the same Xi, one level up in graph complexity
# ----------------------------------------------------------------------------

def tree_e_step(
    grid: np.ndarray,
    weights: np.ndarray,
    log_k: np.ndarray,
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    branching: int,
    depth: int,
):
    """Exact BP E-step on the tree, returning `em.ExpectedStatistics`.

    The argument of `src/em.py` transfers verbatim: the posterior of a tree
    under sitewise likelihood factors is a tree, so BP is exact, and every
    expectation the M-step needs is a sum of pairwise edge beliefs. The E-step
    therefore still compresses to one `M x M` matrix

        Xi[k, j] = sum over edges of the posterior mass on parent u_j -> child u_k,

    which sums to the number of *tree* edges rather than chain edges. Because
    `Xi` is where the graph topology stops being visible, every kernel in
    `src/kernels.py` consumes this without modification -- a chain-trained
    M-step and a tree-trained M-step are the same code.

    The one genuinely new piece of bookkeeping is the evidence. Messages are
    renormalized at every node for numerical range, so `log p(x)` is recovered
    by accumulating the discarded log-scales; that is what makes the monotone
    check available on trees too, and it is verified against the closed-form
    Gaussian evidence in the tests.
    """
    from .em import ExpectedStatistics
    from .noising import log_likelihood_matrix

    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(x)
    n_chains, n_leaves = x.shape
    if n_leaves != ti.n_leaves:
        raise ValueError(f"x has {n_leaves} leaves, tree has {ti.n_leaves}")
    m = grid.size
    k_mat = np.exp(log_k)
    root = np.exp(log_root)
    leaves = ti.leaf_nodes()
    n_nodes = ti.n_nodes

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / np.maximum(s, 1e-300), np.log(np.maximum(s, 1e-300))

    # -- upward pass -------------------------------------------------------
    bu = np.ones((n_chains, n_nodes, m))    # belief-up: evidence * children msgs
    up = np.ones((n_chains, n_nodes, m))    # message node -> parent
    log_scale = np.zeros(n_chains)

    for c in range(n_chains):
        bu[c, leaves] = np.exp(log_likelihood_matrix(grid, x[c], alpha, delta))
    # log_likelihood_matrix subtracts each row's max and drops the Gaussian
    # normalizer; both are constants in `a` and are added back here.
    z = x[:, :, None] - alpha * grid[None, None, :]
    row_max = (-0.5 * z**2 / delta).max(axis=2)
    log_scale += row_max.sum(axis=1) - 0.5 * n_leaves * np.log(2 * np.pi * delta)

    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        msg = np.einsum("cnk,kj->cnj", weights * bu[:, nodes], k_mat)
        msg, ls = norm(msg)
        up[:, nodes] = msg
        log_scale += ls.sum(axis=(1, 2))
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(n_chains, -1, branching, m).prod(axis=2)
        bu[:, parents], ls = norm(bu[:, parents] * prod)
        log_scale += ls.sum(axis=(1, 2))

    log_evidence = float(
        np.sum(log_scale + np.log(np.maximum((weights * bu[:, 0] * root).sum(1), 1e-300)))
    )

    # -- downward pass, accumulating Xi ------------------------------------
    down = np.ones((n_chains, n_nodes, m))
    c_mat = np.zeros((m, m))
    for d in range(depth):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        msgs = up[:, kids].reshape(n_chains, -1, branching, m)
        loo = _leave_one_out_product(msgs)
        extra = down[:, parents]
        if d == 0:
            extra = extra * root
        excl, _ = norm(loo * extra[:, :, None, :])
        excl = excl.reshape(n_chains, -1, m)

        # Pairwise edge belief b(u_j, u_k) ~ excl_parent(j) K[k, j] bu_child(k),
        # in the weighted representation that makes sums quadratures.
        f_all = (weights * excl).reshape(-1, m)               # parent side
        g_all = (weights * bu[:, kids]).reshape(-1, m)        # child side
        partition = np.einsum("ek,ek->e", g_all, f_all @ k_mat.T)
        c_mat += (g_all / np.maximum(partition, 1e-300)[:, None]).T @ f_all

        down[:, kids], _ = norm(np.einsum("cnj,kj->cnk", weights * excl, k_mat))

    xi = c_mat * k_mat
    site1 = np.zeros(m)   # root density is held fixed; nothing to accumulate
    n_edges = n_chains * (ti.n_nodes - 1)
    return ExpectedStatistics(
        xi=xi, site1=site1, log_evidence=log_evidence,
        n_edges=n_edges, n_chains=n_chains,
    )


def fit_em_tree(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    groups,
    branching: int,
    depth: int,
    log_root: np.ndarray | None = None,
    n_iters: int = 50,
    tol: float = 1e-9,
):
    """EM for a tree prior. `groups` is a list of `(x, alpha, delta)` batches.

    Deliberately a thin wrapper: the E-step differs from the chain case, the
    M-step does not, because both hand the kernel the same `Xi`.
    """
    import time

    from .em import EMTrace, ExpectedStatistics

    if log_root is None:
        log_root = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)

    trace = EMTrace(log_evidence=[], theta=[], seconds=[])
    current = kernel
    prev = -np.inf
    for _ in range(n_iters):
        t0 = time.perf_counter()
        log_k = current.log_transition_matrix(grid)
        total: ExpectedStatistics | None = None
        for x, alpha, delta in groups:
            part = tree_e_step(
                grid, weights, log_k, log_root, x, alpha, delta, branching, depth
            )
            total = part if total is None else total + part
        assert total is not None
        trace.log_evidence.append(total.log_evidence)
        trace.theta.append(np.asarray(current.theta, dtype=float).copy())
        current = current.m_step(total, grid)
        trace.seconds.append(time.perf_counter() - t0)
        if np.isfinite(prev) and abs(total.log_evidence - prev) <= tol * abs(prev):
            break
        prev = total.log_evidence
    return current, trace
