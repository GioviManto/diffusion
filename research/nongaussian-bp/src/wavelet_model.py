"""The wavelet hidden-Markov-tree model: fitting it, scoring with it, sampling from it.

This is the assembly layer. `src/wavelet.py` supplies the transform and the tree
indexing, `src/wavelet_bp.py` supplies exact inference, `src/kernels.py` supplies
the M-step; what is left is the bookkeeping that turns them into a model of
images.

Three decisions are recorded here rather than buried:

**Per-subband standardisation.** Each subband is divided by its training
standard deviation s_d, which turns the single pixel-space Delta_t into a
per-depth Delta_d = Delta_t / s_d^2. See `src/wavelet_bp.py` for why that is an
exact reparametrisation and not an approximation. The s_d are estimated once, on
the training split, and stored on the model: they are parameters, and using test
statistics for them would leak.

**One kernel per scale, shared across orientations.** Coefficients are not
scale-stationary, so a shared kernel is misspecified. Whether HL, LH and HH need
*separate* kernels is an empirical question, so `tie_orientations` is a flag and
the default (True) is the more constrained model, which is the one that should
have to be beaten.

**The LL coefficient is modelled separately.** It is a single scalar per image,
disconnected from all three trees. Modelling it as a one-dimensional density
costs nothing and keeps the tree model honest: no part of the image is quietly
dropped, and the held-out likelihood is a likelihood of the whole image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from .bp_grid import make_grid
from .em import ExpectedStatistics
from .hierarchy import TreeIndex
from .noising import alpha_delta
from .wavelet import ORIENTATIONS, WaveletQuadtree, images_to_tree, tree_to_images
from .wavelet_bp import stats_by_level, wavelet_tree_bp

_LOG_2PI = float(np.log(2.0 * np.pi))


def _sample_columns(
    cdf: np.ndarray, column: np.ndarray, rng: np.random.Generator, block: int = 200000
) -> np.ndarray:
    """Sample one state per entry of `column`, from that column's CDF.

    `cdf` is (M, M) with `cdf[:, j]` the cumulative distribution of the child
    given parent state j. Done by broadcast comparison rather than a per-sample
    `searchsorted` loop, in blocks so the (M, N) intermediate stays bounded.
    """
    flat = column.ravel()
    out = np.empty(flat.size, dtype=np.intp)
    for start in range(0, flat.size, block):
        idx = flat[start : start + block]
        u = rng.random(idx.size)
        out[start : start + block] = (cdf[:, idx] < u[None, :]).sum(axis=0)
    np.clip(out, 0, cdf.shape[0] - 1, out=out)
    return out.reshape(column.shape)


# ----------------------------------------------------------------------------
# Standardisation
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubbandScales:
    """Per-(orientation, depth) standard deviations, estimated on training data.

    Stored as (3, depth + 1). `per_node` expands them to one entry per tree node,
    which is the form both the standardiser and the per-depth Delta need.
    """

    scales: np.ndarray

    @classmethod
    def fit(cls, qt: WaveletQuadtree, nodes: np.ndarray) -> "SubbandScales":
        depth_of = qt.node_depth
        out = np.empty((3, qt.depth + 1))
        for oi in range(3):
            for d in range(qt.depth + 1):
                out[oi, d] = nodes[:, oi, depth_of == d].std()
        if not np.all(out > 0):
            raise ValueError("a subband has zero variance; check the input images")
        return cls(out)

    def per_node(self, qt: WaveletQuadtree) -> np.ndarray:
        """(3, n_nodes) scale for every node."""
        depth_of = qt.node_depth
        return self.scales[:, depth_of]

    def standardise(self, qt: WaveletQuadtree, nodes: np.ndarray) -> np.ndarray:
        return nodes / self.per_node(qt)[None, :, :]

    def restore(self, qt: WaveletQuadtree, nodes: np.ndarray) -> np.ndarray:
        return nodes * self.per_node(qt)[None, :, :]

    def delta_by_depth(self, orientation_index: int, delta: float) -> np.ndarray:
        """Delta_d = Delta / s_d^2 for one orientation tree."""
        return delta / self.scales[orientation_index] ** 2


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveletTreeModel:
    """A fitted wavelet HMT: per-scale kernels, root priors, subband scales.

    `kernels[oi][d]` is the kernel for the edge depth d -> depth d+1 of
    orientation tree `oi`. When orientations are tied, the three lists hold the
    same objects.
    """

    qt: WaveletQuadtree
    scales: SubbandScales
    kernels: list[list]
    log_root: np.ndarray          # (3, M) root prior per orientation
    grid: np.ndarray
    weights: np.ndarray
    ll_mean: float                # scaling-coefficient prior
    ll_std: float
    tie_orientations: bool = True

    @property
    def depth(self) -> int:
        return self.qt.depth

    def log_k(self, orientation_index: int) -> list[np.ndarray]:
        return [k.log_transition_matrix(self.grid) for k in self.kernels[orientation_index]]

    # -- inference ---------------------------------------------------------

    def posterior_mean_nodes(
        self, nodes_std: np.ndarray, alpha: float, delta: float, chunk: int = 32
    ) -> tuple[np.ndarray, float]:
        """E[a | x] on standardised nodes, plus the exact log evidence.

        `nodes_std` is (B, 3, n_nodes). The three orientation trees are disjoint,
        so their evidences add and their posteriors are computed independently --
        which is a statement about the model, not a factorisation approximation.
        """
        out = np.empty_like(nodes_std)
        total_log_ev = 0.0
        for oi in range(3):
            res = wavelet_tree_bp(
                self.grid, self.weights, self.log_k(oi), self.log_root[oi],
                nodes_std[:, oi, :], alpha,
                self.scales.delta_by_depth(oi, delta),
                self.qt.branching, self.depth, chunk=chunk,
            )
            out[:, oi, :] = res.posterior_mean
            total_log_ev += res.log_evidence
        return out, total_log_ev

    def denoise_images(
        self, noisy: np.ndarray, t: float, chunk: int = 32
    ) -> np.ndarray:
        """E[clean image | noisy image] at diffusion time t.

        The whole pipeline: transform, standardise, exact BP per orientation,
        unstandardise, inverse transform. Exact at every step but the quadrature,
        because the transform is orthonormal and the trees are loop-free.
        """
        alpha, delta = alpha_delta(t)
        _, nodes, scaling = images_to_tree(noisy, self.qt.levels)
        std_nodes = self.scales.standardise(self.qt, nodes)
        post, _ = self.posterior_mean_nodes(std_nodes, alpha, delta, chunk)
        post = self.scales.restore(self.qt, post)
        # The LL coefficient is a scalar Gaussian problem in closed form.
        post_ll = self._ll_posterior_mean(scaling, alpha, delta)
        return tree_to_images(self.qt, post, post_ll)

    def _ll_posterior_mean(self, scaling, alpha: float, delta: float):
        v = self.ll_std**2
        return (alpha * v * (scaling - alpha * self.ll_mean) / (alpha**2 * v + delta)
                + self.ll_mean)

    # -- resolution -------------------------------------------------------

    def resolution_report(self, t: float, points_per_std: float = 3.0) -> dict:
        """Is the grid fine enough to resolve the likelihood at this `t`?

        This is the failure mode that per-subband standardisation *creates*, and
        it is worth stating plainly because nothing else in the package has it.
        In standardised coordinates the likelihood of node v has standard
        deviation

            sqrt(Delta_t) / (alpha_t s_d),

        so a subband with a *large* scale s_d gets a *narrow* likelihood. The
        coarse subbands of a natural image have the largest scales (LH depth 0 is
        10.7 against 0.14 for HH depth 4, a factor of 76), so on a shared grid
        they are the ones that go under-resolved, and they do it at small t --
        exactly where a reverse sampler spends its last steps.

        The consequence is not a small loss of accuracy: below the resolved t the
        near-delta likelihood is being integrated against a mesh that cannot see
        it, and both the score and the evidence become unreliable. So this is
        reported, and `min_resolved_t` is what the samplers clamp to, rather than
        letting the caller integrate into a regime the discretisation does not
        support.

        The fix is a *per-depth* grid, and it is cheap: coarse subbands have 1, 4,
        16 nodes against 256 at the finest level, so refining exactly where the
        likelihood is narrow costs almost nothing. It needs rectangular
        transition matrices and an M-step that takes a parent grid and a child
        grid separately, which is a real change to `src/kernels.py` and is not
        done here.
        """
        alpha, delta = alpha_delta(t)
        dx = float(self.grid[1] - self.grid[0])
        width = np.sqrt(delta) / (alpha * self.scales.scales)   # (3, depth+1)
        pps = width / dx
        # Smallest t at which every subband clears the threshold:
        # sqrt(exp(2t) - 1) >= points_per_std * dx * s_max.
        need = points_per_std * dx * float(self.scales.scales.max())
        return {
            "t": float(t),
            "grid_spacing": dx,
            "points_per_std": pps,
            "min_points_per_std": float(pps.min()),
            "resolved": bool(pps.min() >= points_per_std),
            "min_resolved_t": float(0.5 * np.log(need**2 + 1.0)),
        }

    def score_images(self, noisy: np.ndarray, t: float, chunk: int = 32) -> np.ndarray:
        """Pixel-space score grad_x log p_t(x), for `src/reverse.py`.

        Built from the posterior mean, which is what BP returns:
        s = (alpha E[a|x] - x) / Delta. Exact up to quadrature, and in *pixel*
        coordinates because the orthonormal transform carries the wavelet-space
        score back by W^T -- which `denoise_images` has already done.
        """
        alpha, delta = alpha_delta(t)
        return (alpha * self.denoise_images(noisy, t, chunk) - noisy) / delta

    # -- sampling ----------------------------------------------------------

    def sample_ancestral(
        self, n: int, rng: np.random.Generator, jitter: bool = True
    ) -> np.ndarray:
        """Draw `n` images from the fitted model directly, without diffusing.

        The model is a Markov tree with a known root prior and known per-level
        kernels, so it can be sampled root-to-leaves in closed form. On the grid
        that is an ordinary discrete Markov chain over `M` states -- which is
        exactly the model BP does inference in, so these samples and the
        reverse-diffusion samples of `sample_reverse` are samples of the *same*
        object. Any disagreement between them is sampler error, not model error,
        which is what makes the comparison worth running.

        `jitter` spreads each drawn state uniformly across its grid cell; without
        it the samples live on a lattice and every downstream statistic inherits
        a spurious discreteness.
        """
        grid, w = self.grid, self.weights
        dx = float(grid[1] - grid[0])
        ti = TreeIndex(self.depth, self.qt.branching)
        nodes = np.empty((n, 3, ti.n_nodes))

        for oi in range(3):
            state = np.empty((n, ti.n_nodes), dtype=np.intp)
            root_p = np.exp(self.log_root[oi] - self.log_root[oi].max()) * w
            root_p /= root_p.sum()
            state[:, 0] = rng.choice(len(grid), size=n, p=root_p)

            for d in range(self.depth):
                k = np.exp(self.kernels[oi][d].log_transition_matrix(grid)) * w[:, None]
                cdf = np.cumsum(k, axis=0)
                cdf /= cdf[-1][None, :]
                parents = ti.nodes_at(d)
                kids = ti.nodes_at(d + 1)
                parent_state = np.repeat(state[:, parents], ti.branching, axis=1)
                state[:, kids] = _sample_columns(cdf, parent_state, rng)

            vals = grid[state]
            if jitter:
                vals = vals + (rng.random(vals.shape) - 0.5) * dx
            nodes[:, oi, :] = vals

        nodes = self.scales.restore(self.qt, nodes)
        scaling = self.ll_mean + self.ll_std * rng.standard_normal((n, 1))
        return tree_to_images(self.qt, nodes, scaling)

    def sample_reverse(
        self,
        n: int,
        rng: np.random.Generator,
        n_steps: int = 200,
        t_max: float = 3.0,
        t_min: float = 0.02,
        chunk: int = 32,
        ode: bool = False,
    ) -> np.ndarray:
        """Generate by reverse diffusion driven by the exact BP score.

        This is the route the project is actually about: no ancestral shortcut,
        just the score integrated backwards. It should agree with
        `sample_ancestral` in distribution, and where it does not, the gap is the
        sampler's discretisation rather than the model's.
        """
        from .reverse import (
            denoising_readout, probability_flow_ode, reverse_sde, time_grid,
        )

        # Never integrate below the resolved t: past it the coarse subbands'
        # likelihood is narrower than a grid cell and the score is not merely
        # inaccurate but meaningless. See `resolution_report`.
        floor = self.resolution_report(t_min)["min_resolved_t"]
        t_min = max(t_min, floor)
        if t_max <= t_min:
            raise ValueError(
                f"grid resolves no t below {floor:.3f}; refine the grid "
                f"(grid_size) or raise t_max above it"
            )

        side = self.qt.side
        x = rng.standard_normal((n, side, side))
        times = time_grid(t_max, t_min, n_steps)

        def score_fn(z, t):
            return self.score_images(z, float(t), chunk=chunk)

        if ode:
            x = probability_flow_ode(x, score_fn, times, heun=False)
        else:
            x = reverse_sde(x, score_fn, times, rng)

        # Read out the posterior mean rather than x(t_min). This is not a
        # cosmetic choice here: `t_min` is floored at the resolved t, which for a
        # natural-image subband set is around 0.9, and x(t_min) still carries
        # Delta_t ~ 0.83 of noise. Comparing *that* against an ancestral sample
        # compares a noisy variable with a clean one, and the disagreement is
        # entirely the noise. The readout is exact for a BP score because the
        # score is built from the posterior mean in the first place.
        return denoising_readout(x, score_fn(x, t_min), t_min)

    def log_likelihood_images(self, images: np.ndarray, t: float, chunk: int = 32) -> float:
        """Exact log p_t(x) for a batch of images, in *pixel* coordinates.

        Two corrections turn the tree evidence into an image likelihood and both
        are exact:

        * standardisation is a diagonal linear map, contributing -sum_v log s_v
          to the log density (change of variables);
        * the Haar transform is orthonormal, so its Jacobian determinant is 1 and
          it contributes nothing at all.

        That second point is the reason this number is comparable across models
        that work in different bases, and it is why the transform had to be
        orthonormal rather than merely invertible.
        """
        alpha, delta = alpha_delta(t)
        _, nodes, scaling = images_to_tree(images, self.qt.levels)
        std_nodes = self.scales.standardise(self.qt, nodes)
        _, log_ev = self.posterior_mean_nodes(std_nodes, alpha, delta, chunk)
        log_jac = -float(np.sum(np.log(self.scales.per_node(self.qt)))) * len(images)
        var_ll = alpha**2 * self.ll_std**2 + delta
        log_ll = float(np.sum(
            -0.5 * ((scaling - alpha * self.ll_mean) ** 2 / var_ll
                    + np.log(var_ll) + _LOG_2PI)
        ))
        return log_ev + log_jac + log_ll


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------

@dataclass
class WaveletEMTrace:
    log_evidence: list[float]
    seconds: list[float]

    @property
    def monotone_violation(self) -> float:
        """Largest decrease in the evidence across iterations; should be 0.

        The same check the chain EM uses. On a tree with per-level kernels the
        argument is unchanged -- every level's M-step increases its own term of
        Q and the terms are separate -- so a violation here means a bug, not a
        property of the model.
        """
        if len(self.log_evidence) < 2:
            return 0.0
        d = np.diff(self.log_evidence)
        return float(max(0.0, -d.min()))


def fit_wavelet_tree(
    images: np.ndarray,
    levels: int,
    t_train,
    kernel_factory,
    n_iters: int = 30,
    half_width: float = 8.0,
    grid_size: int = 401,
    tie_orientations: bool = True,
    chunk: int = 32,
    tol: float = 1e-9,
    verbose: bool = False,
) -> tuple[WaveletTreeModel, WaveletEMTrace]:
    """Generalised EM for the wavelet HMT.

    `t_train` is a list of diffusion times; the model is fitted on noisy
    observations at all of them at once, which is the point of the construction --
    one kernel is a denoiser at *every* noise level, so the training set should
    exercise several.

    `kernel_factory(depth_index, rng)` returns a fresh kernel for one edge level.
    """
    from .utils import rng_for

    rng = rng_for("wavelet-em", levels, tuple(t_train))
    qt, clean_nodes, clean_scaling = images_to_tree(images, levels)
    scales = SubbandScales.fit(qt, clean_nodes)
    std_clean = scales.standardise(qt, clean_nodes)

    grid, weights = make_grid(half_width, grid_size)
    log_root = np.tile(-0.5 * grid**2 - 0.5 * _LOG_2PI, (3, 1))

    n_kern = 1 if tie_orientations else 3
    kern = [[kernel_factory(d, rng) for d in range(qt.depth)] for _ in range(n_kern)]

    # Noisy observations, drawn once so EM sees a fixed dataset.
    obs = []
    for t in t_train:
        alpha, delta = alpha_delta(t)
        noise = rng.standard_normal(std_clean.shape)
        # Standardised coordinates: the noise is scaled by 1 / s_d as well.
        per_node = scales.per_node(qt)[None, :, :]
        obs.append((alpha * std_clean + np.sqrt(delta) * noise / per_node, alpha, delta))

    trace = WaveletEMTrace([], [])
    prev = -np.inf
    for it in range(n_iters):
        t0 = time.perf_counter()
        total = [[None] * qt.depth for _ in range(n_kern)]
        log_ev = 0.0

        for x_std, alpha, delta in obs:
            for oi in range(3):
                ki = 0 if tie_orientations else oi
                log_k = [k.log_transition_matrix(grid) for k in kern[ki]]
                res = wavelet_tree_bp(
                    grid, weights, log_k, log_root[oi], x_std[:, oi, :], alpha,
                    scales.delta_by_depth(oi, delta), qt.branching, qt.depth,
                    want_stats=True, chunk=chunk,
                )
                log_ev += res.log_evidence
                parts = stats_by_level(
                    res.xi_by_level, res.log_evidence, len(images), qt.branching
                )
                for d, p in enumerate(parts):
                    total[ki][d] = p if total[ki][d] is None else total[ki][d] + p

        trace.log_evidence.append(log_ev)
        for ki in range(n_kern):
            for d in range(qt.depth):
                kern[ki][d] = kern[ki][d].m_step(total[ki][d], grid)
        trace.seconds.append(time.perf_counter() - t0)
        if verbose:
            print(f"  EM {it + 1:3d}/{n_iters}  log-ev {log_ev:.6e}  "
                  f"{trace.seconds[-1]:.1f}s")
        if np.isfinite(prev) and abs(log_ev - prev) <= tol * abs(prev):
            break
        prev = log_ev

    kernels = kern * 3 if tie_orientations else kern
    model = WaveletTreeModel(
        qt=qt, scales=scales, kernels=list(kernels), log_root=log_root,
        grid=grid, weights=weights,
        ll_mean=float(clean_scaling.mean()), ll_std=float(clean_scaling.std()),
        tie_orientations=tie_orientations,
    )
    return model, trace
