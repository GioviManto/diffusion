"""
generate_figures.py
===================

Produce the 8 PNG figures referenced by main.tex. Every figure is
computed from the closed-form objects in ar1_utils, bp_score and
laplace_score, without any neural-network or training code.

Run after numerical_audit.py has passed:

    python3 generate_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ar1_utils import (
    ar1_covariance,
    ar1_precision_clean,
    joint_score_matrix,
    ou_params,
    precision_t,
    sigma_t,
    stationary_sigma_eta,
)
from bp_score import bp_score
from laplace_score import (
    gaussian_K1_marginal,
    gaussian_K1_score,
    laplace_marginal_v,
    laplace_score_v,
    variance_matched_b,
)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Common style
# ---------------------------------------------------------------------------

PALETTE = {
    "ink":      "#1a1410",
    "bg":       "#f7f4ef",
    "accent":   "#b5341a",
    "accent2":  "#1a3a5c",
    "muted":    "#6e6055",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})


# ---------------------------------------------------------------------------
# Figure 1: Sigma_0 (dense Toeplitz) vs Q_0 (tridiagonal)
# ---------------------------------------------------------------------------

def fig01_sigma_vs_precision() -> str:
    K, alpha = 20, 0.9
    Sigma_0 = ar1_covariance(K, alpha)
    Q_0 = ar1_precision_clean(K, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    cm = "RdBu_r"

    im = axes[0].imshow(Sigma_0, cmap=cm,
                        vmin=-Sigma_0.max(), vmax=Sigma_0.max())
    axes[0].set_title(r"$\Sigma_0$  (dense Toeplitz)")
    axes[0].set_xticks([0, K // 2, K - 1])
    axes[0].set_yticks([0, K // 2, K - 1])
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    cap = max(abs(Q_0.min()), abs(Q_0.max()))
    im2 = axes[1].imshow(Q_0, cmap=cm, vmin=-cap, vmax=cap)
    axes[1].set_title(r"$Q_0 = \Sigma_0^{-1}$  (tridiagonal)")
    axes[1].set_xticks([0, K // 2, K - 1])
    axes[1].set_yticks([0, K // 2, K - 1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    fig.suptitle(rf"Stationary Gaussian AR(1), $K = {K}$, $\alpha = {alpha}$",
                 y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "fig01_sigma_vs_precision.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Factor graph for the chain  --  variables as circles, factors
# as squares; tree structure, no cycles
# ---------------------------------------------------------------------------

def fig02_factor_graph() -> str:
    K = 5
    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    # Coordinates
    x_a = np.arange(K) * 1.6
    y_a = 0.0
    x_x = x_a
    y_x = 1.6
    y_factor_above = 0.85
    y_factor_below = -0.85

    radius_v = 0.34
    side_f = 0.32

    accent = PALETTE["accent"]
    navy = PALETTE["accent2"]
    ink = PALETTE["ink"]
    muted = PALETTE["muted"]

    # Variable nodes: clean a_k (bottom row) and noisy x_k (top row)
    for k in range(K):
        ax.add_patch(plt.Circle((x_a[k], y_a), radius_v,
                                facecolor="white", edgecolor=ink, lw=1.5))
        ax.text(x_a[k], y_a, rf"$a_{{{k}}}$", ha="center", va="center",
                fontsize=12)
        ax.add_patch(plt.Circle((x_x[k], y_x), radius_v,
                                facecolor="white", edgecolor=navy, lw=1.5,
                                linestyle="--"))
        ax.text(x_x[k], y_x, rf"$x_{{{k}}}$", ha="center", va="center",
                fontsize=12)

    # Prior factor p_0 at the leftmost a_0
    ax.add_patch(plt.Rectangle((x_a[0] - side_f - 0.65, y_factor_below - side_f),
                               2 * side_f, 2 * side_f,
                               facecolor=accent, alpha=0.18, edgecolor=accent))
    ax.text(x_a[0] - 0.65, y_factor_below, r"$p_0$",
            ha="center", va="center", fontsize=10)
    ax.plot([x_a[0] - 0.65 + side_f, x_a[0]], [y_factor_below, y_a],
            color=accent, lw=1.0)

    # Transition factors M(a_{k+1} | a_k)
    for k in range(K - 1):
        cx = (x_a[k] + x_a[k + 1]) / 2.0
        cy = y_factor_below
        ax.add_patch(plt.Rectangle((cx - side_f, cy - side_f),
                                   2 * side_f, 2 * side_f,
                                   facecolor=accent, alpha=0.18,
                                   edgecolor=accent))
        ax.text(cx, cy, rf"$M_{{{k}}}$", ha="center", va="center", fontsize=9)
        ax.plot([cx - side_f, x_a[k]], [cy, y_a], color=accent, lw=1.0)
        ax.plot([cx + side_f, x_a[k + 1]], [cy, y_a], color=accent, lw=1.0)

    # Local evidence factors p_t(x_k | a_k)
    for k in range(K):
        cx = x_a[k]
        cy = y_factor_above
        ax.add_patch(plt.Rectangle((cx - side_f, cy - side_f),
                                   2 * side_f, 2 * side_f,
                                   facecolor=navy, alpha=0.18,
                                   edgecolor=navy))
        ax.text(cx, cy, rf"$\phi_{{{k}}}$", ha="center", va="center",
                fontsize=9)
        ax.plot([cx, cx], [cy + side_f, y_x - radius_v], color=navy, lw=1.0,
                linestyle="--")
        ax.plot([cx, cx], [cy - side_f, y_a + radius_v], color=navy, lw=1.0)

    # Annotations
    ax.text(-1.1, 0.0, r"clean", ha="right", va="center",
            fontsize=11, color=muted)
    ax.text(-1.1, 1.6, r"observed", ha="right", va="center",
            fontsize=11, color=muted)
    ax.annotate(r"$\phi_k(a_k) = p_t(x_k \mid a_k)$",
                xy=(x_a[K - 1] + 0.5, y_factor_above),
                xytext=(x_a[K - 1] + 1.4, y_factor_above),
                fontsize=10, color=navy, va="center",
                arrowprops=dict(arrowstyle="->", color=navy, lw=0.8))
    ax.annotate(r"$M_k(a_{k+1} \mid a_k)$  (Markov)",
                xy=(x_a[K - 1] - 0.4, y_factor_below),
                xytext=(x_a[K - 1] + 1.4, y_factor_below),
                fontsize=10, color=accent, va="center",
                arrowprops=dict(arrowstyle="->", color=accent, lw=0.8))

    ax.set_xlim(-3.1, x_a[-1] + 4.0)
    ax.set_ylim(y_factor_below - 1.2, y_x + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Factor graph of the AR(1) chain with OU corruption: "
                 r"a tree (path), so belief propagation is exact",
                 fontsize=11)

    path = os.path.join(OUT, "fig02_factor_graph.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Annotated precision heatmap at multiple t  --  shows actual
# numerical values in cells; emphasises the dense -> sparse transition.
# ---------------------------------------------------------------------------

def fig03_precision_annotated() -> str:
    K, alpha = 6, 0.9
    Sigma_0 = ar1_covariance(K, alpha)
    ts = (1e-5, 1e-3, 1e-2, 1e-1, 1.0, 5.0)
    titles = [rf"$t = 10^{{{int(np.log10(t))}}}$" if t < 1
              else rf"$t = {t:g}$" for t in ts]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    cmap = "RdBu_r"

    # Use a consistent colour scale across the lifecycle for visual comparison
    Q_first = precision_t(Sigma_0, ts[0])
    cap = max(abs(Q_first.min()), abs(Q_first.max())) * 0.9

    for i, t in enumerate(ts):
        ax = axes[i // 3, i % 3]
        Q = precision_t(Sigma_0, t)
        im = ax.imshow(Q, cmap=cmap, vmin=-cap, vmax=cap)
        ax.set_title(titles[i])
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(range(K))
        ax.set_yticklabels(range(K))

        # Annotate every cell with its value to 1-2 sig figs
        for r in range(K):
            for c in range(K):
                v = Q[r, c]
                if abs(v) >= 1.0:
                    s = f"{v:.2f}"
                elif abs(v) >= 1e-3:
                    s = f"{v:.3f}"
                else:
                    # Scientific notation for tiny entries to show fill-in scale
                    s = f"{v:.0e}"
                # Pick text colour based on cell shade for legibility
                shade = abs(v) / max(cap, 1e-12)
                col = "white" if shade > 0.55 else PALETTE["ink"]
                ax.text(c, r, s, ha="center", va="center", fontsize=7.6,
                        color=col)

        plt.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

    fig.suptitle(rf"Frame-basis precision $Q_t = \Sigma_t^{{-1}}$ at six "
                 rf"diffusion times  ($K = {K}$, $\alpha = {alpha}$). "
                 rf"Tridiagonal $\to$ banded $\to$ identity-floor.",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig03_precision_annotated.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Band-fill |Q_t[i, i + d]|  on log-log axes
# ---------------------------------------------------------------------------

def fig04_band_fill() -> str:
    K, alpha = 25, 0.9
    Sigma_0 = ar1_covariance(K, alpha)
    Q_0 = ar1_precision_clean(K, alpha)
    i = K // 2
    ts = np.logspace(-4, 1, 80)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("plasma")
    for d in range(1, 6):
        vals = np.array([abs(precision_t(Sigma_0, t)[i, i + d]) for t in ts])
        col = cmap((d - 1) / 5)
        ax.loglog(ts, vals, label=rf"$d = {d}$", color=col)
        c_pred = ((-1) ** (d - 1)) * (2.0 ** (d - 1)) \
                 * np.linalg.matrix_power(Q_0, d)[i, i + d]
        anchor = ts[0]
        ax.loglog(ts, abs(c_pred) * (ts / anchor) ** (d - 1) * (anchor ** (d - 1)),
                  ls="--", color=col, alpha=0.55)

    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"$|Q_t[i,\, i + d]|$")
    ax.set_title(r"Band-fill scaling   $|Q_t[i, i + d]| \sim t^{d - 1}$"
                 r"  at small $t$  (dashed: theory, no factorial)")
    ax.legend(ncol=5, fontsize=9, frameon=False)
    ax.set_ylim(1e-12, 1e2)
    ax.grid(alpha=0.25, which="both")

    path = os.path.join(OUT, "fig04_band_fill.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: BP vs matrix joint score  --  audit residuals
# ---------------------------------------------------------------------------

def fig05_bp_audit() -> str:
    rng = np.random.default_rng(7)
    Ks = (2, 3, 5, 8, 16)
    alphas = (0.2, 0.5, 0.9, -0.4)
    ts = (0.05, 0.3, 1.0, 3.0)
    residuals: list[float] = []
    for K in Ks:
        for alpha in alphas:
            for t in ts:
                Sigma_0 = ar1_covariance(K, alpha)
                for _ in range(20):
                    x = rng.standard_normal(K)
                    s_bp = bp_score(x, t, alpha)
                    s_mat = joint_score_matrix(x, t, Sigma_0, alpha)
                    residuals.append(float(np.max(np.abs(s_bp - s_mat))))
    residuals_arr = np.asarray(residuals)

    log_floor = -17.0
    log_residuals = np.log10(np.maximum(residuals_arr, 10.0 ** log_floor))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(log_residuals, bins=np.linspace(log_floor, -13, 25),
            color=PALETTE["accent2"], edgecolor="white")
    ax.set_xlabel(r"$\log_{10}\,\max_k |S^{\rm BP}_k - S^{\rm matrix}_k|$")
    ax.set_ylabel("count")
    ax.set_title(rf"BP vs matrix joint score, "
                 rf"{len(residuals_arr)} configurations  (max disagreement "
                 rf"$ = {residuals_arr.max():.2e}$)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = os.path.join(OUT, "fig05_bp_audit.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: Laplace K=1 noised marginal  vs  Gaussian benchmark
# ---------------------------------------------------------------------------

def fig06_K1_density() -> str:
    """Laplace K=1 noised marginal at several diffusion times, with the
    bare prior at t=0 and the Gaussian attractor at t=infty as
    reference dashed curves -- not a Laplace-vs-Gaussian comparison,
    but the lifecycle of the Laplace marginal itself."""
    b = 1.0
    xs = np.linspace(-3.5, 3.5, 320)
    ts = (0.05, 0.3, 0.8, 1.6, 3.0)

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Bare Laplace prior (t -> 0) as reference
    ax.plot(xs, np.exp(-np.abs(xs) / b) / (2.0 * b),
            color="#1a1410", lw=1.0, ls=":",
            label=r"prior $p_0$ (Laplace)")

    for k, t in enumerate(ts):
        col = cmap(0.12 + 0.75 * k / max(1, len(ts) - 1))
        p_lap = laplace_marginal_v(xs, t, b)
        ax.plot(xs, p_lap, color=col, lw=1.6,
                label=rf"$t = {t}$")

    # Gaussian attractor (t -> infty) as reference
    ax.plot(xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2.0 * np.pi),
            color=PALETTE["accent2"], lw=1.0, ls="--",
            label=r"attractor $N(0, 1)$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p_t(x)$")
    ax.set_title(r"Laplace $K = 1$ noised marginal, "
                 r"interpolating from the Laplace prior (dotted) to "
                 r"the Gaussian attractor (dashed)")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path = os.path.join(OUT, "fig06_K1_density.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 7: Laplace K=1 joint score  vs  Gaussian benchmark
# ---------------------------------------------------------------------------

def fig07_K1_score() -> str:
    """Laplace K=1 joint score across several diffusion times, with the
    Gaussian-attractor limit -x as a dashed reference."""
    b = 1.0
    xs = np.linspace(-3.0, 3.0, 280)
    ts = (0.05, 0.2, 0.6, 1.5, 3.0)

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for k, t in enumerate(ts):
        col = cmap(0.12 + 0.75 * k / max(1, len(ts) - 1))
        s_lap = laplace_score_v(xs, t, b)
        ax.plot(xs, s_lap, color=col, lw=1.6,
                label=rf"$t = {t}$")

    ax.plot(xs, -xs, color=PALETTE["accent2"], lw=1.0, ls="--",
            label=r"attractor $-x$")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$S(x, t) = \partial_x \log p_t(x)$")
    ax.set_title(r"Laplace $K = 1$ joint score: nonlinearity at small "
                 r"$t$, linear attractor as $t \to \infty$")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path = os.path.join(OUT, "fig07_K1_score.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 8: Laplace K=1 score asymptotes (t -> 0 and t -> infty)
# ---------------------------------------------------------------------------

def fig08_K1_score_limits() -> str:
    b = 1.0
    xs = np.linspace(-3.0, 3.0, 180)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    for t in (1.0, 0.4, 0.1, 0.02):
        mu, Delta = ou_params(t)
        s = laplace_score_v(xs, t, b)
        ax.plot(xs, Delta * s, label=rf"$t = {t}$")
    ax.plot(xs, -np.sign(xs), color="k", ls="--",
            label=r"$-\,\mathrm{sign}(x)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Delta_t\,S(x, t)$")
    ax.set_title(r"$t \to 0$:   $\Delta_t\,S(x, t) \to -\,\mathrm{sign}(x)$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for t in (0.5, 1.5, 3.0, 6.0):
        s = laplace_score_v(xs, t, b)
        ax.plot(xs, s, label=rf"$t = {t}$")
    ax.plot(xs, -xs, color="k", ls="--", label=r"$-x$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$S(x, t)$")
    ax.set_title(r"$t \to \infty$:   $S(x, t) \to -x$  "
                 r"(Gaussian attractor of OU)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(OUT, "fig08_K1_score_limits.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    paths = [
        fig01_sigma_vs_precision(),
        fig02_factor_graph(),
        fig03_precision_annotated(),
        fig04_band_fill(),
        fig05_bp_audit(),
        fig06_K1_density(),
        fig07_K1_score(),
        fig08_K1_score_limits(),
    ]
    print("Generated figures:")
    for p in paths:
        print(f"  {p}")
    print(f"Total: {len(paths)} PNGs in {OUT}")


if __name__ == "__main__":
    main()
