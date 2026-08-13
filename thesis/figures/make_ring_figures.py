"""Regenerate the rotating-ring figures of the thesis in the paper style.

Seven figures, one per distinct result, in the same style as make_figures.py:
vector PDF, white background, serif text with cm mathtext, navy/red/olive
palette, no top/right spines, no chart junk.

The closed-form quantities are recomputed here; the measured quantities are
read from the curated CSVs of research/experiment1-rotating-ring/data.

Run:  python make_ring_figures.py        (from thesis/figures/)
"""

import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.join(HERE, "..", "..", "research", "experiment1-rotating-ring")
DATA = os.path.join(RESEARCH, "data")
sys.path.insert(0, os.path.join(RESEARCH, "code"))

from metrics import diagnostics_table  # noqa: E402
from polar_analysis import cavity_mass, exact_score_slice, marginal_score_slice  # noqa: E402
from polar_core import PolarGridSmoother, PolarRingModel, ou_parameters  # noqa: E402
from surrogate_model import (  # noqa: E402
    SurrogateConfig,
    add_ou_noise as add_ou_noise_surrogate,
    exact_score_and_jacobian,
    rotation,
    sample_clean_trajectories,
)

# ---------------- paper style (identical to make_figures.py) ----------------
NAVY = "#1f3a5f"
RED = "#a02c2c"
OLIVE = "#6b6b2a"
GRAY = "#7f7f7f"
CYCLE = [NAVY, RED, OLIVE, "#4a7ba6", "#c07840", GRAY]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=CYCLE),
})

BASELINE = dict(kappa=2.0, D_r=0.015, omega=1.0, D_theta=0.005, T=20)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def read_csv(*parts):
    import pandas as pd
    return pd.read_csv(os.path.join(DATA, *parts))


def unit_circle(ax, alpha=0.8):
    angle = np.linspace(0, 2 * math.pi, 400)
    ax.plot(np.cos(angle), np.sin(angle), ls=":", lw=0.8, color=GRAY, alpha=alpha)


def tag(ax, letter):
    ax.text(0.025, 0.975, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, fontweight="bold")


def bracketed_matrix(ax, x, y, entries, prefix="", width=0.34, height=0.11, fontsize=8.5):
    """A 2x2 matrix in axes coordinates; matplotlib mathtext has no matrix env."""
    kw = dict(transform=ax.transAxes, color="black", clip_on=False)
    if prefix:
        ax.text(x, y, prefix, ha="right", va="center", fontsize=fontsize, **kw)
    left, right = x + 0.015, x + 0.015 + width
    top, bottom = y + height, y - height
    tick = 0.035
    for edge, direction in ((left, +1), (right, -1)):
        ax.plot([edge, edge], [bottom, top], lw=0.7, **kw)
        ax.plot([edge, edge + direction * tick], [top, top], lw=0.7, **kw)
        ax.plot([edge, edge + direction * tick], [bottom, bottom], lw=0.7, **kw)
    cols = (left + 0.28 * width, left + 0.75 * width)
    rows = (y + 0.5 * height, y - 0.5 * height)
    for index, entry in enumerate(entries):
        ax.text(cols[index % 2], rows[index // 2], entry, ha="center", va="center",
                fontsize=fontsize, **kw)


# =====================================================================
def fig_ring_model():
    """The model: polar coordinates, the confining force, one trajectory, two clocks."""
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.85))

    ax = axes[0]
    theta, radius = 0.95, 1.18
    point = radius * np.array([math.cos(theta), math.sin(theta)])
    e_r = np.array([math.cos(theta), math.sin(theta)])
    e_t = np.array([-math.sin(theta), math.cos(theta)])
    unit_circle(ax)
    ax.plot([0, point[0]], [0, point[1]], color=NAVY, lw=1.0)
    ax.plot(*point, "o", ms=3.5, color=NAVY)
    ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=math.degrees(theta),
                     color=RED, lw=0.8))
    ax.plot([0, 0.9], [0, 0], ls=":", lw=0.6, color=GRAY)
    ax.annotate("", xy=point + 0.38 * e_r, xytext=point,
                arrowprops=dict(arrowstyle="-|>", color=OLIVE, lw=0.9, mutation_scale=7))
    ax.annotate("", xy=point + 0.38 * e_t, xytext=point,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, mutation_scale=7))
    ax.text(*(point + 0.46 * e_r), r"$e_r$", fontsize=8, color=OLIVE)
    ax.text(*(point + 0.46 * e_t), r"$e_\theta$", fontsize=8, color=RED)
    ax.text(0.30, 0.55, r"$r$", fontsize=8.5, color=NAVY)
    ax.text(0.33, 0.09, r"$\theta$", fontsize=8.5, color=RED)
    ax.set_xlim(-0.3, 1.8)
    ax.set_ylim(-0.3, 1.8)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("polar coordinates")
    tag(ax, "a")

    ax = axes[1]
    kappa = BASELINE["kappa"]
    grid = np.linspace(0.45, 1.55, 300)
    ax.plot(grid, 0.5 * kappa * (grid - 1.0) ** 2, color=NAVY, label=r"$V(r)$")
    ax.plot(grid, -kappa * (grid - 1.0), color=RED, ls="--", label=r"$-V'(r)$")
    ax.axhline(0.0, color=GRAY, lw=0.5)
    ax.axvline(1.0, color=GRAY, ls=":", lw=0.6)
    for r0 in (0.66, 1.34):
        step = 0.13 * np.sign(-kappa * (r0 - 1.0))
        ax.annotate("", xy=(r0 + step, 0), xytext=(r0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0, mutation_scale=7))
    ax.set_xlabel(r"$r$")
    ax.set_title("radial confinement")
    ax.legend(loc="upper right", handlelength=1.4)
    tag(ax, "b")

    rng = np.random.default_rng(20260726)
    model = PolarRingModel(**BASELINE)
    _, _, clean = model.simulate(rng)
    ax = axes[2]
    unit_circle(ax)
    ax.plot(clean[:, 0], clean[:, 1], "-o", ms=2.2, lw=0.9, color=NAVY)
    ax.plot(*clean[0], "o", ms=4, color=RED)
    ax.annotate("", xy=clean[7], xytext=clean[4],
                arrowprops=dict(arrowstyle="-|>", color=OLIVE, lw=1.0, mutation_scale=7,
                                connectionstyle="arc3,rad=0.35"))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(r"clean trajectory, $t=0$")
    tag(ax, "c")

    ax = axes[3]
    noise = rng.standard_normal(clean.shape)
    attenuation, variance = ou_parameters(0.5)
    noisy = attenuation * clean + math.sqrt(variance) * noise
    unit_circle(ax)
    ax.plot(noisy[:, 0], noisy[:, 1], "-o", ms=2.2, lw=0.9, color=GRAY)
    ax.plot(*noisy[0], "o", ms=4, color=RED)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(r"after the channel, $t=0.5$")
    tag(ax, "d")

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_ring_model.pdf")


# =====================================================================
def _chain(ax, n, target):
    xs = np.linspace(0.09, 0.91, n)
    y_top, y_bot, rw, rh = 0.80, 0.24, 0.048, 0.105
    for i, x in enumerate(xs):
        ax.add_patch(Ellipse((x, y_top), 2 * rw, 2 * rh, facecolor="white",
                             edgecolor=NAVY, lw=1.4 if i == target else 0.8, zorder=3))
        ax.text(x, y_top, rf"$a_{{{i}}}$", ha="center", va="center", fontsize=8, zorder=4)
        ax.add_patch(Ellipse((x, y_bot), 2 * rw, 2 * rh, facecolor="white",
                             edgecolor=GRAY, lw=0.8, zorder=3))
        ax.text(x, y_bot, rf"$x_{{{i}}}$", ha="center", va="center", fontsize=8, zorder=4)
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i + 1] - rw - 0.005, y_top), xytext=(xs[i] + rw + 0.005, y_top),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=0.8, mutation_scale=8))
    return xs, y_top, y_bot, rh


def fig_ring_joint_vs_marginal():
    """Joint score conditions on all frames; the marginal score conditions on one."""
    n, target = 6, 2
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.15))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    xs, y_top, y_bot, rh = _chain(ax, n, target)
    for x in xs:
        ax.annotate("", xy=(x, y_bot + rh + 0.005), xytext=(x, y_top - rh - 0.005),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7, mutation_scale=7))
    for i, x in enumerate(xs):
        rad = 0.30 if i < target else (-0.30 if i > target else 0.0)
        ax.add_patch(FancyArrowPatch((x, y_bot + rh + 0.008),
                                     (xs[target] - 0.010, y_top - rh - 0.010),
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=8, color=NAVY, lw=0.8, zorder=2))
    ax.set_title("joint score")
    ax.text(0.5, 0.03, r"$S_k(x,t)\propto\mathbb{E}[a_k\,|\,x_0,\dots,x_{K-1}]$",
            ha="center", fontsize=8.5)

    ax = axes[1]
    xs, y_top, y_bot, rh = _chain(ax, n, target=-1)
    for x in xs:
        ax.annotate("", xy=(x, y_bot + rh + 0.005), xytext=(x, y_top - rh - 0.005),
                    arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.0, mutation_scale=7))
    ax.set_title("per-frame marginal scores")
    ax.text(0.5, 0.03, r"$s^{\mathrm{marg}}_k(x_k,t)\propto\mathbb{E}[a_k\,|\,x_k]$",
            ha="center", fontsize=8.5)

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_ring_joint_vs_marginal.pdf")


# =====================================================================
def fig_ring_gauge():
    """A known rotation is an orthogonal change of frame that removes the circulation."""
    psi = math.radians(38.0)
    R = rotation(psi)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1))

    ax = axes[0]
    v = np.array([1.02, 0.26])
    w = R @ v
    unit_circle(ax, alpha=0.6)
    for vec, colour, label, offset in ((v, NAVY, r"$z$", (0.04, -0.10)),
                                       (w, RED, r"$R_\psi z$", (-0.02, 0.06))):
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1, mutation_scale=8))
        ax.text(*(vec + np.array(offset)), label, fontsize=8.5, color=colour)
    ax.add_patch(Arc((0, 0), 1.2, 1.2,
                     theta1=math.degrees(math.atan2(v[1], v[0])),
                     theta2=math.degrees(math.atan2(w[1], w[0])), color=OLIVE, lw=0.9))
    ax.text(0.70, 0.30, r"$\psi$", fontsize=8.5, color=OLIVE)
    bracketed_matrix(ax, 0.20, 0.87,
                     (r"$\cos\psi$", r"$-\sin\psi$", r"$\sin\psi$", r"$\cos\psi$"),
                     prefix=r"$R_\psi=$")
    ax.text(0.03, 0.05, r"$R_\psi^{\top}R_\psi=I_2$", transform=ax.transAxes, fontsize=8.5)
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(-0.35, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("one frame rotation")
    tag(ax, "a")

    rng = np.random.default_rng(20260725)
    cfg = SurrogateConfig(T=14, sigma=0.16, psi=2 * math.pi / 14)
    y, z = sample_clean_trajectories(rng, cfg, 3)
    for ax, data, title, letter in ((axes[1], z, r"laboratory frame $z_k$", "b"),
                                    (axes[2], y, r"co-rotating frame $y_k$", "c")):
        unit_circle(ax)
        for i in range(data.shape[0]):
            ax.plot(data[i, :, 0], data[i, :, 1], "-o", ms=1.8, lw=0.8, color=CYCLE[i])
            ax.plot(data[i, 0, 0], data[i, 0, 1], "o", ms=3.5, color=RED)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(title)
        tag(ax, letter)

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_ring_gauge.pdf")


# =====================================================================
def fig_ring_surrogate_response():
    """The exact surrogate response is a near-diagonal chain term plus a rank-one term."""
    rng = np.random.default_rng(7)
    cfg = SurrogateConfig()
    y, _ = sample_clean_trajectories(rng, cfg, 250)
    t = 0.20
    x = add_ou_noise_surrogate(rng, y, t)
    _, jac, pieces = exact_score_and_jacobian(x, np.arange(cfg.T), t, cfg, {})
    chain = np.abs(pieces["Cinv"])
    anchor = np.abs(np.outer(pieces["q"], pieces["q"]))
    full = np.mean(np.linalg.norm(jac, axis=(-2, -1)), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    panels = [
        (chain / chain.max(), r"chain term $|C_t^{-1}|$", "a"),
        (anchor / anchor.max(), r"anchor term $|qq^{\top}|$", "b"),
        (full / full.max(), r"full $\|J_{kj}\|_F$", "c"),
    ]
    for ax, (matrix, title, letter) in zip(axes, panels):
        image = ax.imshow(matrix, cmap="Greys", origin="upper", vmin=0.0, vmax=1.0)
        ax.set_xlabel(r"observed frame $j$")
        ax.set_title(title)
        ax.set_xticks([0, 10, 20, 29])
        ax.set_yticks([0, 10, 20, 29])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        tag(ax, letter)
    axes[0].set_ylabel(r"score block $k$")
    cbar = fig.colorbar(image, ax=axes, fraction=0.020, pad=0.02)
    cbar.set_label("normalised magnitude", fontsize=8)
    cbar.outline.set_linewidth(0.5)
    save(fig, "fig_ring_surrogate_response.pdf")


# =====================================================================
def fig_ring_diagnostics():
    """Reach rises to the finite-chain ceiling, intensity falls, the composite turns."""
    profiles = read_csv("polar", "baseline_response_profiles.csv")
    total = diagnostics_table(profiles, "fro_rms")
    radial = diagnostics_table(profiles, "rr_rms")
    tangential = diagnostics_table(profiles, "tt_rms")
    windows = read_csv("polar", "window_receptive_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15))

    ax = axes[0]
    ax.plot(radial["t"], radial["mean_lag"], "-o", ms=2.6, color=NAVY, label="radial")
    ax.plot(tangential["t"], tangential["mean_lag"], "-s", ms=2.6, color=RED,
            label="tangential")
    ceiling = float(total["flat_mean_lag"].iloc[0])
    ax.axhline(ceiling, color=GRAY, ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0, 6.4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\bar\ell$ (frames)")
    ax.set_title("reach")
    ax.legend(handlelength=1.4)
    tag(ax, "a")

    ax = axes[1]
    ax.plot(total["t"], total["intensity"], "-o", ms=2.6, color=NAVY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$I_{\mathrm{off}}$")
    ax.set_title("intensity")
    tag(ax, "b")

    ax = axes[2]
    ax.plot(total["t"], total["relative_reach"], "-o", ms=2.6, color=NAVY,
            label=r"$\widetilde\Xi$")
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\widetilde\Xi$ (frames)")
    ax.set_title("weighted reach")
    twin = ax.twinx()
    twin.plot(windows["t"], windows["total_L_5pct"], "--s", ms=2.6, color=RED,
              label=r"$L_{5\%}$")
    twin.set_ylabel(r"$L_{5\%}$", color=RED, fontsize=9)
    twin.tick_params(axis="y", labelcolor=RED, labelsize=8, width=0.6)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_linewidth(0.6)
    twin.set_ylim(-0.4, 10.5)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="lower left", handlelength=1.4)
    tag(ax, "c")

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_ring_diagnostics.pdf")


# =====================================================================
def _polar_reference(times):
    rng = np.random.default_rng(20260726)
    model = PolarRingModel(**BASELINE)
    solver = PolarGridSmoother(model,
                              np.linspace(0.52, 1.48, 31),
                              np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False))
    _, _, clean = model.simulate(rng)
    noise = rng.standard_normal(clean.shape)
    noisy = {}
    for t in times:
        attenuation, variance = ou_parameters(t)
        noisy[t] = attenuation * clean + math.sqrt(variance) * noise
    return model, solver, clean, noisy


def _draw_field(ax, xx, yy, log_density, field, title, observed, clean_point):
    n = xx.shape[0]
    density = log_density.reshape(n, n) - np.nanmax(log_density)
    arrows = field.reshape(n, n, 2)
    levels = np.linspace(max(-8.0, float(np.nanmin(density))), 0.0, 13)
    ax.contourf(xx, yy, density, levels=levels, cmap="Greys", alpha=0.55)
    magnitude = np.linalg.norm(arrows, axis=-1, keepdims=True)
    display = arrows / (1.0 + magnitude)
    step = 2
    ax.quiver(xx[::step, ::step], yy[::step, ::step],
              display[::step, ::step, 0], display[::step, ::step, 1],
              angles="xy", scale_units="xy", scale=2.9, width=0.006, color=NAVY)
    unit_circle(ax, alpha=0.7)
    ax.plot(*observed, "x", ms=4.5, mew=1.2, color=RED)
    ax.plot(*clean_point, "o", ms=3.2, color=OLIVE)
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(title)


def fig_ring_score_field():
    """The joint conditional score localises a phase; the marginal stays ring-symmetric."""
    times = (0.25, 0.70)
    model, solver, clean, noisy = _polar_reference(times)
    k = model.T // 2
    axis = np.linspace(-1.5, 1.5, 21)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.9), sharex=True, sharey=True)
    for column, t in enumerate(times):
        x = noisy[t]
        cavity = cavity_mass(solver, x, t, k)
        log_joint, _, joint = exact_score_slice(solver, x, t, k, points, cavity)
        log_marginal, marginal = marginal_score_slice(solver, t, k, points)
        _draw_field(axes[0, column], xx, yy, log_joint, joint,
                    rf"joint, $t={t:.2f}$", x[k], clean[k])
        _draw_field(axes[1, column], xx, yy, log_marginal, marginal,
                    rf"marginal, $t={t:.2f}$", x[k], clean[k])
    for ax in axes[1, :]:
        ax.set_xlabel(r"$x_k^{(1)}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$x_k^{(2)}$")
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "fig_ring_score_field.pdf")


# =====================================================================
def fig_ring_taylor():
    """The linearisation holds while the posterior angular spread stays small."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1))

    ax = axes[0]
    theta = 0.65
    e_r = np.array([math.cos(theta), math.sin(theta)])
    e_t = np.array([-math.sin(theta), math.cos(theta)])
    phis = np.linspace(-0.9, 0.9, 200)
    exact = np.column_stack((np.cos(theta + phis), np.sin(theta + phis)))
    first = e_r[None, :] + phis[:, None] * e_t[None, :]
    second = first - 0.5 * (phis ** 2)[:, None] * e_r[None, :]
    unit_circle(ax)
    ax.plot(*exact.T, color=NAVY, label="exact arc")
    ax.plot(*first.T, color=RED, ls="--", label="first order")
    ax.plot(*second.T, color=OLIVE, ls="-.", label="second order")
    ax.plot(*e_r, "o", ms=3.2, color="black")
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.9, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("expansion geometry")
    ax.legend(loc="lower left", handlelength=1.5)
    tag(ax, "a")

    ax = axes[1]
    grid = np.logspace(-2.1, 0.1, 100)
    exact_pts = np.column_stack((np.cos(grid), np.sin(grid)))
    err_first = np.linalg.norm(exact_pts - np.column_stack((np.ones_like(grid), grid)), axis=1)
    err_second = np.linalg.norm(
        exact_pts - np.column_stack((1 - 0.5 * grid ** 2, grid)), axis=1)
    ax.plot(grid, err_first, color=RED, label=r"first, $\simeq\varphi^2/2$")
    ax.plot(grid, err_second, color=OLIVE, ls="--", label=r"second, $\simeq\varphi^3/6$")
    ax.axhline(0.05, color=GRAY, lw=0.6)
    ax.axvline(math.sqrt(0.1), color=NAVY, ls=":", lw=0.8)
    ax.text(math.sqrt(0.1) * 1.12, 2e-6, r"$\varphi=0.32$", fontsize=7, color=NAVY,
            rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varphi$ (rad)")
    ax.set_ylabel(r"$\|a_k-\widehat a_k\|$")
    ax.set_title("truncation error")
    ax.legend(loc="lower right", handlelength=1.5)
    tag(ax, "b")

    ax = axes[2]
    taylor = read_csv("taylor", "taylor_vs_exact.csv")
    spread = read_csv("taylor", "angular_posterior_spread.csv")
    ax.plot(taylor["t"], taylor["central_relative_error"], "-o", ms=2.6, color=NAVY,
            label="central block")
    ax.plot(taylor["t"], taylor["trajectory_relative_error"], "-s", ms=2.6, color=RED,
            label="trajectory")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.15)
    ax.set_xlabel("$t$")
    ax.set_ylabel("relative RMS error")
    ax.set_title("measured accuracy")
    twin = ax.twinx()
    twin.plot(spread["t"], spread["circular_std"], "--^", ms=2.6, color=OLIVE,
              label="angular spread")
    twin.axhline(0.32, color=OLIVE, ls=":", lw=0.7)
    twin.set_ylabel("posterior s.d. (rad)", color=OLIVE, fontsize=9)
    twin.tick_params(axis="y", labelcolor=OLIVE, labelsize=8, width=0.6)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_linewidth(0.6)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", handlelength=1.5)
    tag(ax, "c")

    fig.tight_layout(w_pad=1.7)
    save(fig, "fig_ring_taylor.pdf")


FIGURES = [
    fig_ring_model,
    fig_ring_joint_vs_marginal,
    fig_ring_gauge,
    fig_ring_surrogate_response,
    fig_ring_diagnostics,
    fig_ring_score_field,
    fig_ring_taylor,
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for builder in FIGURES:
        if only and only not in builder.__name__:
            continue
        builder()


if __name__ == "__main__":
    main()
