#!/usr/bin/env python3
"""All figures of the note.

Seaborn is used for the statistical panels and matplotlib for vector fields,
geometry diagrams, and graphical models. Every panel is written as vector PDF
(for LaTeX) and PNG (for quick inspection). Figure file names carry the number
used in the document so that text and graphics cannot drift apart.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from metrics import diagnostics, diagnostics_table  # noqa: E402
from polar_analysis import (  # noqa: E402
    cavity_mass,
    exact_score_slice,
    linearized_slice,
    marginal_score_slice,
)
from polar_core import PolarGridSmoother, PolarRingModel, ou_parameters  # noqa: E402
from surrogate_model import (  # noqa: E402
    SurrogateConfig,
    add_ou_noise as add_ou_noise_surrogate,
    clean_score as surrogate_clean_score,
    exact_score_and_jacobian,
    rotation,
    sample_clean_trajectories,
)

FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"
FIGDIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1F4E79"
TEAL = "#2A7F62"
BRICK = "#B03A2E"
GOLD = "#B58900"
PURPLE = "#6C5B7B"
GRAY = "#6B7280"
LIGHT = "#EAF0F4"
PALETTE = [NAVY, TEAL, BRICK, GOLD, PURPLE, GRAY]

BASELINE = dict(kappa=2.0, D_r=0.015, omega=1.0, D_theta=0.005, T=20)


def setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=PALETTE,
        font="DejaVu Serif",
        font_scale=1.05,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 150,
        },
    )
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "mathtext.fontset": "dejavuserif",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def unit_circle(ax: plt.Axes, *, alpha: float = 0.8) -> None:
    angle = np.linspace(0, 2 * math.pi, 500)
    ax.plot(np.cos(angle), np.sin(angle), ls="--", lw=1.0, color=GRAY, alpha=alpha)


def panel_tag(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.02, 1.10, f"({letter})", transform=ax.transAxes,
        ha="right", va="top", fontsize=11, fontweight="bold", color=NAVY,
    )


def draw_matrix_2x2(
    ax: plt.Axes,
    x: float,
    y: float,
    entries: tuple[str, str, str, str],
    prefix: str = "",
    *,
    width: float = 0.36,
    height: float = 0.13,
    fontsize: float = 10.5,
    color: str = NAVY,
) -> None:
    """Draw a bracketed 2x2 matrix in axes coordinates.

    Matplotlib's mathtext has no matrix environment, so the brackets are drawn
    as line segments and the entries are placed individually.
    """
    kw = dict(transform=ax.transAxes, color=color, clip_on=False)
    if prefix:
        ax.text(x, y, prefix, ha="right", va="center", fontsize=fontsize, **kw)
    left, right = x + 0.015, x + 0.015 + width
    top, bottom = y + height, y - height
    tick = 0.035 * width / 0.36
    for edge, direction in ((left, +1), (right, -1)):
        ax.plot([edge, edge], [bottom, top], lw=1.1, **kw)
        ax.plot([edge, edge + direction * tick], [top, top], lw=1.1, **kw)
        ax.plot([edge, edge + direction * tick], [bottom, bottom], lw=1.1, **kw)
    cols = (left + 0.28 * width, left + 0.74 * width)
    rows = (y + 0.5 * height, y - 0.5 * height)
    for index, entry in enumerate(entries):
        ax.text(cols[index % 2], rows[index // 2], entry,
                ha="center", va="center", fontsize=fontsize, **kw)


# =====================================================================
# 1. The problem: polar coordinates, the confining potential, the goal
# =====================================================================
def fig01_problem_setup() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.6))

    # (a) polar coordinates and the moving frame
    ax = axes[0]
    theta = 0.95
    r = 1.18
    point = r * np.array([math.cos(theta), math.sin(theta)])
    unit_circle(ax)
    ax.plot([0, point[0]], [0, point[1]], color=NAVY, lw=1.6)
    ax.scatter(*point, s=60, color=NAVY, zorder=5)
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=math.degrees(theta),
                     color=BRICK, lw=1.4))
    ax.annotate("", xy=(0.95, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9, ls=":"))
    e_r = np.array([math.cos(theta), math.sin(theta)])
    e_t = np.array([-math.sin(theta), math.cos(theta)])
    ax.quiver(*point, *(0.42 * e_r), angles="xy", scale_units="xy", scale=1,
              color=TEAL, width=0.011, zorder=6)
    ax.quiver(*point, *(0.42 * e_t), angles="xy", scale_units="xy", scale=1,
              color=PURPLE, width=0.011, zorder=6)
    ax.text(*(point + 0.5 * e_r + np.array([0.02, 0.0])), r"$e_r$", color=TEAL, fontsize=11)
    ax.text(*(point + 0.5 * e_t), r"$e_\theta$", color=PURPLE, fontsize=11)
    ax.text(0.34, 0.58, r"$r$", color=NAVY, fontsize=12)
    ax.text(0.36, 0.10, r"$\theta$", color=BRICK, fontsize=12)
    ax.text(0.02, 0.03, r"$A=r(\cos\theta,\sin\theta)$", transform=ax.transAxes,
            fontsize=9.5, color=NAVY)
    ax.set_xlim(-0.35, 1.85)
    ax.set_ylim(-0.35, 1.85)
    ax.set_aspect("equal")
    ax.set_title("Polar coordinates")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    panel_tag(ax, "a")

    # (b) potential and the force it generates
    ax = axes[1]
    kappa = BASELINE["kappa"]
    grid = np.linspace(0.4, 1.6, 400)
    potential = 0.5 * kappa * (grid - 1.0) ** 2
    force = -kappa * (grid - 1.0)
    ax.plot(grid, potential, color=NAVY, lw=2.1, label=r"$V(r)=\frac{\kappa}{2}(r-1)^2$")
    ax.plot(grid, force, color=BRICK, lw=1.8, ls="--", label=r"force $-V'(r)=-\kappa(r-1)$")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axvline(1.0, color=GRAY, ls=":", lw=1.1)
    for r0 in (0.62, 1.40):
        direction = -kappa * (r0 - 1.0)
        ax.annotate("", xy=(r0 + 0.16 * np.sign(direction), 0), xytext=(r0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=BRICK, lw=1.8,
                                    mutation_scale=13))
    ax.text(1.02, 0.86, "stable radius", transform=ax.transAxes, rotation=90,
            va="top", ha="right", fontsize=8.5, color=GRAY)
    ax.set_title("Radial confinement")
    ax.set_xlabel("radius $r$")
    ax.set_ylabel("potential / force")
    ax.legend(fontsize=8.0, loc="lower left")
    panel_tag(ax, "b")

    # (c) one clean trajectory
    ax = axes[2]
    rng = np.random.default_rng(20260726)
    model = PolarRingModel(**BASELINE)
    _, _, clean = model.simulate(rng)
    unit_circle(ax)
    ax.plot(clean[:, 0], clean[:, 1], "-o", ms=3.6, lw=1.4, color=NAVY)
    ax.scatter(*clean[0], s=62, color=BRICK, zorder=5)
    ax.annotate(r"$A_0$", clean[0], textcoords="offset points", xytext=(6, 6),
                fontsize=10, color=BRICK)
    ax.annotate("", xy=clean[6], xytext=clean[3],
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.6, mutation_scale=13,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(0.03, 0.04, r"$\omega>0$: coherent turn", transform=ax.transAxes,
            fontsize=9.0, color=GOLD)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_title(f"One clean trajectory ($T={model.T}$)")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    panel_tag(ax, "c")

    # (d) the goal
    ax = axes[3]
    t_show = 0.35
    m, delta = ou_parameters(t_show)
    noisy = m * clean + math.sqrt(delta) * rng.standard_normal(clean.shape)
    centre = model.T // 2
    unit_circle(ax, alpha=0.5)
    ax.plot(noisy[:, 0], noisy[:, 1], "-o", ms=2.8, lw=0.9, color=GRAY, alpha=0.8)
    ax.scatter(*noisy[centre], s=100, facecolor="none", edgecolor=BRICK, lw=2.0, zorder=6)
    ax.annotate(r"$x_k$", noisy[centre], textcoords="offset points", xytext=(8, 6),
                fontsize=10, color=BRICK)
    for lag in range(1, 6):
        for j in (centre - lag, centre + lag):
            weight = 1.0 / lag
            ax.annotate(
                "", xy=noisy[centre], xytext=noisy[j],
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=0.5 + 1.9 * weight,
                                alpha=0.25 + 0.55 * weight, mutation_scale=9,
                                connectionstyle="arc3,rad=0.25"),
            )
    ax.text(0.03, 0.04,
            "which frames, and how strongly,\n"
            r"enter the block $S_k$?",
            transform=ax.transAxes, fontsize=8.6, color=PURPLE)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal")
    ax.set_title(rf"The question, at $t={t_show:.2f}$")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    panel_tag(ax, "d")

    fig.tight_layout()
    save(fig, "fig01_problem_setup")


# =====================================================================
# 2. The two clocks
# =====================================================================
def fig02_two_clocks() -> None:
    rng = np.random.default_rng(314159)
    model = PolarRingModel(**BASELINE)
    _, _, clean = model.simulate(rng)
    noise = rng.standard_normal(clean.shape)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), sharex=True, sharey=True)
    for ax, t in zip(axes, (0.0, 0.15, 0.50, 1.50)):
        if t == 0:
            x = clean
        else:
            m, delta = ou_parameters(t)
            x = m * clean + math.sqrt(delta) * noise
        unit_circle(ax, alpha=0.55)
        ax.plot(x[:, 0], x[:, 1], "-o", ms=3, lw=1.2, color=NAVY)
        ax.scatter(x[0, 0], x[0, 1], s=42, color=BRICK, zorder=4)
        ax.set_aspect("equal")
        ax.set_title(rf"$t={t:.2f}$")
        ax.set_xlabel("$x^{(1)}$")
    axes[0].set_ylabel("$x^{(2)}$")
    axes[0].text(0.03, 0.03, "internal time runs\nalong the polyline",
                 transform=axes[0].transAxes, fontsize=8.6, color=GRAY)
    fig.suptitle("Two clocks: internal time orders the frames, diffusion time corrupts them all at once",
                 y=1.04)
    fig.tight_layout()
    save(fig, "fig02_two_clocks")


# =====================================================================
# 3. Joint score versus per-frame marginal scores, as graphical models
# =====================================================================
def _chain_nodes(ax: plt.Axes, n: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.09, 0.91, n)
    y_top, y_bot = 0.80, 0.22
    rw, rh = 0.052, 0.115
    for i, x in enumerate(xs):
        ax.add_patch(Ellipse((x, y_top), rw * 2, rh * 2, facecolor="white",
                             edgecolor=NAVY, lw=1.5 if i != target else 2.4, zorder=3))
        ax.text(x, y_top, rf"$A_{{{i}}}$", ha="center", va="center", fontsize=10.5,
                color=NAVY, zorder=4)
        ax.add_patch(Ellipse((x, y_bot), rw * 2, rh * 2, facecolor="white",
                             edgecolor=TEAL, lw=1.5, zorder=3))
        ax.text(x, y_bot, rf"$X_{{{i}}}$", ha="center", va="center", fontsize=10.5,
                color=TEAL, zorder=4)
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i + 1] - rw - 0.006, y_top), xytext=(xs[i] + rw + 0.006, y_top),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3, mutation_scale=12))
    return xs, np.array([y_top, y_bot])


def fig03_score_graphs() -> None:
    n, target = 6, 2
    rh = 0.115
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # -- joint score: every observation reaches every clean frame
    ax = axes[0]
    xs, (y_top, y_bot) = _chain_nodes(ax, n, target)
    for i, x in enumerate(xs):
        ax.annotate("", xy=(x, y_bot + rh + 0.006), xytext=(x, y_top - rh - 0.006),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1, mutation_scale=11))
    for i, x in enumerate(xs):
        rad = 0.30 if i < target else (-0.30 if i > target else 0.0)
        ax.add_patch(FancyArrowPatch(
            (x, y_bot + rh + 0.01), (xs[target] - 0.012, y_top - rh - 0.012),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=12,
            color=PURPLE, lw=1.15, alpha=0.85, zorder=2,
        ))
    ax.set_title("Joint score", color=NAVY, fontsize=14, pad=14)
    ax.text(0.5, 0.035, r"$S_k(x,t)\ \propto\ \mathbb{E}[A_k \mid X_0,\ldots,X_{T-1}]$",
            ha="center", fontsize=11.5, color=PURPLE)
    ax.text(0.5, 0.955, "every observation can inform every clean frame",
            ha="center", fontsize=9.5, color=GRAY)

    # -- product of marginals: each frame is denoised on its own
    ax = axes[1]
    xs, (y_top, y_bot) = _chain_nodes(ax, n, target=-1)
    for x in xs:
        ax.annotate("", xy=(x, y_bot + rh + 0.006), xytext=(x, y_top - rh - 0.006),
                    arrowprops=dict(arrowstyle="<|-|>", color=BRICK, lw=1.5, mutation_scale=11))
    ax.set_title("Per-frame marginal scores", color=BRICK, fontsize=14, pad=14)
    ax.text(0.5, 0.035, r"$s_k^{\mathrm{marg}}(x_k,t)\ \propto\ \mathbb{E}[A_k \mid X_k]$",
            ha="center", fontsize=11.5, color=BRICK)
    ax.text(0.5, 0.955, "cross-frame information is discarded before denoising",
            ha="center", fontsize=9.5, color=GRAY)

    fig.tight_layout()
    save(fig, "fig03_score_graphs")


# =====================================================================
# 4. Constructing a diagnostic that sees range and intensity at once
# =====================================================================
def _synthetic_profiles(max_lag: int = 10) -> dict[str, np.ndarray]:
    lags = np.arange(0, max_lag + 1)
    short = np.exp(-(lags[1:] - 1) / 1.1)
    long_ = np.exp(-(lags[1:] - 1) / 6.0)
    short = short / short.sum()
    long_ = long_ / long_.sum()
    out = {
        "short, strong": np.concatenate([[1.0], 10.0 * short]),
        "short, weak": np.concatenate([[1.0], 1.0 * short]),
        "long, strong": np.concatenate([[1.0], 10.0 * long_]),
        "long, weak": np.concatenate([[1.0], 1.0 * long_]),
    }
    return {"lags": lags, **out}


def fig04_metric_construction() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    synth = _synthetic_profiles()
    lags = synth.pop("lags")
    colors = {"short, strong": NAVY, "short, weak": TEAL,
              "long, strong": BRICK, "long, weak": GOLD}

    ax = axes[0]
    for name, profile in synth.items():
        ax.plot(lags[1:], profile[1:], "-o", ms=4, lw=1.7, color=colors[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"lag $\ell$")
    ax.set_ylabel(r"$C(\ell)$")
    ax.set_title("Four reference profiles")
    ax.legend(fontsize=8.6, title="shape, amplitude", title_fontsize=8.6)
    panel_tag(ax, "a")

    ax = axes[1]
    records = []
    for name, profile in synth.items():
        d = diagnostics(lags, profile)
        records.append({"profile": name,
                        r"$\xi^{\rm norm}$": d.normalised_range,
                        r"$I_{\rm off}$": d.intensity,
                        r"$\bar\ell$": d.mean_lag,
                        r"$\Xi$": d.weighted_reach})
    frame = pd.DataFrame(records).set_index("profile")
    normalised = frame / frame.max()
    tidy = normalised.reset_index().melt(id_vars="profile", var_name="diagnostic",
                                         value_name="value (scaled)")
    sns.barplot(data=tidy, x="diagnostic", y="value (scaled)", hue="profile",
                palette=[colors[k] for k in frame.index], ax=ax)
    ax.set_title("Which diagnostic separates them")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.42)
    ax.legend(fontsize=8.2, ncol=2, loc="upper center")
    panel_tag(ax, "b")

    ax = axes[2]
    profiles = pd.read_csv(DATADIR / "polar/baseline_response_profiles.csv")
    total = diagnostics_table(profiles, "fro_rms")
    windows = pd.read_csv(DATADIR / "polar/window_receptive_summary.csv")
    ax.plot(total["t"], total["normalised_range"], "-o", ms=4, color=GRAY,
            label=r"$\xi^{\rm norm}$")
    ax.plot(windows["t"], windows["total_L_5pct"], "-s", ms=4.5, color=NAVY,
            label=r"$L_{5\%}$ (measured)")
    ax.axhline(total["flat_normalised_range"].iloc[0], color=GRAY, ls=":", lw=1.0)
    ax.text(0.98, total["flat_normalised_range"].iloc[0], "structureless ceiling",
            ha="right", va="bottom", fontsize=8.2, color=GRAY,
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 11.2)
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("frames")
    twin = ax.twinx()
    twin.plot(total["t"], total["relative_reach"], "-o", ms=4.5, color=BRICK, lw=2.3,
              label=r"$\widetilde\Xi$")
    twin.set_ylabel(r"weighted reach $\widetilde\Xi$", color=BRICK)
    twin.tick_params(axis="y", labelcolor=BRICK)
    twin.set_ylim(0, 2.9)
    twin.grid(False)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8.4, loc="lower left")
    ax.set_title("Measured, original polar model")
    panel_tag(ax, "c")

    fig.tight_layout()
    save(fig, "fig04_metric_construction")


# =====================================================================
# 5. The rotation matrix
# =====================================================================
def fig05_rotation() -> None:
    psi = math.radians(38.0)
    R = rotation(psi)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    # (a) action on a vector
    ax = axes[0]
    v = np.array([1.05, 0.28])
    w = R @ v
    unit_circle(ax, alpha=0.5)
    ax.quiver(0, 0, *v, angles="xy", scale_units="xy", scale=1, color=NAVY, width=0.012)
    ax.quiver(0, 0, *w, angles="xy", scale_units="xy", scale=1, color=BRICK, width=0.012)
    ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=math.degrees(math.atan2(v[1], v[0])),
                     theta2=math.degrees(math.atan2(w[1], w[0])), color=GOLD, lw=1.8))
    ax.text(*(1.06 * v + np.array([0.03, -0.06])), r"$z$", color=NAVY, fontsize=12)
    ax.text(*(1.06 * w + np.array([0.0, 0.04])), r"$R_\psi z$", color=BRICK, fontsize=12)
    ax.text(0.79, 0.30, r"$\psi$", color=GOLD, fontsize=12)
    draw_matrix_2x2(
        ax, 0.15, 0.86,
        (r"$\cos\psi$", r"$-\sin\psi$", r"$\sin\psi$", r"$\cos\psi$"),
        prefix=r"$R_\psi=$", width=0.32, height=0.10, fontsize=10.0,
    )
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylim(-0.4, 1.5)
    ax.set_aspect("equal")
    ax.set_title("A rotation by one frame angle")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    panel_tag(ax, "a")

    # (b) orthonormal columns
    ax = axes[1]
    unit_circle(ax, alpha=0.5)
    ax.quiver(0, 0, *R[:, 0], angles="xy", scale_units="xy", scale=1, color=TEAL, width=0.013)
    ax.quiver(0, 0, *R[:, 1], angles="xy", scale_units="xy", scale=1, color=PURPLE, width=0.013)
    ax.add_patch(Arc((0, 0), 0.34, 0.34,
                     theta1=math.degrees(math.atan2(R[1, 0], R[0, 0])),
                     theta2=math.degrees(math.atan2(R[1, 1], R[0, 1])),
                     color=GRAY, lw=1.2))
    ax.text(*(1.10 * R[:, 0] + np.array([0.0, 0.04])), "column 1", color=TEAL, fontsize=9.5)
    ax.text(*(1.10 * R[:, 1] + np.array([-0.42, 0.06])), "column 2", color=PURPLE, fontsize=9.5)
    ax.text(0.10, 0.24, r"$90^\circ$", color=GRAY, fontsize=9)
    ax.text(0.03, 0.05,
            r"$R_\psi^{\top}R_\psi=I_2,\quad \det R_\psi=1$" "\n"
            r"$\Rightarrow \|R_\psi z\|=\|z\|$ for every $z$",
            transform=ax.transAxes, fontsize=9.8, color=NAVY)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title("The columns are orthonormal")
    ax.set_xlabel("first coordinate")
    panel_tag(ax, "b")

    # (c) the isotropic Gaussian is unchanged
    ax = axes[2]
    rng = np.random.default_rng(11)
    sample = rng.standard_normal((450, 2))
    rotated = sample @ R.T
    ax.scatter(*sample.T, s=8, color=NAVY, alpha=0.30, label=r"$\eta\sim\mathcal{N}(0,I_2)$")
    ax.scatter(*rotated.T, s=8, color=BRICK, alpha=0.30, marker="^",
               label=r"$R_\psi\eta$")
    for radius in (1.0, 2.0, 3.0):
        ax.add_patch(Circle((0, 0), radius, fill=False, ls="--", lw=0.9, color=GRAY, alpha=0.85))
    radii = np.linalg.norm(sample, axis=1)
    highlighted = np.argsort(np.abs(radii - 2.4))[:3]
    for i in highlighted:
        ax.scatter(*sample[i], s=32, color=NAVY, zorder=6)
        ax.scatter(*rotated[i], s=32, color=BRICK, marker="^", zorder=6)
        ax.annotate("", xy=rotated[i], xytext=sample[i],
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.7, mutation_scale=12,
                                    connectionstyle="arc3,rad=0.35", shrinkA=4, shrinkB=4))
    ax.text(0.03, 0.04,
            "the level sets are circles,\nso every point stays on its own circle",
            transform=ax.transAxes, fontsize=8.6, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none",
                      alpha=0.85))
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Rotations preserve $\\mathcal{N}(0,I_2)$")
    ax.set_xlabel("first coordinate")
    ax.legend(fontsize=8.6, loc="upper right", frameon=True, facecolor="white",
              framealpha=0.88, edgecolor="none")
    panel_tag(ax, "c")

    fig.tight_layout()
    save(fig, "fig05_rotation")


# =====================================================================
# 6. The gauge: laboratory frame versus co-rotating frame
# =====================================================================
def fig06_gauge() -> None:
    rng = np.random.default_rng(20260725)
    cfg = SurrogateConfig(T=14, sigma=0.16, psi=2 * math.pi / 14)
    y, z = sample_clean_trajectories(rng, cfg, 4)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharex=True, sharey=True)
    for ax, data, title, colour in (
        (axes[0], z, r"laboratory frame: $Z_k$", TEAL),
        (axes[1], y, r"co-rotating frame: $Y_k=R_{-k\psi}Z_k$", GOLD),
    ):
        unit_circle(ax)
        for i in range(data.shape[0]):
            ax.plot(data[i, :, 0], data[i, :, 1], "-o", ms=2.8, lw=1.15, color=colour, alpha=0.9)
            ax.scatter(data[i, 0, 0], data[i, 0, 1], s=48, color=BRICK, zorder=5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("first coordinate")
    axes[0].set_ylabel("second coordinate")
    axes[0].text(0.03, 0.03, "coherent circulation", transform=axes[0].transAxes,
                 fontsize=9, color=GRAY)
    axes[1].text(0.03, 0.03, "a plain random walk from a ring anchor",
                 transform=axes[1].transAxes, fontsize=9, color=GRAY)
    fig.suptitle(r"$U_\psi$ is orthogonal, so the de-rotation commutes with the isotropic channel",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig06_gauge")


# =====================================================================
# 7. The clean surrogate score is local
# =====================================================================
def fig07_clean_score() -> None:
    cfg = SurrogateConfig(T=7, psi=2 * math.pi / 7)
    path = np.array([
        [1.03, -0.02], [1.04, 0.05], [1.02, 0.10], [1.28, -0.16],
        [1.00, 0.17], [0.97, 0.21], [0.94, 0.24],
    ], dtype=float)[None, ...]
    score = surrogate_clean_score(path, cfg)[0]
    magnitude = np.linalg.norm(score, axis=1)
    # The score spans nearly two decades, so arrow lengths are square-root
    # compressed (directions and the ordering of magnitudes are preserved) and the
    # longest arrow is fixed at 0.13 data units. Panel (b) carries the true values.
    display = score / magnitude[:, None] * np.sqrt(magnitude)[:, None]
    scale = np.sqrt(magnitude).max() / 0.13

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), constrained_layout=True)
    ax = axes[0]
    unit_circle(ax)
    ax.plot(path[0, :, 0], path[0, :, 1], "-o", color=NAVY, lw=1.6, ms=5.0)
    ax.quiver(path[0, :, 0], path[0, :, 1], display[:, 0], display[:, 1],
              angles="xy", scale_units="xy", scale=scale, color=BRICK, width=0.008)
    for k in (0, 3):
        ax.annotate(rf"$y_{{{k}}}$", path[0, k], textcoords="offset points",
                    xytext=(-16, -4), fontsize=10, color=NAVY)
    ax.set_aspect("equal")
    ax.set_xlim(0.86, 1.40)
    ax.set_ylim(-0.24, 0.30)
    ax.set_title("Clean score in the co-rotating frame")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    ax.annotate("one displaced frame", xy=path[0, 3], xytext=(1.13, 0.24),
                arrowprops=dict(arrowstyle="->", color=GRAY), color=GRAY, fontsize=9)
    panel_tag(ax, "a")

    ax = axes[1]
    frames = np.arange(cfg.T)
    ax.vlines(frames, 1e-2, magnitude, color=GRAY, lw=1.4)
    ax.plot(frames, magnitude, "o", ms=7, color=BRICK)
    ax.set_yscale("log")
    ax.set_xticks(frames)
    ax.set_ylim(2e-1, 1e2)
    ax.set_xlabel("frame $k$")
    ax.set_ylabel(r"$\|S_k(y,0)\|$")
    ax.set_title("The clean score is local")
    ax.annotate("displaced frame\nand its two neighbours", xy=(3, magnitude[3]),
                xytext=(3.4, 60), fontsize=8.6, color=BRICK,
                arrowprops=dict(arrowstyle="->", color=BRICK, lw=1.0))
    ax.annotate("ring anchor", xy=(0, magnitude[0]), xytext=(0.15, 8.0),
                fontsize=8.6, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0))
    panel_tag(ax, "b")

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.array([0.10, 0.32, 0.54, 0.76, 0.95])
    ys = np.array([0.52, 0.60, 0.30, 0.61, 0.53])
    for i in range(4):
        ax.plot(xs[i:i + 2], ys[i:i + 2], color=GRAY, lw=2.1)
    ax.scatter(xs, ys, s=130, color=NAVY, edgecolor="white", zorder=3)
    for i, (x0, y0) in enumerate(zip(xs, ys)):
        label = [r"$y_{k-2}$", r"$y_{k-1}$", r"$y_k$", r"$y_{k+1}$", r"$y_{k+2}$"][i]
        ax.text(x0, y0 - 0.11, label, ha="center", fontsize=10)
    ax.annotate("", xy=(xs[2], ys[2]), xytext=(xs[1], ys[1]),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.0, mutation_scale=13))
    ax.annotate("", xy=(xs[2], ys[2]), xytext=(xs[3], ys[3]),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.0, mutation_scale=13))
    ax.text(0.5, 0.88, "only the two incident edges contain $y_k$",
            ha="center", fontsize=9.5, color=TEAL)
    ax.text(0.5, 0.07, r"$S_k(y,0)=\dfrac{y_{k-1}-2y_k+y_{k+1}}{\sigma^2}$",
            ha="center", fontsize=11.5, color=BRICK)
    ax.set_title("Why: a second difference")
    panel_tag(ax, "c")
    save(fig, "fig07_clean_score")


# =====================================================================
# 8. Exact decomposition of the surrogate response
# =====================================================================
def fig08_surrogate_response() -> None:
    rng = np.random.default_rng(7)
    cfg = SurrogateConfig()
    y, _ = sample_clean_trajectories(rng, cfg, 250)
    t = 0.20
    x = add_ou_noise_surrogate(rng, y, t)
    _, jac, pieces = exact_score_and_jacobian(x, np.arange(cfg.T), t, cfg, {})
    chain = np.abs(pieces["Cinv"])
    anchor = np.abs(np.outer(pieces["q"], pieces["q"]))
    full = np.mean(np.linalg.norm(jac, axis=(-2, -1)), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.85))
    for ax, mat, title, cmap, letter in zip(
        axes,
        [chain, anchor / anchor.max(), full],
        [r"random-walk term $|C_t^{-1}|$",
         r"anchor coefficient $|qq^{\top}|$ (scaled)",
         r"full block response $\|J_{kj}\|_F$"],
        ["mako", "crest", "rocket_r"],
        "abc",
    ):
        sns.heatmap(mat, cmap=cmap, square=True, cbar_kws={"shrink": 0.75}, ax=ax,
                    xticklabels=5, yticklabels=5)
        ax.set_title(title)
        ax.set_xlabel("observed frame $j$")
        ax.set_ylabel("score block $k$")
        panel_tag(ax, letter)
    fig.suptitle(rf"Exact surrogate response at $t={t:.2f}$: near-diagonal chain term plus a rank-one anchor term",
                 y=1.04)
    fig.tight_layout()
    save(fig, "fig08_surrogate_response")


# =====================================================================
# 9. Surrogate diagnostics
# =====================================================================
def fig09_surrogate_diagnostics() -> None:
    summary = pd.read_csv(DATADIR / "surrogate/summary.csv")
    summary["relative_rmse"] = np.sqrt(summary["joint_vs_marginal_relative_mse"])
    profiles = pd.read_csv(DATADIR / "surrogate/influence_profiles.csv")
    windows = pd.read_csv(DATADIR / "surrogate/window_errors_full.csv")
    diag = diagnostics_table(profiles, "mean_block_norm")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))

    ax = axes[0, 0]
    sns.lineplot(data=summary, x="t", y="relative_rmse", marker="o", color=BRICK, ax=ax)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cost of deleting temporal context")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("relative RMS score error")
    panel_tag(ax, "a")

    ax = axes[0, 1]
    selected = [0.02, 0.10, 0.40, 0.70, 2.00]
    subset = profiles[profiles["t"].isin(selected)].copy()
    subset["diffusion time"] = subset["t"].map(lambda v: rf"$t={v:g}$")
    sns.lineplot(data=subset[subset["lag"] >= 1], x="lag", y="mean_block_norm",
                 hue="diffusion time", marker="o", markersize=3.2, palette="viridis", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Response by temporal lag")
    ax.set_xlabel(r"lag $\ell=|j-k|$")
    ax.set_ylabel(r"mean $\|J_{kj}\|_F$")
    panel_tag(ax, "b")

    ax = axes[1, 0]
    ax.plot(diag["t"], diag["intensity"], "-o", ms=4, color=NAVY, label=r"$I_{\rm off}$")
    ax.plot(diag["t"], diag["relative_reach"], "-o", ms=4, color=BRICK,
            label=r"$\widetilde\Xi$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Intensity falls, weighted reach peaks")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("diagnostic value")
    ax.legend(fontsize=9)
    panel_tag(ax, "c")

    ax = axes[1, 1]
    windows = windows.copy()
    windows["diffusion time"] = windows["t"].map(lambda v: rf"$t={v:g}$")
    sns.lineplot(data=windows, x="window_radius", y="relative_rmse",
                 hue="diffusion time", marker="o", markersize=3.5,
                 palette=[NAVY, TEAL, GOLD, PURPLE], ax=ax)
    ax.set_yscale("log")
    ax.axhline(0.05, color=GRAY, ls="--", lw=1)
    ax.text(0.99, 0.052, r"$5\%$", ha="right", va="bottom", fontsize=8.5, color=GRAY,
            transform=ax.get_yaxis_transform())
    ax.set_title("Functional receptive field")
    ax.set_xlabel("window radius $L$")
    ax.set_ylabel("relative RMS score error")
    panel_tag(ax, "d")

    fig.suptitle("Exactly solvable surrogate: the four diagnostics", y=1.01)
    fig.tight_layout()
    save(fig, "fig09_surrogate_diagnostics")


# =====================================================================
# 10. One-step transition kernels of the polar model
# =====================================================================
def wrapped_normal_density(delta: np.ndarray, var: float, wraps: int = 5) -> np.ndarray:
    out = np.zeros_like(delta, dtype=float)
    for n in range(-wraps, wraps + 1):
        out += np.exp(-0.5 * (delta + 2 * math.pi * n) ** 2 / var)
    return out / math.sqrt(2 * math.pi * var)


def fig10_transition_kernels() -> None:
    model = PolarRingModel(**BASELINE)
    h = model.h
    a = math.exp(-model.kappa * h)
    q_r = (model.D_r / model.kappa) * (1 - a * a)
    q_theta = 2 * model.D_theta * h

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    grid = np.linspace(0.55, 1.45, 600)
    for i, r0 in enumerate((0.75, 1.00, 1.25)):
        mean = 1 + a * (r0 - 1)
        axes[0].plot(grid, norm.pdf(grid, mean, math.sqrt(q_r)), lw=2.0,
                     color=PALETTE[i], label=rf"$r_k={r0:.2f}$")
        axes[0].axvline(mean, color=PALETTE[i], ls=":", lw=1)
    axes[0].axvline(1.0, color=GRAY, ls="--", lw=1)
    axes[0].set_title("Exact radial transition")
    axes[0].set_xlabel(r"$r_{k+1}$")
    axes[0].set_ylabel("conditional density")
    axes[0].legend(title="current radius", fontsize=9, title_fontsize=9)
    axes[0].text(0.02, 0.97, rf"mean $=1+a(r_k-1)$, $a={a:.3f}$" "\n"
                             rf"variance $q_r={q_r:.2e}$",
                 transform=axes[0].transAxes, va="top", fontsize=9, color=GRAY)
    panel_tag(axes[0], "a")

    delta = np.linspace(-math.pi, math.pi, 700)
    axes[1].plot(delta, wrapped_normal_density(delta, q_theta), color=TEAL, lw=2.1)
    axes[1].axvline(0, color=GRAY, ls="--", lw=1)
    axes[1].set_title("Angular innovation, wrapped to the circle")
    axes[1].set_xlabel(r"$\delta=\theta_{k+1}-\theta_k-\omega h$")
    axes[1].set_ylabel("wrapped-normal density")
    axes[1].set_xticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                       [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"])
    axes[1].text(0.02, 0.97, rf"variance $q_\theta=2D_\theta h={q_theta:.2e}$" "\n"
                             r"one winding dominates here",
                 transform=axes[1].transAxes, va="top", fontsize=9, color=GRAY)
    panel_tag(axes[1], "b")
    fig.tight_layout()
    save(fig, "fig10_transition_kernels")


# =====================================================================
# Shared machinery for the polar score-field figures
# =====================================================================
def make_polar_solver(nr: int = 31, nt: int = 64):
    model = PolarRingModel(**BASELINE)
    solver = PolarGridSmoother(
        model,
        np.linspace(0.52, 1.48, nr),
        np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False),
    )
    return model, solver


def draw_field(ax, xx, yy, logd, scores, title, observed=None, clean=None) -> None:
    n = xx.shape[0]
    density = logd.reshape(n, n).copy()
    field = scores.reshape(n, n, 2)
    density -= np.nanmax(density)
    levels = np.linspace(max(-8, float(np.nanmin(density))), 0, 18)
    ax.contourf(xx, yy, density, levels=levels, cmap="mako")
    magnitude = np.linalg.norm(field, axis=-1, keepdims=True)
    display = field / (1.0 + magnitude)
    stride = 2
    ax.quiver(xx[::stride, ::stride], yy[::stride, ::stride],
              display[::stride, ::stride, 0], display[::stride, ::stride, 1],
              angles="xy", scale_units="xy", scale=1.18, width=0.0042,
              color="white", alpha=0.88)
    unit_circle(ax, alpha=0.7)
    if observed is not None:
        ax.scatter(*observed, marker="x", s=46, color=GOLD, lw=2, zorder=4)
    if clean is not None:
        ax.scatter(*clean, marker="o", s=32, color=BRICK, edgecolor="white", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel(r"candidate $x_k^{(1)}$")
    ax.set_ylabel(r"candidate $x_k^{(2)}$")


def polar_reference_sample(times=(0.05, 0.25, 0.70, 1.50)):
    rng = np.random.default_rng(20260726)
    model, solver = make_polar_solver()
    _, theta, clean = model.simulate(rng)
    noise = rng.standard_normal(clean.shape)
    noisy = {}
    for t in times:
        m, delta = ou_parameters(t)
        noisy[t] = m * clean + math.sqrt(delta) * noise
    return model, solver, theta, clean, noisy


def fig11_joint_score_fields() -> None:
    model, solver, _, clean, noisy = polar_reference_sample()
    k = model.T // 2
    axis = np.linspace(-1.55, 1.55, 23)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), sharex=True, sharey=True)
    for ax, (t, x) in zip(axes, noisy.items()):
        cav = cavity_mass(solver, x, t, k)
        logd, _, score = exact_score_slice(solver, x, t, k, points, cav)
        draw_field(ax, xx, yy, logd, score, rf"$t={t:.2f}$", x[k], clean[k])
    fig.suptitle("A two-dimensional conditional slice of the joint score, central frame", y=1.03)
    fig.tight_layout()
    save(fig, "fig11_joint_score_fields")


def fig12_joint_vs_marginal_fields() -> None:
    model, solver, _, clean, noisy = polar_reference_sample()
    k = model.T // 2
    axis = np.linspace(-1.55, 1.55, 23)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.1), sharex=True, sharey=True)
    for col, t in enumerate((0.25, 0.70)):
        x = noisy[t]
        cav = cavity_mass(solver, x, t, k)
        log_joint, _, joint = exact_score_slice(solver, x, t, k, points, cav)
        log_marg, marg = marginal_score_slice(solver, t, k, points)
        draw_field(axes[0, col], xx, yy, log_joint, joint, rf"joint, $t={t:.2f}$", x[k], clean[k])
        draw_field(axes[1, col], xx, yy, log_marg, marg, rf"marginal, $t={t:.2f}$", x[k], clean[k])
    fig.suptitle("The joint score localises a phase; the one-frame marginal keeps the ring symmetry",
                 y=1.01)
    fig.tight_layout()
    save(fig, "fig12_joint_vs_marginal_fields")


# =====================================================================
# 13. The Taylor expansion, and when it is allowed
# =====================================================================
def angular_posterior_spread(times) -> pd.DataFrame:
    """Circular standard deviation of the smoothed angular posterior."""
    model, solver = make_polar_solver(nr=41, nt=128)
    rng = np.random.default_rng(4242)
    k = model.T // 2
    records = []
    for t in times:
        spreads = []
        for _ in range(6):
            _, _, clean = model.simulate(rng)
            m, delta = ou_parameters(t)
            x = m * clean + math.sqrt(delta) * rng.standard_normal(clean.shape)
            belief = solver.posterior_beliefs(x, t)[k]
            angular = belief.sum(axis=0)
            angular = angular / angular.sum()
            resultant = abs(np.sum(angular * np.exp(1j * solver.theta_grid)))
            resultant = min(max(resultant, 1e-12), 1 - 1e-12)
            spreads.append(math.sqrt(-2.0 * math.log(resultant)))
        records.append({"t": t, "circular_std": float(np.mean(spreads))})
    return pd.DataFrame.from_records(records)


def fig13_taylor() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    # (a) geometry of the expansion
    ax = axes[0]
    theta = 0.65
    e_r = np.array([math.cos(theta), math.sin(theta)])
    e_t = np.array([-math.sin(theta), math.cos(theta)])
    phis = np.linspace(-0.95, 0.95, 300)
    exact = np.column_stack((np.cos(theta + phis), np.sin(theta + phis)))
    first = e_r[None, :] + phis[:, None] * e_t[None, :]
    second = first - 0.5 * (phis**2)[:, None] * e_r[None, :]
    unit_circle(ax)
    ax.plot(*exact.T, color=NAVY, lw=2.2, label="exact arc")
    ax.plot(*first.T, color=BRICK, lw=1.8, label="first order")
    ax.plot(*second.T, color=GOLD, lw=1.5, ls="--", label="second order")
    ax.scatter(*e_r, s=45, color=TEAL, zorder=5)
    ax.quiver([e_r[0]] * 2, [e_r[1]] * 2, [0.32 * e_r[0], 0.32 * e_t[0]],
              [0.32 * e_r[1], 0.32 * e_t[1]], angles="xy", scale_units="xy", scale=1,
              color=[TEAL, PURPLE], width=0.008)
    ax.set_aspect("equal")
    ax.set_title("Expanding around one rotating branch")
    ax.set_xlabel("first coordinate")
    ax.set_ylabel("second coordinate")
    ax.legend(fontsize=8.6, loc="lower left")
    panel_tag(ax, "a")

    # (b) truncation error is quadratic, and the posterior spread decides
    ax = axes[1]
    grid = np.logspace(-2.2, 0.15, 120)
    exact_pts = np.column_stack((np.cos(grid), np.sin(grid)))
    first_pts = np.column_stack((np.ones_like(grid), grid))
    err_first = np.linalg.norm(exact_pts - first_pts, axis=1)
    second_pts = np.column_stack((1 - 0.5 * grid**2, grid))
    err_second = np.linalg.norm(exact_pts - second_pts, axis=1)
    ax.plot(grid, err_first, color=BRICK, lw=2.0,
            label=r"first order, $\simeq\varphi^2/2$")
    ax.plot(grid, err_second, color=GOLD, lw=1.8, ls="--",
            label=r"second order, $\simeq\varphi^3/6$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(0.05, color=GRAY, lw=0.9)
    ax.text(0.0075, 0.062, r"$5\%$ of the ring radius", fontsize=8.4, color=GRAY)
    ax.axvline(math.sqrt(0.1), color=TEAL, ls="-.", lw=1.2)
    ax.text(math.sqrt(0.1) * 1.09, 3e-6, r"$\varphi\approx0.32$", fontsize=8.6, color=TEAL,
            rotation=90)
    ax.set_xlabel(r"angular deviation $\varphi$ (rad)")
    ax.set_ylabel(r"$\|A-\widehat A\|$")
    ax.set_title("Truncation error of the linearisation")
    ax.legend(fontsize=8.6, loc="upper left")
    panel_tag(ax, "b")

    # (c) measured error against the measured posterior spread
    ax = axes[2]
    metrics = pd.read_csv(DATADIR / "taylor/taylor_vs_exact.csv")
    spread = angular_posterior_spread(tuple(metrics["t"].tolist()))
    spread.to_csv(DATADIR / "taylor/angular_posterior_spread.csv", index=False)
    ax.plot(metrics["t"], metrics["central_relative_error"], "-o", ms=4.5, color=NAVY,
            label="central block")
    ax.plot(metrics["t"], metrics["trajectory_relative_error"], "-o", ms=4.5, color=BRICK,
            label="full trajectory")
    ax.set_xscale("log")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("relative RMS score error")
    ax.set_ylim(0, 0.15)
    twin = ax.twinx()
    twin.plot(spread["t"], spread["circular_std"], "-s", ms=4.5, color=TEAL,
              label="posterior angular spread")
    twin.axhline(0.32, color=TEAL, ls="-.", lw=1.1)
    twin.set_ylabel("posterior circular s.d. (rad)", color=TEAL)
    twin.tick_params(axis="y", labelcolor=TEAL)
    twin.grid(False)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8.4, loc="upper left")
    ax.set_title("Measured accuracy and its cause")
    panel_tag(ax, "c")

    fig.tight_layout()
    save(fig, "fig13_taylor")


# =====================================================================
# 14-17. Original-model measurements
# =====================================================================
def fig14_response_profiles() -> None:
    frame = pd.read_csv(DATADIR / "polar/baseline_response_profiles.csv")
    selected = [0.03, 0.10, 0.40, 0.70, 1.50]
    subset = frame[frame["t"].isin(selected) & (frame["lag"] >= 1)].copy()
    subset["diffusion time"] = subset["t"].map(lambda v: rf"$t={v:g}$")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.25), sharey=True)
    sns.lineplot(data=subset, x="lag", y="rr_rms", hue="diffusion time", marker="o",
                 markersize=3.2, palette="viridis", ax=axes[0])
    sns.lineplot(data=subset, x="lag", y="tt_rms", hue="diffusion time", marker="o",
                 markersize=3.2, palette="viridis", ax=axes[1], legend=False)
    for ax, title, letter in zip(axes, ["Radial channel", "Tangential channel"], "ab"):
        ax.set_yscale("log")
        ax.set_xlabel(r"temporal lag $\ell=|j-k|$")
        ax.set_title(title)
        panel_tag(ax, letter)
    axes[0].set_ylabel("RMS projected response")
    axes[0].legend(loc="lower left", fontsize=8.4, title="diffusion time",
                   title_fontsize=8.4, ncol=2)
    fig.suptitle(r"Response of score block $k$ to perturbing the observation at frame $j$", y=1.03)
    fig.tight_layout()
    save(fig, "fig14_response_profiles")


def fig15_range_intensity() -> None:
    profiles = pd.read_csv(DATADIR / "polar/baseline_response_profiles.csv")
    radial = diagnostics_table(profiles, "rr_rms")
    tangential = diagnostics_table(profiles, "tt_rms")
    total = diagnostics_table(profiles, "fro_rms")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    ax = axes[0]
    ax.plot(radial["t"], radial["mean_lag"], "-o", ms=4, color=NAVY, label="radial")
    ax.plot(tangential["t"], tangential["mean_lag"], "-o", ms=4, color=TEAL, label="tangential")
    ax.axhline(total["flat_mean_lag"].iloc[0], color=GRAY, ls=":", lw=1.1)
    ax.text(0.02, total["flat_mean_lag"].iloc[0] + 0.12, "structureless ceiling",
            ha="left", va="bottom", fontsize=8.2, color=GRAY,
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 6.2)
    ax.set_title(r"Reach: weighted mean lag $\bar\ell$")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("frames")
    ax.legend(fontsize=9)
    panel_tag(ax, "a")

    ax = axes[1]
    ax.plot(total["t"], total["intensity"], "-o", ms=4, color=BRICK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Strength: intensity $I_{\rm off}$")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel(r"$\sum_{\ell\geq1} C_t(\ell)$")
    panel_tag(ax, "b")

    ax = axes[2]
    ax.plot(radial["t"], radial["relative_reach"], "-o", ms=4, color=NAVY, label="radial")
    ax.plot(tangential["t"], tangential["relative_reach"], "-o", ms=4, color=TEAL,
            label="tangential")
    ax.plot(total["t"], total["relative_reach"], "-o", ms=4, color=BRICK, lw=2.2,
            label="total")
    peak = total.loc[total["relative_reach"].idxmax()]
    ax.axvline(peak["t"], color=GRAY, ls="--", lw=1.0)
    ax.annotate(rf"peak at $t={peak['t']:g}$", xy=(peak["t"], peak["relative_reach"]),
                xytext=(8, 8), textcoords="offset points", fontsize=8.6, color=GRAY)
    ax.set_xscale("log")
    ax.set_title(r"Both at once: $\widetilde\Xi=I_{\rm off}\bar\ell / C_t(0)$")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("frames")
    ax.legend(fontsize=9)
    panel_tag(ax, "c")

    fig.suptitle("Reach and strength move in opposite directions; the weighted diagnostic is non-monotone",
                 y=1.04)
    fig.tight_layout()
    save(fig, "fig15_range_intensity")


def fig16_receptive_field() -> None:
    windows = pd.read_csv(DATADIR / "polar/window_receptive_field.csv")
    summary = pd.read_csv(DATADIR / "polar/window_receptive_summary.csv")
    selected = [0.03, 0.10, 0.40, 0.70, 1.50, 2.00]
    subset = windows[windows["t"].isin(selected)].copy()
    subset["diffusion time"] = subset["t"].map(lambda v: rf"$t={v:g}$")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    for ax, column, title, legend, letter in (
        (axes[0], "radial_relative_error", "Radial component", True, "a"),
        (axes[1], "tangential_relative_error", "Tangential component", False, "b"),
    ):
        sns.lineplot(data=subset, x="radius", y=column, hue="diffusion time", marker="o",
                     markersize=3.0, palette="viridis", ax=ax, legend=legend)
        ax.set_yscale("log")
        ax.axhline(0.05, color=GRAY, ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("window radius $L$")
        ax.set_ylabel("relative RMS error")
        panel_tag(ax, letter)

    ax = axes[2]
    ax.plot(summary["t"], summary["radial_L_5pct"], "-o", ms=4, color=NAVY, label="radial")
    ax.plot(summary["t"], summary["tangential_L_5pct"], "-o", ms=4, color=TEAL,
            label="tangential")
    ax.plot(summary["t"], summary["total_L_5pct"], "-o", ms=4, color=BRICK, label="total")
    ax.set_xscale("log")
    ax.set_ylim(-0.4, 10.4)
    ax.set_title(r"Radius needed for $5\%$ accuracy")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel(r"minimum radius $L_{5\%}$")
    ax.legend(fontsize=9)
    panel_tag(ax, "c")
    fig.suptitle("Functional receptive field of the joint score", y=1.04)
    fig.tight_layout()
    save(fig, "fig16_receptive_field")


def fig17_parameter_sweeps() -> None:
    frame = pd.read_csv(DATADIR / "polar/parameter_sweeps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    for ax, sweep, xcol, ycol, xlabel, title, letter in (
        (axes[0], "kappa", "parameter", "rr_length", r"confinement $\kappa$",
         "Radial reach versus confinement", "a"),
        (axes[1], "D_theta", "parameter", "tt_length", r"angular diffusivity $D_\theta$",
         "Tangential reach versus angular noise", "b"),
    ):
        subset = frame[frame["sweep"] == sweep].copy()
        subset["diffusion time"] = subset["t"].map(lambda v: rf"$t={v:g}$")
        sns.lineplot(data=subset, x=xcol, y=ycol, hue="diffusion time", marker="o",
                     palette="viridis", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"normalised range $\xi^{\rm norm}$ (frames)")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8.2, title="diffusion time",
                  title_fontsize=8.2, ncol=2)
        panel_tag(ax, letter)
    axes[1].set_xscale("log")
    fig.suptitle("The clean one-step dynamics set the low-noise response channels", y=1.03)
    fig.tight_layout()
    save(fig, "fig17_parameter_sweeps")


def fig18_validation() -> None:
    frame = pd.read_csv(DATADIR / "polar/fluctuation_response_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.0))
    sns.scatterplot(data=frame, x="fd_norm", y="covariance_identity_norm", hue="t",
                    style="lag", palette="viridis", s=70, ax=axes[0])
    lo = min(frame["fd_norm"].min(), frame["covariance_identity_norm"].min())
    hi = max(frame["fd_norm"].max(), frame["covariance_identity_norm"].max())
    axes[0].plot([lo, hi], [lo, hi], ls="--", color=GRAY, lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Two independent evaluations agree")
    axes[0].set_xlabel(r"finite-difference $\|J_{kj}\|_F$")
    axes[0].set_ylabel(r"posterior-covariance $\|J_{kj}\|_F$")
    panel_tag(axes[0], "a")

    subset = frame.copy()
    subset["diffusion time"] = subset["t"].map(lambda v: rf"$t={v:g}$")
    sns.lineplot(data=subset, x="lag", y="relative_error", hue="diffusion time",
                 marker="o", palette=[NAVY, TEAL, BRICK], ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("Relative discrepancy")
    axes[1].set_xlabel("temporal lag")
    axes[1].set_ylabel("relative error")
    panel_tag(axes[1], "b")
    fig.suptitle("Numerical check of the posterior response identity", y=1.04)
    fig.tight_layout()
    save(fig, "fig18_validation")


FIGURES = [
    fig01_problem_setup,
    fig02_two_clocks,
    fig03_score_graphs,
    fig04_metric_construction,
    fig05_rotation,
    fig06_gauge,
    fig07_clean_score,
    fig08_surrogate_response,
    fig09_surrogate_diagnostics,
    fig10_transition_kernels,
    fig11_joint_score_fields,
    fig12_joint_vs_marginal_fields,
    fig13_taylor,
    fig14_response_profiles,
    fig15_range_intensity,
    fig16_receptive_field,
    fig17_parameter_sweeps,
    fig18_validation,
]


def main(only: str | None = None) -> None:
    setup_style()
    print("writing figures:")
    for builder in FIGURES:
        if only and only not in builder.__name__:
            continue
        builder()
    print(f"done -> {FIGDIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
