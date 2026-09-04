#!/usr/bin/env python3
"""Rebuild the thesis figures into ``overleaf/thesis/figures``.

Writes only inside the thesis folder. ``overleaf/shared/figures`` is left alone
so the paper, the workshop note and the compendium keep the figures they were
built against.

    python3 tools/make_thesis_figures.py [name ...]

With no arguments it rebuilds everything it knows how to build.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BLUE, FULL, GREEN, GREY, VERM,  # noqa: E402
                      label_lines, new_figure, ramp, save)

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"
OUT = THESIS.parents[1] / "research" / "nongaussian-bp" / "outputs"


def read(rel: str) -> list[dict]:
    p = OUT / rel
    if not p.exists():
        raise FileNotFoundError(f"missing committed output: {p}")
    with p.open() as fh:
        return list(csv.DictReader(fh))


def col(rows, name, cast=float):
    return [cast(r[name]) for r in rows]


# ===========================================================================
# Chapter 2 — the three scores, memorisation, and the speciation cascade
# ===========================================================================

def fig_forward_corruption() -> None:
    """One clean AR(1) trajectory, and the same trajectory at four noise levels.

    The reader should see the problem before reading a description of it.
    """
    rng = np.random.default_rng(0)
    alpha, L = 0.85, 60
    a = np.empty(L)
    a[0] = rng.normal(0, 1)
    for k in range(1, L):
        a[k] = alpha * a[k - 1] + np.sqrt(1 - alpha**2) * rng.normal()

    times = [0.0, 0.25, 0.75, 2.0]
    fig, axes = new_figure(1, 4, width=FULL, height=1.85, sharey=True)
    cols = ramp(len(times))
    for ax, t, c in zip(axes, times, cols):
        mu, dt = np.exp(-t), 1 - np.exp(-2 * t)
        x = mu * a + np.sqrt(dt) * rng.normal(size=L)
        ax.plot(a, color=GREY, lw=0.8, alpha=0.45)
        ax.plot(x, color=c, lw=1.3)
        ax.set_title(rf"$t={t:g}$")
        ax.set_xlabel("site $k$")
        ax.set_xticks([0, 30, 60])
    axes[0].set_ylabel("$x_k$")
    axes[0].annotate("clean chain", xy=(0.04, 0.06), xycoords="axes fraction",
                     fontsize=7.5, color=GREY)
    save(fig, FIG / "fig_forward_corruption")


def fig_three_scores() -> None:
    """Exact, empirical and BP score for a one-dimensional Gaussian at three times.

    The empirical score is the score of the measure putting mass 1/N on each
    training point; away from the data it points at the nearest one, which is
    the mechanism behind collapse. The exact score is linear. The BP score is
    computed from two fitted numbers and tracks the exact one everywhere.
    """
    rng = np.random.default_rng(3)
    n_train = 12
    train = rng.normal(0, 1, n_train)
    xs = np.linspace(-3.2, 3.2, 601)
    times = [0.15, 0.5, 1.5]

    fig, axes = new_figure(1, 3, width=FULL, height=2.15, sharey=True)
    for ax, t in zip(axes, times):
        mu, dt = np.exp(-t), 1 - np.exp(-2 * t)
        var = mu**2 + dt                       # clean variance 1
        exact = -xs / var
        # empirical score: grad log (1/N) sum N(x; mu a_i, dt)
        d = xs[:, None] - mu * train[None, :]
        logw = -0.5 * d**2 / dt
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        emp = -(w * d).sum(axis=1) / dt
        ax.axhline(0, color=GREY, lw=0.5, alpha=0.4)
        ax.plot(xs, emp, color=VERM, lw=1.3)
        ax.plot(xs, exact, color=BLUE, lw=1.7)
        ax.plot(mu * train, np.zeros(n_train), "|", color=VERM,
                markersize=7, markeredgewidth=1.0)
        ax.set_title(rf"$t={t:g}$")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel(r"score $\partial_x \log p_t(x)$")
    axes[0].set_ylim(-9, 9)
    handles = [
        axes[0].plot([], [], color=BLUE, lw=1.7, label="exact score")[0],
        axes[0].plot([], [], color=VERM, lw=1.3, label="empirical score, $N=12$")[0],
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save(fig, FIG / "fig_three_scores")


def fig_collapse_time() -> None:
    """Measured collapse time against dataset size, with the closed form overlaid.

    Data: exp_14. ``t_collapse_train_entropy`` is the measured crossing;
    ``t_collapse_predicted`` is the excess-entropy criterion, which exists only
    where the criterion has a solution.
    """
    rows = read("exp_14_memorization_collapse/collapse_time_summary.csv")
    by_n = defaultdict(list)
    for r in rows:
        by_n[int(r["n_sites"])].append(r)

    fig, ax = new_figure(width=4.4, height=2.7)
    sizes = sorted(by_n)
    cols = ramp(len(sizes), "viridis")
    for n, c in zip(sizes, cols):
        rs = sorted(by_n[n], key=lambda r: int(r["n_train"]))
        N = [int(r["n_train"]) for r in rs]
        meas = [float(r["t_collapse_train_entropy"]) for r in rs]
        ax.plot(N, meas, marker="o", color=c, lw=1.5, label=rf"$L={n}$")
        pred = [(int(r["n_train"]), float(r["t_collapse_predicted"])) for r in rs
                if r["t_collapse_predicted"] not in ("", "nan")]
        if pred:
            ax.plot(*zip(*pred), ls=":", marker="^", markersize=3.5,
                    color=c, lw=1.1, alpha=0.85)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training sequences $N$")
    ax.set_ylabel("collapse time $t_{\\mathrm{C}}$")
    ax.annotate("solid: measured\ndotted: excess-entropy criterion",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=GREY)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(sizes), frameon=False)
    save(fig, FIG / "fig_collapse_time")


def fig_speciation_cascade() -> None:
    """Predicted against measured speciation times, one per level of the hierarchy.

    Data: exp_13. The prediction is read off the covariance spectrum alone.
    """
    rows = read("exp_13_speciation_cascade/cascade_times.csv")
    rows = sorted(rows, key=lambda r: -float(r["eigenvalue"]))
    lev = [int(r["level"]) for r in rows]
    pred = col(rows, "t_speciation_predicted")
    fwd = col(rows, "t_crossing_forward")
    rev = col(rows, "t_crossing_reverse")
    eig = col(rows, "eigenvalue")

    fig, (ax0, ax1) = new_figure(1, 2, width=FULL, height=2.7)

    # left: the ladder itself, one rung per level
    y = np.arange(len(lev))
    ax0.hlines(y, np.minimum(fwd, rev), np.maximum(fwd, rev),
               color=GREY, lw=1.0, alpha=0.5)
    ax0.plot(pred, y, "D", color=BLUE, markersize=5.5, label="predicted")
    ax0.plot(fwd, y, "o", color=VERM, markersize=4.5, label="measured, forward")
    ax0.plot(rev, y, "s", color=GREEN, markersize=4.0, label="measured, reverse")
    ax0.set_yticks(y)
    ax0.set_yticklabels([rf"$\lambda={e:.2f}$" for e in eig])
    ax0.invert_yaxis()
    ax0.set_xlabel("speciation time $t_{\\mathrm{S}}$")
    ax0.set_title("the cascade, level by level")
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False)

    # right: prediction against measurement, with the identity line
    lim = [0, max(pred + fwd + rev) * 1.08]
    ax1.plot(lim, lim, ls="--", color=GREY, lw=0.9)
    ax1.plot(pred, fwd, "o", color=VERM, markersize=5)
    ax1.plot(pred, rev, "s", color=GREEN, markersize=4.5)
    err = max(abs(m - p) / p for p, m in zip(pred, fwd + rev) if p > 0)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("predicted from the spectrum")
    ax1.set_ylabel("measured")
    ax1.set_title(rf"agreement, worst case {100*err:.0f}\%"
                  .replace("\\%", "%"))
    save(fig, FIG / "fig_speciation_cascade")


# ===========================================================================
# Rebuilds of the two figures whose legends collided
# ===========================================================================

def fig_em_diagnostics() -> None:
    """EM convergence and parameter recovery.

    Replaces the version whose legend read ``init.\\\\ 0.0`` -- an r-string
    backslash pair reaching mathtext verbatim -- and whose third panel drew its
    legend on top of the y-axis tick labels.
    """
    trace = read("exp_18/em_trace.csv")
    rate = read("frozen/exp_06_rate16/em_rate.csv")

    fig, ax = new_figure(1, 3, width=FULL, height=2.5)

    by = defaultdict(list)
    for r in trace:
        by[(int(r["n_components"]), int(r["init_id"]), float(r["rho_init"]))].append(
            (int(r["iteration"]), float(r["log_evidence"]), float(r["rho_hat"])))
    series = [(rho0, pts) for (c, _, rho0), pts in sorted(by.items()) if c == 4]
    cols = ramp(len(series), "viridis")
    for (rho0, pts), c in zip(series, cols):
        pts.sort()
        it, ll, rh = zip(*pts)
        ax[0].plot(it, ll, lw=1.3, color=c, label=rf"$\rho_0={rho0:g}$")
        ax[1].plot(it, rh, lw=1.3, color=c)

    ax[0].set_xlabel("EM iteration")
    ax[0].set_ylabel("marginal log-likelihood")
    ax[0].set_title(r"objective, $C=4$")

    ax[1].axhline(0.85, color=GREY, ls="--", lw=1.0)
    ax[1].annotate(r"true $\rho$", xy=(0.97, 0.85), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", fontsize=8, color=GREY)
    ax[1].set_xlabel("EM iteration")
    ax[1].set_ylabel("estimate")
    ax[1].set_title("autoregressive coefficient")

    n = np.array(col(rate, "n_chains"))
    rho_rmse = np.array(col(rate, "mixture_rho_rmse"))
    var_rmse = np.array(col(rate, "mixture_var_rmse"))
    ax[2].loglog(n, rho_rmse, marker="s", color=BLUE, lw=1.4)
    ax[2].loglog(n, var_rmse, marker="o", color=VERM, lw=1.4)
    ax[2].loglog(n, rho_rmse[0] * (n[0] / n) ** 0.5, ls=":", color=GREY, lw=1.1)
    ax[2].set_xlabel("training sequences")
    ax[2].set_ylabel("RMSE")
    ax[2].set_title("parameter recovery")
    # Labelled in place: three curves do not need a legend, and a legend here
    # is what overdrew the tick labels in the previous version.
    label_lines(ax[2], [
        (r"$\rho$", rho_rmse[-1], BLUE),
        (r"$\sigma_\eta^2$", var_rmse[-1], VERM),
        (r"$-1/2$", rho_rmse[0] * (n[0] / n[-1]) ** 0.5, GREY),
    ], dx=1.04, fontsize=8)
    # One legend for the whole row, above the titles. Placing it on ax[0]
    # put it in the same band as that panel's title and overprinted both.
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(series), frameon=False)
    save(fig, FIG / "fig_em_diagnostics")


def fig_nonmarkov() -> None:
    """Where each violation of the Markov assumption stops paying.

    Same data as before. The legend moves out of the data area and the
    break-even crossing is annotated on the curve instead of described below it.
    """
    import glob as _glob
    import re as _re

    src = OUT / ("frozen/exp_21_clean" if (OUT / "frozen" / "exp_21_clean").exists()
                 else "exp_21_frozen")
    arms = {"cnn": ("local CNN", VERM, "^"), "mlp": ("global MLP", GREEN, "v")}
    panels = (("beta", r"rank-one strength $\beta$", "rank-one global latent"),
              ("gamma", r"long-range coupling $\gamma$", "long-range precision coupling"))

    fig, ax = new_figure(1, 2, width=FULL, height=2.75, sharey=True)
    for k, (mech, xlabel, title) in enumerate(panels):
        for family, dash in (("gauss", "-"), ("laplace", "--")):
            cells = {}
            for d in sorted(_glob.glob(str(src / f"{family}_{mech}*"))):
                m = _re.search(rf"{mech}([0-9.]+)$", d)
                f = Path(d) / f"nonmarkov_{family}.csv"
                if not m or not f.exists():
                    continue
                cells[float(m.group(1))] = list(csv.DictReader(f.open()))
            if not cells:
                if family == "gauss":
                    raise FileNotFoundError(f"missing committed output: {src}")
                continue
            xs = sorted(cells)
            for arm, (lab, colr, mk) in arms.items():
                ys = [float(np.mean([float(r["ratio_to_em"]) for r in cells[x]
                                     if r["arm"] == arm])) for x in xs]
                ax[k].plot(xs, ys, ls=dash, color=colr, marker=mk, lw=1.4,
                           markerfacecolor="white" if family == "laplace" else colr,
                           label=f"{lab}, {'Laplace' if family == 'laplace' else 'Gaussian'}")
                # Mark where the structured estimator stops winning. The two
                # arms cross within 0.05 of each other, so the labels are
                # staggered: side by side they overprinted into one unreadable
                # line.
                if family == "gauss":
                    for i in range(len(xs) - 1):
                        if (ys[i] - 1) * (ys[i + 1] - 1) < 0:
                            f_ = (ys[i] - 1) / (ys[i] - ys[i + 1])
                            xc = xs[i] + f_ * (xs[i + 1] - xs[i])
                            ax[k].plot([xc], [1.0], "o", color=colr, markersize=6,
                                       markerfacecolor="white", zorder=5)
                            dy = -14 if arm == "cnn" else 10
                            ha = "right" if arm == "cnn" else "left"
                            ax[k].annotate(rf"{lab.split()[0]} break-even {xc:.2f}",
                                           xy=(xc, 1.0), xytext=(-2 if arm == "cnn" else 4, dy),
                                           textcoords="offset points", ha=ha,
                                           fontsize=7.0, color=colr, zorder=6)
                            break
        ax[k].axhline(1.0, color=GREY, ls="--", lw=0.9)
        ax[k].set_yscale("log")
        ax[k].set_xlabel(xlabel)
        ax[k].set_title(title)
    ax[0].set_ylabel("baseline error / EM--BP error")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False)
    save(fig, FIG / "fig_nonmarkov")


def fig_screening() -> None:
    """Risk against parameter count for the three screened architectures.

    Replaces the version whose ``legend(loc="upper left")`` was drawn on top of
    the window head's own configurations, which are exactly the points in the
    cheap, low-risk corner the figure exists to show.
    """
    rows = [r for r in read("frozen/exp_31_screen/screening.csv")
            if r["region"] == "bulk"]
    arch_style = {
        "window": (BLUE, "s", "weight-shared window head"),
        "conv": (VERM, "^", "dilated convolutional stack"),
        "bimp": (GREEN, "v", "bidirectional message passing"),
    }

    fig, ax = new_figure(width=4.9, height=2.9)
    for arch, (colr, mk, lab) in arch_style.items():
        by_hp = defaultdict(list)
        for r in rows:
            if r["arch"] == arch:
                by_hp[r["hp"]].append((float(r["n_params"]), float(r["risk"])))
        if not by_hp:
            continue
        pts = sorted((float(np.mean([a for a, _ in v])),
                      float(np.mean([b for _, b in v]))) for v in by_hp.values())
        ax.plot([a for a, _ in pts], [b for _, b in pts], ls="none", marker=mk,
                color=colr, alpha=0.8, markersize=4.5, label=lab)
        best = min(pts, key=lambda z: z[1])
        ax.plot(*best, marker=mk, color=colr, markersize=9, ls="none",
                markeredgecolor="#222222", markeredgewidth=0.9, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("mean risk on the bulk region")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=1, frameon=False,
               handletextpad=0.5)
    save(fig, FIG / "fig_screening")


BUILDERS = {
    "fig_forward_corruption": fig_forward_corruption,
    "fig_three_scores": fig_three_scores,
    "fig_collapse_time": fig_collapse_time,
    "fig_speciation_cascade": fig_speciation_cascade,
    "fig_em_diagnostics": fig_em_diagnostics,
    "fig_nonmarkov": fig_nonmarkov,
    "fig_screening": fig_screening,
}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(BUILDERS)
    bad = [n for n in names if n not in BUILDERS]
    if bad:
        print(f"unknown figure(s): {', '.join(bad)}", file=sys.stderr)
        print(f"known: {', '.join(BUILDERS)}", file=sys.stderr)
        return 2
    for n in names:
        print(n)
        BUILDERS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
