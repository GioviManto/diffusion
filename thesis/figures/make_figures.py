"""Regenerate every thesis figure in a uniform paper style.

All figures are vector PDFs on a white background, serif fonts, no chart
junk. Everything except fig_laplace_closure is recomputed from closed
forms / quadrature; fig_laplace_closure reads the exp_02 per-trial CSV of
research/nongaussian-bp.

Run:  python make_figures.py            (from thesis/figures/)
"""

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP02 = os.path.join(
    HERE, "..", "..", "research", "nongaussian-bp", "outputs",
    "exp_02_laplace_gaussian_message_error", "laplace_per_trial.csv",
)

# ---------------- paper style ----------------
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


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", name)


# ---------------- model helpers ----------------
def sigma0(K, a):
    idx = np.arange(K)
    return a ** np.abs(idx[:, None] - idx[None, :])


def Q0(K, a):
    s2 = 1.0 - a * a
    Q = np.zeros((K, K))
    np.fill_diagonal(Q, (1 + a * a) / s2)
    Q[0, 0] = Q[-1, -1] = 1.0 / s2
    for k in range(K - 1):
        Q[k, k + 1] = Q[k + 1, k] = -a / s2
    return Q


def sigt(K, a, t):
    return np.exp(-2 * t) * sigma0(K, a) + (1 - np.exp(-2 * t)) * np.eye(K)


def Qt(K, a, t):
    return np.linalg.inv(sigt(K, a, t))


def bulk_params(a, t):
    s2 = 1.0 - a * a
    Dt = 1.0 - np.exp(-2 * t)
    Jd = np.exp(-2 * t) / Dt + (1 + a * a) / s2
    beta = -a / s2
    return Jd, beta


def q_of(a, t):
    Jd, b = bulk_params(a, t)
    return (Jd - np.sqrt(Jd * Jd - 4 * b * b)) / (2 * abs(b))


# ================= fig_spectral =================
def fig_spectral():
    K, a = 40, 0.8
    S0 = sigma0(K, a)
    w, U = np.linalg.eigh(S0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ts = np.linspace(0, 3, 200)
    for wi in w:
        ax1.plot(ts, np.exp(-2 * ts) * wi + (1 - np.exp(-2 * ts)),
                 color=NAVY, lw=0.6, alpha=0.45)
    ax1.axhline(1.0, color=GRAY, lw=0.6, ls="--")
    ax1.set_xlabel(r"diffusion time $t$")
    ax1.set_ylabel(r"eigenvalues $\lambda_i(t)$")
    ax1.set_title(r"$\lambda_i(t) = e^{-2t}\omega_i + \Delta_t \to 1$")
    for j, col in zip([K - 1, K - 3, K - 6], [NAVY, RED, OLIVE]):
        ax2.plot(np.arange(K), U[:, j], color=col, lw=1.1,
                 label=rf"$u_{{{K - j}}}$")
    ax2.set_xlabel(r"frame index $k$")
    ax2.set_ylabel("eigenvector component")
    ax2.set_title(r"top eigenvectors of $\Sigma_0$ (near-Fourier)")
    ax2.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_spectral.pdf")


# ================= fig_precision_lifecycle =================
def fig_precision_lifecycle():
    K, a = 15, 0.8
    times = [0.0, 0.15, 0.6, 3.0]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.0))
    for ax, t in zip(axes, times):
        Q = Qt(K, a, t) if t > 0 else Q0(K, a)
        v = np.max(np.abs(Q))
        im = ax.imshow(Q, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(rf"$t = {t:g}$")
        ax.set_xticks([0, K - 1])
        ax.set_yticks([0, K - 1])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "fig_precision_lifecycle.pdf")


# ================= fig_band_fill =================
def fig_band_fill():
    K, a = 25, 0.9
    i0 = K // 2 - 2
    Q0m = Q0(K, a)
    ts = np.logspace(-3, -0.6, 25)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for d in range(1, 6):
        meas = np.array([abs(Qt(K, a, t)[i0, i0 + d]) for t in ts])
        pred = np.array([
            (2 * t) ** (d - 1) *
            abs(np.linalg.matrix_power(Q0m, d)[i0, i0 + d]) for t in ts
        ])
        col = CYCLE[d - 1]
        ax.loglog(ts, meas, "o", ms=2.6, color=col, mew=0)
        ax.loglog(ts, pred, "-", lw=1.0, color=col,
                  label=rf"$d = {d}$")
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"$|(Q_t)_{i,i+d}|$")
    ax.legend(ncol=2)
    save(fig, "fig_band_fill.pdf")


# ================= fig_tridiag_loss =================
def fig_tridiag_loss():
    K, a = 40, 0.8
    ts = np.linspace(1e-3, 4, 160)
    covdiff, offmass = [], []
    S0 = sigma0(K, a)
    tri = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]) <= 1
    for t in ts:
        St = sigt(K, a, t)
        Q = np.linalg.inv(St)
        covdiff.append(np.linalg.norm(St - S0))
        offmass.append(np.linalg.norm(np.where(tri, 0.0, Q)))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ts, covdiff, color=NAVY,
            label=r"$\|\Sigma_t - \Sigma_0\|_F$")
    ax2 = ax.twinx()
    ax2.plot(ts, offmass, color=RED,
             label=r"off-tridiagonal mass of $Q_t$")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"$\|\Sigma_t - \Sigma_0\|_F$", color=NAVY)
    ax2.set_ylabel(r"$\|Q_t - \mathrm{tri}(Q_t)\|_F$", color=RED)
    ax.tick_params(axis="y", colors=NAVY)
    ax2.tick_params(axis="y", colors=RED)
    save(fig, "fig_tridiag_loss.pdf")


# ================= fig_local_vs_full =================
def _local_rms(K, a, t, r):
    """Exact RMS error of the radius-r local posterior-mean estimator."""
    Dt = 1 - np.exp(-2 * t)
    g = np.exp(-t) / Dt
    k = K // 2
    J = (np.exp(-2 * t) / Dt) * np.eye(K) + Q0(K, a)
    w_full = g * np.linalg.inv(J)[k, :]
    n = 2 * r + 1
    Jw = (np.exp(-2 * t) / Dt) * np.eye(n) + Q0(n, a)
    w_loc = g * np.linalg.inv(Jw)[r, :]
    w_win = np.zeros(K)
    w_win[k - r:k + r + 1] = w_loc
    e = w_full - w_win
    St = sigt(K, a, t)
    return float(np.sqrt(e @ St @ e))


def fig_local_vs_full():
    K, a = 121, 0.8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    rs = np.arange(1, 14)
    for t, col in zip([0.1, 0.5, 2.0], [NAVY, RED, OLIVE]):
        rms = np.array([_local_rms(K, a, t, r) for r in rs])
        q = q_of(a, t)
        ax1.semilogy(rs, rms, "o", ms=3, color=col, mew=0,
                     label=rf"$t = {t:g}$")
        anchor = rms[1] / q ** rs[1]
        ax1.semilogy(rs, anchor * q ** rs, "--", lw=0.9, color=col)
    ax1.set_xlabel(r"radius $r$")
    ax1.set_ylabel(r"RMS truncation error")
    ax1.set_title(r"error $\propto q(\alpha,t)^r$ (dashed: predicted)")
    ax1.legend()
    ts = np.linspace(0.02, 4, 60)
    for r, col in zip([1, 2, 4], [NAVY, RED, OLIVE]):
        ax2.plot(ts, [_local_rms(K, a, t, r) for t in ts],
                 color=col, label=rf"$r = {r}$")
    ax2.set_xlabel(r"diffusion time $t$")
    ax2.set_ylabel(r"RMS truncation error")
    ax2.set_title(r"fixed radius: peak at intermediate $t$")
    ax2.legend()
    fig.tight_layout()
    save(fig, "fig_local_vs_full.pdf")


# ================= Laplace K=1 =================
def _laplace_pt(x, t, b=1.0):
    mu, Dt = np.exp(-t), 1 - np.exp(-2 * t)
    av = np.linspace(-30, 30, 6001)
    pa = np.exp(-np.abs(av) / b) / (2 * b)
    ker = np.exp(-(x[:, None] - mu * av[None, :]) ** 2 / (2 * Dt)) \
        / np.sqrt(2 * np.pi * Dt)
    return np.trapz(ker * pa[None, :], av, axis=1)


def _laplace_score(x, t, b=1.0):
    mu, Dt = np.exp(-t), 1 - np.exp(-2 * t)
    av = np.linspace(-30, 30, 6001)
    pa = np.exp(-np.abs(av) / b) / (2 * b)
    ker = np.exp(-(x[:, None] - mu * av[None, :]) ** 2 / (2 * Dt)) \
        / np.sqrt(2 * np.pi * Dt)
    w = ker * pa[None, :]
    Z = np.trapz(w, av, axis=1)
    Ea = np.trapz(w * av[None, :], av, axis=1) / Z
    return (mu * Ea - x) / Dt


def fig_laplace_k1():
    x = np.linspace(-5, 5, 401)
    times = [0.05, 0.2, 0.6, 1.5]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(x, np.exp(-np.abs(x)) / 2, ":", color="k", lw=1.0,
            label=r"Laplace prior")
    for t, col in zip(times, CYCLE):
        ax.plot(x, _laplace_pt(x, t), color=col, label=rf"$t = {t:g}$")
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "--",
            color=GRAY, lw=1.0, label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p_t^{(1)}(x)$")
    ax.legend(ncol=2)
    save(fig, "fig06_K1_density.pdf")

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for t, col in zip(times, CYCLE):
        ax.plot(x, _laplace_score(x, t), color=col, label=rf"$t = {t:g}$")
    ax.plot(x, -x, "--", color=GRAY, lw=1.0, label=r"$-x$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$S(x, t)$")
    ax.legend(ncol=2)
    save(fig, "fig07_K1_score.pdf")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for t, col in zip([0.3, 0.1, 0.03, 0.01], CYCLE):
        ax1.plot(x, _laplace_score(x, t), color=col, label=rf"$t = {t:g}$")
    ax1.plot(x, -np.sign(x), "--", color="k", lw=1.0,
             label=r"$-\mathrm{sign}(x)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$S(x,t)$")
    ax1.set_title(r"$t \to 0$: prior score")
    ax1.legend(fontsize=7)
    for t, col in zip([1.0, 2.0, 4.0, 6.0], CYCLE):
        ax2.plot(x, _laplace_score(x, t), color=col, label=rf"$t = {t:g}$")
    ax2.plot(x, -x, "--", color="k", lw=1.0, label=r"$-x$")
    ax2.set_xlabel(r"$x$")
    ax2.set_title(r"$t \to \infty$: Gaussian attractor")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig08_K1_score_limits.pdf")


# ================= fig_bulk_variance =================
def _closures(a, ts):
    Jd, b = bulk_params(a, ts)
    Vex = 1.0 / np.sqrt(Jd ** 2 - 4 * b ** 2)
    disc = Jd ** 2 - 8 * b ** 2
    Vamp = np.where(disc >= 0,
                    (Jd - np.sqrt(np.clip(disc, 0, None))) / (4 * b ** 2),
                    np.nan)
    Vmf = 1.0 / Jd
    return Vex, Vamp, Vmf


def t_c(a):
    g = (2 * np.sqrt(2) * abs(a) - 1 - a * a) / (1 - a * a)
    if g <= 0:
        return np.inf
    return -0.5 * np.log(g / (1 + g))


def fig_bulk_variance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, a, tmax in [(ax1, 0.8, 1.2), (ax2, 0.3, 4.0)]:
        ts = np.linspace(0.005, tmax, 500)
        Vex, Vamp, Vmf = _closures(a, ts)
        ax.plot(ts, Vex, color=NAVY, label=r"exact $=$ BP")
        ax.plot(ts, Vamp, color=RED, label=r"AMP")
        ax.plot(ts, Vmf, color=OLIVE, lw=1.0, label=r"mean field")
        tc = t_c(a)
        if np.isfinite(tc) and tc < tmax:
            ax.axvspan(tc, tmax, color=RED, alpha=0.08, lw=0)
            ax.axvline(tc, color=RED, lw=0.7, ls=":")
            ax.text(tc, ax.get_ylim()[1] * 0.55, rf"$t_c = {tc:.3f}$",
                    color=RED, fontsize=7.5, ha="left", rotation=90,
                    va="center")
        ax.set_xlabel(r"diffusion time $t$")
        ax.set_ylabel(r"bulk posterior variance")
        ax.set_title(rf"$\alpha = {a}$")
    ax1.legend()
    fig.tight_layout()
    save(fig, "fig_bulk_variance.pdf")


# ================= fig_bp_vs_amp =================
def _amp_var_iter(a, t, iters=4000, damp=0.5):
    Jd, b = bulk_params(a, t)
    V = 1.0 / Jd
    for _ in range(iters):
        denom = Jd - 2 * b * b * V
        if denom <= 0:
            return np.nan
        V = (1 - damp) * V + damp / denom
        if V <= 0 or V > 1e6:
            return np.nan
    return V


def fig_bp_vs_amp():
    a = 0.8
    K = 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ts = np.linspace(0.02, 0.6, 60)
    mean_err, var_err = [], []
    for t in ts:
        Dt = 1 - np.exp(-2 * t)
        J = (np.exp(-2 * t) / Dt) * np.eye(K) + Q0(K, a)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(K)
        h = (np.exp(-t) / Dt) * x
        m_exact = np.linalg.solve(J, h)
        m = np.zeros(K)
        for _ in range(20000):
            m_new = (h - (J - np.diag(np.diag(J))) @ m) / np.diag(J)
            if np.max(np.abs(m_new - m)) < 1e-15:
                m = m_new
                break
            m = 0.7 * m + 0.3 * m_new
        mean_err.append(np.max(np.abs(m - m_exact)) /
                        max(np.max(np.abs(m_exact)), 1e-30))
        Jd, b = bulk_params(a, t)
        Vex = 1.0 / np.sqrt(Jd ** 2 - 4 * b ** 2)
        disc = Jd ** 2 - 8 * b ** 2
        if disc >= 0:
            Vamp = (Jd - np.sqrt(disc)) / (4 * b ** 2)
            var_err.append(abs(Vamp - Vex) / Vex)
        else:
            var_err.append(np.nan)
    ax1.semilogy(ts, mean_err, color=NAVY, label="AMP mean error")
    ax1.semilogy(ts, var_err, color=RED, label="AMP variance error")
    tc = t_c(a)
    ax1.axvspan(tc, ts[-1], color=RED, alpha=0.08, lw=0)
    ax1.axvline(tc, color=RED, lw=0.7, ls=":")
    ax1.set_xlabel(r"diffusion time $t$")
    ax1.set_ylabel("relative error")
    ax1.set_title(rf"$K = {K}$, $\alpha = {a}$")
    ax1.legend()

    alphas = np.linspace(0.05, 0.98, 90)
    tgrid = np.linspace(0.01, 1.5, 90)
    exist = np.zeros((len(tgrid), len(alphas)))
    for j, al in enumerate(alphas):
        for i, t in enumerate(tgrid):
            exist[i, j] = 1.0 if np.isfinite(_amp_var_iter(
                al, t, iters=800)) else 0.0
    ax2.imshow(exist, origin="lower", aspect="auto",
               extent=[alphas[0], alphas[-1], tgrid[0], tgrid[-1]],
               cmap=matplotlib.colors.ListedColormap(
                   ["#f3dede", "#e8eef5"]))
    aa = np.linspace(np.sqrt(2) - 1 + 1e-4, 0.98, 200)
    ax2.plot(aa, [t_c(v) for v in aa], color="k", lw=1.2,
             label=r"$t_c(\alpha)$ (closed form)")
    ax2.axvline(np.sqrt(2) - 1, color="k", lw=0.8, ls="--",
                label=r"$\alpha_c = \sqrt{2} - 1$")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$t$")
    ax2.set_ylim(tgrid[0], tgrid[-1])
    ax2.set_title("AMP variance fixed point exists (blue)")
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_bp_vs_amp.pdf")


# ================= fig_laplace_closure =================
def fig_laplace_closure():
    by_t = defaultdict(lambda: {"err": [], "grid": []})
    with open(EXP02) as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            by_t[t]["err"].append(float(row["score_rel_error"]))
            by_t[t]["grid"].append(float(row["grid_self_conv_rel_error"]))
    ts = sorted(by_t)
    med = [np.median(by_t[t]["err"]) for t in ts]
    p10 = [np.percentile(by_t[t]["err"], 10) for t in ts]
    p90 = [np.percentile(by_t[t]["err"], 90) for t in ts]
    gmed = [np.median(by_t[t]["grid"]) for t in ts]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.fill_between(ts, p10, p90, color=NAVY, alpha=0.15, lw=0,
                    label=r"closure error, 10--90\%")
    ax.loglog(ts, med, "o-", color=NAVY, ms=3, mew=0,
              label="closure error, median")
    ax.loglog(ts, gmed, "s--", color=GRAY, ms=2.5, mew=0,
              label="grid error budget")
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel("relative score error")
    ax.legend()
    save(fig, "fig_laplace_closure.pdf")


if __name__ == "__main__":
    fig_spectral()
    fig_precision_lifecycle()
    fig_band_fill()
    fig_tridiag_loss()
    fig_local_vs_full()
    fig_laplace_k1()
    fig_bulk_variance()
    fig_bp_vs_amp()
    fig_laplace_closure()
    print("all figures regenerated")
