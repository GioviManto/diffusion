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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax1.plot(x, np.exp(-np.abs(x)) / 2, ":", color="k", lw=1.0,
             label=r"prior")
    for t, col in zip(times, CYCLE):
        ax1.plot(x, _laplace_pt(x, t), color=col, label=rf"$t = {t:g}$")
    ax1.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "--",
             color=GRAY, lw=1.0, label=r"$\mathcal{N}(0,1)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$p_t^{(1)}(x)$")
    ax1.legend(fontsize=7, ncol=2)
    for t, col in zip(times, CYCLE):
        ax2.plot(x, _laplace_score(x, t), color=col, label=rf"$t = {t:g}$")
    ax2.plot(x, -x, "--", color=GRAY, lw=1.0, label=r"$-x$")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$S(x, t)$")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig_laplace_k1.pdf")


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
        ax.plot(ts, Vex, color="k", label=r"exact (BP)")
        ax.plot(ts, Vamp, color="k", ls="--", label=r"AMP")
        ax.plot(ts, Vmf, color=GRAY, lw=1.0, ls="-.", label=r"mean field")
        tc = t_c(a)
        if np.isfinite(tc) and tc < tmax:
            ax.axvline(tc, color="k", lw=0.6, ls=":")
            ax.annotate(rf"$t_c = {tc:.3f}$", xy=(tc, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, ha="left", va="bottom")
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
    ax1.semilogy(ts, mean_err, color="k", label="AMP mean error")
    ax1.semilogy(ts, var_err, color="k", ls="--",
                 label="AMP variance error")
    tc = t_c(a)
    ax1.axvline(tc, color="k", lw=0.6, ls=":")
    ax1.annotate(r"$t_c$", xy=(tc, 0.03), xycoords=("data", "axes fraction"),
                 xytext=(4, 0), textcoords="offset points", fontsize=8,
                 ha="left", va="bottom")
    ax1.set_xlabel(r"diffusion time $t$")
    ax1.set_ylabel("relative error")
    ax1.set_title(rf"$K = {K}$, $\alpha = {a}$")
    ax1.legend()

    # phase diagram: closed-form boundary, with the fixed-point iteration
    # sampled on a coarse grid as an independent check (markers)
    aa = np.linspace(np.sqrt(2) - 1 + 1e-4, 0.98, 200)
    ax2.plot(aa, [t_c(v) for v in aa], color="k", lw=1.2,
             label=r"$t_c(\alpha)$ (closed form)")
    ax2.axvline(np.sqrt(2) - 1, color="k", lw=0.8, ls="--",
                label=r"$\alpha_c = \sqrt{2} - 1$")
    a_chk = np.linspace(0.45, 0.96, 12)
    t_bound = []
    for al in a_chk:
        tg = np.linspace(0.01, 1.5, 300)
        ok = np.array([np.isfinite(_amp_var_iter(al, t, iters=800))
                       for t in tg])
        t_bound.append(tg[np.argmax(~ok)] if (~ok).any() else np.nan)
    ax2.plot(a_chk, t_bound, "o", color="k", ms=3.5, mfc="white",
             mew=0.8, label="iteration boundary")
    ax2.text(0.30, 1.05, "fixed point\nexists", fontsize=8, ha="center")
    ax2.text(0.87, 1.05, "no fixed\npoint", fontsize=8, ha="center")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$t$")
    ax2.set_xlim(0.05, 0.98)
    ax2.set_ylim(0.0, 1.5)
    ax2.legend(fontsize=7, loc="lower left")
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
                    label="closure error, 10–90%")
    ax.loglog(ts, med, "o-", color=NAVY, ms=3, mew=0,
              label="closure error, median")
    ax.loglog(ts, gmed, "s--", color=GRAY, ms=2.5, mew=0,
              label="grid error budget")
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel("relative score error")
    ax.legend()
    save(fig, "fig_laplace_closure.pdf")


# ================= fig_toymodel_score =================
def fig_toymodel_score():
    """Two-frame toy model (TM2/TM7): trimodal prior, additive dynamics,
    joint density and joint score field after OU diffusion time t.
    Everything is an explicit 2-D Gaussian mixture, so density and score
    are evaluated in closed form."""
    mus = np.array([-2.0, 0.0, 2.0])
    ws = np.array([0.35, 0.30, 0.35])
    s2 = 0.4 ** 2          # prior component variance
    c = 1.0                # drift
    sig2 = 0.45 ** 2       # innovation variance
    t = 0.15
    e = np.exp(-t)
    Dt = 1 - np.exp(-2 * t)

    means, covs = [], []
    for mu in mus:
        m = np.array([mu, mu + c])
        C = np.array([[s2, s2], [s2, s2 + sig2]])
        means.append(e * m)
        covs.append(np.exp(-2 * t) * C + Dt * np.eye(2))

    def mixture(pts):
        dens = np.zeros(len(pts))
        comp = []
        for w, m, C in zip(ws, means, covs):
            Ci = np.linalg.inv(C)
            d = pts - m
            q = np.einsum("ni,ij,nj->n", d, Ci, d)
            g = w * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(C)))
            comp.append((g, Ci, m))
            dens += g
        score = np.zeros_like(pts)
        for g, Ci, m in comp:
            score += (g / dens)[:, None] * ((m - pts) @ Ci.T)
        return dens, score

    lim = 4.2
    gx = np.linspace(-lim, lim, 220)
    gy = np.linspace(-lim, lim, 220)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dens, _ = mixture(pts)
    Z = dens.reshape(X.shape)

    qx = np.linspace(-lim, lim, 23)
    qy = np.linspace(-lim, lim, 23)
    QX, QY = np.meshgrid(qx, qy)
    qpts = np.column_stack([QX.ravel(), QY.ravel()])
    qdens, sc = mixture(qpts)
    # show the field only where the density is non-negligible; the score
    # of a Gaussian tail grows linearly and would dominate the picture
    mask = qdens > 0.015 * qdens.max()
    U = np.where(mask, sc[:, 0], np.nan).reshape(QX.shape)
    V = np.where(mask, sc[:, 1], np.nan).reshape(QX.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.1))
    levels = np.linspace(Z.max() * 0.04, Z.max(), 9)
    ax1.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.6)
    ax1.set_xlabel(r"$x_0$")
    ax1.set_ylabel(r"$x_1$")
    ax1.set_title(r"noised joint density $p_t(x_0, x_1)$")
    ax1.set_aspect("equal")

    ax2.contour(X, Y, Z, levels=levels, colors=GRAY, linewidths=0.4)
    ax2.quiver(QX, QY, U, V, color="k", width=0.0035, scale=40)
    ax2.set_xlabel(r"$x_0$")
    ax2.set_ylabel(r"$x_1$")
    ax2.set_title(r"joint score $\nabla \log p_t$")
    ax2.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig_toymodel_score.pdf")


if __name__ == "__main__":
    fig_band_fill()
    fig_tridiag_loss()
    fig_local_vs_full()
    fig_laplace_k1()
    fig_bulk_variance()
    fig_bp_vs_amp()
    fig_laplace_closure()
    fig_toymodel_score()
    print("all figures regenerated")
