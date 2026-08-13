"""Every figure of the note, as a vector PDF in the thesis paper style.

Run:  python make_note_figures.py            (from note/)

Core model functions are imported from ../code/verify_scaling.py so the figures
use exactly the audited implementation.  A self-check at the top re-derives the
closed-form score against finite differences before anything is plotted; the
script aborts if it fails.

Heavy sweeps (speciation, sample complexity) are cached in data/*.json.
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from verify_scaling import (                                    # noqa: E402
    LAM, SIGMA, RHO, WQ, LOGRING, RHO2, K_min, rot, sample_ring,
    sample_two_species, gauge_batch, log_pt_walk, logZ_table,
)

OUT = HERE
DATA = os.path.join(HERE, "data")
FIGDATA = os.path.join(HERE, "..", "outputs", "figdata.json")
P3DATA = os.path.join(HERE, "..", "outputs", "p3prof.json")

NAVY, RED, OLIVE, GRAY = "#1f3a5f", "#a02c2c", "#6b6b2a", "#7f7f7f"
BLUE, ORANGE = "#4a7ba6", "#c07840"
CYCLE = [NAVY, RED, OLIVE, BLUE, ORANGE, GRAY]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=CYCLE),
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def _ticks(ax, xt, xl, yt, yl):
    """Explicit major ticks on log axes; log minor labels collide otherwise."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter(xl))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(yl))
    ax.yaxis.set_minor_formatter(NullFormatter())


def cached(name, fn):
    p = os.path.join(DATA, name)
    if os.path.exists(p):
        return json.load(open(p))
    d = fn()
    json.dump(d, open(p, "w"))
    return d


# ----------------------------------------------------------- model core

def deltas(t):
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    return m, Dl


def A_of(T, t):
    m, Dl = deltas(t)
    return m ** 2 * SIGMA ** 2 * K_min(T) + Dl * np.eye(T)


def Zt_and_Phi(beta, kappa, m):
    """Z_t(beta) (true log) and Phi_t = dlogZ/dbeta, for scalar beta."""
    z = m * RHO * beta
    e = LOGRING - 0.5 * m ** 2 * kappa * RHO ** 2 + np.abs(z)
    sh = e.max()
    base = WQ * np.exp(e - sh)
    Z = np.sum(base * RHO * ive(0, z))
    Zp = m * np.sum(base * RHO ** 2 * ive(1, z))
    return np.log(Z) + sh, Zp / Z


def joint_score(X, t):
    """Exact joint score, walk frame.  X: (T,2) -> (T,2)."""
    T = X.shape[0]
    m, _ = deltas(t)
    A = A_of(T, t)
    Ai = np.linalg.inv(A)
    g = Ai @ np.ones(T)
    kappa = float(np.ones(T) @ g)
    b = X.T @ g
    beta = float(np.linalg.norm(b))
    _, phi = Zt_and_Phi(beta, kappa, m)
    return -Ai @ X + phi * np.outer(g, b / max(beta, 1e-300))


def self_check():
    """Closed-form score vs finite differences of a 2-D quadrature reference."""
    T, t = 3, 0.35
    m, Dl = deltas(t)
    A = A_of(T, t)
    Ai = np.linalg.inv(A)
    nr, nph = 220, 192
    xr, wr = np.polynomial.legendre.leggauss(nr)
    rmax = 1.0 + 9.0 * np.sqrt(LAM)
    rr = 0.5 * rmax * (xr + 1.0)
    wrr = 0.5 * rmax * wr
    ph = 2 * np.pi * np.arange(nph) / nph
    R, P = np.meshgrid(rr, ph, indexing="ij")
    W = (wrr[:, None] * (2 * np.pi / nph)) * R
    z0 = np.stack([R.ravel() * np.cos(P.ravel()), R.ravel() * np.sin(P.ravel())], 1)
    w, rho = W.ravel(), R.ravel()

    def logp(X):
        acc = np.zeros(z0.shape[0])
        for i in range(2):
            d = X[:, i][None, :] - m * z0[:, i][:, None]
            acc += np.einsum("gu,uv,gv->g", d, Ai, d)
        e = -(rho - 1.0) ** 2 / (2 * LAM) - 0.5 * acc
        mx = e.max()
        return np.log(np.sum(w * np.exp(e - mx))) + mx

    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, 2)) * 0.8
    s_cf = joint_score(X, t)
    h, s_fd = 1e-4, np.zeros((T, 2))
    for u in range(T):
        for i in range(2):
            Xp, Xm = X.copy(), X.copy()
            Xp[u, i] += h
            Xm[u, i] -= h
            s_fd[u, i] = (logp(Xp) - logp(Xm)) / (2 * h)
    rel = np.abs(s_cf - s_fd).max() / np.abs(s_fd).max()
    print(f"self-check: closed-form score vs finite differences  rel={rel:.2e}")
    assert rel < 5e-5, "figure script core disagrees with the audited formula"


# ------------------------------------------------------------- figures

def fig01_two_clocks():
    """One trajectory, noised at several diffusion times: the two clocks."""
    rng = np.random.default_rng(5)
    T, psi, sig = 12, 2 * np.pi / 12, 0.10
    z = sample_ring(1, rng)[0]
    a = [z.copy()]
    for _ in range(T - 1):
        z = rot(psi) @ z + sig * rng.normal(size=2)
        a.append(z.copy())
    a = np.stack(a)
    ts = [0.0, 0.15, 0.5, 1.5]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
    for ax, t in zip(axes, ts):
        m, Dl = deltas(t)
        x = m * a + np.sqrt(Dl) * rng.normal(size=a.shape)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":",
                                lw=0.6, color=GRAY))
        ax.plot(x[:, 0], x[:, 1], "-o", color=NAVY, ms=2.6, lw=1.0)
        ax.plot(x[0, 0], x[0, 1], "o", color=RED, ms=4.5)
        ax.set_title(rf"$t={t}$", pad=3)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.3, 2.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].text(-2.1, -2.15, r"$u=0$ marked red", color=RED, fontsize=7.5)
    save(fig, "fig01_two_clocks.pdf")


def fig02_ring():
    rng = np.random.default_rng(2)
    z = sample_ring(20000, rng)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a1.plot(z[:2500, 0], z[:2500, 1], ".", ms=1.0, color=NAVY, alpha=0.5)
    a1.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":", lw=0.8, color=RED))
    a1.set_aspect("equal")
    a1.set_xlabel(r"$x$")
    a1.set_ylabel(r"$y$")
    a1.set_title(r"$p_{\rm ring}(z)\propto e^{-(\|z\|-1)^2/2\lambda}$", pad=4)
    r = np.linalg.norm(z, axis=1)
    a2.hist(r, bins=90, density=True, color=BLUE, alpha=0.55,
            edgecolor="none", label="samples")
    gr = RHO * np.exp(LOGRING)
    a2.plot(RHO, gr / np.trapezoid(gr, RHO), color=RED,
            label=r"$\propto \rho\,e^{-(\rho-1)^2/2\lambda}$")
    a2.set_xlabel(r"$\rho=\|z\|$")
    a2.set_ylabel("density")
    a2.set_xlim(0.2, 1.9)
    a2.legend()
    a2.set_title(rf"radial law, $\lambda={LAM}$", pad=4)
    save(fig, "fig02_ring.pdf")


def fig03_gauge():
    d = json.load(open(FIGDATA))
    data = np.array(d["traj"]["data"])
    walk = np.array(d["traj"]["walk"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, tr, ttl in ((a1, data, r"rotating frame: $a$"),
                        (a2, walk, r"gauged frame: $U_\psi a$")):
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":", lw=0.7,
                                color=GRAY))
        for k in range(4):
            ax.plot(tr[k][:, 0], tr[k][:, 1], "-o", ms=2.2, lw=1.0,
                    color=CYCLE[k])
            ax.plot(tr[k][0, 0], tr[k][0, 1], "o", ms=4.5, color=CYCLE[k])
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(ttl, pad=4)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
    save(fig, "fig03_gauge.pdf")


def fig04_joint_vs_marginal():
    d = json.load(open(FIGDATA))
    st = d["stats"]
    keys = [("data", "data"), ("joint", "exact joint score"),
            ("marginal", "per-frame marginal scores")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55))
    for ax, (k, ttl) in zip(axes, keys):
        tr = np.array(d["traj"][k])
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":", lw=0.7,
                                color=GRAY))
        for j in range(4):
            ax.plot(tr[j][:, 0], tr[j][:, 1], "-o", ms=2.0, lw=0.9,
                    color=CYCLE[j])
        R = st[k]["turn_concentration"]
        ax.set_title(f"{ttl}\n" + rf"$R={R:.3f}$", pad=4, fontsize=8.6)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "fig04_joint_vs_marginal.pdf")


def fig05_phi():
    T = 6
    betas = np.linspace(0, 14, 260)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for t in (0.1, 0.3, 0.8, 2.0):
        m, _ = deltas(t)
        A = A_of(T, t)
        g = np.linalg.solve(A, np.ones(T))
        kappa = float(np.ones(T) @ g)
        phi = np.array([Zt_and_Phi(b, kappa, m)[1] for b in betas])
        a1.plot(betas, phi, label=rf"$t={t}$")
        a2.plot(betas, phi / m, label=rf"$t={t}$")
    a1.set_xlabel(r"$\beta$")
    a1.set_ylabel(r"$\Phi_t(\beta)$")
    a1.set_title(r"$\Phi_t=\partial_\beta\log\mathcal{Z}_t$", pad=4)
    a1.legend()
    a2.axhline(1.0, color=GRAY, lw=0.7, ls="--")
    a2.set_xlabel(r"$\beta$")
    a2.set_ylabel(r"$\|\mathbb{E}[z_0\mid x]\|$")
    a2.set_title(r"denoised anchor radius $=\Phi_t(\beta)/m$", pad=4)
    a2.set_ylim(0, 1.6)
    save(fig, "fig05_phi.pdf")


def fig06_bandfill():
    T = 8
    ts = [3e-3, 0.1, 0.6]
    fig = plt.figure(figsize=(6.6, 2.35))
    for j, t in enumerate(ts):
        ax = fig.add_subplot(1, 4, j + 1)
        Ai = np.linalg.inv(A_of(T, t))
        M = np.abs(Ai) / np.abs(np.diag(Ai)).max()
        im = ax.imshow(np.log10(M + 1e-16), vmin=-8, vmax=0, cmap="magma_r")
        ax.set_title(rf"$t={t}$", pad=3)
        ax.set_xticks([0, 4, 7])
        ax.set_yticks([0, 4, 7])
        if j == 0:
            ax.set_ylabel(r"$u$")
        if j == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label(r"$\log_{10}|A_t^{-1}|/{\rm max\,diag}$", fontsize=7)
            cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(1, 4, 4)
    for t in (1e-4, 1e-3, 1e-2):
        m, Dl = deltas(t)
        Ai = np.linalg.inv(A_of(T, t))
        d0 = np.abs(np.diag(Ai)).max()
        ds = np.arange(1, 5)
        r = [np.abs(np.diag(Ai, d)).max() / d0 for d in ds]
        eps = Dl / (m ** 2 * SIGMA ** 2)
        ax.semilogy(ds, r, "o-", ms=3, label=rf"$\Delta/m^2\sigma^2={eps:.1e}$")
        ax.semilogy(ds, eps ** ds, ":", color=GRAY, lw=0.8)
    ax.set_xlabel(r"band index $d$")
    ax.set_ylabel("off-band / diag")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title(r"$\sim(\Delta/m^2\sigma^2)^d$ (dotted)", pad=3, fontsize=8.4)
    ax.legend(fontsize=6.4)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig06_bandfill.pdf")


def fig07_psi_posterior():
    """T=2: p(psi|x,t) concentrates on the observed turning angle gamma."""
    rng = np.random.default_rng(9)
    gamma_true = 0.9
    z0 = sample_ring(1, rng)[0]
    z1 = rot(gamma_true) @ z0 + SIGMA * rng.normal(size=2)
    psis = np.linspace(-np.pi, np.pi, 721)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for t in (0.05, 0.3, 0.8, 1.6):
        m, Dl = deltas(t)
        D1, D2 = Dl, m ** 2 * SIGMA ** 2 + Dl
        x0 = m * z0 + np.sqrt(Dl) * rng.normal(size=2)
        x1 = m * z1 + np.sqrt(Dl) * rng.normal(size=2)
        kappa = 1 / D1 + 1 / D2
        gam = np.arctan2(x1[1], x1[0]) - np.arctan2(x0[1], x0[0])
        lz = []
        for p in psis:
            b = x0 / D1 + rot(-p) @ x1 / D2
            lz.append(Zt_and_Phi(float(np.linalg.norm(b)), kappa, m)[0])
        lz = np.array(lz)
        pp = np.exp(lz - lz.max())
        pp /= np.trapezoid(pp, psis)
        a1.plot(psis, pp, label=rf"$t={t}$")
        if t == 0.05:
            a1.axvline(np.angle(np.exp(1j * gam)), color=GRAY, lw=0.7, ls="--")
    a1.set_xlabel(r"$\psi$")
    a1.set_ylabel(r"$p(\psi\mid x,t)$")
    a1.set_title(r"posterior peaks at $\gamma$ (dashed)", pad=4)
    a1.legend()
    a1.set_xlim(-np.pi, np.pi)

    om = 0.8
    ts = np.linspace(0.02, 4.0, 160)
    for gt in (0.9, 0.4):
        lo = []
        for t in ts:
            m, Dl = deltas(t)
            D1, D2 = Dl, m ** 2 * SIGMA ** 2 + Dl
            kappa = 1 / D1 + 1 / D2
            v = []
            for s in (+1, -1):
                b = m * z0 / D1 + rot(-s * om) @ (m * rot(gt) @ z0) / D2
                v.append(Zt_and_Phi(float(np.linalg.norm(b)), kappa, m)[0])
            lo.append(v[0] - v[1])
        a2.semilogy(ts, np.abs(lo), label=rf"$\gamma={gt}$")
    a2.semilogy(ts, 0.35 * np.exp(-2 * ts), ":", color=GRAY, lw=0.9,
                label=r"$\propto e^{-2t}$")
    a2.set_xlabel(r"$t$")
    a2.set_ylabel("|log-odds|")
    a2.set_title(r"two-species log-odds, $\omega=0.8$", pad=4)
    a2.legend()
    save(fig, "fig07_psi_posterior.pdf")


def fig08_09_speciation():
    def compute():
        rng = np.random.default_rng(21)
        om, n = 0.8, 4000
        tg = np.concatenate([np.linspace(0.03, 1.0, 34),
                             np.linspace(1.06, 4.0, 34)])
        Ts = [2, 4, 8, 16, 32]
        out = {"t": tg.tolist(), "T": Ts, "post": {}}
        for T in Ts:
            a, sgn = sample_two_species(n, T, om, rng)
            row = []
            for t in tg:
                m, Dl = deltas(t)
                X = m * a + np.sqrt(Dl) * rng.normal(size=a.shape)
                lp = np.stack([log_pt_walk(gauge_batch(X, s * om), t)
                               for s in (1.0, -1.0)])
                row.append(float(np.mean(
                    1.0 / (1.0 + np.exp(-(lp[0] - lp[1]) * sgn)))))
            out["post"][str(T)] = row
            print("   speciation T =", T, "done")
        return out

    d = cached("spec.json", compute)
    tg = np.array(d["t"])
    Ts = d["T"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    tsp = []
    for T in Ts:
        p = np.array(d["post"][str(T)])
        a1.plot(tg, p, label=rf"$T={T}$")
        i = int(np.argmax(p < 0.75))
        t0, t1, p0, p1 = tg[i - 1], tg[i], p[i - 1], p[i]
        tsp.append(t0 + (p0 - 0.75) * (t1 - t0) / (p0 - p1))
    a1.axhline(0.75, color=GRAY, lw=0.7, ls="--")
    a1.axhline(0.5, color=GRAY, lw=0.5, ls=":")
    a1.set_xlabel(r"$t$")
    a1.set_ylabel("P(true species)")
    a1.set_xlim(0, 3)
    a1.legend(ncol=2)
    a1.set_title(r"commitment to the sense of rotation", pad=4)

    lg = np.log(np.array(Ts, float))
    tsp = np.array(tsp)
    a2.plot(lg, tsp, "o-", color=NAVY, ms=4, label=r"measured $t_{\rm spec}$")
    a2.plot(lg, 0.5 * lg + (tsp[0] - 0.5 * lg[0]), "--", color=RED,
            label=r"$\frac{1}{2}\log T$ (rejected)")
    a2.set_xlabel(r"$\log T$")
    a2.set_ylabel(r"$t_{\rm spec}(0.75)$")
    a2.legend()
    a2.set_title("no single power law", pad=4)
    ax3 = a2.inset_axes([0.56, 0.13, 0.4, 0.36])
    loc = np.diff(tsp) / np.diff(lg)
    ax3.plot(0.5 * (lg[1:] + lg[:-1]), loc, "s-", color=OLIVE, ms=3)
    ax3.axhline(0.5, color=RED, lw=0.7, ls="--")
    ax3.set_ylim(0.2, 1.0)
    ax3.tick_params(labelsize=6)
    ax3.set_title("local slope", fontsize=6.5, pad=2)
    save(fig, "fig08_speciation.pdf")
    return tsp


def fig10_p2_coupling():
    v0, c, sig2 = 0.7, 1.0, SIGMA ** 2
    ts = np.logspace(-3.3, 1.1, 400)
    off, rel = [], []
    for t in ts:
        m, Dl = deltas(t)
        C0 = np.array([[v0, v0], [v0, v0 + sig2]])
        Ct = m ** 2 * C0 + Dl * np.eye(2)
        Q = np.linalg.inv(Ct)
        off.append(Q[0, 1])
        rel.append(abs(Q[0, 1]) / np.sqrt(Q[0, 0] * Q[1, 1]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a1.semilogx(ts, np.abs(off), color=NAVY, label=r"$|(C_t^{-1})_{01}|$")
    a1.axhline(1 / sig2, color=RED, lw=0.8, ls="--", label=r"$1/\sigma^2$")
    a1.plot(ts, v0 * np.exp(-2 * ts), ":", color=GRAY, lw=0.9,
            label=r"$e^{-2t}v_0$")
    a1.set_yscale("log")
    a1.set_xlabel(r"$t$")
    a1.set_ylabel("coupling")
    a1.set_ylim(1e-3, 40)
    a1.legend()
    a1.set_title(r"the whole joint/marginal gap", pad=4)
    a2.semilogx(ts, rel, color=NAVY)
    a2.set_xlabel(r"$t$")
    a2.set_ylabel("normalised coupling")
    a2.set_title("temporal structure is a low-noise feature", pad=4)
    save(fig, "fig10_p2_coupling.pdf")


def fig11_p2_field():
    """The board's middle panel: two-bump prior, joint score field vs marginal."""
    mus, ws, v, c, t = np.array([-1.5, 1.4]), np.array([0.45, 0.55]), 0.10, 1.0, 0.2
    m, Dl = deltas(t)
    C0 = np.array([[v, v], [v, v + SIGMA ** 2]])
    Ct = m ** 2 * C0 + Dl * np.eye(2)
    Cti = np.linalg.inv(Ct)
    nus = np.stack([np.array([mu, mu + c]) for mu in mus])
    g = np.linspace(-3.2, 3.6, 220)
    XX, YY = np.meshgrid(g, g)
    pts = np.stack([XX.ravel(), YY.ravel()], 1)

    def dens_and_score(P):
        ll = np.stack([np.log(w) - 0.5 * np.einsum(
            "ni,ij,nj->n", P - m * nu, Cti, P - m * nu)
            for w, nu in zip(ws, nus)])
        mx = ll.max(0)
        dens = np.exp(mx) * np.sum(np.exp(ll - mx), 0)
        r = np.exp(ll - mx)
        r /= r.sum(0)
        s = np.zeros_like(P)
        for k, nu in enumerate(nus):
            s += r[k][:, None] * (-(P - m * nu) @ Cti)
        return dens, s

    dens, sj = dens_and_score(pts)
    Dg = np.diag(np.diag(Ct))
    Dgi = np.linalg.inv(Dg)
    ll = np.stack([np.log(w) - 0.5 * np.einsum(
        "ni,ij,nj->n", pts - m * nu, Dgi, pts - m * nu)
        for w, nu in zip(ws, nus)])
    mx = ll.max(0)
    rm = np.exp(ll - mx)
    rm /= rm.sum(0)
    sm = np.zeros_like(pts)
    for k, nu in enumerate(nus):
        sm += rm[k][:, None] * (-(pts - m * nu) @ Dgi)

    q = np.linspace(-3.0, 3.4, 15)
    QX, QY = np.meshgrid(q, q)
    qp = np.stack([QX.ravel(), QY.ravel()], 1)
    _, sjq = dens_and_score(qp)
    llq = np.stack([np.log(w) - 0.5 * np.einsum(
        "ni,ij,nj->n", qp - m * nu, Dgi, qp - m * nu)
        for w, nu in zip(ws, nus)])
    mxq = llq.max(0)
    rq = np.exp(llq - mxq)
    rq /= rq.sum(0)
    smq = np.zeros_like(qp)
    for k, nu in enumerate(nus):
        smq += rq[k][:, None] * (-(qp - m * nu) @ Dgi)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.95))
    for ax, sq, ttl in ((a1, sjq, r"joint score $\nabla\log P_t$"),
                        (a2, smq, "sum of per-frame marginal scores")):
        ax.contour(XX, YY, dens.reshape(XX.shape), levels=7,
                   colors=GRAY, linewidths=0.5)
        ax.quiver(QX, QY, sq[:, 0].reshape(QX.shape), sq[:, 1].reshape(QX.shape),
                  color=NAVY, width=0.004, scale=95)
        ax.set_xlabel(r"$x_0$")
        ax.set_title(ttl, pad=4, fontsize=8.8)
        ax.set_aspect("equal")
    a1.set_ylabel(r"$x_1$")
    save(fig, "fig11_p2_field.pdf")


def fig12_p3_ellipse():
    """The board's ellipse sketch: conditioning the last frame on a target."""
    rho0, s2, mstar = 0.5, 0.25, 1.7
    C0 = np.array([[rho0, rho0], [rho0, rho0 + SIGMA ** 2]])
    E = np.array([[0.0, 1.0]])
    C0r = np.linalg.inv(np.linalg.inv(C0) + E.T @ E / s2)
    mu0r = (C0r @ E.T * mstar / s2).ravel()

    def ell(mu, C, col, ls, lab):
        w, V = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
        for k in (1.0, 2.0):
            e = Ellipse(mu, 2 * k * np.sqrt(w[-1]), 2 * k * np.sqrt(w[0]),
                        angle=ang, fill=False, edgecolor=col, ls=ls, lw=1.2,
                        label=lab if k == 1.0 else None)
            ax.add_patch(e)

    fig, (ax, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ell(np.zeros(2), C0, NAVY, "-", r"prior $P_0$")
    ell(mu0r, C0r, RED, "--", r"tilted $P_0^{r}$")
    ax.axhline(mstar, color=OLIVE, lw=0.8, ls=":")
    ax.text(-1.9, mstar + 0.08, r"target $m_*$ on frame 1", color=OLIVE,
            fontsize=7.5)
    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"$a_1$")
    ax.set_xlim(-2.1, 2.6)
    ax.set_ylim(-2.1, 3.1)
    ax.legend(loc="lower right")
    ax.set_title(r"$T=2$: reward on the last frame", pad=4)

    Ts = 16
    u = np.arange(Ts)
    for al, lab, col in ((None, r"walk: $\rho_0+\sigma^2\min(u,v)$", NAVY),
                         (0.8, r"AR(1): $\alpha^{|u-v|}$", RED)):
        C = (rho0 + SIGMA ** 2 * np.minimum.outer(u, u) if al is None
             else al ** np.abs(np.subtract.outer(u, u)))
        Ei = np.zeros((1, Ts))
        Ei[0, -1] = 1.0
        Cr = np.linalg.inv(np.linalg.inv(C) + Ei.T @ Ei / s2)
        a2.semilogy(u, np.diag(C - Cr) / np.diag(C), "o-", ms=3,
                    color=col, label=lab)
    a2.set_xlabel(r"frame index $u$")
    a2.set_ylabel("fractional variance reduction")
    a2.legend(fontsize=7.2)
    a2.set_title(r"how far back the reward reaches, $T=16$", pad=4)
    save(fig, "fig12_p3_reach.pdf")


def fig14_h2():
    def compute():
        rng = np.random.default_rng(17)
        rho0, target = RHO2 / 2.0, 0.10

        def nreq(T, t, structured):
            D = 2 * T
            m, Dl = deltas(t)
            C0 = np.kron(rho0 + SIGMA ** 2 * K_min(T), np.eye(2))
            La = np.linalg.cholesky(C0)
            Ct = m ** 2 * C0 + Dl * np.eye(D)
            Cti = np.linalg.inv(Ct)
            xt = (np.linalg.cholesky(Ct) @ rng.normal(size=(D, 3000))).T
            se = -xt @ Cti
            nrm = np.sqrt(np.mean(np.sum(se ** 2, 1)))

            def err(N):
                a = (La @ rng.normal(size=(D, N))).T
                if structured:
                    r0h = np.mean(a[:, 0:2] ** 2)
                    inc = (a[:, 2:].reshape(N, T - 1, 2)
                           - a[:, :-2].reshape(N, T - 1, 2))
                    C0h = np.kron(r0h + np.mean(inc ** 2) * K_min(T), np.eye(2))
                else:
                    C0h = (a.T @ a) / N
                sh = -xt @ np.linalg.inv(m ** 2 * C0h + Dl * np.eye(D))
                return np.sqrt(np.mean(np.sum((sh - se) ** 2, 1))) / nrm

            for N in np.unique(np.round(
                    np.geomspace(D + 2, 200_000, 44)).astype(int)):
                if np.mean([err(N) for _ in range(3)]) < target:
                    return float(N)
            return float("nan")

        tl = [0.02, 0.035, 0.05, 0.1, 0.2, 0.4, 0.8]
        Tl = [2, 4, 8, 16]
        return {"t": tl,
                "uns": [nreq(8, t, False) for t in tl],
                "str": [nreq(8, t, True) for t in tl],
                "T": Tl,
                "unsD": [nreq(T, 0.05, False) for T in Tl],
                "strD": [nreq(T, 0.05, True) for T in Tl]}

    d = cached("h2.json", compute)
    tl = np.array(d["t"])
    Dl = 1 - np.exp(-2 * tl)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a1.loglog(Dl, d["uns"], "o-", color=NAVY, ms=4, label="unstructured")
    a1.loglog(Dl, d["str"], "s-", color=RED, ms=4, label="structured (2 scalars)")
    a1.loglog(Dl, 1.46 * 16 / Dl, ":", color=GRAY, lw=1.0,
              label=r"$1.46\,D/\Delta_t$")
    a1.set_xlabel(r"$\Delta_t$")
    a1.set_ylabel("N for 10% score error")
    a1.set_title(r"$T=8$, $D=16$; $D(D{+}1)/2=136$", pad=4)
    a1.legend(fontsize=7.2)
    _ticks(a1, [0.05, 0.1, 0.2, 0.5], ["0.05", "0.1", "0.2", "0.5"],
           [20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])

    Ds = 2 * np.array(d["T"], float)
    a2.loglog(Ds, d["unsD"], "o-", color=NAVY, ms=4, label="unstructured")
    a2.loglog(Ds, d["strD"], "s-", color=RED, ms=4, label="structured")
    a2.loglog(Ds, d["unsD"][0] * (Ds / Ds[0]), ":", color=GRAY, lw=1.0,
              label=r"$\propto D$")
    a2.loglog(Ds, d["unsD"][0] * (Ds / Ds[0]) ** 2, "--", color=OLIVE, lw=0.9,
              label=r"$\propto D^2$ (rejected)")
    a2.set_xlabel(r"$D=2T$")
    a2.set_ylabel("N for 10% score error")
    a2.set_title(r"$t=0.05$", pad=4)
    a2.set_ylim(4, 3000)
    a2.legend(fontsize=7.2)
    _ticks(a2, [4, 8, 16, 32], ["4", "8", "16", "32"],
           [10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig14_h2.pdf")


if __name__ == "__main__":
    self_check()
    fig01_two_clocks()
    fig02_ring()
    fig03_gauge()
    fig04_joint_vs_marginal()
    fig05_phi()
    fig06_bandfill()
    fig07_psi_posterior()
    fig08_09_speciation()
    fig10_p2_coupling()
    fig11_p2_field()
    fig12_p3_ellipse()
    fig14_h2()
    print("all figures written")
