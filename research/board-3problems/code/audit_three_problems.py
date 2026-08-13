"""
Audit of the three board problems, each in its simplest setting (two frames).

P1  rotating ring, random rotation angle.  a = (z0, R_psi z0 + sigma eta) in R^4,
    z0 on the noisy unit ring, psi ~ p(psi).  Three latents (r0, theta0, psi).
    Claims:
      P1a  tr(X^T A_t^{-1} X) does not depend on psi
      P1b  beta(psi)^2 = |x0|^2/D1^2 + |x1|^2/D2^2 + 2|x0||x1|cos(gamma-psi)/(D1 D2)
      P1c  m E[z0 | x, psi] = Phi_t(beta) bhat
      P1d  p(psi|x,t) proportional to p(psi) Z_t(beta(psi))     [vs 3-D quadrature]
      P1e  score of the psi-mixture = posterior average of gauged scores
      P1f  two-species log-odds -> (m^2 <rho^2>) |x0||x1| sin(gamma) sin(omega)
           / (D1 D2)  as m -> 0

P2  additive two-frame chain, x1 = x0 + c + eta (the board's middle panel).
    Claims:
      P2a  Gaussian prior: s = -C_t^{-1}(x - m E[a])            [vs finite diff]
      P2b  off-diagonal coupling closed form and its two limits
      P2c  Gaussian-mixture prior: exact responsibility-weighted score
      P2d  the sum of per-frame marginal scores has zero coupling at every t

P3  terminal-frame reward / stochastic optimal control (the board's right panel).
    Claims:
      P3a  Woodbury form of the tilted covariance C0^r
      P3b  back-propagation profile of the reward along the internal time
      P3c  log h_t closed form and its gradient                 [vs finite diff]
      P3d  the controlled reverse SDE samples N(mu0^r, C0^r)
"""

import numpy as np
from scipy.special import ive
from numpy.polynomial.legendre import leggauss

LAM, SIGMA = 0.05, 0.30
FAIL = []


def check(name, err, tol):
    ok = np.isfinite(err) and err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56s} err={err:.3e} tol={tol:.1e}")
    if not ok:
        FAIL.append(name)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# --------------------------------------------------------------- P1 machinery

_NR = 900
_xr, _wr = leggauss(_NR)
_RMAX = 1.0 + 12.0 * np.sqrt(LAM)
RHO = 0.5 * _RMAX * (_xr + 1.0)
WQ = 0.5 * _RMAX * _wr
LOGRING = -(RHO - 1.0) ** 2 / (2 * LAM)


def Zt(beta, kappa, m, want_mean=False):
    """Z_t(beta) with a common exponent shift; optionally Z'_t too."""
    z = m * RHO * beta
    e = LOGRING - 0.5 * m ** 2 * kappa * RHO ** 2 + np.abs(z)
    shift = e.max()
    base = WQ * np.exp(e - shift)
    Z = np.sum(base * RHO * ive(0, z))
    if not want_mean:
        return Z, shift
    Zp = m * np.sum(base * RHO ** 2 * ive(1, z))
    return Z, shift, Zp / Z


def deltas(t):
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    return m, Dl, Dl, m ** 2 * SIGMA ** 2 + Dl      # m, Delta, Delta_1, Delta_2


def beta_of_psi(x0, x1, psi, D1, D2):
    b = x0 / D1 + rot(-psi) @ x1 / D2
    return np.linalg.norm(b), b


def walk_logp_and_score(X, t):
    """log P_t^walk(X) (up to a psi-independent constant) and its score. X:(2,2)."""
    m, Dl, D1, D2 = deltas(t)
    A = np.diag([D1, D2])
    Ainv = np.diag([1 / D1, 1 / D2])
    g = np.array([1 / D1, 1 / D2])
    kappa = g.sum()
    b = X.T @ g
    beta = np.linalg.norm(b)
    Z, shift, phi = Zt(beta, kappa, m, want_mean=True)
    quad = np.trace(X.T @ Ainv @ X)
    logp = -0.5 * quad + np.log(Z) + shift - 0.5 * np.log(np.linalg.det(A))
    bhat = b / max(beta, 1e-300)
    s = -Ainv @ X + phi * np.outer(g, bhat)
    return logp, s, beta, phi, bhat


def ref_logp_per_psi(x, t, psis, nodes):
    """log P_t(x | psi) for each psi, by 2-D quadrature over z0."""
    z0, w, rho = nodes
    m, Dl, D1, D2 = deltas(t)
    Sig = np.array([D1, D1, D2, D2])
    out = []
    for psi in psis:
        mu = m * np.concatenate([z0, z0 @ rot(psi).T], 1)          # (G,4)
        quad = np.sum((x[None, :] - mu) ** 2 / Sig[None, :], 1)
        e = (LOGRING_of(rho) - 0.5 * quad
             - 0.5 * np.log(Sig).sum() - 2 * np.log(2 * np.pi))
        mx = e.max()
        out.append(np.log(np.sum(w * np.exp(e - mx))) + mx)
    return np.array(out)


def ref_logp_mixture(x, t, psis, wpsi, nodes):
    """log P_t(x) for the psi-mixture by direct quadrature over (z0, psi)."""
    tot = ref_logp_per_psi(x, t, psis, nodes) + np.log(wpsi)
    mx = tot.max()
    return mx + np.log(np.sum(np.exp(tot - mx)))


def LOGRING_of(rho):
    return -(rho - 1.0) ** 2 / (2 * LAM)


def polar_nodes(n_rho=260, n_phi=256):
    xr, wr = leggauss(n_rho)
    rmax = 1.0 + 9.0 * np.sqrt(LAM)
    rho = 0.5 * rmax * (xr + 1.0)
    wrho = 0.5 * rmax * wr
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    R, P = np.meshgrid(rho, phi, indexing="ij")
    W = (wrho[:, None] * (2 * np.pi / n_phi)) * R
    z0 = np.stack([R.ravel() * np.cos(P.ravel()), R.ravel() * np.sin(P.ravel())], 1)
    return z0, W.ravel(), R.ravel()


def audit_P1(rng):
    print("\nP1  rotating ring, two frames, random psi   (D=4, latents=3)")
    t = 0.30
    m, Dl, D1, D2 = deltas(t)
    nodes = polar_nodes()
    x0, x1 = rng.normal(size=2) * 0.8, rng.normal(size=2) * 0.8
    X = np.stack([x0, x1])

    # P1a  quadratic form is psi-free -- ONLY at T=2, where A_t is diagonal.
    # U_psi conjugates A_t (x) I_2 into itself iff A_t is diagonal in u, and
    # K_{uv}=min(u,v) is diagonal only for T=2.  Pin both sides of this.
    q = [np.trace((np.stack([x0, rot(-p) @ x1])).T
                  @ np.diag([1 / D1, 1 / D2])
                  @ np.stack([x0, rot(-p) @ x1])) for p in (0.0, 0.9, 2.4)]
    check("P1a  tr(X^T A^-1 X) psi-free at T=2",
          max(abs(q[0] - q[1]), abs(q[0] - q[2])) / abs(q[0]), 1e-14)
    for TT in (3, 5):
        A = m ** 2 * SIGMA ** 2 * np.minimum.outer(
            np.arange(TT), np.arange(TT)).astype(float) + Dl * np.eye(TT)
        Ai = np.linalg.inv(A)
        XT = rng.normal(size=(TT, 2))
        qq = []
        for p in (0.0, 0.6, 1.9, -1.1):
            Xg = np.stack([rot(-u * p) @ XT[u] for u in range(TT)])
            qq.append(np.trace(Xg.T @ Ai @ Xg))
        check(f"P1a  psi-freeness FAILS at T={TT} (scope pinned)",
              1e-2 / (float(np.ptp(qq)) / abs(qq[0])), 1.0)

    # P1b  beta(psi) identity
    gam = np.arctan2(x1[1], x1[0]) - np.arctan2(x0[1], x0[0])
    r0, r1 = np.linalg.norm(x0), np.linalg.norm(x1)
    err = 0.0
    for p in (0.0, 0.7, 2.0, -1.3):
        b, _ = beta_of_psi(x0, x1, p, D1, D2)
        pred = np.sqrt(r0 ** 2 / D1 ** 2 + r1 ** 2 / D2 ** 2
                       + 2 * r0 * r1 * np.cos(gam - p) / (D1 * D2))
        err = max(err, abs(b - pred))
    check("P1b  beta(psi) closed form", err, 1e-13)

    # P1c  Phi bhat = m E[z0 | x, psi]
    for p in (0.0, 1.1):
        Xg = np.stack([x0, rot(-p) @ x1])
        _, _, beta, phi, bhat = walk_logp_and_score(Xg, t)
        z0, w, rho = nodes
        lw = (LOGRING_of(rho)
              - 0.5 * (np.sum((Xg[0] - m * z0) ** 2, 1) / D1
                       + np.sum((Xg[1] - m * z0) ** 2, 1) / D2))
        lw -= lw.max()
        pw = w * np.exp(lw)
        post = (pw[:, None] * z0).sum(0) / pw.sum()
        check(f"P1c  Phi*bhat = m E[z0|x]  (psi={p})",
              np.abs(phi * bhat - m * post).max(), 1e-9)

    # P1d/P1e  psi-mixture: density and score against 3-D quadrature
    npsi = 96
    psis = 2 * np.pi * np.arange(npsi) / npsi
    wpsi = np.full(npsi, 1.0 / npsi)
    lw = np.array([walk_logp_and_score(np.stack([x0, rot(-p) @ x1]), t)[0]
                   for p in psis])
    mx = lw.max()
    x4 = np.concatenate([x0, x1])

    # the claim is normalisation-free: p(psi|x,t) prop. to p(psi) Z_t(beta(psi)).
    # (i) the psi-posterior matches quadrature exactly
    post_cf = wpsi * np.exp(lw - mx)
    post_cf /= post_cf.sum()
    lref = ref_logp_per_psi(x4, t, psis, nodes)
    post_ref = wpsi * np.exp(lref - lref.max())
    post_ref /= post_ref.sum()
    check("P1d  psi-posterior = quadrature", np.abs(post_cf - post_ref).max(), 1e-9)
    # (ii) the closed-form density differs from the reference by one constant only
    offs = []
    for _ in range(4):
        y = rng.normal(size=4) * 0.8
        Y0, Y1 = y[:2], y[2:]
        lwy = np.array([walk_logp_and_score(np.stack([Y0, rot(-p) @ Y1]), t)[0]
                        for p in psis])
        my = lwy.max()
        offs.append(my + np.log(np.sum(wpsi * np.exp(lwy - my)))
                    - ref_logp_mixture(y, t, psis, wpsi, nodes))
    check("P1d  density offset is x-independent (ring norm.)",
          float(np.ptp(offs)), 1e-8)
    # the posterior peaks at the observed turning angle gamma
    check("P1d  argmax_psi p(psi|x,t) = gamma",
          abs(np.angle(np.exp(1j * (psis[post_cf.argmax()] - gam)))),
          2 * np.pi / npsi)

    post_psi = wpsi * np.exp(lw - mx)
    post_psi /= post_psi.sum()
    s_mix = np.zeros((2, 2))
    for p, wp in zip(psis, post_psi):
        U = rot(-p)
        _, sg, *_ = walk_logp_and_score(np.stack([x0, U @ x1]), t)
        s_mix += wp * np.stack([sg[0], U.T @ sg[1]])
    h = 1e-4
    s_fd = np.zeros(4)
    for i in range(4):
        xp, xm = x4.copy(), x4.copy()
        xp[i] += h
        xm[i] -= h
        s_fd[i] = (ref_logp_mixture(xp, t, psis, wpsi, nodes)
                   - ref_logp_mixture(xm, t, psis, wpsi, nodes)) / (2 * h)
    check("P1e  mixture score = posterior average of gauged scores",
          np.abs(s_mix.ravel() - s_fd).max() / np.abs(s_fd).max(), 3e-5)

    # P1g  R3 (score = psi-posterior average) at T=3, where the T=2
    # simplification is unavailable.  Reference: quadrature over (z0, psi).
    TT, tt = 3, 0.35
    mm, DDl = np.exp(-tt), 1 - np.exp(-2 * tt)
    A3 = mm ** 2 * SIGMA ** 2 * np.minimum.outer(
        np.arange(TT), np.arange(TT)).astype(float) + DDl * np.eye(TT)
    A3i = np.linalg.inv(A3)
    g3 = A3i @ np.ones(TT)
    kap3 = float(np.ones(TT) @ g3)
    z0g, wg, rhog = nodes

    def walk_cf(X):
        b = X.T @ g3
        beta = np.linalg.norm(b)
        Z, sh, phi = Zt(beta, kap3, mm, want_mean=True)
        lp = -0.5 * np.trace(X.T @ A3i @ X) + np.log(Z) + sh
        return lp, -A3i @ X + phi * np.outer(g3, b / max(beta, 1e-300))

    def walk_ref(X):
        acc = np.zeros(z0g.shape[0])
        for i in range(2):
            d = X[:, i][None, :] - mm * z0g[:, i][:, None]
            acc += np.einsum("gu,uv,gv->g", d, A3i, d)
        e = LOGRING_of(rhog) - 0.5 * acc
        mx = e.max()
        return np.log(np.sum(wg * np.exp(e - mx))) + mx

    def ug(p, X):
        return np.stack([rot(-u * p) @ X[u] for u in range(TT)])

    npsi3 = 128
    ps3 = 2 * np.pi * np.arange(npsi3) / npsi3
    w3 = np.full(npsi3, 1.0 / npsi3)
    X3 = rng.normal(size=(TT, 2)) * 0.8

    def mix_ref_logp(Xf):
        v = np.array([walk_ref(ug(p, Xf)) for p in ps3]) + np.log(w3)
        mx = v.max()
        return mx + np.log(np.sum(np.exp(v - mx)))

    lcf = np.array([walk_cf(ug(p, X3))[0] for p in ps3]) + np.log(w3)
    pp = np.exp(lcf - lcf.max())
    pp /= pp.sum()
    s_avg = np.zeros((TT, 2))
    for p, wp in zip(ps3, pp):
        _, sg = walk_cf(ug(p, X3))
        s_avg += wp * np.stack([rot(-u * p).T @ sg[u] for u in range(TT)])
    hh = 1e-4
    s_ref = np.zeros((TT, 2))
    for u in range(TT):
        for i in range(2):
            Xp, Xm = X3.copy(), X3.copy()
            Xp[u, i] += hh
            Xm[u, i] -= hh
            s_ref[u, i] = (mix_ref_logp(Xp) - mix_ref_logp(Xm)) / (2 * hh)
    check("P1g  R3 holds at T=3 (score = psi-posterior average)",
          np.abs(s_avg - s_ref).max() / np.abs(s_ref).max(), 5e-5)

    # P1f  two-species log-odds asymptotics
    om = 0.8
    rho2 = np.sum(WQ * RHO ** 3 * np.exp(LOGRING)) / np.sum(WQ * RHO * np.exp(LOGRING))
    worst = 0.0
    for tt in (2.6, 3.2, 3.8):
        mm, _, dd1, dd2 = deltas(tt)
        kap = 1 / dd1 + 1 / dd2
        lo = []
        for sgn in (+1, -1):
            b, _ = beta_of_psi(x0, x1, sgn * om, dd1, dd2)
            Z, sh = Zt(b, kap, mm)
            lo.append(np.log(Z) + sh)
        exact = lo[0] - lo[1]
        pred = (mm ** 2 * rho2 * r0 * r1 * np.sin(gam) * np.sin(om)) / (dd1 * dd2)
        worst = max(worst, abs(exact - pred) / abs(exact))
    check("P1f  log-odds ~ m^2<rho^2>|x0||x1|sin(gam)sin(om)/(D1D2)", worst, 0.06)


# ------------------------------------------------------------------------ P2

def audit_P2(rng):
    print("\nP2  additive two-frame chain  x1 = x0 + c + eta   (D=2)")
    v0, c, t = 0.7, 1.0, 0.25
    m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
    C0 = np.array([[v0, v0], [v0, v0 + SIGMA ** 2]])
    Ct = m ** 2 * C0 + Dl * np.eye(2)
    Ea = np.array([0.0, c])

    def logp_gauss(x):
        d = x - m * Ea
        return -0.5 * d @ np.linalg.solve(Ct, d)

    x = rng.normal(size=2)
    s_cf = -np.linalg.solve(Ct, x - m * Ea)
    h = 1e-5
    s_fd = np.array([(logp_gauss(x + h * e) - logp_gauss(x - h * e)) / (2 * h)
                     for e in np.eye(2)])
    check("P2a  s = -C_t^-1 (x - m E[a])", np.abs(s_cf - s_fd).max(), 1e-6)

    det = (m ** 4 * v0 * SIGMA ** 2 + 2 * Dl * m ** 2 * v0
           + Dl * m ** 2 * SIGMA ** 2 + Dl ** 2)
    check("P2b  off-diagonal coupling closed form",
          abs(np.linalg.inv(Ct)[0, 1] - (-m ** 2 * v0 / det)), 1e-13)
    # t -> 0 limit is -1/sigma^2.  Using m^2 = 1 - Delta,
    #   coupling = -(1/sigma^2)[1 + Delta(1 - 2/sigma^2 - 1/v0)] + O(Delta^2),
    # so the remainder is O(Delta) with prefactor (2/sigma^2 + 1/v0 - 1).
    # The naive prefactor (2/sigma^2 + 1/v0) is wrong by 4.2% here and is
    # rejected below.
    pref = 2 / SIGMA ** 2 + 1 / v0 - 1
    ratios, ratios_naive = [], []
    for t0 in (1e-5, 1e-6, 1e-7, 1e-8):
        m0, Dl0 = np.exp(-t0), 1 - np.exp(-2 * t0)
        Ct0 = m0 ** 2 * C0 + Dl0 * np.eye(2)
        rel = abs(np.linalg.inv(Ct0)[0, 1] + 1 / SIGMA ** 2) * SIGMA ** 2
        ratios.append(rel / (Dl0 * pref))
        ratios_naive.append(rel / (Dl0 * (2 / SIGMA ** 2 + 1 / v0)))
    check("P2b  t->0 remainder O(Delta), prefactor 2/s^2+1/v0-1",
          max(abs(np.mean(ratios) - 1.0), float(np.ptp(ratios))), 1e-3)
    check("P2b  naive prefactor 2/s^2+1/v0 rejected",
          1.0 / abs(np.mean(ratios_naive) - 1.0), 1e3)
    tb = 6.0
    mb, Dlb = np.exp(-tb), 1 - np.exp(-2 * tb)
    Ctb = mb ** 2 * C0 + Dlb * np.eye(2)
    check("P2b  t->inf coupling ~ -e^{-2t} v0",
          abs(np.linalg.inv(Ctb)[0, 1] + mb ** 2 * v0) / (mb ** 2 * v0), 5e-3)

    # P2c mixture prior
    mus, ws, vv = np.array([-1.6, 1.4]), np.array([0.45, 0.55]), 0.09
    C0m = np.array([[vv, vv], [vv, vv + SIGMA ** 2]])
    Ctm = m ** 2 * C0m + Dl * np.eye(2)
    nus = np.stack([np.array([mu, mu + c]) for mu in mus])

    def logp_mix(x):
        ll = []
        for w, nu in zip(ws, nus):
            d = x - m * nu
            ll.append(np.log(w) - 0.5 * d @ np.linalg.solve(Ctm, d))
        ll = np.array(ll)
        mx = ll.max()
        return mx + np.log(np.sum(np.exp(ll - mx)))

    x = rng.normal(size=2) * 1.2
    ll = np.array([np.log(w) - 0.5 * (x - m * nu) @ np.linalg.solve(Ctm, x - m * nu)
                   for w, nu in zip(ws, nus)])
    r = np.exp(ll - ll.max())
    r /= r.sum()
    s_cf = sum(rk * (-np.linalg.solve(Ctm, x - m * nu)) for rk, nu in zip(r, nus))
    s_fd = np.array([(logp_mix(x + h * e) - logp_mix(x - h * e)) / (2 * h)
                     for e in np.eye(2)])
    check("P2c  mixture score = responsibility-weighted",
          np.abs(s_cf - s_fd).max(), 1e-5)

    marg = np.array([-1 / Ct[0, 0], -1 / Ct[1, 1]])
    check("P2d  marginal score is diagonal (zero coupling)",
          abs(np.diag(np.diag(np.linalg.inv(np.diag(np.diag(Ct)))))[0, 1]), 1e-15)
    print(f"       joint coupling {np.linalg.inv(Ct)[0,1]:+.4f} vs marginal 0; "
          f"diag {np.linalg.inv(Ct)[0,0]:+.4f} vs {marg[0]:+.4f}")


# ------------------------------------------------------------------------ P3

def audit_P3(rng):
    print("\nP3  terminal-frame reward, Gaussian trajectory   (T=6 frames, d=1)")
    T, rho0, s2 = 6, 0.5, 0.4
    K = np.minimum.outer(np.arange(T), np.arange(T)).astype(float)
    C0 = rho0 + SIGMA ** 2 * K                      # ring-anchored walk kernel
    E = np.zeros((1, T))
    E[0, T - 1] = 1.0
    mtgt = np.array([1.8])

    C0r_direct = np.linalg.inv(np.linalg.inv(C0) + E.T @ E / s2)
    C0r_wood = C0 - C0 @ E.T @ np.linalg.inv(
        s2 * np.eye(1) + E @ C0 @ E.T) @ E @ C0
    check("P3a  Woodbury tilted covariance",
          np.abs(C0r_direct - C0r_wood).max(), 1e-11)

    prof_pred = (C0[:, T - 1] ** 2) / (s2 + C0[T - 1, T - 1])
    prof_act = np.diag(C0 - C0r_direct)
    check("P3b  variance reduction profile = C0[u,T-1]^2/(s2+C0[T-1,T-1])",
          np.abs(prof_pred - prof_act).max(), 1e-11)
    print("       profile over u: " + " ".join(f"{p:.3f}" for p in prof_act))

    mu0r = C0r_direct @ E.T @ mtgt / s2

    t = 0.5
    m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
    Sp = np.linalg.inv(np.linalg.inv(C0) + (m ** 2 / Dl) * np.eye(T))
    St = E @ Sp @ E.T

    def log_h(x):
        nu = E @ Sp @ ((m / Dl) * x)
        d = nu - mtgt
        return (-0.5 * np.log(np.linalg.det(np.eye(1) + St / s2))
                - 0.5 * d @ np.linalg.solve(St + s2 * np.eye(1), d))

    x = rng.normal(size=T)
    grad_cf = -(m / Dl) * Sp @ E.T @ np.linalg.solve(
        St + s2 * np.eye(1), E @ Sp @ ((m / Dl) * x) - mtgt)
    hh = 1e-5
    grad_fd = np.array([(log_h(x + hh * e) - log_h(x - hh * e)) / (2 * hh)
                        for e in np.eye(T)])
    check("P3c  grad log h_t closed form", np.abs(grad_cf - grad_fd).max(), 1e-7)

    # P3d  controlled reverse SDE reproduces N(mu0^r, C0^r)
    n, nst, tmax, tmin = 60000, 3000, 5.0, 2e-3
    X = rng.normal(size=(n, T))
    grid = tmin * (tmax / tmin) ** np.linspace(1, 0, nst + 1)
    for i in range(nst):
        tt, dt = grid[i], grid[i] - grid[i + 1]
        mm, DD = np.exp(-tt), 1 - np.exp(-2 * tt)
        Ct = mm ** 2 * C0 + DD * np.eye(T)
        sc = -X @ np.linalg.inv(Ct)
        Spp = np.linalg.inv(np.linalg.inv(C0) + (mm ** 2 / DD) * np.eye(T))
        Stt = E @ Spp @ E.T
        nu = (mm / DD) * (X @ Spp @ E.T)
        gh = -(mm / DD) * ((nu - mtgt) / (Stt[0, 0] + s2)) @ (E @ Spp)
        X = X + dt * (X + 2 * (sc + gh)) + np.sqrt(2 * dt) * rng.normal(size=X.shape)
    check("P3d  controlled SDE mean = mu0^r",
          np.abs(X.mean(0) - mu0r).max(), 3e-2)
    check("P3d  controlled SDE covariance = C0^r",
          np.abs(np.cov(X.T) - C0r_direct).max() / np.abs(C0r_direct).max(), 6e-2)


def audit_bandfill():
    """R14: structure of A_t^{-1}, A_t = m^2 sigma^2 K + Delta I, K = min(u,v).

    (a) K is singular (rank T-1) and row 0 vanishes, because frame 0 IS z_0 and
        carries no walk noise.  So A_t is exactly block diagonal,
        [Delta] (+) (rest), and frame 0 decouples in A_t^{-1} at every t.
        In particular A_0^{-1} does not exist -- the naive statement
        "A_0^{-1} = K^{-1}/sigma^2 is tridiagonal" is meaningless.
    (b) On the remaining block the d-th off-diagonal of A_t^{-1} scales as
        (Delta/(m^2 sigma^2))^d, so A_t^{-1} becomes tridiagonal as t -> 0.
        This is the band-fill law of G8 in this model.
    """
    print("\nR14  band structure of A_t^{-1}")
    T = 8
    K = np.minimum.outer(np.arange(T), np.arange(T)).astype(float)
    check("R14a  K singular with rank T-1",
          float(abs(np.linalg.matrix_rank(K) - (T - 1))), 0.0)
    worst_dec, worst_pow = 0.0, 0.0
    for t in (1e-4, 1e-3):
        m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
        Ai = np.linalg.inv(m ** 2 * SIGMA ** 2 * K + Dl * np.eye(T))
        worst_dec = max(worst_dec, np.abs(Ai[0, 1:]).max() / Ai[0, 0])
        eps = Dl / (m ** 2 * SIGMA ** 2)
        d0 = np.abs(np.diag(Ai)).max()
        for d in (1, 2, 3, 4):
            r = np.abs(np.diag(Ai, d)).max() / d0
            worst_pow = max(worst_pow, abs(np.log(r / eps ** d)))
    check("R14a  frame 0 decouples exactly, all t", worst_dec, 1e-14)
    check("R14b  d-th off-band ~ (Delta/m^2 sigma^2)^d", worst_pow, 0.40)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    print("=" * 74)
    print("AUDIT: the three board problems, simplest settings")
    print("=" * 74)
    audit_P1(rng)
    audit_P2(rng)
    audit_P3(rng)
    audit_bandfill()
    print("\n" + "=" * 74)
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
