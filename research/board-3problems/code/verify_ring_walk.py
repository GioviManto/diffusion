"""
Audit of three claims about the board model (rotating point near a circle,
whole trajectory = one data point in R^{2T}, VP/OU generative diffusion).

MODEL (discrete internal time u = 0..T-1, plane coords z_u = (x_u,y_u)):
    z_0 ~ ring:  density in R^2 proportional to g(|z|) = exp(-(|z|-1)^2/(2 lam))
    z_{u+1} = R_psi z_u + sigma * eta_{u+1},   eta ~ N(0, I_2) iid
    a = (z_0,...,z_{T-1}) in R^{2T},  D = 2T
Forward diffusion (VP / OU), acting coordinatewise on all of R^D:
    X_t = m_t a + sqrt(Delta_t) xi,  m_t = e^{-t}, Delta_t = 1 - e^{-2t}

CLAIM A (rotation is a gauge).  With U = blockdiag(R_{-u psi}):
    P_t^rot(x) = P_t^walk(U x)      and     s^rot(x,t) = U^T s^walk(U x, t)
  where "walk" is the SAME model with psi = 0, i.e. a 2D random walk started
  on the ring.  (U orthogonal => commutes with the isotropic channel.)

CLAIM B (exact joint score in closed form).  In the walk frame, writing the
  state as a matrix X in R^{T x 2}, with
    A_t = m_t^2 sigma^2 K + Delta_t I_T,   K_{uv} = min(u,v)
    kappa_t = 1^T A_t^{-1} 1,   b = X^T A_t^{-1} 1 in R^2,   beta = |b|
    Z_t(beta) = int_0^inf rho g(rho) exp(-m_t^2 kappa_t rho^2/2) I_0(m_t rho beta) drho
    Phi_t(beta) = d/dbeta log Z_t(beta)
  the exact joint score is
    grad_X log P_t(X) = -A_t^{-1} X + Phi_t(beta) * (A_t^{-1} 1) bhat^T
  i.e. Gaussian/linear part + RANK-ONE non-Gaussian correction.

CLAIM C (marginal blindness).  Every single-frame marginal of P_t is
  rotationally symmetric in the plane and does NOT depend on psi.  So a
  per-frame (marginal) score model is provably blind to the dynamics.

References used: polar Gauss-Legendre x trapezoid quadrature of the exact
2-parameter mixture, and central finite differences of log P_t.
"""

import numpy as np
from scipy.special import ive
from numpy.polynomial.legendre import leggauss

LAM = 0.05        # ring width parameter
SIGMA = 0.30      # per-step innovation scale
PSI = 0.7         # rotation per frame (rad)

FAILURES = []


def check(name, err, tol):
    ok = np.isfinite(err) and err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58s} err={err:.3e}  tol={tol:.1e}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- model pieces

def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def gauge(T, psi):
    """U = blockdiag(R_{-u psi}), acting on row-major flattened R^{T x 2}."""
    U = np.zeros((2 * T, 2 * T))
    for u in range(T):
        U[2 * u:2 * u + 2, 2 * u:2 * u + 2] = rot(-u * psi)
    return U


def K_min(T):
    u = np.arange(T)
    return np.minimum.outer(u, u).astype(float)


def cond_cov(T, psi, sigma):
    """Cov(z_u, z_v | z_0) = sigma^2 min(u,v) R_{(u-v)psi}, row-major flatten."""
    S = np.zeros((2 * T, 2 * T))
    for u in range(T):
        for v in range(T):
            S[2 * u:2 * u + 2, 2 * v:2 * v + 2] = (
                sigma ** 2 * min(u, v) * rot((u - v) * psi)
            )
    return S


def cond_mean(T, psi, z0):
    """E[a | z_0] = (R^u z_0)_u, row-major flatten."""
    return np.concatenate([rot(u * psi) @ z0 for u in range(T)])


def polar_nodes(n_rho=220, n_phi=256, rho_max=None):
    """Nodes/weights for int_{R^2} f dz = sum w f, polar GL x trapezoid."""
    if rho_max is None:
        rho_max = 1.0 + 9.0 * np.sqrt(LAM)
    xr, wr = leggauss(n_rho)
    rho = 0.5 * rho_max * (xr + 1.0)
    wrho = 0.5 * rho_max * wr
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    wphi = 2 * np.pi / n_phi
    R, P = np.meshgrid(rho, phi, indexing="ij")
    W = (wrho[:, None] * wphi) * R          # includes the rho Jacobian
    z0 = np.stack([R.ravel() * np.cos(P.ravel()), R.ravel() * np.sin(P.ravel())], 1)
    return z0, W.ravel(), R.ravel()


def log_pt_reference(x, T, psi, sigma, t, nodes):
    """log P_t(x) by direct quadrature of the exact 2-parameter mixture."""
    z0, w, rho = nodes
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    Sig = m ** 2 * cond_cov(T, psi, sigma) + Dl * np.eye(2 * T)
    L = np.linalg.cholesky(Sig)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    mu = m * np.stack([cond_mean(T, psi, z) for z in z0])        # (G, 2T)
    d = x[None, :] - mu
    sol = np.linalg.solve(L, d.T)                                # (2T, G)
    quad = np.sum(sol ** 2, 0)
    log_ring = -(rho - 1.0) ** 2 / (2 * LAM)                     # unnormalised
    e = log_ring - 0.5 * quad - 0.5 * logdet - T * np.log(2 * np.pi)
    mx = e.max()
    return mx + np.log(np.sum(w * np.exp(e - mx)))


def score_reference(x, T, psi, sigma, t, nodes, h=1e-4):
    """Central-difference score of the reference log P_t."""
    s = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        s[i] = (log_pt_reference(xp, T, psi, sigma, t, nodes)
                - log_pt_reference(xm, T, psi, sigma, t, nodes)) / (2 * h)
    return s


# ------------------------------------------------- Claim B: the closed form

def radial_integrals(beta, kappa, m, n_rho=4000, rho_max=None):
    """Z_t(beta) and Z_t'(beta), computed with a shared exponent shift."""
    if rho_max is None:
        rho_max = 1.0 + 12.0 * np.sqrt(LAM)
    xr, wr = leggauss(n_rho)
    rho = 0.5 * rho_max * (xr + 1.0)
    w = 0.5 * rho_max * wr
    zarg = m * rho * beta
    # I_nu(z) = ive(nu, z) * e^{|z|}
    e = -(rho - 1.0) ** 2 / (2 * LAM) - 0.5 * m ** 2 * kappa * rho ** 2 + np.abs(zarg)
    mx = e.max()
    base = w * np.exp(e - mx)
    Z = np.sum(base * rho * ive(0, zarg))
    Zp = m * np.sum(base * rho ** 2 * ive(1, zarg))
    return Z, Zp, mx


def score_closed_form(X, T, sigma, t):
    """Exact joint score in the WALK frame (psi = 0). X is (T,2)."""
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    A = m ** 2 * sigma ** 2 * K_min(T) + Dl * np.eye(T)
    Ainv = np.linalg.inv(A)
    one = np.ones(T)
    g = Ainv @ one                       # A^{-1} 1
    kappa = one @ g
    b = X.T @ g                          # (2,)
    beta = np.linalg.norm(b)
    Z, Zp, _ = radial_integrals(beta, kappa, m)
    Phi = Zp / Z
    bhat = b / beta if beta > 0 else np.zeros(2)
    return -Ainv @ X + Phi * np.outer(g, bhat)


# ------------------------------------------------------------------- audits

def audit_gauge(T, t, rng):
    print(f"\nCLAIM A  gauge reduction   (T={T}, D={2*T}, t={t})")
    U = gauge(T, PSI)
    check("U orthogonal", np.abs(U @ U.T - np.eye(2 * T)).max(), 1e-13)
    # conditional covariance maps to the walk one
    Sr = cond_cov(T, PSI, SIGMA)
    Sw = np.kron(K_min(T), np.eye(2)) * SIGMA ** 2
    check("U Cov_rot U^T = sigma^2 K (x) I_2", np.abs(U @ Sr @ U.T - Sw).max(), 1e-13)
    # densities
    nodes = polar_nodes()
    for trial in range(3):
        x = rng.normal(size=2 * T) * 0.8
        lr = log_pt_reference(x, T, PSI, SIGMA, t, nodes)
        lw = log_pt_reference(U @ x, T, 0.0, SIGMA, t, nodes)
        check(f"log P_t^rot(x) = log P_t^walk(Ux)  [trial {trial}]",
              abs(lr - lw), 1e-9)
    # scores
    x = rng.normal(size=2 * T) * 0.8
    sr = score_reference(x, T, PSI, SIGMA, t, nodes)
    sw = score_reference(U @ x, T, 0.0, SIGMA, t, nodes)
    check("s^rot(x) = U^T s^walk(Ux)",
          np.abs(sr - U.T @ sw).max() / max(np.abs(sr).max(), 1e-30), 1e-5)


def audit_closed_form(T, t, rng):
    print(f"\nCLAIM B  exact closed-form joint score   (T={T}, D={2*T}, t={t})")
    nodes = polar_nodes()
    # Phi = d/dbeta log Z, against finite differences
    m = np.exp(-t)
    A = m ** 2 * SIGMA ** 2 * K_min(T) + (1 - np.exp(-2 * t)) * np.eye(T)
    one = np.ones(T)
    kappa = one @ np.linalg.inv(A) @ one
    for beta in (0.3, 1.7):
        Z, Zp, mx = radial_integrals(beta, kappa, m)
        h = 1e-5
        Zp_num = 0.0
        lg = []
        for bb in (beta + h, beta - h):
            Zb, _, mxb = radial_integrals(bb, kappa, m)
            lg.append(np.log(Zb) + mxb)
        Zp_num = (lg[0] - lg[1]) / (2 * h)
        check(f"Phi_t(beta) = dlogZ/dbeta   [beta={beta}]",
              abs(Zp / Z - Zp_num) / abs(Zp_num), 1e-6)
    # the score itself, against finite differences of the reference
    for trial in range(4):
        X = rng.normal(size=(T, 2)) * 0.8
        s_cf = score_closed_form(X, T, SIGMA, t)
        s_fd = score_reference(X.ravel(), T, 0.0, SIGMA, t, nodes).reshape(T, 2)
        rel = np.abs(s_cf - s_fd).max() / max(np.abs(s_fd).max(), 1e-30)
        check(f"closed form = finite-difference score  [trial {trial}]", rel, 2e-5)
    # structure: the non-Gaussian part is rank one
    X = rng.normal(size=(T, 2)) * 0.8
    Ainv = np.linalg.inv(A)
    corr = score_closed_form(X, T, SIGMA, t) + Ainv @ X
    sv = np.linalg.svd(corr, compute_uv=False)
    check("non-Gaussian correction has rank 1", sv[1] / max(sv[0], 1e-30), 1e-12)


def audit_marginal_blindness(T, t, rng):
    print(f"\nCLAIM C  marginal blindness   (T={T}, t={t})")
    m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
    n = 400_000
    for psi in (0.0, PSI, 2.31):
        rg = np.random.default_rng(7)
        # sample ring: radial density ~ rho * exp(-(rho-1)^2/2lam) by rejection
        rho = rg.normal(1.0, np.sqrt(LAM), size=int(n * 1.4))
        rho = rho[rho > 0]
        keep = rg.random(rho.size) < rho / rho.max()
        rho = rho[keep][:n]
        th = rg.uniform(0, 2 * np.pi, rho.size)
        z = np.stack([rho * np.cos(th), rho * np.sin(th)], 1)
        # propagate to frame u = T-1
        for u in range(T - 1):
            z = z @ rot(psi).T + SIGMA * rg.normal(size=z.shape)
        xu = m * z + np.sqrt(Dl) * rg.normal(size=z.shape)
        r = np.linalg.norm(xu, axis=1)
        if psi == 0.0:
            ref_q = np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9])
        else:
            q = np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9])
            check(f"radial quantiles of frame {T-1} independent of psi={psi}",
                  np.abs(q - ref_q).max() / ref_q.mean(), 8e-3)
        ang = np.arctan2(xu[:, 1], xu[:, 0])
        # phase uniform => |mean of e^{i ang}| ~ 0
        check(f"frame {T-1} phase uniform (psi={psi})",
              abs(np.mean(np.exp(1j * ang))), 6e-3)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("AUDIT: ring-anchored rotating walk under VP diffusion")
    print(f"lam={LAM}  sigma={SIGMA}  psi={PSI}")
    print("=" * 78)
    for T, t in ((3, 0.35), (5, 0.12)):
        audit_gauge(T, t, rng)
        audit_closed_form(T, t, rng)
    audit_marginal_blindness(4, 0.25, rng)
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
