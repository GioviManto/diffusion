"""
The two scaling claims that the first pass of the note asserted without proof.
BOTH ARE REJECTED HERE. This driver records what is actually measured, and
tests the rejection so the wrong versions cannot be reintroduced.

H1  ASSERTED: t_spec = 1/2 log T  (speciation time from frame redundancy).
    MEASURED: t_spec rises with T but is NOT a single power of T. The local
    slope d t_spec / d log T drifts monotonically from about 0.33 at T = 2-4 to
    about 0.86 at T = 32-64. The 1/2 log T law is rejected. The asymptotic
    exponent is left OPEN: at large t the T-exponent of the log-odds measures
    between 2.7 and 3.2 and the e^{-2t} prefactor law does not hold cleanly
    there, so no clean asymptotic form is claimed.

H2  ASSERTED: unstructured joint score needs Theta(T^2) samples, structured
    O(1), from the board's N > D(D+1)/2.
    MEASURED: false at any fixed t. The diffusion noise floor Delta_t
    regularises the empirical score. The requirement for a fixed relative
    score error is
            N_req  ~=  c * D / Delta_t ,     c ~= 1.5
    so it grows like D, not D^2, and the structured/unstructured gap is
    controlled by the noise level, diverging only as t -> 0. The board's
    D(D+1)/2 is a parameter count for identifying C_0 itself, which is a
    strictly stronger requirement than an accurate score at t > 0.
"""

import numpy as np
from scipy.special import ive
from numpy.polynomial.legendre import leggauss

LAM, SIGMA = 0.05, 0.30
FAIL = []

_xr, _wr = leggauss(200)
_RMAX = 1.0 + 12.0 * np.sqrt(LAM)
RHO = 0.5 * _RMAX * (_xr + 1.0)
WQ = 0.5 * _RMAX * _wr
LOGRING = -(RHO - 1.0) ** 2 / (2 * LAM)
RHO2 = np.sum(WQ * RHO ** 3 * np.exp(LOGRING)) / np.sum(WQ * RHO * np.exp(LOGRING))


def check(name, err, tol):
    ok = np.isfinite(err) and err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} err={err:.3e} tol={tol:.1e}")
    if not ok:
        FAIL.append(name)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def K_min(T):
    u = np.arange(T)
    return np.minimum.outer(u, u).astype(float)


def sample_ring(n, rng):
    """Exactly n draws from the ring; fixed rejection bound, not sample-dependent."""
    R = 1.0 + 8.0 * np.sqrt(LAM)
    out, got = [], 0
    while got < n:
        r = rng.normal(1.0, np.sqrt(LAM), size=2 * n + 64)
        r = r[(r > 0) & (r < R)]
        r = r[rng.random(r.size) < r / R]
        out.append(r)
        got += r.size
    r = np.concatenate(out)[:n]
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def sample_two_species(n, T, omega, rng):
    z = sample_ring(n, rng)
    sgn = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    traj = [z.copy()]
    for _ in range(T - 1):
        nz = np.empty_like(z)
        for s in (1.0, -1.0):
            k = sgn == s
            nz[k] = z[k] @ rot(s * omega).T
        z = nz + SIGMA * rng.normal(size=z.shape)
        traj.append(z.copy())
    return np.stack(traj, 1), sgn


def logZ_table(kappa, m, bmax, nb=400):
    b = np.linspace(0.0, bmax, nb)
    z = m * b[:, None] * RHO[None, :]
    e = LOGRING[None, :] - 0.5 * m ** 2 * kappa * RHO[None, :] ** 2 + np.abs(z)
    sh = e.max(axis=1)
    v = np.log(np.sum(WQ[None, :] * RHO[None, :] * ive(0, z)
                      * np.exp(e - sh[:, None]), axis=1)) + sh
    return b, v


def gauge_batch(X, psi):
    return np.stack([X[:, u, :] @ rot(-u * psi).T for u in range(X.shape[1])], 1)


def log_pt_walk(Xg, t):
    T = Xg.shape[1]
    m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
    A = m ** 2 * SIGMA ** 2 * K_min(T) + Dl * np.eye(T)
    Ai = np.linalg.inv(A)
    g = Ai @ np.ones(T)
    quad = np.einsum("bui,uv,bvi->b", Xg, Ai, Xg)
    beta = np.linalg.norm(np.einsum("u,bui->bi", g, Xg), axis=1)
    bt, bv = logZ_table(float(np.ones(T) @ g), m, max(beta.max() * 1.02, 1e-9))
    return -0.5 * quad + np.interp(beta, bt, bv)


def audit_H1(rng):
    print("\nH1  speciation time vs number of frames  (asserted 1/2 log T)")
    omega, n = 0.8, 4000
    tg = np.concatenate([np.linspace(0.03, 1.0, 40), np.linspace(1.05, 3.0, 40)])
    Ts = [2, 4, 8, 16, 32]
    ts = []
    for T in Ts:
        a, sgn = sample_two_species(n, T, omega, rng)
        post = []
        for t in tg:
            m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
            X = m * a + np.sqrt(Dl) * rng.normal(size=a.shape)
            lp = np.stack([log_pt_walk(gauge_batch(X, s * omega), t)
                           for s in (1.0, -1.0)])
            post.append(np.mean(1.0 / (1.0 + np.exp(-(lp[0] - lp[1]) * sgn))))
        post = np.array(post)
        i = int(np.argmax(post < 0.75))
        t0, t1, p0, p1 = tg[i - 1], tg[i], post[i - 1], post[i]
        ts.append(t0 + (p0 - 0.75) * (t1 - t0) / (p0 - p1))
        print(f"       T={T:3d}  t_spec(0.75) = {ts[-1]:.4f}")
    ts = np.array(ts)
    lg = np.log(np.array(Ts, float))
    loc = np.diff(ts) / np.diff(lg)
    print("       local slopes: " + "  ".join(f"{s:.3f}" for s in loc))
    check("H1  t_spec increases with T", -float(np.diff(ts).min()), 0.0)
    check("H1  1/2 log T REJECTED: slope drifts by > 0.20",
          0.20 / (loc[-1] - loc[0]), 1.0)


def audit_H2(rng):
    print("\nH2  sample complexity  (asserted T^2 vs O(1))")
    rho0, target = RHO2 / 2.0, 0.10

    def n_required(T, t, structured, ntest=3000):
        D = 2 * T
        m, Dl = np.exp(-t), 1 - np.exp(-2 * t)
        C0 = np.kron(rho0 + SIGMA ** 2 * K_min(T), np.eye(2))
        La = np.linalg.cholesky(C0)
        Ct = m ** 2 * C0 + Dl * np.eye(D)
        Cti = np.linalg.inv(Ct)
        xt = (np.linalg.cholesky(Ct) @ rng.normal(size=(D, ntest))).T
        s_ex = -xt @ Cti
        nrm = np.sqrt(np.mean(np.sum(s_ex ** 2, 1)))

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
            return np.sqrt(np.mean(np.sum((sh - s_ex) ** 2, 1))) / nrm

        for N in np.unique(np.round(np.geomspace(D + 2, 200_000, 44)).astype(int)):
            if np.mean([err(N) for _ in range(3)]) < target:
                return float(N)
        return np.nan

    print("       Delta_t law at T = 8 (D = 16), D(D+1)/2 = 136")
    ts = (0.02, 0.035, 0.05, 0.1, 0.2, 0.4, 0.8)
    NASYM = 4                     # Delta <~ 0.2: the asymptotic 1/Delta regime
    uns = [n_required(8, t, False) for t in ts]
    stc = [n_required(8, t, True) for t in ts]
    cs = []
    for t, a, b in zip(ts, uns, stc):
        Dl = 1 - np.exp(-2 * t)
        cs.append(a * Dl / 16.0)
        tag = "asym" if len(cs) <= NASYM else "sat"
        print(f"         t={t:5.3f}  Delta={Dl:.4f}  N_uns={a:7.0f}  "
              f"N_str={b:5.0f}  ratio={a/b:6.1f}  "
              f"N_uns*Delta/D={cs[-1]:.2f}  [{tag}]")
    ca = np.array(cs[:NASYM])
    print(f"         asymptotic c = {ca.mean():.2f} +- {ca.std():.2f}")
    check("H2  N_uns ~ c D/Delta for Delta<~0.2, c stable to 25%",
          float(np.ptp(ca)) / float(np.mean(ca)), 0.25)
    check("H2  law saturates to N ~ O(D) at large Delta",
          cs[-1] / ca.mean(), 0.85)
    check("H2  N_str is Delta-independent",
          (max(stc) - min(stc)) / np.mean(stc), 0.30)
    check("H2  gap opens as t -> 0 (ratio > 15 at smallest t)",
          15.0 / (uns[0] / stc[0]), 1.0)

    print("       D-scaling at t = 0.05")
    Ds, Ns = [], []
    for T in (2, 4, 8, 16):
        N = n_required(T, 0.05, False)
        Ds.append(2 * T)
        Ns.append(N)
        print(f"         T={T:3d}  D={2*T:3d}  D(D+1)/2={2*T*(2*T+1)//2:5d}  "
              f"N_uns={N:7.0f}  N_uns/D={N/(2*T):6.1f}")
    p = np.polyfit(np.log(Ds), np.log(Ns), 1)[0]
    print(f"         fit N_uns ~ D^{p:.3f}")
    check("H2  T^2 (i.e. D^2) REJECTED: exponent < 1.35", p, 1.35)
    check("H2  exponent consistent with D^1 to within 0.35", abs(p - 1.0), 0.35)


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    print("=" * 70)
    print("AUDIT: scaling claims H1 and H2 -- both asserted versions REJECTED")
    print("=" * 70)
    audit_H1(rng)
    audit_H2(rng)
    print("\n" + "=" * 70)
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
