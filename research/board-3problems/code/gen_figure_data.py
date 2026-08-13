"""
Four-panel comparison for the board model, plus an end-to-end validation that
the closed-form joint score generates the right law.

 (a) data trajectories        (b) same, after the gauge U (de-rotated)
 (c) reverse SDE, EXACT JOINT score
 (d) reverse SDE, PER-FRAME MARGINAL scores  (the early-work object)

Radial functions are tabulated once per SDE step and interpolated, so the
Bessel quadrature is not re-run per sample.
"""
import json
import numpy as np
from scipy.special import ive
from numpy.polynomial.legendre import leggauss

LAM, SIGMA, PSI, T = 0.05, 0.22, 2 * np.pi / 12, 12
TMAX, TMIN, NSTEP, NSHOW, NSAMP = 4.0, 5e-3, 1500, 6, 3000

_NR = 200
_XR, _WR = leggauss(_NR)
_RMAX = 1.0 + 12.0 * np.sqrt(LAM)
_RHO = 0.5 * _RMAX * (_XR + 1.0)
_WQ = 0.5 * _RMAX * _WR
_LOGRING = -(_RHO - 1.0) ** 2 / (2 * LAM)
NB = 400


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def K_min(T):
    u = np.arange(T)
    return np.minimum.outer(u, u).astype(float)


def phi_tables(kappa, m, beta_max):
    """Phi(beta) = dlogZ/dbeta on a grid, for a batch of kappa values.

    kappa, beta_max: shape (M,).  Returns betas (M,NB), phis (M,NB).
    """
    kappa = np.atleast_1d(kappa)
    beta_max = np.atleast_1d(beta_max)
    betas = np.linspace(0, 1, NB)[None, :] * beta_max[:, None]      # (M,NB)
    z = m * betas[:, :, None] * _RHO[None, None, :]                 # (M,NB,NR)
    e = (_LOGRING[None, None, :]
         - 0.5 * m ** 2 * kappa[:, None, None] * _RHO[None, None, :] ** 2
         + np.abs(z))
    e -= e.max(axis=2, keepdims=True)
    base = _WQ[None, None, :] * np.exp(e)
    Z = np.sum(base * _RHO[None, None, :] * ive(0, z), axis=2)
    Zp = m * np.sum(base * _RHO[None, None, :] ** 2 * ive(1, z), axis=2)
    return betas, Zp / np.maximum(Z, 1e-300)


def joint_score(X, t):
    """Exact joint score, walk frame. X: (B,T,2)."""
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    A = m ** 2 * SIGMA ** 2 * K_min(T) + Dl * np.eye(T)
    Ainv = np.linalg.inv(A)
    g = Ainv @ np.ones(T)
    kappa = float(np.ones(T) @ g)
    b = np.einsum("u,bui->bi", g, X)
    beta = np.linalg.norm(b, axis=1)
    bt, pt = phi_tables(kappa, m, max(beta.max() * 1.02, 1e-9))
    ph = np.interp(beta, bt[0], pt[0])
    bhat = b / np.maximum(beta, 1e-300)[:, None]
    return (-np.einsum("uv,bvi->bui", Ainv, X)
            + ph[:, None, None] * g[None, :, None] * bhat[:, None, :])


def marginal_score(X, t):
    """Per-frame marginal score. X: (B,T,2)."""
    m, Dl = np.exp(-t), 1.0 - np.exp(-2 * t)
    v = m ** 2 * SIGMA ** 2 * np.arange(T) + Dl                     # (T,)
    r = np.linalg.norm(X, axis=2)                                   # (B,T)
    beta = r / v[None, :]
    bt, pt = phi_tables(1.0 / v, m, beta.max(axis=0) * 1.02 + 1e-9)
    ph = np.stack([np.interp(beta[:, u], bt[u], pt[u]) for u in range(T)], 1)
    rad = X / np.maximum(r, 1e-300)[:, :, None]
    return -X / v[None, :, None] + (ph / v[None, :])[:, :, None] * rad


def sample_data(n, rng):
    rho = rng.normal(1.0, np.sqrt(LAM), size=int(n * 2))
    rho = rho[rho > 0]
    rho = rho[rng.random(rho.size) < rho / rho.max()][:n]
    th = rng.uniform(0, 2 * np.pi, rho.size)
    z = np.stack([rho * np.cos(th), rho * np.sin(th)], 1)
    traj, R = [z.copy()], rot(PSI)
    for _ in range(T - 1):
        z = z @ R.T + SIGMA * rng.normal(size=z.shape)
        traj.append(z.copy())
    return np.stack(traj, 1)


def reverse_sde(score_fn, n, rng):
    X = rng.normal(size=(n, T, 2))
    grid = TMIN * (TMAX / TMIN) ** (np.linspace(1, 0, NSTEP + 1))
    for i in range(NSTEP):
        t, dt = grid[i], grid[i] - grid[i + 1]
        s = score_fn(X, t)
        X = X + dt * (X + 2 * s) + np.sqrt(2 * dt) * rng.normal(size=X.shape)
    return X


def regauge(X):
    return np.stack([X[:, u, :] @ rot(u * PSI).T for u in range(T)], 1)


def degauge(X):
    return np.stack([X[:, u, :] @ rot(-u * PSI).T for u in range(T)], 1)


def coherence(X):
    w = X[..., 0] + 1j * X[..., 1]
    prev = np.where(np.abs(w[:, :-1]) < 1e-12, 1e-12, w[:, :-1])
    ang = np.angle(w[:, 1:] / prev).ravel()
    z = np.mean(np.exp(1j * ang))
    return float(np.angle(z)), float(np.abs(z))


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    data_r = sample_data(NSAMP, rng)
    joint_r = regauge(reverse_sde(joint_score, NSAMP, rng))
    marg_r = reverse_sde(marginal_score, NSAMP, rng)

    stats = {"psi_deg": round(np.degrees(PSI), 2),
             "params": {"T": T, "D": 2 * T, "lam": LAM, "sigma": SIGMA,
                        "n_samples": NSAMP, "n_steps": NSTEP}}
    for k, v in (("data", data_r), ("joint", joint_r), ("marginal", marg_r)):
        mu, conc = coherence(v)
        rad = np.linalg.norm(v, axis=2)
        stats[k] = {"turn_mean_deg": round(np.degrees(mu), 2),
                    "turn_concentration": round(conc, 4),
                    "radius_first_frame": round(float(rad[:, 0].mean()), 4),
                    "radius_last_frame": round(float(rad[:, -1].mean()), 4)}
    # end-to-end validation: per-frame radial moments, data vs joint-score SDE
    rd, rj = np.linalg.norm(data_r, axis=2), np.linalg.norm(joint_r, axis=2)
    stats["validation"] = {
        "max_rel_err_radius_mean_per_frame":
            round(float(np.abs(rd.mean(0) - rj.mean(0)).max() / rd.mean()), 4),
        "max_rel_err_radius_std_per_frame":
            round(float(np.abs(rd.std(0) - rj.std(0)).max() / rd.std()), 4),
        "turn_angle_err_deg":
            round(abs(stats["joint"]["turn_mean_deg"] - stats["psi_deg"]), 3)}

    json.dump({"stats": stats, "traj": {
        "data": data_r[:NSHOW].round(4).tolist(),
        "walk": degauge(data_r[:NSHOW]).round(4).tolist(),
        "joint": joint_r[:NSHOW].round(4).tolist(),
        "marginal": marg_r[:NSHOW].round(4).tolist()}},
        open("figdata.json", "w"))
    print(json.dumps(stats, indent=2))
