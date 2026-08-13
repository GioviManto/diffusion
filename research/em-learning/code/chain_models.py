"""Clean Markov-chain data models and the OU noising channel.

The data model throughout this project is a scalar AR(1)-type chain with i.i.d.
additive innovations,

    a_1 ~ mu,           a_{i+1} = rho * a_i + eps_{i+1},

observed through the coordinatewise Ornstein-Uhlenbeck channel of a diffusion model,

    x_i = alpha_t a_i + sqrt(Delta_t) z_i,
    alpha_t = exp(-t),  Delta_t = 1 - exp(-2t),  z_i ~ N(0,1).

Three distinct sources of randomness appear in this project and are easy to confuse,
so they are named explicitly here and kept separate everywhere in the code:

    1. the *innovation* eps      -- enters the clean chain, via the transition kernel K
    2. the *diffusion noise* z   -- enters the observation, via the likelihood ell
    3. the *reverse-SDE noise*   -- enters only when generating samples (see sampling.py)

Belief propagation never samples 1 or 2.  It integrates over them: eps through K and z
through ell.

Innovation families
-------------------
All families are parameterised so that Var(eps) = q exactly, which keeps the *second-order*
structure identical across families:

    Var(a_i) = q / (1 - rho^2),      Cov(a_i, a_{i-k}) = rho^k Var(a_i).

This is deliberate and is the point of the whole design.  Because the covariance is shared,
any estimator that uses only first and second moments -- in particular the Gaussian message
closure, which is exactly the LMMSE denoiser -- cannot tell these families apart.  Every
difference between families is therefore a pure test of non-Gaussian structure.

With the stationary initial condition mu = N(0, q/(1-rho^2)) the chain is stationary; taking
q = 1 - rho^2 additionally normalises Var(a_i) = 1, which is the convention used in the
earlier reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Innovation families
# -----------------------------------------------------------------------------

#: Names of every supported innovation family.
INNOVATION_FAMILIES = ("gaussian", "laplace", "student", "mixture", "uniform")


@dataclass(frozen=True)
class ChainConfig:
    """Parameters of a clean AR(1)-type Markov chain.

    Attributes
    ----------
    n:
        Chain length (number of sites).
    rho:
        Lag-one correlation of the stationary chain.
    innovation:
        One of :data:`INNOVATION_FAMILIES`.
    q:
        Innovation variance.  ``None`` selects ``1 - rho**2``, which normalises the
        stationary variance of the chain to one.
    student_df:
        Degrees of freedom for the Student-t family.  Must exceed 2 for finite variance.
    mixture_sep:
        Separation of the symmetric two-component Gaussian mixture, in units of the
        innovation standard deviation.  Larger values make the innovation strongly
        bimodal and are the hardest case for a single-Gaussian closure.
    mixture_weight:
        Weight of the positive component; 0.5 gives a symmetric mixture.
    """

    n: int = 50
    rho: float = 0.85
    innovation: str = "gaussian"
    q: float | None = None
    student_df: float = 5.0
    mixture_sep: float = 1.5
    mixture_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.innovation not in INNOVATION_FAMILIES:
            raise ValueError(
                f"innovation must be one of {INNOVATION_FAMILIES}, got {self.innovation!r}"
            )
        if not 0.0 <= abs(self.rho) < 1.0:
            raise ValueError(f"|rho| must be < 1 for a stationary chain, got {self.rho}")
        if self.innovation == "student" and self.student_df <= 2.0:
            raise ValueError("student_df must exceed 2 so that the variance is finite")
        if not 0.0 < self.mixture_weight < 1.0:
            raise ValueError("mixture_weight must lie strictly between 0 and 1")

    @property
    def innovation_variance(self) -> float:
        """Innovation variance q, defaulting to 1 - rho^2."""
        return float(1.0 - self.rho ** 2) if self.q is None else float(self.q)

    @property
    def stationary_variance(self) -> float:
        """Stationary variance of the clean chain, q / (1 - rho^2)."""
        return self.innovation_variance / (1.0 - self.rho ** 2)


def sample_innovations(
    rng: np.random.Generator,
    size: tuple[int, ...],
    config: ChainConfig,
) -> np.ndarray:
    """Draw innovations with mean zero and variance exactly ``config.innovation_variance``."""
    q = config.innovation_variance
    fam = config.innovation

    if fam == "gaussian":
        return rng.normal(0.0, np.sqrt(q), size)

    if fam == "laplace":
        # Laplace(0,b) has variance 2b^2.
        return rng.laplace(0.0, np.sqrt(q / 2.0), size)

    if fam == "student":
        # Student-t with nu d.o.f. has variance nu/(nu-2); rescale to q.
        nu = config.student_df
        raw = rng.standard_t(nu, size)
        return raw * np.sqrt(q * (nu - 2.0) / nu)

    if fam == "mixture":
        # Symmetric two-component Gaussian mixture with means +-m and common std s.
        # Var = m^2 * 4w(1-w) + s^2 for weight w on +m and (1-w) on -m, with the mean
        # removed. We parameterise m = mixture_sep * s and solve for s so that Var = q.
        w = config.mixture_weight
        sep = config.mixture_sep
        # mean = (2w-1) m ; Var = m^2 (1 - (2w-1)^2) + s^2 = m^2 * 4w(1-w) + s^2
        # with m = sep * s  =>  s^2 (sep^2 * 4w(1-w) + 1) = q
        s = np.sqrt(q / (sep ** 2 * 4.0 * w * (1.0 - w) + 1.0))
        m = sep * s
        signs = np.where(rng.random(size) < w, 1.0, -1.0)
        return signs * m - (2.0 * w - 1.0) * m + rng.normal(0.0, s, size)

    if fam == "uniform":
        # Uniform(-h,h) has variance h^2/3.
        h = np.sqrt(3.0 * q)
        return rng.uniform(-h, h, size)

    raise ValueError(f"unknown innovation family {fam!r}")


def innovation_pdf(grid: np.ndarray, config: ChainConfig) -> np.ndarray:
    """Density of the innovation on a grid.

    Used to build the exact transition kernel and as the ground truth against which
    nonparametric EM estimates of the innovation density are compared.
    """
    q = config.innovation_variance
    fam = config.innovation

    if fam == "gaussian":
        return np.exp(-0.5 * grid ** 2 / q) / np.sqrt(2.0 * np.pi * q)

    if fam == "laplace":
        b = np.sqrt(q / 2.0)
        return np.exp(-np.abs(grid) / b) / (2.0 * b)

    if fam == "student":
        nu = config.student_df
        scale = np.sqrt(q * (nu - 2.0) / nu)
        u = grid / scale
        from math import lgamma, pi

        log_norm = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * np.log(nu * pi)
        return np.exp(log_norm - 0.5 * (nu + 1.0) * np.log1p(u ** 2 / nu)) / scale

    if fam == "mixture":
        w = config.mixture_weight
        sep = config.mixture_sep
        s = np.sqrt(q / (sep ** 2 * 4.0 * w * (1.0 - w) + 1.0))
        m = sep * s
        shift = (2.0 * w - 1.0) * m
        g = lambda c: np.exp(-0.5 * (grid - c) ** 2 / s ** 2) / np.sqrt(2.0 * np.pi * s ** 2)
        return w * g(m - shift) + (1.0 - w) * g(-m - shift)

    if fam == "uniform":
        h = np.sqrt(3.0 * q)
        return np.where(np.abs(grid) <= h, 1.0 / (2.0 * h), 0.0)

    raise ValueError(f"unknown innovation family {fam!r}")


# -----------------------------------------------------------------------------
# Sampling clean chains and noisy observations
# -----------------------------------------------------------------------------

def sample_chains(rng: np.random.Generator, n_chains: int, config: ChainConfig) -> np.ndarray:
    """Sample ``n_chains`` independent clean chains, shape ``(n_chains, config.n)``.

    The chain is started from its stationary distribution so that every site has the
    same marginal variance and Cov(a_i, a_j) = rho^{|i-j|} * stationary_variance exactly.
    """
    n = config.n
    a = np.empty((n_chains, n))
    a[:, 0] = rng.normal(0.0, np.sqrt(config.stationary_variance), n_chains)
    eps = sample_innovations(rng, (n_chains, n - 1), config)
    for i in range(1, n):
        a[:, i] = config.rho * a[:, i - 1] + eps[:, i - 1]
    return a


def ou_coefficients(t: float) -> tuple[float, float]:
    """Return ``(alpha_t, Delta_t)`` for the OU noising channel at diffusion time ``t``."""
    if t < 0.0:
        raise ValueError("diffusion time must be non-negative")
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    return alpha, delta


def noise_chains(
    rng: np.random.Generator,
    a: np.ndarray,
    t: float,
) -> tuple[np.ndarray, float, float]:
    """Apply the OU channel at diffusion time ``t``.

    Returns ``(x, alpha_t, Delta_t)``.  At ``t = 0`` this returns the clean chains and
    ``Delta_t = 0``; callers that divide by ``Delta_t`` must guard against that case.
    """
    alpha, delta = ou_coefficients(t)
    x = alpha * a + np.sqrt(delta) * rng.normal(size=a.shape)
    return x, alpha, delta


def ar1_covariance(config: ChainConfig) -> np.ndarray:
    """Stationary covariance Sigma_0 with entries ``stationary_variance * rho^{|i-j|}``.

    This matrix depends only on ``rho`` and ``q`` -- never on the innovation family --
    which is exactly why second-moment estimators cannot distinguish the families.
    """
    idx = np.arange(config.n)
    return config.stationary_variance * config.rho ** np.abs(idx[:, None] - idx[None, :])


def noisy_covariance(config: ChainConfig, t: float) -> np.ndarray:
    """Covariance of the noisy observation, ``Sigma_t = alpha_t^2 Sigma_0 + Delta_t I``."""
    alpha, delta = ou_coefficients(t)
    return alpha ** 2 * ar1_covariance(config) + delta * np.eye(config.n)
