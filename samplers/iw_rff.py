"""Safeguarded importance-weighted RFF (IW-RFF) sampler."""

import numpy as np
from typing import Callable, Tuple

from ._utils import spectral_sampler, NPInputMat, NPSample, NPKernel


# ---------------------------------------------------------------------------
# Spectral-density helpers (private)
# ---------------------------------------------------------------------------

def _log_spectral_density(
    omega: np.ndarray,
    kind: str,
    l: float,
    nu: float,
    d: int,
) -> np.ndarray:
    """Log spectral density log p(omega) for a batch of frequencies.

    RBF   : p = N(0, I/l^2), normalised.
    Matern: p = multivariate-t(2*nu, 0, I/l^2), normalised.

    Parameters
    ----------
    omega : (F, d) frequency array
    kind  : 'rbf' or 'matern'
    l     : kernel lengthscale
    nu    : Matern smoothness (ignored for RBF)
    d     : input dimension

    Returns
    -------
    log_p : (F,) log-density values
    """
    from scipy.special import gammaln as _gammaln
    sq = np.sum(omega ** 2, axis=1)  # (F,)
    if kind in ("rbf", "se"):
        # N(0, I/l^2):  log p = d*log(l) - d/2*log(2*pi) - l^2/2 * sq
        return d * np.log(l) - 0.5 * d * np.log(2.0 * np.pi) - 0.5 * l**2 * sq
    if kind == "matern":
        # multivariate-t(2*nu, 0, I/l^2):
        # log p = log Gamma((2nu+d)/2) - log Gamma(nu)
        #       + d*log(l) - d/2*log(2*nu*pi)
        #       - (2nu+d)/2 * log(1 + l^2*sq/(2*nu))
        log_norm = (
            _gammaln(0.5 * (2.0 * nu + d))
            - _gammaln(nu)
            + d * np.log(l)
            - 0.5 * d * np.log(2.0 * nu * np.pi)
        )
        return log_norm - 0.5 * (2.0 * nu + d) * np.log(
            1.0 + l**2 * sq / (2.0 * nu)
        )
    raise ValueError(f"_log_spectral_density: unknown kernel kind {kind!r}")


# ---------------------------------------------------------------------------
# Sampler: Safeguarded importance-weighted RFF (IW-RFF)
# ---------------------------------------------------------------------------


def _iw_rff_draw_frequencies(
    d: int,
    l: float,
    rng: np.random.Generator,
    m: int,
    eta: float = 0.5,
    guard_scale: float = None,
    g_sampler: Callable = None,
    g_logpdf: Callable = None,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw m frequencies and compute per-frequency IW-RFF amplitudes.

    Returns (omega, a) where omega is (m, d) and a is (m,).
    The amplitudes encode sqrt(2 * p/q / D) with D = 2*m, so that
        K_acc = sum_j a_j^2 (cos_j cos_j^T + sin_j sin_j^T)
    satisfies E[K_acc] = K (unit output scale).
    """
    D = 2 * m
    kind = "rbf" if kernel_type in ("rbf", "se") else kernel_type

    if eta >= 1.0:
        omega = spectral_sampler(m, d, kind, l, nu, rng)
        a = np.full(m, np.sqrt(2.0 / D), dtype=np.float64)
        return omega, a

    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0, 1]; got {eta}")

    have_custom_g = (g_sampler is not None and g_logpdf is not None)
    if not have_custom_g:
        if guard_scale is None:
            guard_scale = 1.0
        if not (0.0 < guard_scale <= 1.0):
            raise ValueError(f"guard_scale must be in (0, 1]; got {guard_scale}")
        l_guard = l * guard_scale

        def _g_sampler(n_samp, _d, _rng):
            return spectral_sampler(n_samp, _d, kind, l_guard, nu, _rng)

        def _g_logpdf(omega):
            return _log_spectral_density(omega, kind, l_guard, nu, omega.shape[1])

        g_sampler = _g_sampler
        g_logpdf = _g_logpdf

    # Draw from mixture q_eta = (1-eta)*g + eta*p
    from_p = rng.uniform(size=m) < eta
    n_from_p = int(from_p.sum())
    n_from_g = m - n_from_p

    omega = np.empty((m, d), dtype=np.float64)
    if n_from_p > 0:
        omega[from_p] = spectral_sampler(n_from_p, d, kind, l, nu, rng)
    if n_from_g > 0:
        omega[~from_p] = g_sampler(n_from_g, d, rng)

    # IS weights: a_j = sqrt(2 p / (D * q_eta))
    log_p = _log_spectral_density(omega, kind, l, nu, d)
    log_g = g_logpdf(omega)
    log_q = np.logaddexp(
        np.log(1.0 - eta) + log_g,
        np.log(eta) + log_p,
    )
    a = np.sqrt(2.0 * np.exp(log_p - log_q) / D)  # (m,)

    return omega, a


def _build_iw_rff_features(
    x: np.ndarray,
    l: float,
    rng: np.random.Generator,
    D: int,
    eta: float = 0.5,
    guard_scale: float = None,
    g_sampler: Callable = None,
    g_logpdf: Callable = None,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> np.ndarray:
    """Build IW-RFF feature matrix Z such that E[Z @ Z.T] = K.

    Frequencies are drawn from q_eta = (1 - eta)*g + eta*p where p is the
    kernel spectral density and g is a heavier-tailed guard.  Default g = p
    recovers plain RFF (the safest choice; supply guard_scale < 1 for a
    broadened guard).

    Returns Z : (n, D) feature matrix with block layout [cos | sin].
    """
    if D % 2 != 0:
        raise ValueError("D must be even")

    n, d = x.shape
    m = D // 2

    omega, a = _iw_rff_draw_frequencies(
        d, l, rng, m,
        eta=eta, guard_scale=guard_scale,
        g_sampler=g_sampler, g_logpdf=g_logpdf,
        kernel_type=kernel_type, nu=nu,
    )

    proj = x.astype(np.float64) @ omega.T  # (n, m)
    Z = np.empty((n, D), dtype=np.float64)
    Z[:, :m] = a[None, :] * np.cos(proj)
    Z[:, m:] = a[None, :] * np.sin(proj)
    return Z


def sample_iw_rff_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    eta: float = 0.5,
    guard_scale: float = None,
    g_sampler: Callable = None,
    g_logpdf: Callable = None,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """Safeguarded importance-weighted RFF GP prior sampler.

    Draws D//2 frequencies from the defensive mixture

        q_eta(omega) = (1 - eta) * g(omega) + eta * p(omega)

    where p is the kernel spectral density and g is a guard proposal.
    Each frequency is importance-weighted by p / q_eta so that
    E[Z Z^T] = K exactly.

    Returns (y_noise, np.nan) — no n×n matrix is formed.
    """
    n = x.shape[0]
    Z = _build_iw_rff_features(
        x, l, rng, D,
        eta=eta,
        guard_scale=guard_scale,
        g_sampler=g_sampler,
        g_logpdf=g_logpdf,
        kernel_type=kernel_type,
        nu=nu,
    )
    Z = float(np.sqrt(sigma)) * Z
    w = rng.standard_normal(D)
    y_noise = Z @ w + rng.normal(scale=float(np.sqrt(noise_var)), size=n)
    return y_noise, np.nan
