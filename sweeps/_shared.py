"""Shared utilities for sweep harnesses.

Deduplicates kernel construction, effective-dimension estimation,
LRFF setup, and K̂_ξ accumulation used by both sweeps/matern_bayes
and sweeps/cvm_hyp.
"""

import numpy as np
import scipy.linalg as linalg

from gpsampler.maths import k_se, k_mat
from gpsampler.samplers._utils import kernel_matrix as _km
from gpsampler.samplers.lrff import (
    recursive_rls,
    nystrom_factor,
    ApproxLeverage,
)


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def build_K(x: np.ndarray, nu: float, ell: float, sigma: float = 1.0) -> np.ndarray:
    """Stationary kernel matrix — Matérn (finite ν) or RBF (ν ≥ 1000)."""
    if nu >= 1000.0:
        return k_se(x, x, sigma, ell)
    return k_mat(x, x, sigma, ell, nu=nu)


def kernel_kind(nu: float):
    """Return (kind, nu_effective) for the lrff / spectral_sampler API."""
    if nu >= 1000.0:
        return "rbf", 1.5  # nu unused for rbf
    return "matern", float(nu)


# ---------------------------------------------------------------------------
# Effective dimension
# ---------------------------------------------------------------------------

def neff_hutchinson(
    K: np.ndarray,
    L_xi: np.ndarray,
    n_probes: int = 30,
    rng: np.random.Generator = None,
) -> float:
    """Estimate Tr(K K_ξ⁻¹) via Hutchinson trace estimator.

    Reuses the Cholesky factor L_xi of K_ξ already held by the caller.
    """
    rng = rng or np.random.default_rng()
    n = K.shape[0]
    total = 0.0
    for _ in range(n_probes):
        v = rng.standard_normal(n)
        total += float(np.dot(K @ v, linalg.cho_solve((L_xi, True), v)))
    return total / n_probes


def neff_exact(K: np.ndarray, noise_var: float) -> float:
    """Exact Tr(K(K+σ²I)⁻¹) via eigendecomposition (O(n³), affordable n≤2048)."""
    eigs = np.maximum(np.linalg.eigvalsh(K), 0.0)
    return float(np.sum(eigs / (eigs + noise_var)))


# ---------------------------------------------------------------------------
# LRFF setup (cached once per config)
# ---------------------------------------------------------------------------

def lrff_setup(x: np.ndarray, nu: float, ell: float, noise_var: float):
    """Build ApproxLeverage callable and related objects for an (x, ν, ℓ) config.

    Returns (K_unit, alpha_fn, r_landmarks).
    """
    kind, nu_eff = kernel_kind(nu)
    K_unit = _km(x, kind=kind, ell=ell, nu=nu_eff)
    S = recursive_rls(K_unit, lam=noise_var, rng=np.random.default_rng(99))
    B = nystrom_factor(K_unit, S)
    alpha_fn = ApproxLeverage(x, B, noise_var)
    return K_unit, alpha_fn, len(S)


# ---------------------------------------------------------------------------
# Chunked K̂_ξ accumulation from (omega, a) frequency-amplitude pairs
# ---------------------------------------------------------------------------

def accumulate_khat(
    x: np.ndarray,
    omega: np.ndarray,
    a: np.ndarray,
    sigma: float,
    noise_var: float,
    chunk_size: int = 512,
    dtype: type = np.float64,
) -> np.ndarray:
    """Build K̂_ξ = σ·∑ a²·[cos cosᵀ + sin sinᵀ] + σ²_ξ·I via chunks.

    Parameters
    ----------
    x         : (n, d) input locations
    omega     : (m, d) frequencies
    a         : (m,) per-frequency amplitudes (encode sqrt(2·r/D) normalisation)
    sigma     : kernel output scale σ²
    noise_var : noise variance σ²_ξ
    chunk_size: frequencies per chunk (controls peak memory)
    dtype     : dtype for per-chunk intermediates

    Returns
    -------
    Khat_xi : (n, n) realised covariance with E[K̂_ξ] = σ·K + σ²_ξ·I.
    """
    n = x.shape[0]
    m = omega.shape[0]
    K_acc = np.zeros((n, n), dtype=np.float64)
    for start in range(0, m, chunk_size):
        b = min(chunk_size, m - start)
        w_b = omega[start:start + b]
        a_b = a[start:start + b]
        v = (x @ w_b.T).astype(dtype)
        cv = (np.cos(v) * a_b).astype(dtype)
        sv = (np.sin(v) * a_b).astype(dtype)
        K_acc += cv @ cv.T + sv @ sv.T  # upcasts to float64
    return sigma * K_acc + noise_var * np.eye(n)
