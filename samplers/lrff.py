"""Leverage-reweighted RFF (LRFF) sampler.

Absorbed from leverage_reweighted_rff.py plus sample_lrff_from_x from
the original samplers.py monolith.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from typing import Tuple

from ._utils import kernel_matrix, spectral_sampler, NPInputMat, NPSample, NPKernel


# ---------------------------------------------------------------------------
# Recursive ridge-leverage-score Nystrom landmark selection (no K_xi^{-1})
# ---------------------------------------------------------------------------
def _approx_rls(K, idx, landmarks, lam):
    """Dictionary estimate of the lam-ridge leverage scores for points `idx`.

    tilde_l_i = (1/lam) ( K_ii - K_iD (K_DD + lam I)^{-1} K_Di ),  D = landmarks.
    Exact (= lam * true RLS) when D is the full index set; an over-estimate for
    subsets, which is what valid leverage sampling requires.
    """
    diag = np.diag(K)
    if len(landmarks) == 0:
        return np.minimum(diag[idx] / lam, 1.0)
    KDD = K[np.ix_(landmarks, landmarks)] + lam * np.eye(len(landmarks))
    c = cho_factor(KDD)
    KiD = K[np.ix_(idx, landmarks)]
    rho = diag[idx] - np.sum(KiD * cho_solve(c, KiD.T).T, axis=1)
    return np.clip(rho / lam, 0.0, 1.0)


def recursive_rls(K, lam, oversample=3.0, base=128, rng=None):
    """Recursive RLS-Nystrom landmark selection (Musco & Musco, 2017).

    Returns a landmark index array S of size O~(n_eff).  The residual bound
    0 <= K - K_S K_SS^+ K_S^T <= lam I  holds w.h.p. for `oversample` large
    enough, giving factor-2-accurate leverage downstream.
    """
    rng = np.random.default_rng() if rng is None else rng

    def rec(idx):
        m = len(idx)
        if m <= base:
            return list(idx)
        half = idx[rng.random(m) < 0.5]
        dict_landmarks = rec(half)
        tl = _approx_rls(K, idx, dict_landmarks, lam)
        keep = idx[rng.random(m) < np.clip(oversample * tl, 0.0, 1.0)]
        return list(keep) if len(keep) > 0 else list(idx[:base])

    n = K.shape[0]
    return np.unique(rec(np.arange(n)))


# ---------------------------------------------------------------------------
# Nystrom factor and Woodbury handle for the approximate leverage
# ---------------------------------------------------------------------------
def nystrom_factor(K, S, jitter=1e-12):
    """B (n x r) with  Khat = K[:,S] K[S,S]^+ K[S,:] = B B^T."""
    KSS = K[np.ix_(S, S)]
    w, U = eigh(KSS)
    w = np.maximum(w, jitter)
    return K[:, S] @ (U / np.sqrt(w))


class ApproxLeverage:
    """Callable  alpha_t(W)  using  Khat_xi^{-1} = sig2^{-1}(I - B (sig2 I + B^T B)^{-1} B^T).

    Evaluating alpha_t for F frequencies costs O(n r F + r^3), never O(n^2).
    """

    def __init__(self, X, B, sigma2):
        self.X = X
        self.B = B
        self.sigma2 = sigma2
        self.n = X.shape[0]
        r = B.shape[1]
        self._chol = cho_factor(sigma2 * np.eye(r) + B.T @ B)

    def __call__(self, W, chunk=2000):
        """W: (F, d) frequencies -> alpha_t: (F,) approximate leverage.

        Frequencies are processed in batches of `chunk` to bound peak memory
        to O(n * chunk) rather than O(n * F) for large pools.  cho_solve is
        applied separately to real and imaginary parts for scipy < 1.7
        compatibility (scipy 1.6.x rejects complex RHS from a real factor).
        """
        F = W.shape[0]
        out = np.empty(F)
        for s in range(0, F, chunk):
            Wc = W[s:s + chunk]
            Uc = np.exp(1j * (self.X @ Wc.T))           # (n, C)
            BtUc = self.B.T @ Uc                         # (r, C)
            solved = (cho_solve(self._chol, np.real(BtUc))
                      + 1j * cho_solve(self._chol, np.imag(BtUc)))
            out[s:s + chunk] = np.real(np.sum(np.conj(BtUc) * solved, axis=0))
        return (self.n - out) / self.sigma2


# ---------------------------------------------------------------------------
# Frequency sampling from qt  proportional to  alpha_t * p   (SIR)
# ---------------------------------------------------------------------------
def compute_sir_pool(n_freq, d, kind, ell, nu, alpha_fn, rng,
                     pool_factor=20, pool_min=4000):
    """Draw a pool of candidate frequencies and evaluate their leverage scores.

    This is the expensive O(n * r * P) step.  Call once per (dataset, D) and
    reuse the result across many trials via resample_from_pool.

    Returns
    -------
    pool   : (P, d)  candidate frequencies drawn from p
    a_pool : (P,)    their approximate ridge-leverage scores
    Z_hat  : scalar  estimate of E_p[alpha_t]
    """
    P = max(pool_factor * n_freq, pool_min)
    pool = spectral_sampler(P, d, kind, ell, nu, rng)
    a_pool = alpha_fn(pool)
    a_pool = np.maximum(a_pool, 1e-12)
    Z_hat = a_pool.mean()
    return pool, a_pool, Z_hat


def resample_from_pool(pool, a_pool, Z_hat, n_freq, rng):
    """SIR resampling from a precomputed pool — cheap per-trial step.

    Returns
    -------
    W      : (n_freq, d) resampled frequencies
    alpha  : (n_freq,)   their approximate leverage scores
    Z_hat  : scalar      (passed through unchanged)
    """
    probs = a_pool / a_pool.sum()
    sel = rng.choice(len(pool), size=n_freq, replace=True, p=probs)
    return pool[sel], a_pool[sel], Z_hat


def sample_frequencies(n_freq, d, kind, ell, nu, alpha_fn, rng,
                       pool_factor=20, pool_min=4000):
    """Sampling-importance-resampling draw of n_freq frequencies from
    qt(w) proportional to alpha_t(w) p(w).

    Returns
    -------
    W      : (n_freq, d) resampled frequencies (approx ~ qt)
    alpha  : (n_freq,)   their approximate leverage alpha_t(w_j)
    Z_hat  : scalar      estimate of  Zt = E_p[alpha_t]  (pool mean)
    """
    pool, a_pool, Z_hat = compute_sir_pool(
        n_freq, d, kind, ell, nu, alpha_fn, rng, pool_factor, pool_min)
    return resample_from_pool(pool, a_pool, Z_hat, n_freq, rng)


# ---------------------------------------------------------------------------
# Full scheme: build features and draw a sample
# ---------------------------------------------------------------------------
def reweighted_rff_sampler(X, kind="rbf", ell=0.1, nu=1.5, sigma2=1e-2,
                           n_freq=512, oversample=3.0, base=128,
                           pool_factor=20, rng=None, return_diagnostics=False,
                           alpha_fn=None, pool_cache=None):
    """Construct the leverage-reweighted RFF feature matrix Phi (n x D), D = 2*n_freq,
    such that  Phi Phi^T  is an unbiased, low-variance estimate of the kernel K.

    A prior sample is then  f = Phi @ z  with  z ~ N(0, I_D)  (see draw_sample).

    Returns Phi, or (Phi, diagnostics) if return_diagnostics=True.

    alpha_fn : callable, optional
        Pre-built ApproxLeverage object.  When provided the kernel matrix,
        recursive-RLS and Nystrom-factor steps are skipped (useful when calling
        reweighted_rff_sampler many times for the same data set X but varying
        n_freq, e.g. in a sweep over D).  If None (default) these are computed
        internally.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.atleast_2d(X)
    n, d = X.shape

    if alpha_fn is None:
        # (1) kernel matrix (used only for the O~(n n_eff^2) sketch, never inverted)
        K = kernel_matrix(X, kind, ell, nu)

        # (2) recursive-RLS landmarks at ridge level sigma2, then Nystrom factor B
        S = recursive_rls(K, lam=sigma2, oversample=oversample, base=base, rng=rng)
        B = nystrom_factor(K, S)
        alpha_fn = ApproxLeverage(X, B, sigma2)
    else:
        K = None  # not available when caller pre-built alpha_fn

    # (3) draw frequencies from qt ~ alpha_t * p  (SIR), get weights
    if pool_cache is None:
        W, alpha_sel, Z_hat = sample_frequencies(
            n_freq, d, kind, ell, nu, alpha_fn, rng, pool_factor=pool_factor)
    else:
        # Pool already evaluated by caller — just resample (cheap per-trial step)
        _pool, _a_pool, _Z_hat = pool_cache
        W, alpha_sel, Z_hat = resample_from_pool(_pool, _a_pool, _Z_hat, n_freq, rng)

    # (4) feature map.  Per-frequency gain  g_j = sqrt( Z_hat / (n_freq * alpha_j) ),
    #     two columns (cos, sin) per frequency -> D = 2 n_freq features.
    g = np.sqrt(Z_hat / (n_freq * alpha_sel)).astype(np.float32)  # (n_freq,)
    proj = X.astype(np.float32) @ W.T.astype(np.float32)          # (n, n_freq) float32
    Phi = np.empty((n, 2 * n_freq), dtype=np.float32)
    Phi[:, 0::2] = g * np.cos(proj)
    Phi[:, 1::2] = g * np.sin(proj)

    if not return_diagnostics:
        return Phi

    diag = {
        "Z_hat": Z_hat,
        "alpha_min": alpha_sel.min(),
        "alpha_max": alpha_sel.max(),
    }
    if K is not None:
        diag["n_landmarks"] = len(S)
        diag["neff_proxy"] = np.trace(K @ cho_solve(cho_factor(K + sigma2 * np.eye(n)), np.eye(n)))
        diag["K"] = K
    return Phi, diag


def draw_sample(Phi, n_samples=1, sigma_obs=0.0, rng=None):
    """Draw GP prior samples from the feature matrix:  f = Phi z,  z ~ N(0, I_D).

    Optionally adds observation noise of std `sigma_obs` (set to sqrt(sigma2) to
    obtain draws from the noisy GP marginal N(0, K + sigma2 I)).
    Returns array of shape (n,) if n_samples == 1, else (n, n_samples).
    """
    rng = np.random.default_rng() if rng is None else rng
    n, Dfeat = Phi.shape
    z = rng.standard_normal((Dfeat, n_samples))
    f = Phi @ z
    if sigma_obs > 0:
        f = f + sigma_obs * rng.standard_normal((n, n_samples))
    return f[:, 0] if n_samples == 1 else f


# ---------------------------------------------------------------------------
# Public sampler API (moved from samplers.py)
# ---------------------------------------------------------------------------

def sample_lrff_from_x(
        x: NPInputMat, sigma: float, noise_var: float, l: float,
        rng: np.random.Generator, D: int, kernel_type: str = "rbf",
        **kwargs) -> Tuple[NPSample, NPKernel]:
    """Leverage-reweighted RFF sample at points x.  Same external interface as
    sample_rff_from_x so the same sweep harness (sweep.py) drives both methods.

    D is the total number of RFF features (D = 2 * n_freq, must be even).
    The outputscale sigma and noise variance noise_var match sample_se_rff_from_x:
      - Phi from reweighted_rff_sampler has k(0)=1 (no sigma); scaled by sqrt(sigma)
        so that Cov(y_noisefree) ≈ sigma * K_RBF.
      - Additive noise ε ~ N(0, noise_var · I) is drawn with the same rng.

    Note: reweighted_rff_sampler forms the full n×n kernel matrix K for the
    Nyström sketch — O(n²) cost identical to the whitening step already done by
    the harness.  No additional O(n³) work is introduced beyond what the harness
    already performs.
    """
    n = x.shape[0]
    kind = "rbf" if kernel_type in ("rbf", "se") else kernel_type
    nu = kwargs.get("nu", 1.5)
    n_freq = D // 2
    # Build the (n, D) feature matrix.  leverage_reweighted_rff normalises so
    # that Phi @ Phi^T ≈ K with k(0)=1; sigma is applied below.
    Phi = reweighted_rff_sampler(
        X=x, kind=kind, ell=l, nu=nu, sigma2=noise_var,
        n_freq=n_freq, rng=rng,
        alpha_fn=kwargs.get("alpha_fn"),
        pool_factor=kwargs.get("pool_factor", 5),
        pool_cache=kwargs.get("pool_cache"))
    # Apply output scale so Cov(y) ≈ sigma * K.
    # Keep Phi in float32 to avoid upcasting to float64 (halves peak memory).
    Phi = Phi * np.float32(np.sqrt(sigma))
    # Draw prior sample: z ~ N(0, I_D), y = Phi z
    # z is float32 to avoid upcasting Phi; y will be float64 after noise addition.
    z = rng.standard_normal(Phi.shape[1]).astype(np.float32)
    y = (Phi @ z).astype(np.float64)
    # Add observation noise, identical convention to sample_se_rff_from_x
    y_noise = y + rng.normal(scale=np.sqrt(noise_var), size=(n,))
    return y_noise, np.nan
