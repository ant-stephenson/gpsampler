"""Nystrom-preconditioned CG / Lanczos GP prior sampler.

Absorbed from the CG/Lanczos sections of the original samplers.py monolith.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional

from gpsampler.maths import k_se, k_mat
from ._utils import NPInputMat, NPSample, NPKernel


# ---------------------------------------------------------------------------
# Nystrom preconditioner
# ---------------------------------------------------------------------------

class NystromPreconditioner:
    """Nyström preconditioner P = K̃ + η σ_ξ² I for K_ηξ = K + η σ_ξ² I.

    K̃ = V Σ² V^T is a rank-m Nyström approximation of K (not K_ηξ), built
    from random landmark columns.  Cheap O(nm) P^{±1/2} applies via the
    Sherman-Morrison-Woodbury low-rank structure:

        P^{1/2}  v = √(ησ_ξ²) v  + V [(√(Σ²+ησ_ξ²) − √(ησ_ξ²)) ⊙ (V^T v)]
        P^{-1/2} v = v/√(ησ_ξ²)  + V [(1/√(Σ²+ησ_ξ²) − 1/√(ησ_ξ²)) ⊙ (V^T v)]

    Parameters
    ----------
    K        : (n, n) kernel matrix (without noise).
    eta      : noise-split parameter η ∈ (0, 1).
    noise_var: σ_ξ² — observation noise variance.
    rank     : Nyström rank m; defaults to ⌊√n⌋.
    landmarks: (m,) integer index array; if None, chosen uniformly at random.
    rng      : numpy random Generator for landmark selection.
    jitter   : small positive value clipped onto negative eigenvalues of K[I,I].
    """

    def __init__(
        self,
        K: np.ndarray,
        eta: float,
        noise_var: float,
        rank: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        jitter: float = 1e-12,
    ) -> None:
        n = K.shape[0]
        if rank is None:
            rank = max(1, int(np.sqrt(n)))
        rank = min(rank, n)

        self.eta = float(eta)
        self.noise_var = float(noise_var)
        self.reg = eta * noise_var  # η σ_ξ²

        if landmarks is None:
            _rng = rng if rng is not None else np.random.default_rng()
            landmarks = _rng.choice(n, size=rank, replace=False)
        landmarks = np.asarray(landmarks)

        K_II = K[np.ix_(landmarks, landmarks)]
        K_nI = K[:, landmarks]  # (n, m)

        w, R = np.linalg.eigh(K_II)
        w = np.maximum(w, jitter)
        U = K_nI @ (R * (1.0 / np.sqrt(w)))  # (n, m)

        V_u, s, _ = np.linalg.svd(U, full_matrices=False)

        self.V: np.ndarray = V_u          # (n, r)
        self.sigma2: np.ndarray = s ** 2  # (r,) eigenvalues of K̃

        a = float(np.sqrt(self.reg))
        sqrt_sum = np.sqrt(self.sigma2 + self.reg)

        self._a = a
        self._scale_sqrt = sqrt_sum - a
        self._scale_invsqrt = 1.0 / sqrt_sum - 1.0 / a

    def apply_sqrt(self, v: np.ndarray) -> np.ndarray:
        """Apply P^{1/2} to vector v.  Cost O(nm)."""
        coords = self.V.T @ v
        return self._a * v + self.V @ (self._scale_sqrt * coords)

    def apply_inv_sqrt(self, v: np.ndarray) -> np.ndarray:
        """Apply P^{-1/2} to vector v.  Cost O(nm)."""
        coords = self.V.T @ v
        return (1.0 / self._a) * v + self.V @ (self._scale_invsqrt * coords)

    @property
    def dense_P(self) -> np.ndarray:
        """Dense (n×n) representation of P — for testing only."""
        return self.reg * np.eye(len(self.V)) + self.V @ np.diag(self.sigma2) @ self.V.T


# ---------------------------------------------------------------------------
# Lanczos step-count heuristic
# ---------------------------------------------------------------------------

def suggest_k(
    n: int,
    eta: float,
    noise_var: float,
    eps: float = 0.01,
    lambda1: Optional[float] = None,
) -> int:
    """Suggest the number of Lanczos steps to achieve approximation error ε.

    Bound (Chebyshev analysis):

        k ≥ log[ n (λ₁ + η σ_ξ²) / ((1-η) ε² σ_ξ²) ]
            / ( 2 log[ (√κ_η + 1) / (√κ_η − 1) ] )

    with κ_η = (λ₁ + η σ_ξ²) / (η σ_ξ²).  Uses trace bound λ₁ ≤ n when
    lambda1 is not supplied.
    """
    sigma_xi_sq = float(noise_var)
    lam1 = float(n) if lambda1 is None else float(lambda1)

    lam1_reg = lam1 + eta * sigma_xi_sq
    lam_n_reg = eta * sigma_xi_sq

    if lam_n_reg <= 0.0:
        raise ValueError("eta * noise_var must be strictly positive")

    kappa = lam1_reg / lam_n_reg
    sqrt_kappa = float(np.sqrt(kappa))

    if sqrt_kappa <= 1.0 + 1e-12:
        return 1

    log_num = np.log(n * lam1_reg / ((1.0 - eta) * eps ** 2 * sigma_xi_sq))
    log_denom = 2.0 * np.log((sqrt_kappa + 1.0) / (sqrt_kappa - 1.0))

    if log_denom <= 0.0 or not np.isfinite(log_denom):
        return 1

    return max(1, int(np.ceil(log_num / log_denom)))


# ---------------------------------------------------------------------------
# Lanczos core
# ---------------------------------------------------------------------------

def _lanczos_core(
    matvec: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    k: int,
    tol: float = 1e-14,
    reortho: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """k-step Lanczos factorisation of a symmetric positive-definite operator.

    Returns Q (n, j), alpha (j,), beta (j-1,), j (steps taken).
    Full two-pass Gram-Schmidt re-orthogonalisation when reortho='full'.
    Stops early on lucky breakdown (‖w‖ < tol * ‖u‖).
    """
    n = len(u)
    k = min(k, n)

    Q = np.empty((n, k), dtype=float)
    alpha = np.empty(k, dtype=float)
    beta = np.empty(k - 1, dtype=float)

    norm_u = np.linalg.norm(u)
    breakdown_tol = tol * norm_u

    Q[:, 0] = u / norm_u
    beta_prev = 0.0
    q_prev = np.zeros(n, dtype=float)

    for j in range(k):
        q = Q[:, j]
        w = matvec(q) - beta_prev * q_prev

        alpha[j] = float(q @ w)
        w -= alpha[j] * q

        if reortho == "full":
            for _pass in range(2):
                for i in range(j + 1):
                    w -= (Q[:, i] @ w) * Q[:, i]

        beta_j = np.linalg.norm(w)

        if j < k - 1:
            if beta_j < breakdown_tol:
                return Q[:, : j + 1], alpha[: j + 1], beta[:j], j + 1
            beta[j] = beta_j
            q_prev = q
            beta_prev = beta_j
            Q[:, j + 1] = w / beta_j

    return Q, alpha, beta, k


def _tsqrt_times_e1(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute T_k^{1/2} e₁ for symmetric tridiagonal T_k via eigendecomposition."""
    k = len(alpha)
    T = np.diag(alpha.copy())
    if len(beta) > 0:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    theta, S = np.linalg.eigh(T)
    theta = np.maximum(theta, 0.0)
    return S @ (np.sqrt(theta) * S[0, :])


# ---------------------------------------------------------------------------
# Lanczos GP prior sampler
# ---------------------------------------------------------------------------

def sample_lanczos_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    k: int,
    kernel_type: str = "rbf",
    eta: float = 0.8,
    preconditioner: Optional[NystromPreconditioner] = None,
    reortho: str = "full",
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """Lanczos GP prior sampler.

    Draws ŷ ~ GP(0, K_ξ) using a k-step Lanczos approximation to K_ηξ^{1/2} u.

    Unpreconditioned (Part A)
    --------------------------
    1. K_ηξ = K(x,x) + η σ_ξ² I.
    2. u ~ N(0, I_n).
    3. k-step Lanczos on K_ηξ: Q_k, T_k.
    4. f̂ = ‖u‖ · Q_k · T_k^{1/2} · e₁.
    5. ŷ = f̂ + ξ,  ξ ~ N(0, (1-η) σ_ξ² I).

    Preconditioned (Part B)
    ------------------------
    Same but Lanczos runs on W = P^{-1/2} K_ηξ P^{-1/2}, then f̂ = P^{1/2} ĝ.

    Parameters
    ----------
    x            : (n, d) input locations.
    sigma        : kernel output scale.
    noise_var    : observation noise variance σ_ξ².
    l            : kernel lengthscale.
    rng          : numpy random Generator.
    k            : number of Lanczos steps.
    kernel_type  : 'rbf'/'se', 'exp', 'matern32', 'matern52'.
    eta          : noise-split parameter η ∈ (0,1).
    preconditioner: NystromPreconditioner or None.
    reortho      : re-orthogonalisation strategy ('full' or 'none').

    Returns
    -------
    y_noise : (n,) sample with covariance ≈ K_ξ.
    np.nan  : placeholder (approx_cov not computed).
    """
    n = x.shape[0]

    kt = kernel_type.lower()
    if kt in ("rbf", "se"):
        K = k_se(x, x, sigma, l)
    elif kt == "exp":
        K = k_mat(x, x, sigma, l, nu=0.5)
    elif kt == "matern32":
        K = k_mat(x, x, sigma, l, nu=1.5)
    elif kt == "matern52":
        K = k_mat(x, x, sigma, l, nu=2.5)
    else:
        raise ValueError(
            f"Unsupported kernel_type {kernel_type!r}. "
            "Options: 'rbf'/'se', 'exp', 'matern32', 'matern52'."
        )

    K_etaxi = K + eta * noise_var * np.eye(n)
    u = rng.standard_normal(n)

    if preconditioner is None:
        def _mv(v: np.ndarray) -> np.ndarray:
            return K_etaxi @ v

        Q, alpha, beta, _k = _lanczos_core(_mv, u, k, reortho=reortho)
        tsqrt_e1 = _tsqrt_times_e1(alpha, beta)
        f_hat = np.linalg.norm(u) * (Q @ tsqrt_e1)
    else:
        pre = preconditioner

        def _mv_W(v: np.ndarray) -> np.ndarray:
            return pre.apply_inv_sqrt(K_etaxi @ pre.apply_inv_sqrt(v))

        Q, alpha, beta, _k = _lanczos_core(_mv_W, u, k, reortho=reortho)
        tsqrt_e1 = _tsqrt_times_e1(alpha, beta)
        g_hat = np.linalg.norm(u) * (Q @ tsqrt_e1)
        f_hat = pre.apply_sqrt(g_hat)

    xi = rng.standard_normal(n) * np.sqrt((1.0 - eta) * noise_var)
    y_noise = f_hat + xi
    return y_noise, np.nan


def sample_cg_from_x(x: NPInputMat, sigma: float, noise_var: float, l: float,
                     rng: np.random.Generator, k: int) -> Tuple[NPSample, float]:
    """Lanczos GP prior sampler — delegates to sample_lanczos_from_x."""
    return sample_lanczos_from_x(x, sigma, noise_var, l, rng, k)
