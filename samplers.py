from re import I
import numpy as np
from numba import jit, prange
from scipy.special import ellipj, ellipk
from functools import partial
from itertools import repeat
import torch
import gpytorch
from joblib import Parallel, delayed
# from gpytorch.utils import contour_integral_quad
from typing import Callable, Tuple, Optional, Union
from nptyping import NDArray, Shape, Float
from contextlib import ExitStack
import warnings

import math
import warnings
import copy

import torch

try:
    from linear_operator.utils.broadcasting import _matmul_broadcast_shape
    from linear_operator.utils.linear_cg import linear_cg
    from linear_operator.utils.minres import minres
except ImportError:
    pass  # only needed for CIQ sampler; lrff/rff paths work without it
from gpytorch.utils.warnings import NumericalWarning

from gpsampler.utils import msqrt
from gpsampler.maths import k_se, k_mat

try:
    from gpprediction.kernels.keops_kernels import RBFKernel
except ImportError:
    pass  # only needed for KeOps-based CIQ sampler

# warnings.simplefilter("error")

rng = np.random.default_rng(1)
T_TYPE = torch.cuda.DoubleTensor if torch.cuda.is_available(
) else torch.DoubleTensor  # type: ignore

torch.set_default_tensor_type(T_TYPE)

NPInputVec = NDArray[Shape["P,1"], Float]
NPInputMat = NDArray[Shape["N,P"], Float]
NPSample = NDArray[Shape["N,1"], Float]
NPKernel = NDArray[Shape["N,N"], Float]


# @jit(nopython=True)
def k_true(sigma: float, l: float, xp: np.ndarray, xq: np.ndarray) -> float:
    return sigma * np.exp(-0.5*np.dot(xp-xq, xp-xq)/l**2)  # true kernel


@jit(nopython=True, fastmath=True)
def zrf(omega: NDArray[Shape["D, P"],
                       Float],
        D: int, x: NPInputVec) -> NDArray[Shape["[cos,sin] x n_rff"],
                                          Float]:
    if x.ndim == 1:
        n = 1
    else:
        n = x.shape[0]
    v = np.dot(omega, x.T)  # omega @ x.T
    return np.sqrt(2/D) * np.concatenate((np.cos(v), np.sin(v)))


@jit(nopython=True, fastmath=True)
def f_rf(
    omega: NDArray[Shape["D, P"],
                   Float],
    D: int, w: NDArray[Shape["2 x n_rff"],
                       Float],
    x: NPInputVec) -> float: return np.sum(
    w * zrf(omega, D, x))  # GP approximation


# @jit(nopython=True)
def estimate_rff_kernel(
        X: NPInputMat, D: int, ks: float, l: float) -> NPKernel:
    N, d = X.shape
    cov_omega = np.eye(d)/l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)
    Z = zrf(omega, D, X)*np.sqrt(ks)
    approx_cov = np.inner(Z, Z)
    return approx_cov


def construct_kernels(
        l: float, b: float = 1.0, kernel=gpytorch.kernels.RBFKernel(),
        issparse=False) -> gpytorch.kernels.Kernel:
    if issparse:
        kernel = SparseKernel(kernel)
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        kernel = gpytorch.kernels.MultiDeviceKernel(
            kernel, device_ids=range(n_gpus), output_device="cuda:0")
        kernel.base_kernel.base_kernel.lengthscale = l
        kernel.base_kernel.outputscale = b
    else:
        kernel.base_kernel.lengthscale = l
        kernel.outputscale = b
    return kernel


def approx_extreme_eigs(X, noise_var=None):
    max_eig = X.shape[0]
    if noise_var is not None:
        min_eig = noise_var
    else:
        min_eig = 1/max_eig
    return min_eig, max_eig
    raise NotImplementedError


def matsqrt(X, J, Q, reg=1e-6):
    """Calculates the matrix sqrt of a symmetric matrix X using method 3 in
    Hale2008. Note that this implementation is not computationally efficient as
    it directly inverts an nxn matrix. 
    Assumes we have X = X + s_n^2I

    Args:
        X (_type_): _description_
        J (_type_): _description_
        Q (_type_): _description_
        reg (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    n = X.shape[0]
    I = np.eye(n)
    m, M = approx_extreme_eigs(X, reg)
    k2 = m/M
    Kp = ellipk(1 - k2)
    # for N in range(5,25,5):
    for N in [Q]:
        t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
        sn, cn, dn, _ = ellipj(np.imag(t), 1 - k2)
        cn = 1.0 / cn
        dn = dn * cn
        sn = 1j * sn * cn
        w = np.sqrt(m) * sn
        dzdt = cn * dn
        S = np.zeros_like(X)
        for j in range(N):
            S = S - np.linalg.solve(X-w[j]**2 * I, I) * dzdt[j]
        S = -2 * Kp * np.sqrt(m) / (np.pi * N) * X @ S
    return S


def estimate_ciq_kernel(
        X: NPInputMat, J: int, Q: int, ks: float, l: float, nv=None) -> NPKernel:
    kernel = construct_kernels(l, ks)
    n, d = X.shape
    K = kernel(torch.tensor(X)).detach().numpy()
    rootK = matsqrt(K, J, Q, nv)
    return np.real(rootK @ rootK)


def generate_ciq_data(n: int, xmean: np.ndarray, xcov_diag: np.ndarray,
                      noise_var: float, kernelscale: float, lenscale: float, kernel_type: str,
                      J: int, Q: int, checkpoint_size: int = 1500,
                      max_preconditioner_size: int = 0) -> Tuple[NPInputMat, NPSample]:
    """ Generates a data sample from a MVN and a sample from an approximate GP
    using CIQ to approximate K^1/2 b

    Args:
        n (int): Length of sample
        xmean (np.ndarray): Mean of x distribution
        xcov_diag (np.ndarray): Variances of x values
        noise_var (float): Noise variance of GP
        kernelscale (float): scaling factor for GP kernel
        lenscale (float): RBF lengthscale
        J (int): # Lanczsos iterations
        Q (int): # Quadrature points
        checkpoint_size (int): Kernel checkpointing size. Larger is faster, but more memory.
                               0 means no checkpointing and should be used if possible.
                               Otherwise choose largest value that memory allows.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: sampled x values, noisy GP sample
    """
    input_dim = xmean.shape[0]
    assert input_dim == xcov_diag.shape[0]

    cov_diag = torch.as_tensor(xcov_diag[0].reshape((1, -1)))
    mean = torch.as_tensor(xmean.reshape((1, -1)))
    x = torch.randn(n, input_dim) * cov_diag + mean

    sample, approx_cov = sample_ciq_from_x(
        x, kernelscale, noise_var, lenscale, kernel_type, rng, J, Q,
        checkpoint_size, max_preconditioner_size)

    return x.cpu().numpy(), sample


def generate_rff_data(n: int, xmean: np.ndarray, xcov_diag: np.ndarray,
                      noise_var: float, kernelscale: float, lenscale: float,
                      D: int, kernel_type: str = "rbf", **kwargs) -> Tuple[NPInputMat, NPSample]:
    """ Generates a data sample from a MVN and a sample from an approximate GP using RFF

    Args:
        n (int): Length of sample
        xmean (np.ndarray): Mean of x distribution
        xcov_diag (np.ndarray): Variances of x values
        noise_var (float): Noise variance of GP
        kernelscale (float): scaling factor for GP kernel
        lenscale (float): RBF lengthscale
        D (int): # RFF

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: sampled x values, noise-free
   sample and noisy GP sample
    """
    assert D % 2 == 0
    input_dim = xmean.shape[0]
    assert input_dim == xcov_diag.shape[0]

    xcov = np.diag(xcov_diag)
    x = rng.multivariate_normal(xmean, xcov, n)

    noisy_sample, approx_cov = sample_rff_from_x(
        x, kernelscale, noise_var, lenscale, rng, D, kernel_type, **kwargs)
    return x, noisy_sample


def sample_chol_from_x(x: NPInputMat, sigma: float, noise_var: float, l: float,
                       rng: np.random.Generator, L: np.ndarray) -> Tuple[NPSample, NPKernel]:
    n, d = x.shape
    u = rng.standard_normal(n)
    y_noise = L @ u
    approx_cov = L @ L.T
    return y_noise, approx_cov


# ---------------------------------------------------------------------------
# Lanczos (Krylov-subspace) GP prior sampler
# ---------------------------------------------------------------------------
#
# Notation
# --------
# K        : (n×n) SPD kernel matrix with k(0) = σ_f² = 1.
# σ_ξ²     : observation noise variance.
# η        : split parameter, η ∈ (0, 1).
# K_ηξ     = K + η σ_ξ² I   — regularised kernel operated on by Lanczos.
# K_ξ      = K + σ_ξ² I     — target observation covariance.
#
# Exactness identity (preconditioned case)
# -----------------------------------------
# Let P = K̃ + η σ_ξ² I be the Nyström preconditioner, where K̃ is a
# rank-m Nyström approximation of K.  Define W = P^{-1/2} K_ηξ P^{-1/2}.
# Then (P^{1/2} W^{1/2})(P^{1/2} W^{1/2})^T = K_ηξ exactly.
#
# κ̃ = κ(W) = max/min generalised eigenvalue of (K_ηξ, P).
# TRAP: np.linalg.cond(P^{-1} K_ηξ) returns the singular-value ratio of
# the non-symmetric product — a different quantity.


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


def sample_rff_from_x(x: NPInputMat, sigma: float, noise_var: float, l: float,
                      rng: np.random.Generator, D: int, kernel_type: str = "rbf",
                      **kwargs) -> Tuple[NPSample, NPKernel]:
    """ Generates sample from approximate GP using RFF method at points x

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw; 1D array of length n and approx cov
    """
    if kernel_type == "rbf":
        return sample_se_rff_from_x(x, sigma, noise_var, l, rng, D)
    elif kernel_type == "matern" or kernel_type == "exp":
        kargs = {**kwargs}
        if "G" in kargs.keys():
            G = kargs["G"]
        else:
            G = int(D**0.4)
            D = D // G
        if kernel_type == "matern":
            nu = kargs["nu"]
        else:
            nu = 0.5

        print(f"Using {D} RFFs and {G} Gamma samples")

        return sample_mat_rff_from_x(x, sigma, noise_var, l, rng, D, G, nu)
    elif kernel_type == "laplacian":
        return sample_lap_rff_from_x(x, sigma, noise_var, l, rng, D)
    else:
        raise NotImplementedError


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
    from gpsampler.leverage_reweighted_rff import reweighted_rff_sampler
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

def sample_iw_rff_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    rho: float = 0.1,
    guard_scale: float = 0.5,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Safeguarded importance-weighted RFF (IW-RFF) GP prior sampler.

    Draws D//2 frequencies from the mixture proposal

        q_rho(omega) = (1 - rho) * p(omega) + rho * g(omega)

    where p is the kernel's spectral density and g is the same spectral family
    with lengthscale l_guard = l * guard_scale < l (guard_scale < 1 gives g
    heavier tails than p in frequency space).  Each frequency is importance-
    weighted by the bounded ratio p(omega) / q_rho(omega) in [1-rho, 1], giving
    a covariance-unbiased feature matrix: E[Phi @ Phi.T] = sigma * K.

    Parameters
    ----------
    x           : (n, d) input locations
    sigma       : kernel output scale
    noise_var   : observation noise variance
    l           : kernel lengthscale
    rng         : numpy random Generator
    D           : number of RFF features (must be even)
    rho         : guard mixture weight rho in (0, 1); default 0.1
    guard_scale : l_guard = l * guard_scale; must be in (0, 1).  Default 0.5.
    kernel_type : 'rbf'/'se' or 'matern'
    nu          : Matern smoothness (ignored for RBF); default 1.5

    Returns
    -------
    y_noise    : (n,) sample with covariance approx sigma*K + noise_var*I
    approx_cov : (n, n) IS-reweighted Phi @ Phi.T + noise_var * I
    """
    if D % 2 != 0:
        raise ValueError("D must be even")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0, 1); got {rho}")
    if not (0.0 < guard_scale < 1.0):
        raise ValueError(
            f"guard_scale must be in (0, 1) for heavier guard tails; got {guard_scale}"
        )

    n, d = x.shape
    n_freq = D // 2
    kind = "rbf" if kernel_type in ("rbf", "se") else kernel_type
    l_guard = l * guard_scale

    # ------------------------------------------------------------------
    # 1. Sample n_freq frequencies from mixture q_rho = (1-rho)*p + rho*g
    # ------------------------------------------------------------------
    from_p = rng.uniform(size=n_freq) < (1.0 - rho)  # True -> from p
    n_from_p = int(from_p.sum())
    n_from_g = n_freq - n_from_p

    from gpsampler.leverage_reweighted_rff import spectral_sampler

    omega_p = spectral_sampler(n_from_p, d, kind, l,       nu, rng)  # (n_from_p, d)
    omega_g = spectral_sampler(n_from_g, d, kind, l_guard, nu, rng)  # (n_from_g, d)

    omega = np.empty((n_freq, d), dtype=np.float64)
    omega[from_p]  = omega_p
    omega[~from_p] = omega_g

    # ------------------------------------------------------------------
    # 2. IS weights r(omega) = p(omega) / q_rho(omega)
    #    q_rho >= (1-rho)*p  so  r <= 1/(1-rho)  always.
    # ------------------------------------------------------------------
    log_p = _log_spectral_density(omega, kind, l,       nu, d)  # (n_freq,)
    log_g = _log_spectral_density(omega, kind, l_guard, nu, d)  # (n_freq,)

    log_q = np.logaddexp(np.log1p(-rho) + log_p, np.log(rho) + log_g)
    log_r = log_p - log_q  # log IS weight, in [log(1-rho), 0]
    r = np.exp(log_r)       # in [1-rho, 1]

    # ------------------------------------------------------------------
    # 3. IS-reweighted feature matrix Phi (n x D)
    #    g_j = sqrt(r_j / n_freq)  so  Phi @ Phi.T = sigma * sum_j r_j/n_freq M_j
    #    E[Phi @ Phi.T] = sigma * E_p[M(omega)] = sigma * K
    # ------------------------------------------------------------------
    g_j = np.sqrt(r / n_freq)                      # (n_freq,)
    proj = x.astype(np.float64) @ omega.T          # (n, n_freq)
    sq_sigma = float(np.sqrt(sigma))
    Phi = np.empty((n, D), dtype=np.float64)
    Phi[:, 0::2] = sq_sigma * g_j * np.cos(proj)
    Phi[:, 1::2] = sq_sigma * g_j * np.sin(proj)

    # ------------------------------------------------------------------
    # 4. Draw prior sample and add observation noise
    # ------------------------------------------------------------------
    z = rng.standard_normal(D)
    y_noise = Phi @ z + rng.normal(scale=float(np.sqrt(noise_var)), size=n)
    approx_cov = Phi @ Phi.T + noise_var * np.eye(n)
    return y_noise, approx_cov


# ---------------------------------------------------------------------------
# Sampler: Stratified truncated-Taylor leverage-reweighted RFF
# ---------------------------------------------------------------------------

def sample_stratified_rff_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    taylor_order: int = 2,
    nystrom_rank: int = 50,
    pool_factor: int = 5,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified truncated-Taylor leverage-reweighted RFF GP prior sampler.

    Extends sample_lrff_from_x with two improvements:

    1. **Stratified pool**: the P = pool_factor * D//2 candidate frequencies
       are drawn from a stratified base distribution rather than i.i.d.
       Equal-probability radial strata under the spectral CDF ensure uniform
       coverage, giving a lower-variance estimate of Z_hat = E_p[alpha].

    2. **Truncated-Taylor surrogate + rejection step**: a degree-taylor_order
       polynomial alpha_hat(r) = sum_{k=0}^{T} c_k r^{2k}  (r = ||omega||)
       is fit by least squares to the pool's leverage scores.  Each pool
       candidate is then accepted/rejected with probability
       alpha(omega) / max(alpha_hat(||omega||), alpha(omega)), thinning the
       pool towards high-leverage frequencies before the SIR step.

    After rejection, n_freq frequencies are drawn by SIR proportional to
    exact leverage and corrected by Z_hat/alpha IS weights for unbiasedness.

    Parameters
    ----------
    x            : (n, d) input locations
    sigma        : kernel output scale
    noise_var    : observation noise variance
    l            : kernel lengthscale
    rng          : numpy random Generator
    D            : number of RFF features (must be even)
    taylor_order : degree T for the radial leverage surrogate; default 2
    nystrom_rank : Nyström rank for the Woodbury leverage; default 50
    pool_factor  : pool size P = pool_factor * D//2; default 5
    kernel_type  : 'rbf'/'se' or 'matern'
    nu           : Matern smoothness (ignored for RBF); default 1.5

    Returns
    -------
    y_noise    : (n,) sample approx ~ GP(0, K_xi)
    approx_cov : (n, n) Phi @ Phi.T + noise_var * I
    """
    if D % 2 != 0:
        raise ValueError("D must be even")

    n, d = x.shape
    n_freq = D // 2
    kind = "rbf" if kernel_type in ("rbf", "se") else kernel_type
    P = max(pool_factor * n_freq, n_freq + 1)

    # ------------------------------------------------------------------
    # 1. Stratified radial pool from p
    #    Divide [0,1) into P equal strata, place one stratified-uniform
    #    point per stratum, map through the spectral radial quantile.
    # ------------------------------------------------------------------
    u_strat = (np.arange(P) + rng.uniform(size=P)) / P  # (P,) in (0, 1)

    if kind in ("rbf", "se"):
        from scipy.stats import chi
        radii = chi.ppf(u_strat, df=d) / l          # (P,) chi(d)/l quantiles
    elif kind == "matern":
        from scipy.stats import chi2, chi as _chi
        # Matérn: ||omega|| = chi(d)/l * sqrt(2*nu / u_scale), u_scale ~ chi2(2*nu).
        # Stratify the heavy-tailed chi2(2*nu) component.
        u_scale = np.maximum(chi2.ppf(u_strat, df=2.0 * nu), 1e-10)
        g_norms = _chi.rvs(df=d, size=P, random_state=rng)
        radii = (g_norms / l) * np.sqrt(2.0 * nu / u_scale)
    else:
        raise ValueError(
            f"unsupported kernel_type {kernel_type!r}; choose 'rbf' or 'matern'"
        )

    dirs = rng.standard_normal((P, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30
    pool = radii[:, None] * dirs          # (P, d)

    # ------------------------------------------------------------------
    # 2. Woodbury leverage scoring for all pool candidates
    # ------------------------------------------------------------------
    from gpsampler.leverage_reweighted_rff import (
        kernel_matrix, nystrom_factor, ApproxLeverage
    )
    K_mat = kernel_matrix(x, kind, l, nu)
    rank = min(nystrom_rank, n)
    landmarks = rng.choice(n, size=rank, replace=False)
    B = nystrom_factor(K_mat, landmarks)
    alpha_fn = ApproxLeverage(x, B, noise_var)

    alpha_pool = np.maximum(alpha_fn(pool), 1e-12)   # (P,)

    # ------------------------------------------------------------------
    # 3. Fit truncated-Taylor surrogate  alpha_hat(r) = sum_k c_k r^{2k}
    #    by least squares in the variable r^2.
    # ------------------------------------------------------------------
    sq_radii = radii ** 2                             # (P,)
    T = int(taylor_order)
    V = np.column_stack([sq_radii ** k for k in range(T + 1)])  # (P, T+1)
    coeffs, *_ = np.linalg.lstsq(V, alpha_pool, rcond=None)
    alpha_hat = np.maximum(V @ coeffs, 1e-12)        # (P,) Taylor surrogate

    # ------------------------------------------------------------------
    # 4. Rejection step: accept pool[j] with probability
    #       alpha_pool[j] / max(alpha_pool[j], alpha_hat[j])
    #    The pointwise max gives a valid upper bound, so acceptance <= 1.
    # ------------------------------------------------------------------
    alpha_bound = np.maximum(alpha_pool, alpha_hat)
    accept_prob = alpha_pool / alpha_bound            # in (0, 1]
    mask = rng.uniform(size=P) < accept_prob

    if mask.sum() < n_freq:                          # fallback: full pool
        mask = np.ones(P, dtype=bool)

    survivors  = pool[mask]         # (S, d),  S >= n_freq
    alpha_surv = alpha_pool[mask]   # (S,)

    # ------------------------------------------------------------------
    # 5. SIR resample n_freq from survivors proportional to leverage
    # ------------------------------------------------------------------
    probs = alpha_surv / alpha_surv.sum()
    sel   = rng.choice(len(survivors), size=n_freq, replace=True, p=probs)
    omega   = survivors[sel]         # (n_freq, d)
    alpha_j = alpha_surv[sel]        # (n_freq,)

    # ------------------------------------------------------------------
    # 6. IS-reweighted feature matrix
    #    Z_hat estimated from stratified pool for lower variance.
    #    weight_j = Z_hat / (n_freq * alpha_j)  -> E[Phi @ Phi.T] = sigma*K
    # ------------------------------------------------------------------
    Z_hat = float(alpha_pool.mean())
    if Z_hat < 1e-15:
        Z_hat = 1.0

    g_j = np.sqrt(Z_hat / (n_freq * alpha_j))    # (n_freq,)
    proj = x.astype(np.float64) @ omega.T         # (n, n_freq)
    sq_sigma = float(np.sqrt(sigma))
    Phi = np.empty((n, D), dtype=np.float64)
    Phi[:, 0::2] = sq_sigma * g_j * np.cos(proj)
    Phi[:, 1::2] = sq_sigma * g_j * np.sin(proj)

    # ------------------------------------------------------------------
    # 7. Draw prior sample and add observation noise
    # ------------------------------------------------------------------
    z = rng.standard_normal(D)
    y_noise = Phi @ z + rng.normal(scale=float(np.sqrt(noise_var)), size=n)
    approx_cov = Phi @ Phi.T + noise_var * np.eye(n)
    return y_noise, approx_cov


def sample_mat_rff_from_x1(x: NPInputMat, sigma: float, noise_var: float, l:
                           float, rng: np.random.Generator, D: int, G: int,
                           nu: float) -> Tuple[NPSample, NPKernel]:
    n, d = x.shape
    w = rng.standard_normal((D, ))
    s = rng.gamma(shape=nu, scale=l**2/nu, size=G)
    # omega = rng.standard_normal((D//2, d, G))
    N = int(1e6)
    y, C = np.zeros(n,), np.nan

    # n_jobs = 4

    # def func(s): return _par_sampler(x, D, s, w, sigma)

    # def worker(func, args_batch):
    #     y = np.zeros((n, 1))
    #     for args in args_batch:
    #         y_new = func(args).reshape(-1, 1)
    #         np.sum(np.hstack([y, y_new]), axis=1, keepdims=True, out=y)

    #     return y
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     funcs = repeat(func, n_jobs)
    #     s_batches = np.array_split(s, n_jobs, axis=0)
    #     jobs = zip(funcs, s_batches)
    #     y = np.sum(parallel(delayed(worker)(*job) for job in jobs), axis=0).flatten()

    for ss in s:
        omega = rng.standard_normal((D//2, d))
        if n > N:
            ys, Cs = np.zeros(n,), np.nan
            parts = int(np.ceil(n/N))
            for p in range(parts):
                idx = np.s_[(p*N):((p+1)*N)]
                ys[idx], Cp = _sample_se_rff_from_x(
                    x[idx, :], sigma, omega/np.sqrt(ss), w)
        else:
            ys, Cs = _sample_se_rff_from_x(x, sigma, omega/np.sqrt(ss), w)
        y += ys
        C += Cs

    y /= np.sqrt(G)
    C /= G
    noise = rng.normal(scale=np.sqrt(noise_var), size=n)
    y_noise = y + noise
    return y_noise, C


def sample_mat_rff_from_x(x, sigma: float, noise_var: float, l:
                          float, rng: np.random.Generator, D: int, G: int,
                          nu: float):
    n, d = x.shape
    w = rng.standard_normal((D, ))
    y, C = np.zeros(n,), np.nan

    omega_y = rng.standard_normal((D//2, d)) * np.sqrt(2)/l
    omega_u = rng.chisquare(2*nu, size=(D//2,))
    omega = np.sqrt(2*nu/np.tile(omega_u, (d, 1)).T) * omega_y
    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n, ))
    y_noise = y + noise
    return y_noise, approx_cov


def sample_se_rff_from_x(
        x: NPInputMat, sigma: float, noise_var: float, l: float,
        rng: np.random.Generator, D: int) -> Tuple[
        NPSample, NPKernel]:
    """ Generates sample from approximate GP using RFF method at points x

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw; 1D array of length n and approx cov
    """
    n, d = x.shape
    cov_omega = np.eye(d)/l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)

    w = rng.standard_normal((D, ))

    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n, ))
    # print(y.shape, noise.shape, flush=True)
    y_noise = y + noise
    return y_noise, approx_cov


def sample_lap_rff_from_x(
        x: NPInputMat, sigma: float, noise_var: float, l: float,
        rng: np.random.Generator, D: int) -> Tuple[
        NPSample, NPKernel]:
    """ Generates sample from approximate Laplacian-kernel GP using RFF method
    at points x
    See classic Random Features for large-Scale Kernel Machiens (Rahimi 2009) 

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw; 1D array of length n and approx cov
    """
    n, d = x.shape
    cov_omega = np.eye(d)/l**2
    omega = np.zeros((D//2, d))
    for di in range(d):
        omega[:, di] = np.tan(np.pi*(rng.uniform(size=D//2) - 0.5))

    w = rng.standard_normal((D, 1))

    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n, ))
    # print(y.shape, noise.shape, flush=True)
    y_noise = y + noise
    return y_noise, approx_cov


@jit(nopython=True, parallel=True, fastmath=True)
def _sample_se_rff_from_x(x: NPInputMat, sigma: float,
                          omega: NDArray[Shape["N,D"],
                                         Float],
                          w: NDArray[Shape["D,1"],
                                     Float],
                          compute_cov=False) -> Tuple[NPSample, NPKernel]:
    D = w.shape[0]
    # Z = zrf(omega, D, x)*np.sqrt(sigma)
    if compute_cov:
        pass
        # approx_cov = Z @ Z.T
    else:
        approx_cov = np.nan
    # y = (Z @ w).flatten()
    n = x.shape[0]
    y = np.zeros((n, ))
    for i in prange(n):
        y[i] = f_rf(omega, D, w, x[i, :]) * np.sqrt(sigma)
    return y, approx_cov


def sample_ciq_from_x(x: Union[torch.Tensor, NPInputMat],
                      sigma: float, noise_var: float, l: float,
                      kernel_type: str, rng: np.random.Generator, J: int,
                      Q: Optional[int] = None, checkpoint_size: int = 1500,
                      max_preconditioner_size: int = 0) -> Tuple[NPSample,
                                                                 Union[NPKernel, float]]:
    """ Generates sample from approximate GP using CIQ method at points x

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw with noise; 1D array of length n and approx cov
    """
    n, d = x.shape
    u = rng.standard_normal(n)

    eta = 0.8

    if kernel_type.lower() == 'rbf':
        base_kernel = gpytorch.kernels.RBFKernel()
    elif kernel_type.lower() == 'exp':
        base_kernel = gpytorch.kernels.MaternKernel(0.5)
    elif kernel_type.lower() == 'matern32':
        base_kernel = gpytorch.kernels.MaternKernel(1.5)
    elif kernel_type.lower() == 'matern52':
        base_kernel = gpytorch.kernels.MaternKernel(2.5)
    else:
        raise ValueError(
            "Unsupported kernel or incorrect name. Options: 'rbf', 'exp', 'matern32', 'matern52'.")

    kernel = construct_kernels(
        l, sigma, base_kernel)(
        torch.as_tensor(x)).add_diag(torch.as_tensor(eta*noise_var))
    kernel.preconditioner_override = ID_Preconditioner

    # not sure why I need this yet but...
    if max_preconditioner_size == 0:
        ciqfun = contour_integral_quad
    else:
        ciqfun = gpytorch.utils.contour_integral_quad

    with ExitStack() as stack:
        checkpoint_size = stack.enter_context(
            gpytorch.beta_features.checkpoint_kernel(checkpoint_size))
        max_preconditioner_size = stack.enter_context(
            gpytorch.settings.max_preconditioner_size(max_preconditioner_size))
        min_preconditioning_size = stack.enter_context(
            gpytorch.settings.min_preconditioning_size(100))
        minres_tol = stack.enter_context(
            gpytorch.settings.minres_tolerance(1e-10))
        # _use_eval_tolerance = stack.enter_context(
        #     gpytorch.settings._use_eval_tolerance(True))
        eval_cg_tolerance = stack.enter_context(
            gpytorch.settings.eval_cg_tolerance(1e-10))
        max_cg_iterations = stack.enter_context(
            gpytorch.settings.max_cg_iterations(J))
        solves, weights, _, _ = contour_integral_quad(
            kernel,
            torch.as_tensor(u.reshape(-1, 1)),
            max_lanczos_iter=J, num_contour_quadrature=Q)
    f = (solves * weights).sum(0).squeeze()
    y_noise = (f + torch.sqrt(torch.tensor((1-eta)*noise_var))
               * torch.randn(n)).detach().numpy()
    # approx_cov = estimate_ciq_kernel(x, J, Q, sigma, l)
    approx_cov = np.nan
    return y_noise, approx_cov


def sample_sparse_from_x(x: NPInputMat, sigma: float, noise_var: float,
                         l: float, kernel_type: str, rng: np.random.Generator,
                         m: int) -> Tuple[NPSample, NPKernel]:
    n, d = x.shape
    u = rng.standard_normal(m)
    eta = 0.8

    if kernel_type.lower() == 'rbf':
        base_kernel = RBFKernel
    elif kernel_type.lower() == 'exp':
        base_kernel = gpytorch.kernels.MaternKernel(0.5)
    elif kernel_type.lower() == 'matern32':
        base_kernel = gpytorch.kernels.MaternKernel(1.5)
    elif kernel_type.lower() == 'matern52':
        base_kernel = gpytorch.kernels.MaternKernel(2.5)
    else:
        raise ValueError(
            "Unsupported kernel or incorrect name. Options: 'rbf', 'exp', 'matern32', 'matern52'.")

    sind = rng.choice(n, m)

    inducing_points = torch.as_tensor(x[sind, :])
    base_kernel = gpytorch.kernels.InducingPointKernel(
        base_kernel,
        inducing_points=inducing_points,
        likelihood=gpytorch.likelihoods.Likelihood,
    )

    kernel = construct_kernels(
        l, sigma, base_kernel)

    rootKmm = kernel._inducing_inv_root
    Knm = kernel(torch.as_tensor(x), inducing_points)

    # TODO: use gpytorch/keops to exploit GPUs
    # rootKmm = msqrt(kernel(inducing_points, inducing_points))

    y_noise = Knm @ rootKmm @ torch.as_tensor(u)
    approx_cov = Knm @ rootKmm @ Knm.T
    return y_noise, approx_cov


def contour_integral_quad(
    lazy_tensor,
    rhs,
    inverse=False,
    weights=None,
    shifts=None,
    max_lanczos_iter=20,
    num_contour_quadrature=None,
    shift_offset=0,
):
    r"""
    Performs :math:`\mathbf K^{1/2} \mathbf b` or `\mathbf K^{-1/2} \mathbf b`
    using contour integral quadrature.

    :param gpytorch.lazy.LazyTensor lazy_tensor: LazyTensor representing :math:`\mathbf K`
    :param torch.Tensor rhs: Right hand side tensor :math:`\mathbf b`
    :param bool inverse: (default False) whether to compute :math:`\mathbf K^{1/2} \mathbf b` (if False)
        or `\mathbf K^{-1/2} \mathbf b` (if True)
    :param int max_lanczos_iter: (default 10) Number of Lanczos iterations to run (to estimate eigenvalues)
    :param int num_contour_quadrature: How many quadrature samples to use for approximation. Default is in settings.
    :rtype: torch.Tensor
    :return: Approximation to :math:`\mathbf K^{1/2} \mathbf b` or :math:`\mathbf K^{-1/2} \mathbf b`.
    """
    if num_contour_quadrature is None:
        num_contour_quadrature = gpytorch.settings.num_contour_quadrature.value()

    # output_batch_shape = _matmul_broadcast_shape(
    #     lazy_tensor.batch_shape, rhs.shape[:-2])
    output_batch_shape = torch.broadcast_shapes(
        lazy_tensor.batch_shape, rhs.shape[:-2])
    preconditioner, preconditioner_lt, _ = lazy_tensor._preconditioner()

    def sqrt_precond_matmul(rhs):
        if preconditioner_lt is not None:
            solves, weights, _, _ = contour_integral_quad(
                preconditioner_lt, rhs, inverse=False)
            return (solves * weights).sum(0)
        else:
            return rhs

    # if not inverse:
    rhs = sqrt_precond_matmul(rhs)

    if shifts is None:
        # Determine if init_vecs has extra_dimensions
        num_extra_dims = max(0, rhs.dim() - lazy_tensor.dim())
        lanczos_init = rhs.__getitem__(
            (*([0] * num_extra_dims),
             Ellipsis, slice(None, None, None),
             slice(None, 1, None))).expand(
            *lazy_tensor.shape[: -1],
            1)
        with warnings.catch_warnings(), torch.no_grad():
            # Supress CG stopping warning
            warnings.simplefilter("ignore", NumericalWarning)
            _, lanczos_mat = linear_cg(
                lambda v: lazy_tensor._matmul(v),
                rhs=lanczos_init,
                n_tridiag=1,
                max_iter=max_lanczos_iter,
                tolerance=1e-10,
                max_tridiag_iter=max_lanczos_iter,
                preconditioner=preconditioner,
            )
            # We have an extra singleton batch dimension from the Lanczos init
            lanczos_mat = lanczos_mat.squeeze(0)

        """
        K^{-1/2} b = 2/pi \int_0^\infty (K - t^2 I)^{-1} dt
        We'll approximate this integral as a sum using quadrature
        We'll determine the appropriate values of t, as well as their weights using elliptical integrals
        """

        # Compute an approximate condition number
        # We'll do this with Lanczos
        try:
            approx_eigs = lanczos_mat.symeig()[0]
            if approx_eigs.min() <= 0:
                raise RuntimeError
        except RuntimeError:
            approx_eigs = lazy_tensor.diag()

        max_eig = approx_eigs.max(dim=-1)[0]
        min_eig = approx_eigs.min(dim=-1)[0]
        k2 = min_eig / max_eig

        # Compute the shifts needed for the contour
        flat_shifts = torch.zeros(
            num_contour_quadrature + 1, k2.numel(),
            dtype=k2.dtype, device=k2.device)
        flat_weights = torch.zeros(
            num_contour_quadrature, k2.numel(),
            dtype=k2.dtype, device=k2.device)

        # For loop because numpy
        for i, (sub_k2, sub_min_eig) in enumerate(
            zip(k2.flatten().tolist(),
                min_eig.flatten().tolist())):
            # Compute shifts
            Kp = ellipk(1 - sub_k2)  # Elliptical integral of the first kind
            N = num_contour_quadrature
            t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
            # Jacobi elliptic functions
            sn, cn, dn, _ = ellipj(np.imag(t), 1 - sub_k2)
            cn = 1.0 / cn
            dn = dn * cn
            sn = 1j * sn * cn
            w = np.sqrt(sub_min_eig) * sn
            w_pow2 = np.real(np.power(w, 2))
            sub_shifts = torch.tensor(
                w_pow2, dtype=rhs.dtype, device=rhs.device)

            # Compute weights
            constant = -2 * Kp * np.sqrt(sub_min_eig) / (math.pi * N)
            dzdt = torch.tensor(cn * dn, dtype=rhs.dtype, device=rhs.device)
            dzdt.mul_(constant)
            sub_weights = dzdt

            # Store results
            flat_shifts[1:, i].copy_(sub_shifts)
            flat_weights[:, i].copy_(sub_weights)

        weights = flat_weights.view(num_contour_quadrature, *k2.shape, 1, 1)
        shifts = flat_shifts.view(num_contour_quadrature + 1, *k2.shape)
        shifts.sub_(shift_offset)

        # Make sure we have the right shape
        if k2.shape != output_batch_shape:
            weights = torch.stack(
                [w.expand(*output_batch_shape, 1, 1) for w in weights], 0)
            shifts = torch.stack([s.expand(output_batch_shape)
                                  for s in shifts], 0)

    # Compute the solves at the given shifts
    # Do one more matmul if we don't want to include the inverse
    with torch.no_grad():
        solves = minres(lambda v: lazy_tensor._matmul(v),
                        rhs, value=-1, shifts=shifts,
                        preconditioner=preconditioner,
                        max_iter=max_lanczos_iter)
    no_shift_solves = solves[0]
    solves = solves[1:]
    if not inverse:
        solves = lazy_tensor._matmul(solves)

    return solves, weights, no_shift_solves, shifts


def ID_Preconditioner(self):
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(
            -1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None

    if self._q_cache is None:

        import scipy.linalg.interpolative as sli

        # get quantities & form sample matrix
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()

        M = self._lazy_tensor.evaluate().detach().numpy()

        U, s, V = sli.svd(M, k)

        #L = V @ S^0.5
        L = V * (s ** 0.5)

        self._piv_chol_self = torch.as_tensor(L)

        if torch.any(torch.isnan(self._piv_chol_self)).item():
            warnings.warn(
                "NaNs encountered in preconditioner computation. Attempting to continue without preconditioning."
            )
            return None, None, None
        self._init_cache()

    def precondition_closure(tensor):
        # This makes it fast to compute solves with it
        qqt = self._q_cache.matmul(
            self._q_cache.transpose(-2, -1).matmul(tensor))
        if self._constant_diag:
            return (1 / self._noise) * (tensor - qqt)
        return (tensor / self._noise) - qqt

    return (precondition_closure, self._precond_lt, self._precond_logdet_cache)


class SparseRBFKernel(gpytorch.kernels.RBFKernel):
    is_stationary = True
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        dist = super().forward(x1, x2, diag=diag, **params)
        dist.where(dist.abs() < 1e-16, torch.as_tensor(0.0))
        return dist


class SparseKernel(gpytorch.kernels.Kernel):
    """Wrapper similar to ScaleKernel to sparsify off-diag kernel elements if
    they have value less than double precision epsilon (1e-16).
    """

    def __init__(self, base_kernel, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        self.has_lengthscale = base_kernel.has_lengthscale
        super(SparseKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    @property
    def lengthscale(self):
        return self.base_kernel.lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        self.base_kernel._set_lengthscale(value)

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        orig_output.where(orig_output.abs() < 1e-16, torch.as_tensor(0.0))
        if diag:
            return gpytorch.delazify(orig_output)
        else:
            return orig_output

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(
            self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(
            train_inputs, train_prior_dist, train_labels, likelihood)


if __name__ == '__main__':
    N = 500  # no. of data points
    d = 2  # input (x) dimensionality
    D = 100  # no.of fourier features
    J = int(np.sqrt(N) * np.log(N))
    Q = int(np.log(N))
    l = 1.1  # lengthscale
    sigma = 0.7  # kernel scale
    noise_var = 0.2  # noise variance

    xmean = np.zeros(d)
    xcov_diag = np.ones(d)/d

    print(
        """
data_size %d
xmean %s
xcov_diag %s
noise_var %.2f
kernelscale %.2f
lenscale %.2f
    """
        % (
            N,
            str(xmean),
            str(xcov_diag),
            noise_var,
            sigma,
            l
        )
    )

    # x, sample = generate_ciq_data(
    # N, xmean, xcov_diag, noise_var, sigma, l, J, Q)
    x, sample = generate_rff_data(N, xmean, xcov_diag, noise_var, sigma, l, D)
    # np.savetxt("x.out.gz", x)
    # np.savetxt("sample.out.gz", sample)
    # np.savetxt("noisy_sample.out.gz", noisy_sample)

    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("samples have been generated")
    print("peak memory usage: %s kb" % mem)
