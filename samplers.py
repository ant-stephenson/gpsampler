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
except (ImportError, AttributeError):
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

    Parameters
    ----------
    x           : (n, d) input locations
    l           : kernel lengthscale
    rng         : numpy Generator
    D           : total feature dimension (must be even)
    eta         : mixture fraction on p, in (0, 1].  eta=1 ⟹ plain RFF.
    guard_scale : if given, g = spectral density with lengthscale l*guard_scale
    g_sampler   : callable(n_samples, d, rng) -> (n_samples, d) frequencies
    g_logpdf    : callable(omega) -> (F,) log-density of g
    kernel_type : 'rbf'/'se' or 'matern'
    nu          : Matern smoothness (ignored for RBF)

    Returns
    -------
    Z : (n, D) feature matrix with block layout [cos | sin].
    """
    if D % 2 != 0:
        raise ValueError("D must be even")
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0, 1]; got {eta}")

    n, d = x.shape
    m = D // 2
    kind = "rbf" if kernel_type in ("rbf", "se") else kernel_type

    from gpsampler.leverage_reweighted_rff import spectral_sampler

    # ---- Build guard if needed ------------------------------------------
    if eta >= 1.0:
        # Plain RFF: all from p, uniform weight
        omega = spectral_sampler(m, d, kind, l, nu, rng)
        Z = np.empty((n, D), dtype=np.float64)
        proj = x.astype(np.float64) @ omega.T
        Z[:, :m] = np.sqrt(2.0 / D) * np.cos(proj)
        Z[:, m:] = np.sqrt(2.0 / D) * np.sin(proj)
        return Z

    have_custom_g = (g_sampler is not None and g_logpdf is not None)
    if not have_custom_g:
        if guard_scale is None:
            guard_scale = 1.0  # g = p when no guard specified
        if not (0.0 < guard_scale <= 1.0):
            raise ValueError(f"guard_scale must be in (0, 1]; got {guard_scale}")
        l_guard = l * guard_scale

        def _g_sampler(n_samp, _d, _rng):
            return spectral_sampler(n_samp, _d, kind, l_guard, nu, _rng)

        def _g_logpdf(omega):
            return _log_spectral_density(omega, kind, l_guard, nu, omega.shape[1])

        g_sampler = _g_sampler
        g_logpdf = _g_logpdf

    # ---- 1. Draw from mixture q_eta = (1-eta)*g + eta*p -----------------
    from_p = rng.uniform(size=m) < eta
    n_from_p = int(from_p.sum())
    n_from_g = m - n_from_p

    omega = np.empty((m, d), dtype=np.float64)
    if n_from_p > 0:
        omega[from_p] = spectral_sampler(n_from_p, d, kind, l, nu, rng)
    if n_from_g > 0:
        omega[~from_p] = g_sampler(n_from_g, d, rng)

    # ---- 2. IS weights: a_j = sqrt(2 p / (D * q_eta)) ------------------
    log_p = _log_spectral_density(omega, kind, l, nu, d)
    log_g = g_logpdf(omega)
    log_q = np.logaddexp(
        np.log(1.0 - eta) + log_g,
        np.log(eta) + log_p,
    )
    a = np.sqrt(2.0 * np.exp(log_p - log_q) / D)  # (m,)

    # ---- 3. Feature matrix Z = [a*cos(X omega^T) | a*sin(X omega^T)] ---
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


# ---------------------------------------------------------------------------
# Sampler: Stratified truncated-Taylor RFF
# ---------------------------------------------------------------------------

def _enumerate_multi_indices(d: int, R: int):
    """All d-tuples alpha with |alpha| <= R, sorted by total degree."""
    from itertools import product as _product
    return sorted(
        [a for a in _product(range(R + 1), repeat=d) if sum(a) <= R],
        key=lambda a: (sum(a), a),
    )


def _monomial_design(X: np.ndarray, alphas: list) -> np.ndarray:
    """Build monomial design matrix Phi (n, r) for multi-indices alphas."""
    n, d = X.shape
    r = len(alphas)
    Phi = np.ones((n, r), dtype=np.float64)
    for col, a in enumerate(alphas):
        for j in range(d):
            if a[j] > 0:
                Phi[:, col] *= X[:, j] ** a[j]
    return Phi


def _taylor_coeffs_batch(omega: np.ndarray, alphas: list) -> np.ndarray:
    """Taylor coefficients c_alpha(omega) = i^{|alpha|} / alpha! * omega^alpha.

    Parameters
    ----------
    omega  : (F, d) frequency array
    alphas : list of d-tuples

    Returns
    -------
    C : (F, r) complex array
    """
    from math import factorial as _fact
    F, d = omega.shape
    r = len(alphas)
    C = np.empty((F, r), dtype=np.complex128)
    for col, a in enumerate(alphas):
        afact = 1
        for ai in a:
            afact *= _fact(ai)
        coeff = (1j ** sum(a)) / afact
        col_val = np.ones(F, dtype=np.complex128)
        for j in range(d):
            if a[j] > 0:
                col_val *= omega[:, j] ** a[j]
        C[:, col] = coeff * col_val
    return C


def _choose_taylor_order(Z_max: float, eps: float = 0.4) -> int:
    """Smallest R such that sup_{|z|<=Z_max} |e^{iz} - T_R(iz)| <= eps."""
    zs = np.linspace(-Z_max, Z_max, 400)
    for R in range(1, 200):
        term = np.ones_like(zs, dtype=complex)
        acc = term.copy()
        for k in range(1, R + 1):
            term = term * (1j * zs) / k
            acc = acc + term
        if np.max(np.abs(np.exp(1j * zs) - acc)) <= eps:
            return R
    return 200


def _raw_gaussian_moments(s: float, B: float, max_deg: int) -> np.ndarray:
    """Compute M_k = int_{-B}^{B} w^k * N(w; 0, s^2) dw for k=0..max_deg.

    Uses scipy.integrate.quad for accuracy.  For SE kernels the spectral
    density is N(0, 1/l^2) so s = 1/l.
    """
    from scipy.integrate import quad
    from scipy.stats import norm as _norm
    moments = np.zeros(max_deg + 1)
    for k in range(max_deg + 1):
        integrand = lambda w, _k=k: w**_k * _norm.pdf(w, scale=s)
        moments[k], _ = quad(integrand, -B, B)
    return moments


def _build_H_matrix(alphas: list, raw_mom: np.ndarray, d: int) -> np.ndarray:
    """Build H (r, r) real matrix where H[a,b] = prod_j M_{a_j+b_j}.

    For SE kernels H is real because odd moments vanish and the complex
    pre-factor i^{|a|+|b|} / (a! b!) produces real entries when combined
    with the moment parity.
    """
    from math import factorial as _fact
    r = len(alphas)
    H = np.zeros((r, r), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            val = 1.0
            for dim in range(d):
                val *= raw_mom[a[dim] + b[dim]]
            # pre-factor: i^{|a|+|b|} / (a! b!)
            total_deg = sum(a) + sum(b)
            # i^k is real iff k is even — and M_k = 0 for odd k (symmetric),
            # so the product is always real for SE.
            i_pow = (1j ** total_deg)
            afact = 1
            bfact = 1
            for ai in a:
                afact *= _fact(ai)
            for bi in b:
                bfact *= _fact(bi)
            H[i, j] = np.real(i_pow * val / (afact * bfact))
    return H


def _build_B_via_woodbury(Phi: np.ndarray, H: np.ndarray,
                          s2: float) -> np.ndarray:
    """B = Phi^T (Phi H Phi^T + s2 I)^{-1} Phi via Woodbury.

    Cost: O(n r^2 + r^3) — never forms n×n matrices.

    Returns B (r, r) symmetric positive semi-definite.
    """
    # Woodbury: (s2 I + Phi H Phi^T)^{-1} = s2^{-1} I - s2^{-2} Phi (H^{-1} + s2^{-1} Phi^T Phi)^{-1} Phi^T
    # B = Phi^T inv(A_R) Phi = s2^{-1} Phi^T Phi - s2^{-2} Phi^T Phi (H^{-1} + s2^{-1} Phi^T Phi)^{-1} Phi^T Phi
    # Let G = Phi^T Phi (r, r).  Then B = G/s2 - G/s2^2 (H^{-1} + G/s2)^{-1} G
    # = G/s2 (I - (s2 H^{-1} + G)^{-1} G)
    # Simpler: B = Phi^T inv(A_R) Phi.  With A_R = Phi H Phi^T + s2 I:
    #   inv(A_R) Phi = s2^{-1}(Phi - Phi (H^{-1} + G/s2)^{-1} G/s2)  ... messy.
    # Direct: B = (H^{-1} + G/s2)^{-1} / s2  ... let's derive cleanly.
    #
    # From Woodbury on the r×r side:
    #   Phi^T (Phi H Phi^T + s2 I)^{-1} Phi = H^{-1} (H^{-1} + Phi^T Phi / s2)^{-1} Phi^T Phi / s2
    #
    # Actually the standard push-through identity gives:
    #   Phi^T (Phi H Phi^T + s2 I)^{-1} = (H^{-1} + Phi^T Phi / s2)^{-1} Phi^T / s2  ... when H invertible.
    #
    # So B = Phi^T inv(A_R) Phi = (H^{-1} + G/s2)^{-1} G / s2
    # But we want numerical stability when H is near-singular.
    #
    # Best approach: form M = s2 * H^{-1} + G, then B = H (M^{-1}) G = (M^{-1} G)^T s2 ... no.
    # Let's just use the matrix inversion lemma more carefully:
    #   B = Phi^T inv(A_R) Phi
    # Let L = cholesky(H).  Let V = Phi L (n, r).  Then A_R = V V^T + s2 I.
    # B = L^T V^T (V V^T + s2 I)^{-1} V L
    # Inner part: V^T (V V^T + s2 I)^{-1} V = I - s2 (V^T V + s2 I)^{-1}   [push-through]
    # So B = L^T (I - s2 (V^T V + s2 I)^{-1}) L = L^T L - s2 L^T (V^T V + s2 I)^{-1} L
    # = H - s2 L^T (V^T V + s2 I)^{-1} L

    r = Phi.shape[1]
    G = Phi.T @ Phi  # (r, r)

    # Regularise H for Cholesky
    H_reg = H + 1e-12 * np.eye(r)
    try:
        L = np.linalg.cholesky(H_reg)
    except np.linalg.LinAlgError:
        # H may be singular/near-singular; fall back to eigendecomposition
        evals, evecs = np.linalg.eigh(H_reg)
        evals = np.maximum(evals, 1e-12)
        L = evecs * np.sqrt(evals)  # (r, r) "pseudo-Cholesky" V = Phi @ L

    VtV = L.T @ G @ L  # (r, r)  = (Phi L)^T (Phi L)
    M = VtV + s2 * np.eye(r)
    # B = Phi^T inv(A_R) Phi = H - s2 L^T inv(V^T V + s2 I) L
    M_inv = np.linalg.solve(M, np.eye(r))
    B = H_reg - s2 * (L.T @ M_inv @ L)
    # Symmetrise
    B = 0.5 * (B + B.T)
    return B


def _leverage_batch(C: np.ndarray, B_mat: np.ndarray) -> np.ndarray:
    """Compute leverage a(omega) = Re(c)^T B Re(c) + Im(c)^T B Im(c).

    Parameters
    ----------
    C     : (F, r) complex Taylor coefficient matrix
    B_mat : (r, r) real symmetric Woodbury factor

    Returns
    -------
    a : (F,) leverage scores (non-negative)
    """
    Cr = C.real  # (F, r)
    Ci = C.imag  # (F, r)
    # a_j = Cr[j] @ B @ Cr[j] + Ci[j] @ B @ Ci[j]
    a = np.sum((Cr @ B_mat) * Cr, axis=1) + np.sum((Ci @ B_mat) * Ci, axis=1)
    return np.maximum(a, 0.0)


def _rejection_sample_vectorised(
    d: int, s: float, B: float,
    B_mat: np.ndarray, alphas: list, M_bound: float,
    n_accept: int, rng: np.random.Generator,
    batch: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rejection-sample frequencies from box-truncated Gaussian weighted by leverage.

    Proposes from N(0, s^2 I_d) | [-B,B]^d, accepts with prob a(omega)/M_bound.

    Returns
    -------
    omega_acc : (n_accept, d) accepted frequencies
    a_acc     : (n_accept,) leverage at accepted frequencies
    n_proposed: total proposals made
    """
    from scipy.stats import truncnorm as _tn
    a_tn, b_tn = -B / s, B / s
    collected_w = []
    collected_a = []
    total_proposed = 0

    while sum(len(c) for c in collected_w) < n_accept:
        # Propose a batch
        omega = np.empty((batch, d), dtype=np.float64)
        for j in range(d):
            omega[:, j] = _tn.rvs(a_tn, b_tn, loc=0, scale=s, size=batch,
                                   random_state=rng)
        C = _taylor_coeffs_batch(omega, alphas)
        a_vals = _leverage_batch(C, B_mat)
        accept = rng.uniform(size=batch) * M_bound <= a_vals
        total_proposed += batch
        if accept.any():
            collected_w.append(omega[accept])
            collected_a.append(a_vals[accept])

    omega_all = np.concatenate(collected_w, axis=0)[:n_accept]
    a_all = np.concatenate(collected_a)[:n_accept]
    return omega_all, a_all, total_proposed


def _build_stratified_rff_features(
    x: np.ndarray,
    l: float,
    noise_var: float,
    rng: np.random.Generator,
    D: int,
    eps: float = 0.4,
    box_scale: float = 3.0,
    eta: float = None,
    rank_cap: int = 5000,
) -> np.ndarray:
    """Build stratified truncated-Taylor RFF features for SE kernel.

    SE-only (single spectral component).  The algorithm:
    1. Choose Taylor order R from input-box radius and eps.
    2. Build monomial design Phi (n, r), moment matrix H, Woodbury B.
    3. Rejection-sample frequencies from box-truncated Gaussian weighted by leverage.
    4. Compute optimal safeguard eta* if not given.
    5. Build IS-weighted features Z with E[Z Z^T] = K_trunc ≈ K.

    Returns Z (n, D).
    """
    from scipy.stats import norm as _norm

    if D % 2 != 0:
        raise ValueError("D must be even")
    n, d = x.shape
    m = D // 2
    s2 = noise_var

    # ---- Spectral scale and box -----------------------------------------
    s = 1.0 / l  # SE spectral std
    B = box_scale * s  # frequency box half-width

    # ---- Taylor order ---------------------------------------------------
    Bx = np.max(np.abs(x))
    Z_max = Bx * B * np.sqrt(d) if d > 1 else Bx * B
    R = _choose_taylor_order(Z_max, eps)
    alphas = _enumerate_multi_indices(d, R)
    r = len(alphas)
    if r > rank_cap:
        # Reduce R until r fits
        while r > rank_cap and R > 1:
            R -= 1
            alphas = _enumerate_multi_indices(d, R)
            r = len(alphas)

    # ---- Monomial design Phi (n, r) -------------------------------------
    Phi = _monomial_design(x, alphas)

    # ---- Moment matrix H (r, r) -----------------------------------------
    max_deg = 2 * R
    raw_mom = _raw_gaussian_moments(s, B, max_deg)
    H = _build_H_matrix(alphas, raw_mom, d)

    # ---- Woodbury B = Phi^T inv(A_R) Phi --------------------------------
    B_mat = _build_B_via_woodbury(Phi, H, s2)

    # ---- Leverage bound M -----------------------------------------------
    # Verify eps: compute actual Taylor error on a small grid
    zs = np.linspace(-Z_max, Z_max, 400) if Z_max > 0 else np.array([0.0])
    term = np.ones_like(zs, dtype=complex)
    acc = term.copy()
    for k in range(1, R + 1):
        term = term * (1j * zs) / k
        acc = acc + term
    actual_eps = np.max(np.abs(np.exp(1j * zs) - acc))
    M_bound = n * (1.0 + actual_eps) ** 2 / s2

    # ---- Rejection-sample pilot for eta estimation ------------------------
    n_pilot = min(m, 5000)
    omega_pilot, a_pilot, _ = _rejection_sample_vectorised(
        d, s, B, B_mat, alphas, M_bound, n_pilot, rng,
    )

    # ---- d_l = E_p[a * 1_box]: average leverage under FULL p ------------
    # The pilot samples come from the box-truncated Gaussian (in-box only).
    # E_p[a * 1_box] = E_{p|box}[a] * pi_box = mean(a_pilot) * pi_box.
    pi_box = (_norm.cdf(B, scale=s) - _norm.cdf(-B, scale=s)) ** d
    d_l = float(np.mean(a_pilot)) * pi_box
    T_l = float(np.var(a_pilot)) * pi_box**2  # Var under full p (approx)

    if eta is None:
        if T_l > 0 and d_l > 0:
            eta = np.sqrt(T_l) / (d_l + np.sqrt(T_l))
            eta = np.clip(eta, 0.01, 0.99)
        else:
            eta = 0.5

    # ---- Draw m frequencies from mixture q = (1-eta)*g + eta*p ----------
    # eta fraction: from FULL spectral density p (can be outside box)
    # (1-eta) fraction: from rejection sampler (in box, proportional to a*p)
    from gpsampler.leverage_reweighted_rff import spectral_sampler

    from_p = rng.uniform(size=m) < eta
    n_from_p = int(from_p.sum())
    n_from_g = m - n_from_p

    omega_all = np.empty((m, d), dtype=np.float64)
    a_all = np.zeros(m, dtype=np.float64)

    # eta fraction: draw from full p
    if n_from_p > 0:
        omega_all[from_p] = spectral_sampler(n_from_p, d, "rbf", l, 1.5, rng)
        # Compute leverage for those that land in box
        in_box_p = np.all(np.abs(omega_all[from_p]) <= B, axis=1)
        if in_box_p.any():
            C_p = _taylor_coeffs_batch(omega_all[from_p][in_box_p], alphas)
            a_p = _leverage_batch(C_p, B_mat)
            a_all_p = np.zeros(n_from_p)
            a_all_p[in_box_p] = a_p
            a_all[from_p] = a_all_p

    # (1-eta) fraction: draw from rejection sampler (in box)
    if n_from_g > 0:
        # Use remaining pilot or sample more
        if n_from_g <= len(omega_pilot):
            omega_all[~from_p] = omega_pilot[:n_from_g]
            a_all[~from_p] = a_pilot[:n_from_g]
        else:
            omega_rej, a_rej, _ = _rejection_sample_vectorised(
                d, s, B, B_mat, alphas, M_bound, n_from_g, rng,
            )
            omega_all[~from_p] = omega_rej
            a_all[~from_p] = a_rej

    # ---- IS weights: w = p / q with NO pi_box factor --------------------
    # q(w) = (1-eta)*g(w) + eta*p(w)
    # where g(w) = p(w)*a(w)/d_l for w in box, 0 outside
    # For in-box: q = p*((1-eta)*a/d_l + eta), so p/q = 1/((1-eta)*a/d_l + eta)
    # For out-of-box: q = eta*p (g=0), so p/q = 1/eta
    in_box = np.all(np.abs(omega_all) <= B, axis=1)
    ratio = np.full(m, 1.0 / eta)  # default: out-of-box weight
    ratio[in_box] = 1.0 / (eta + (1.0 - eta) * a_all[in_box] / d_l)
    a_feat = np.sqrt(2.0 * ratio / D)  # (m,)  NO pi_box here

    # ---- Feature matrix Z = [a*cos | a*sin] -----------------------------
    proj = x.astype(np.float64) @ omega_all.T  # (n, m)
    Z = np.empty((n, D), dtype=np.float64)
    Z[:, :m] = a_feat[None, :] * np.cos(proj)
    Z[:, m:] = a_feat[None, :] * np.sin(proj)
    return Z


def sample_stratified_rff_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    eps: float = 0.4,
    box_scale: float = 3.0,
    eta: float = None,
    rank_cap: int = 5000,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """Stratified truncated-Taylor RFF GP prior sampler (SE-only).

    Implements the paper's Algorithm 1: Taylor polynomial feature map,
    Woodbury leverage scoring, rejection sampling from box-truncated
    Gaussian, and safeguarded importance weighting.

    Returns (y_noise, np.nan) — no n×n matrix is formed.
    """
    if kernel_type not in ("rbf", "se"):
        raise NotImplementedError(
            f"Stratified Taylor RFF is SE-only for now; got kernel_type={kernel_type!r}"
        )
    n = x.shape[0]
    Z = _build_stratified_rff_features(
        x, l, noise_var, rng, D,
        eps=eps, box_scale=box_scale, eta=eta, rank_cap=rank_cap,
    )
    Z = float(np.sqrt(sigma)) * Z
    w = rng.standard_normal(D)
    y_noise = Z @ w + rng.normal(scale=float(np.sqrt(noise_var)), size=n)
    return y_noise, np.nan




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
