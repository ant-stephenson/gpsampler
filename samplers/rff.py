"""Plain RFF, Cholesky, and legacy Matérn/SE/Laplace RFF samplers."""

import numpy as np
from numba import jit, prange
from typing import Tuple
from nptyping import NDArray, Shape, Float

from ._utils import (
    rng as _module_rng,
    k_true,
    construct_kernels,
    NPInputMat,
    NPSample,
    NPKernel,
)


@jit(nopython=True, fastmath=True)
def zrf(
    omega: NDArray[Shape["D, P"], Float],
    D: int,
    x: NDArray[Shape["P,1"], Float],
) -> NDArray[Shape["[cos,sin] x n_rff"], Float]:
    if x.ndim == 1:
        n = 1
    else:
        n = x.shape[0]
    v = np.dot(omega, x.T)  # omega @ x.T
    return np.sqrt(2 / D) * np.concatenate((np.cos(v), np.sin(v)))


@jit(nopython=True, fastmath=True)
def f_rf(
    omega: NDArray[Shape["D, P"], Float],
    D: int,
    w: NDArray[Shape["2 x n_rff"], Float],
    x: NDArray[Shape["P,1"], Float],
) -> float:
    return np.sum(w * zrf(omega, D, x))  # GP approximation


# @jit(nopython=True)
def estimate_rff_kernel(X: NPInputMat, D: int, ks: float, l: float) -> NPKernel:
    N, d = X.shape
    cov_omega = np.eye(d) / l**2
    omega = _module_rng.multivariate_normal(np.zeros(d), cov_omega, D // 2)
    Z = zrf(omega, D, X) * np.sqrt(ks)  # (D, N)
    approx_cov = Z.T @ Z  # (N, N) kernel approximation
    return approx_cov


def generate_rff_data(
    n: int,
    xmean: np.ndarray,
    xcov_diag: np.ndarray,
    noise_var: float,
    kernelscale: float,
    lenscale: float,
    D: int,
    kernel_type: str = "rbf",
    **kwargs,
) -> Tuple[NPInputMat, NPSample]:
    """Generates a data sample from a MVN and a sample from an approximate GP using RFF

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
    x = _module_rng.multivariate_normal(xmean, xcov, n)

    noisy_sample, approx_cov = sample_rff_from_x(
        x,
        kernelscale,
        noise_var,
        lenscale,
        _module_rng,
        D,
        kernel_type,
        **kwargs,
    )
    return x, noisy_sample


def sample_chol_from_x(
    x: NPInputMat,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    L: np.ndarray,
) -> Tuple[NPSample, NPKernel]:
    n, d = x.shape
    u = rng.standard_normal(n)
    y_noise = L @ u
    approx_cov = L @ L.T
    return y_noise, approx_cov


def sample_rff_from_x(
    x: NPInputMat,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    kernel_type: str = "rbf",
    **kwargs,
) -> Tuple[NPSample, NPKernel]:
    """Generates sample from approximate GP using RFF method at points x

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


def sample_mat_rff_from_x(
    x,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    G: int,
    nu: float,
):
    n, d = x.shape
    w = rng.standard_normal((D,))
    y, C = (
        np.zeros(
            n,
        ),
        np.nan,
    )

    omega_y = rng.standard_normal((D // 2, d)) * np.sqrt(2) / l
    omega_u = rng.chisquare(2 * nu, size=(D // 2,))
    omega = np.sqrt(2 * nu / np.tile(omega_u, (d, 1)).T) * omega_y
    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n,))
    y_noise = y + noise
    return y_noise, approx_cov


def sample_se_rff_from_x(
    x: NPInputMat,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
) -> Tuple[NPSample, NPKernel]:
    """Generates sample from approximate GP using RFF method at points x

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
    cov_omega = np.eye(d) / l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D // 2)

    w = rng.standard_normal((D,))

    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n,))
    y_noise = y + noise
    return y_noise, approx_cov


def sample_lap_rff_from_x(
    x: NPInputMat,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
) -> Tuple[NPSample, NPKernel]:
    """Generates sample from approximate Laplacian-kernel GP using RFF method
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
    cov_omega = np.eye(d) / l**2
    omega = np.zeros((D // 2, d))
    for di in range(d):
        omega[:, di] = np.tan(np.pi * (rng.uniform(size=D // 2) - 0.5))

    w = rng.standard_normal((D, 1))

    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n,))
    y_noise = y + noise
    return y_noise, approx_cov


@jit(nopython=True, parallel=True, fastmath=True)
def _sample_se_rff_from_x(
    x: NPInputMat,
    sigma: float,
    omega: NDArray[Shape["N,D"], Float],
    w: NDArray[Shape["D,1"], Float],
    compute_cov=False,
) -> Tuple[NPSample, NPKernel]:
    D = w.shape[0]
    if compute_cov:
        pass
    else:
        approx_cov = np.nan
    n = x.shape[0]
    y = np.zeros((n,))
    for i in prange(n):
        y[i] = f_rf(omega, D, w, x[i, :]) * np.sqrt(sigma)
    return y, approx_cov
