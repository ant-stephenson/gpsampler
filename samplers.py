from re import I
import numpy as np
# from numba import jit
from scipy.special import ellipj, ellipk
import scipy.linalg as linalg
from functools import partial
from itertools import repeat
import torch
import gpytorch
from joblib import Parallel, delayed
# from gpytorch.utils import contour_integral_quad
from typing import Tuple, Optional, Union
from nptyping import NDArray, Shape, Float
from contextlib import ExitStack
import warnings

import math
import warnings
import copy

import torch

from linear_operator.utils.broadcasting import _matmul_broadcast_shape
from linear_operator.utils.linear_cg import linear_cg
from linear_operator.utils.minres import minres
from gpytorch.utils.warnings import NumericalWarning

# warnings.simplefilter("error")

rng = np.random.default_rng()
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


# @jit(nopython=True)
def zrf(omega: NDArray[Shape["D, P"],
                       Float],
        D: int, x: NPInputVec) -> NDArray[Shape["[cos,sin] x n_rff"],
                                          Float]:
    if x.ndim == 1:
        n = 1
    else:
        n = x.shape[0]
    v = omega @ x.T
    return np.sqrt(2/D) * np.hstack((np.cos(v.T), np.sin(v.T))).reshape(-1, 1)


# @jit(nopython=True)
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


def sample_cg_from_x(x: NPInputMat, sigma: float, noise_var: float, l: float,
                     rng: np.random.Generator, m: int) -> Tuple[NPSample, NPKernel]:
    from scipy.sparse import diags
    from gpytools.maths import invmsqrt
    n, d = x.shape
    z = rng.standard_normal(n)
    y0 = np.zeros(n)

    Q = construct_kernels(
        l, sigma)(
        torch.as_tensor(x)).add_jitter(
        1 * noise_var).evaluate().detach().numpy()

    r0 = z - Q @ y0

    beta = np.zeros(m)
    alpha = np.zeros(m)

    V = np.zeros((n, m))

    beta[0] = linalg.norm(r0)
    V[:, 0] = r0/beta[0]

    e1 = np.concatenate(([1], np.zeros(m-1, dtype=int)))

    for j in range(0, m-1):
        wj = Q @ V[:, j] - beta[j] * V[:, j-1]
        alpha[j] = wj.T @ V[:, j]
        beta[j+1] = linalg.norm(wj)
        V[:, j+1] = wj/beta[j+1]

    offset = [-1, 0, 1]
    # not sure if we should exclude the first or last beta...
    T = diags([beta[:-1], alpha, beta[:-1]], offset).toarray()
    y = y0 + beta[0] * V @ invmsqrt(T) @ e1
    y = Q @ y
    approx_cov = V @ T @ V.T
    return y, approx_cov


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
    elif kernel_type == "matern":
        kargs = {**kwargs}
        if "G" in kargs.keys():
            G = kargs["G"]
        else:
            G = int(np.sqrt(D))
            D = D // G
        nu = kargs["nu"]

        return sample_mat_rff_from_x(x, sigma, noise_var, l, rng, D, G, nu)
    elif kernel_type == "laplacian":
        return sample_lap_rff_from_x(x, sigma, noise_var, l, rng, D)
    else:
        raise NotImplementedError


def sample_mat_rff_from_x(x: NPInputMat, sigma: float, noise_var: float, l:
                          float, rng: np.random.Generator, D: int, G: int,
                          nu: float) -> Tuple[NPSample, NPKernel]:
    n, d = x.shape
    w = rng.standard_normal((D, 1))
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

    for i, ss in enumerate(s):
        omega = rng.standard_normal((D//2, d))
        if n > N:
            ys, Cs = np.zeros(n,), np.nan
            parts = int(np.ceil(n/N))
            for p in range(parts):
                idx = np.s_[(p*N):((p+1)*N)]
                ys[idx], Cp = _sample_se_rff_from_x(
                    x[idx, :], sigma, omega[:, :]/np.sqrt(ss), w)
        else:
            ys, Cs = _sample_se_rff_from_x(x, sigma, omega[:, :]/np.sqrt(ss), w)
        y += ys
        C += Cs

    y /= np.sqrt(G)
    C /= G
    noise = rng.normal(scale=np.sqrt(noise_var), size=n)
    y_noise = y + noise
    return y_noise, C


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

    w = rng.standard_normal((D, 1))

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


# @jit(nopython=True)
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
    for i in range(n):
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
