"""Contour-integral quadrature (CIQ) sampler and sparse kernel helpers.

Absorbed from the CIQ sections of the original samplers.py monolith.
"""

from __future__ import annotations
import math
import warnings
import numpy as np
import torch
import gpytorch
from scipy.special import ellipj, ellipk
from contextlib import ExitStack
from typing import Tuple, Optional, Union

try:
    from linear_operator.utils.broadcasting import _matmul_broadcast_shape
    from linear_operator.utils.linear_cg import linear_cg
    from linear_operator.utils.minres import minres
except ImportError:
    pass  # only needed for CIQ sampler; lrff/rff paths work without it
from gpytorch.utils.warnings import NumericalWarning

try:
    from gpprediction.kernels.keops_kernels import RBFKernel
except (ImportError, AttributeError):
    pass  # only needed for KeOps-based CIQ sampler

from ._utils import construct_kernels, approx_extreme_eigs, rng as _module_rng, NPInputMat, NPSample, NPKernel


# ---------------------------------------------------------------------------
# Matrix square root via contour integration (Hale 2008)
# ---------------------------------------------------------------------------

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
        x, kernelscale, noise_var, lenscale, kernel_type, _module_rng, J, Q,
        checkpoint_size, max_preconditioner_size)

    return x.cpu().numpy(), sample


# ---------------------------------------------------------------------------
# CIQ sampler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Contour integral quadrature (local implementation)
# ---------------------------------------------------------------------------

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

        r"""
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


# ---------------------------------------------------------------------------
# ID Preconditioner
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sparse kernel classes
# ---------------------------------------------------------------------------

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
