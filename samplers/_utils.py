"""Shared state, type aliases, and utility functions used across sampler modules.

torch and gpytorch are imported lazily so that pure-numpy code paths
(lrff, iw_rff, stratified_rff, cg) never pay the ~1 GB torch/gpytorch cost.
"""

import numpy as np
from typing import Tuple, Optional, Union
from nptyping import NDArray, Shape, Float
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

from gpsampler.utils import msqrt
from gpsampler.maths import k_se, k_mat

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1)

# Lazy handles — populated by _ensure_torch() on first use.
_torch = None
_gpytorch = None
T_TYPE = None


def _ensure_torch():
    """Import torch + gpytorch and set the default tensor type (once)."""
    global _torch, _gpytorch, T_TYPE
    if _torch is not None:
        return
    import torch as _t
    import gpytorch as _g
    _torch = _t
    _gpytorch = _g
    T_TYPE = (_t.cuda.DoubleTensor if _t.cuda.is_available()
              else _t.DoubleTensor)
    _t.set_default_tensor_type(T_TYPE)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NPInputVec = NDArray[Shape["P,1"], Float]
NPInputMat = NDArray[Shape["N,P"], Float]
NPSample = NDArray[Shape["N,1"], Float]
NPKernel = NDArray[Shape["N,N"], Float]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

# @jit(nopython=True)
def k_true(sigma: float, l: float, xp: np.ndarray, xq: np.ndarray) -> float:
    return sigma * np.exp(-0.5*np.dot(xp-xq, xp-xq)/l**2)  # true kernel


def construct_kernels(
        l: float, b: float = 1.0, kernel=None,
        issparse=False):
    _ensure_torch()
    if kernel is None:
        kernel = _gpytorch.kernels.RBFKernel()
    if issparse:
        from .ciq import SparseKernel
        kernel = SparseKernel(kernel)
    kernel = _gpytorch.kernels.ScaleKernel(kernel)
    n_gpus = _torch.cuda.device_count()
    if n_gpus > 1:
        kernel = _gpytorch.kernels.MultiDeviceKernel(
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


# ---------------------------------------------------------------------------
# Spectral utilities (moved from leverage_reweighted_rff.py)
# ---------------------------------------------------------------------------

def kernel_matrix(X, kind="rbf", ell=0.1, nu=1.5):
    """Stationary kernel Gram matrix K with k(0)=1.  X is (n, d)."""
    D = cdist(X, X)
    if kind == "rbf":
        return np.exp(-(D ** 2) / (2 * ell ** 2))
    if kind == "matern":
        Dz = np.where(D == 0.0, 1e-12, D)
        f = np.sqrt(2 * nu) * Dz / ell
        K = (2 ** (1 - nu) / gamma(nu)) * (f ** nu) * kv(nu, f)
        np.fill_diagonal(K, 1.0)
        return K
    raise ValueError(f"unknown kernel {kind!r}")


def spectral_sampler(n_freq, d, kind="rbf", ell=0.1, nu=1.5, rng=None):
    """Draw n_freq frequencies from the kernel's spectral density p(omega).

    RBF:    omega ~ N(0, ell^{-2} I_d).
    Matern: omega ~ multivariate-t with 2*nu dof and scale ell^{-1}
            (verified to satisfy E_p[cos(w.tau)] = k(tau)).
    """
    rng = np.random.default_rng() if rng is None else rng
    g = rng.standard_normal((n_freq, d))
    if kind == "rbf":
        return g / ell
    if kind == "matern":
        u = rng.chisquare(2 * nu, size=(n_freq, 1))
        return (g / ell) * np.sqrt(2 * nu / u)
    raise ValueError(f"unknown kernel {kind!r}")
