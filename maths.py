import numpy as np
import torch
from functools import singledispatch
from scipy import linalg
from scipy.special import gamma, kv
from scipy.spatial import distance_matrix as _distance_matrix


@singledispatch
def msqrt(M: np.ndarray) -> np.ndarray:
    U, s, V = linalg.svd(M)
    return U @ np.diag(np.sqrt(s)) @ V


@msqrt.register
def _msqrt_torch(M: torch.Tensor) -> torch.Tensor:
    U, s, V = torch.linalg.svd(M)
    return U @ torch.diag(torch.sqrt(s)) @ V


def invmsqrt(M: np.ndarray) -> np.ndarray:
    U, s, V = linalg.svd(M)
    return U @ np.diag(1 / np.sqrt(s)) @ V


@singledispatch
def id_inv(M: np.ndarray, nv: float, k: int) -> np.ndarray:
    import scipy.linalg.interpolative as sli

    n = M.shape[0]
    U, s, V = sli.svd(M, k)
    out = nv ** (-1) * (
        np.eye(n) - U @ (_inv(nv * np.diag(1 / s) + V.T @ U) @ V.T)
    )
    return out


@id_inv.register
def _id_inv_torch(M: torch.Tensor, nv: float, k: int) -> torch.Tensor:
    Minv = id_inv(M.detach().numpy(), nv, k)
    return torch.as_tensor(Minv)


def _inv(M: np.ndarray, thresh=1e-9) -> np.ndarray:
    U, s, V = linalg.svd(M)
    sinv = 1 / s
    sinv[s < thresh] = 0.0
    return U @ np.diag(sinv) @ V


def k_se(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-_distance_matrix(x1, x2) ** 2 / (2 * ls**2))


def k_mat_half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-_distance_matrix(x1, x2) / ls)


def k_mat_3half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = _distance_matrix(x1, x2)
    return sigma * (1 + np.sqrt(3) * D / ls) * np.exp(-np.sqrt(3) * D / ls)


def k_mat_5half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = _distance_matrix(x1, x2)
    return (
        sigma
        * (1 + np.sqrt(5) * D / ls + 5 / 3 * D**2 / ls**2)
        * np.exp(-np.sqrt(5) * D / ls)
    )


def k_mat(x1: np.ndarray, x2: np.ndarray, sigma, ls, nu) -> np.ndarray:
    import warnings

    if nu == 0.5:
        return k_mat_half(x1, x2, sigma, ls)
    if nu == 1.5:
        return k_mat_3half(x1, x2, sigma, ls)
    if nu == 2.5:
        return k_mat_5half(x1, x2, sigma, ls)
    if nu >= 1000:
        warnings.warn("Large nu; treating as squared exp.")
        return k_se(x1, x2, sigma, ls)
    D = _distance_matrix(x1, x2)
    return (
        sigma
        * 2 ** (1 - nu)
        / gamma(nu)
        * (np.sqrt(2 * nu) * D / ls) ** nu
        * kv(nu, np.sqrt(2 * nu) * D / ls)
    )
