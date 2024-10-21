#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import linalg
from scipy.special import gamma
from scipy.spatial import distance_matrix

from typing import Tuple
from nptyping import NDArray, Shape, Float

n,d = 100,5
ks = 0.9
ls=3.0
nu=1.5
noise_var = 0.1
rng = np.random.default_rng(1)

D = 12 * int(n/noise_var)

#%%
rng = np.random.default_rng(1)
T_TYPE = torch.cuda.DoubleTensor if torch.cuda.is_available(
) else torch.DoubleTensor  # type: ignore

torch.set_default_tensor_type(T_TYPE)

NPInputVec = NDArray[Shape["P,1"], Float]
NPInputMat = NDArray[Shape["N,P"], Float]
NPSample = NDArray[Shape["N,1"], Float]
NPKernel = NDArray[Shape["N,N"], Float]

def k_se(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1,x2)**2/(2*ls**2))

def k_per(x1: np.ndarray, x2: np.ndarray, sigma, ls, p) -> np.ndarray:
    return sigma * np.exp(-2*np.sin(np.pi/p * distance_matrix(x1,x2)/(ls**2))**2)

def k_mat_half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1,x2)/ls)

def k_mat_3half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = distance_matrix(x1,x2)
    return sigma * (1 + np.sqrt(3) * D/ls) * np.exp(-np.sqrt(3)*D/ls)

def k_mat_5half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = distance_matrix(x1,x2)
    return sigma * (1 + np.sqrt(5) * D/ls + 5/3 * D**2/ls**2) * np.exp(-np.sqrt(5)*D/ls)

def k_mat(x1: np.ndarray, x2: np.ndarray, sigma, ls, nu) -> np.ndarray:
    if nu == 0.5:
        return k_mat_half(x1,x2,sigma,ls)
    if nu == 1.5:
        return k_mat_3half(x1,x2,sigma,ls)
    if nu == 2.5:
        return k_mat_5half(x1,x2,sigma,ls)
    if nu >= 1000:
        warnings.warn("Large nu; treating as squared exp.")
        return k_se(x1,x2,sigma,ls)
    else:
        D = distance_matrix(x1,x2)
        return sigma * 2**(1-nu) / gamma(nu) * (np.sqrt(2*nu) * D/ls)**nu * kv(nu,np.sqrt(2*nu)*D/ls)

def msqrt(M: np.ndarray) -> np.ndarray:
    U, s, V = linalg.svd(M)
    Msqrt = U @ np.diag(np.sqrt(s)) @ V
    return Msqrt

def _sample_se_rff_from_x(x: NPInputMat, sigma: float,
                          omega: NDArray[Shape["N,D"],
                                         Float],
                          w: NDArray[Shape["D,1"],
                                     Float],
                          compute_cov=False) -> Tuple[NPSample, NPKernel]:
    D = w.shape[0]
    if compute_cov:
        Z = zrf(omega, D, x)*np.sqrt(sigma)
        approx_cov = Z.T @ Z
        # y = (Z @ w).flatten()
    else:
        approx_cov = np.nan
    
    n = x.shape[0]
    y = np.zeros((n, ))
    for i in range(n):
        y[i] = f_rf(omega, D, w, x[i, :]) * np.sqrt(sigma)
    return y, approx_cov

# @jit(nopython=True, fastmath=True)
def zrf(omega: NDArray[Shape["D, P"],
                       Float],
        D: int, x: NPInputVec) -> NDArray[Shape["[cos,sin] x n_rff"],
                                          Float]:
    if x.ndim == 1:
        n = 1
    else:
        n = x.shape[0]
    v = np.dot(omega, x.T) #omega @ x.T
    return np.sqrt(2/D) * np.concatenate((np.cos(v), np.sin(v))) 


# @jit(nopython=True, fastmath=True)
def f_rf(
    omega: NDArray[Shape["D, P"],
                   Float],
    D: int, w: NDArray[Shape["2 x n_rff"],
                       Float],
    x: NPInputVec) -> float: return np.sum(
    w * zrf(omega, D, x))  # GP approximation
#%%
def sample_mat_rff_from_x3(x, sigma: float, noise_var: float, l:
                          float, rng: np.random.Generator, D: int,
                          nu: float):
    n, d = x.shape
    w = rng.standard_normal((D, ))
    y, C = np.zeros(n,), np.nan

    omega_y = rng.standard_normal((D//2, d)) * np.sqrt(2)/l
    omega_u = rng.chisquare(2*nu, size=(D//2,))
    omega = np.sqrt(2*nu/np.tile(omega_u,(d,1)).T) * omega_y
    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n, ))
    y_noise = y + noise
    return y_noise, approx_cov


def estimate_matern_rff_kernel(
        X, D: int, ks: float, l: float, nu: float):
    N, d = X.shape
    omega_y = rng.standard_normal((D//2, d)) * np.sqrt(2)/l
    omega_u = rng.chisquare(2*nu, size=(D//2,))
    omega = np.sqrt(2*nu/np.tile(omega_u,(d,1)).T) * omega_y
    Z = zrf(omega, D, X).T*np.sqrt(ks)
    approx_cov = np.inner(Z, Z)
    return approx_cov
#%%
# def test_sample3(x, sigma, noise_var, l, rng, D, nu):
#     n,d = x.shape
#     ym = sample_mat_rff_from_x3(x,sigma, noise_var, l, rng, D,nu)
#     y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
#     # y_approx = ym[0]
#     err = np.linalg.norm(y_true-y_approx)
#     return err

# err_sqrt3 = test_sample3(x, ks, noise_var, l, rng, D, nu)
# %%
x = rng.standard_normal((n,d)) / np.sqrt(d)
K = k_mat_3half(x,x,ks,ls)
Krff = estimate_matern_rff_kernel(x,D,ks,ls,nu)

u = rng.standard_normal(n)

y = msqrt(K) @ u
yrff = msqrt(Krff) @ u
ychol = linalg.cholesky(K) @ u

gram_mat_err = linalg.norm(K-Krff)
sample_err = linalg.norm(y-yrff)
chol_sample_err = linalg.norm(y-ychol)
# %%
print("||K-Krff||:",gram_mat_err)
print("||y-yrff||:",sample_err)
print("||y-Lu||:",chol_sample_err)
# %%
