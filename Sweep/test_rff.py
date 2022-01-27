#%%
from numpy.linalg.linalg import _assert_finite
from numpy.testing._private.utils import assert_array_less
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
from gpybench.maths import safe_logdet, Tr
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

#%% define parameters
nv = 0.0
l = 1.1
b = 1

rng = np.random.default_rng(1)
d = 10

n = 100
#%% define functions to compute/generate actual data
kernel = gpytorch.kernels.RBFKernel()
kernel.lengthscale = l
kernel = gpytorch.kernels.ScaleKernel(kernel)
kernel.outputscale = b

covariate_params = (np.zeros((d,)), 1 / d * np.eye(d))
covariate_distributions = ds.MultivariateNormal(*covariate_params, rng=rng)
# setup sampler
sampler = ds.GPGenerator(
    covariate_dists=[covariate_distributions],
    kernel=kernel,
    noise_var=torch.tensor(nv),
    rng=rng,
)
# %% test KL bounds with eigenvalues
# assumptions on structure/distribution of E?
def construct_kernels(n):
    X = sampler.sample_covariates(n)

    # K1 = ds.sample_with_correlation(n)
    K1 = numpify(kernel)(X)
    K2 = ds.sample_with_correlation(n)

    K = K1
    E = (1)**np.random.randint(2)*1e-4 * K2
    Khat = K + E
    return K, Khat, E

def kldiv(K, Khat, lam=None, lhat=None):
    n = K.shape[0]
    if lam is not None:
        kl = 0.5 * (Tr(np.linalg.solve(K,Khat)) - n + np.log(lam).sum() - np.log(lhat).sum())
    else:
        kl = 0.5 * (Tr(np.linalg.solve(K,Khat)) - n + safe_logdet(K) - safe_logdet(Khat))
    return kl

#%% check KL approximation
K, Khat, E = construct_kernels(n)

lam, U = np.linalg.eigh(K)
lhat, Uhat = np.linalg.eigh(Khat)

Kinv = U @ np.diag(1/lam) @ U.T

Delta = Kinv @ E

KL = kldiv(Khat, K)
KL_Delta = Tr(Delta @ Delta)/4
#%% - still gets worse with n :/
def test_KL_bounds(n: int) -> Tuple[float, float, float]:

    K, Khat, E = construct_kernels(n)

    M2 = np.linalg.norm(E, ord=2)
    lam, U = np.linalg.eigh(K)
    lhat, Uhat = np.linalg.eigh(Khat)

    Kinv = U @ np.diag(1/lam) @ U.T

    # KL_L = n/2 * (1/(1+M2/lam.min()) - 1 + np.log2(1 + M2/lam.max()))
    # KL_U = n/2 *(np.log2(1 + M2/lam.min()) - M2/lam.max())
    KL_L = 0
    # KL_U = np.linalg.norm(Kinv)**2 * np.linalg.norm(E)**2/4
    #better:
    KL_U = Tr(Kinv) * M2/4

    KL = kldiv(Khat, K)

    return KL_L,KL,KL_U

n_trials = 100
sizes_n = [10, 20, 50, 100, 500]
gaps = np.empty((len(sizes_n), n_trials))
for j, n in enumerate(sizes_n):
    for i in range(n_trials):
        KL_L,KL,KL_U = test_KL_bounds(n)
        gaps[j, i] = KL_U-KL

print_mean_std(gaps, 1, sizes_n)
# %% - Tr bound tightness: tight  - err scales with n
# def test_Tr_bound(n):
#     K, Khat, E = construct_kernels(n)

#     M2 = np.linalg.norm(E, ord=2)
#     lam,U = np.linalg.eigh(K)
#     Kinv = U @ np.diag(1/lam) @ U.T
#     TrKinvKhat = np.diag(Kinv @ Khat).sum()

#     tr_err = n*(1-M2/lam.max()) - TrKinvKhat
#     return tr_err


# tr_gaps = np.empty((len(sizes_n), n_trials))
# for j, n in enumerate(sizes_n):
#     for i in range(n_trials):
#         tr_gaps[j, i] = test_Tr_bound(n)

# print_mean_std(tr_gaps, 1, sizes_n)
#%% logdet bound tightness: slack - err scales with n
# def test_logdet_bound(n):
#     K, Khat, E = construct_kernels(n)

#     M2 = np.linalg.norm(E, ord=2)

#     lam,U = np.linalg.eigh(K)
#     lhat = np.linalg.eigvalsh(Khat)

#     logdetKhatinvK = np.log(lam).sum()-np.log(lhat).sum()
#     det_err = n*np.log(1+M2/lam.min())-logdetKhatinvK

#     return det_err

# ld_gaps = np.empty((len(sizes_n), n_trials))
# for j, n in enumerate(sizes_n):
#     for i in range(n_trials):
#         ld_gaps[j, i] = test_logdet_bound(n)

# print_mean_std(ld_gaps, 1, sizes_n)

#%% - fDelta seems to upper bound, but using the Tr(Delta) bound DOESN'T work and scales VERY badly with n
# def test_Delta_Tr_bound(n):
#     K, Khat, E = construct_kernels(n)
#     Delta = np.linalg.solve(K, E)

#     MF = np.linalg.norm(E)

#     lam,U = np.linalg.eigh(K)
#     Kinv = U @ np.diag(1/lam) @ U.T

#     TrKinvKhat = np.diag(Kinv @ Khat).sum()
#     TrDelta_bound = np.linalg.norm(Kinv) * MF
#     fDelta = n - Tr(Delta) + Tr(Delta @ Delta)
#     fDelta_bound = n - TrDelta_bound + TrDelta_bound ** 2
#     tr_err = fDelta_bound - TrKinvKhat

#     return tr_err

# trd_gaps = np.empty((len(sizes_n), n_trials))
# for j, n in enumerate(sizes_n):
#     for i in range(n_trials):
#         trd_gaps[j, i] = test_Delta_Tr_bound(n)

# print_mean_std(trd_gaps, 1, sizes_n)
# %% detKhatinvK = np.exp(np.log(lam).sum()-np.log(lhat).sum())
# Delta bound here does seem to work
# def test_Delta_logdet_bound(n):
#     K, Khat, E = construct_kernels(n)
#     Delta = np.linalg.solve(K, E)

#     MF = np.linalg.norm(E)

#     lam,U = np.linalg.eigh(K)
#     Kinv = U @ np.diag(1/lam) @ U.T

# # fDelta = lambda a: np.exp(n*np.log(a) + safe_logdet(Delta)) - Tr(Delta) + 0.5*Tr(Delta)**2 - 0.5*Tr(Delta @ Delta)
# # fDelta = fDelta(1)
#     TrDelta_bound = np.linalg.norm(Kinv) * MF
#     fDelta = np.log(1 + Tr(Delta) + 0.5*Tr(Delta)**2 - 0.5*Tr(Delta @ Delta))
#     fDelta_bound = np.log(1 + TrDelta_bound)

#     det_err = fDelta_bound - (safe_logdet(K) - safe_logdet(Khat))

#     return det_err

# ldd_gaps = np.empty((len(sizes_n), n_trials))
# for j, n in enumerate(sizes_n):
#     for i in range(n_trials):
#         ldd_gaps[j, i] = test_Delta_logdet_bound(n)

# print_mean_std(ldd_gaps, 1, sizes_n)

