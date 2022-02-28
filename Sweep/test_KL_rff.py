#%%
from numpy.linalg.linalg import _assert_finite
from numpy.testing._private.utils import assert_array_less
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
from gpytools.maths import safe_logdet, Tr
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt

#%% define parameters
nv = 0.2
l = 1.1
b = 2

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

#%%
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

def kldiv(K, Khat):
    kl = 0.5 * (Tr(np.linalg.solve(K,Khat)) - n + safe_logdet(K) - safe_logdet(Khat))
    return kl


#%%
J, Jhat, E = construct_kernels(n)

assert kldiv(J,J) < 1e-6
assert kldiv(Jhat, Jhat) < 1e-6

M2 = np.linalg.norm(E, ord=2)
lam, U = np.linalg.eigh(J)
lhat, Uhat = np.linalg.eigh(Jhat)

Jinv = U @ np.diag(1/lam) @ U.T

#%%
KL_J = kldiv(J, Jhat)
approx_klj = 0.25 * Tr(Jinv @ E @ Jinv @ E)
print(KL_J)
print(approx_klj)
# %%
K = b * J + np.eye(n) * nv
Khat = b * Jhat + np.eye(n) * nv

Kinv = U @ np.diag(1/(b*lam + nv)) @ U.T
#%%
KL_K = kldiv(K, Khat)
approx_klk = 0.25* Tr(Kinv @ E @ Kinv @ E)
print(KL_K)
print(approx_klk)
# %%
