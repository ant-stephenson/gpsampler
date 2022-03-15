#%%
from numpy.linalg.linalg import _assert_finite
from numpy.testing._private.utils import assert_array_less
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
from gpytools.maths import safe_logdet, Tr
from gpytools.utils import check_bounds
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import kl_div

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
    E = (1)**np.random.randint(2)*1e-5 * K2
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
Mf = np.linalg.norm(E)
lam, U = np.linalg.eigh(J)
lhat, Uhat = np.linalg.eigh(Jhat)

Jinv = U @ np.diag(1/lam) @ U.T

#%% - lower bound not working - why??
KL_J = kldiv(Jhat, J)
DeltaJ = Jinv @ E
upper_klj = 0.25 * Tr(DeltaJ @ DeltaJ)
lower_klj = upper_klj - 1/6 * Tr(DeltaJ @ DeltaJ @ DeltaJ)
check_bounds(KL_J, lower_klj, upper_klj, warn_mode=True)
# %%
K = b * J + np.eye(n) * nv
Khat = b * Jhat + np.eye(n) * nv

Kinv = U @ np.diag(1/(b*lam + nv)) @ U.T
#%% - w outputscale and noise
Ek = Khat - K
KL_K = kldiv(Khat, K)
DeltaK = Kinv @ Ek
upper_klk = 0.25 * Tr(DeltaK @ DeltaK)
lower_klk = upper_klk - 1/6 * Tr(DeltaK @ DeltaK @ DeltaK)
check_bounds(KL_K, lower_klk, upper_klk, warn_mode=True)
# %% - with bounds on the trace terms - these at least appear to work
upper_klk_2 = Mf**2/(4*nv**2)
lower_klk_2 = Mf**2/(4*n**2)
check_bounds(KL_K, lower_klk_2, upper_klk_2, warn_mode=True)
# %% empirical kl calculation on a sample, does kl_div take cdf or pdf? (looks
# like cdf))
from sklearn.neighbors import KernelDensity
x = np.random.randn(10000) * 2
y = np.random.rand(10000) * 3.5

# kde_x = KernelDensity().fit(x.reshape(-1,1))
# kde_y = KernelDensity().fit(y.reshape(-1,1))

ecdf_x = ECDF(x)
ecdf_y = ECDF(y)

kl_emp_xy = kl_div(ecdf_x(x), ecdf_y(y)).mean()
# kl_emp_xy = kl_div(np.exp(kde_x.score_samples(x.reshape(-1,1))), np.exp(kde_y.score_samples(y.reshape(-1,1)))).mean()
kl_the_xy = np.log(3.5/2) + (2**2 + (0-0)**2)/(2*3.5**2) - 0.5
print(kl_emp_xy, kl_the_xy)
# %%
