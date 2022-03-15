#%%
from numpy.linalg.linalg import _assert_finite
from numpy.testing._private.utils import assert_array_less
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
from gpytools.maths import safe_logdet, Tr, partial_svd_inv, low_rank_inv_approx, k_true, zrf, estimate_rff_kernel, estimate_rff_kernel_inv, f_gp_approx, am
import gpybench.plotting as gplt
import gpytorch
from gpytorch.utils import contour_integral_quad
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
from pathlib import Path

#%%
if isnotebook():
    path = Path("..")
else:
    path = Path(".")
#%% define parameters
nv = 0.0
ls = 1.1
b = 1

rng = np.random.default_rng(1)
d = 10

n = 100
#%% define functions to compute/generate actual data
kernel = gpytorch.kernels.RBFKernel()
kernel.lengthscale = ls
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
    if kl < 0:
        warnings.warn("Negative KL")
        kl = np.nan
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
    KL_U = np.linalg.norm(Kinv)**2 * np.linalg.norm(E)**2/4
    #better? not sure if true...seems to hold empirically so far, but not better? Both scale with n
    # KL_U = (Tr(Kinv) * M2)**2/4

    KL = kldiv(Khat, K)

    return KL_L,KL,KL_U

n_trials = 10
sizes_n = [10, 100, 500]
gaps = np.empty((len(sizes_n), n_trials))
# for j, n in enumerate(sizes_n):
#     for i in range(n_trials):
#         KL_L,KL,KL_U = test_KL_bounds(n)
#         gaps[j, i] = KL_U-KL

# print_mean_std(gaps, 1, sizes_n)
#%% - construct RFF approx

get_D = lambda n: int(n**(3/2))

X = sampler.sample_covariates(n)
x_star = sampler.sample_covariates(1)
k_star = numpify(kernel)(X, x_star)
D = get_D(n)
K = numpify(kernel)(X)
Krff = estimate_rff_kernel(X, D, ls, b)
reg = 1e-9
Krff += reg * np.eye(n)
E = Krff-K

M2 = np.linalg.norm(E, ord=2)

#%% lower bound on M2: (s2 from On Error of RFF paper, M2_lower from
#concentration of evals)
s2 = 1/D * (1 + 0.5*am(1, ls=ls, b=2) - am(2, ls=ls))
M2_lower = 2 * np.sqrt(s2 * n)
assert M2 > M2_lower
#%% - sample z ~ N(0,1)
z = np.random.multivariate_normal(np.zeros(n), np.eye(n))
# %% try again; compute objects: Khat = Krff = Urff Lrff Urff.T, Urff = U + V, Lrff = L + De
# compute error: ||K^1/2 - Krff^1/2||_2
nv = 1e-3
lam, U = np.linalg.eigh(K)
lrff, Urff = np.linalg.eigh(Krff)
De = lrff-lam
V = Urff - U

rootK = U @ np.diag(np.sqrt(lam)) @ U.T
rootKrff = Urff @ np.diag(np.sqrt(lrff)) @ Urff.T

Delta = U @ np.diag(1/lam) @ U.T @ E

sqrt_err = np.linalg.norm(rootK - rootKrff, ord=2)

#%% error in draws
f = rootK @ z
frff = rootKrff @ z
err_draw_rff = np.linalg.norm(f - frff)
# %% - theoretical bound on bias (?)
Elam1 = (n-1) * am(1) + b + (am(1)**2 - am(2))/am(1)
bias2 = Elam1 * (2+M2/reg + 2*np.sqrt(1+M2/reg))
#%% ciq draw comparison
J = int(np.sqrt(n) * np.log(n))
Q = int(np.log(n))
solves, weights, _, _ = contour_integral_quad(kernel(X).evaluate_kernel(), torch.tensor(z.reshape(-1,1)), max_lanczos_iter=J, num_contour_quadrature=Q)
fciq = (solves * weights).sum(0).detach().numpy().squeeze()
err_draw_ciq = np.linalg.norm(rootK @ z - fciq)
#%% plot error in draws
with gplt.LaTeX() as _:
    plt.plot(f-frff)
    plt.plot(f-fciq)
    plt.legend(["RFF err", "CIQ err"])
    plt.xlabel("n")
    plt.ylabel("$f_{true}-f_{approx}$")
gplt.save_fig(path, "GP_draw_error_comp", suffix="jpg", show=True)
# %% calculate error bound
# sqrt_err_bound = 0.5 * np.max(De/lam) + 3*np.max(np.sqrt(lam + De))
sqrt_err_bound = np.sqrt(np.linalg.norm(K-Krff, ord=2))
print(sqrt_err)
print(sqrt_err_bound)
# %%
with gplt.LaTeX() as _:
    fig,ax = plt.subplots()
    ax.plot(De)
    ax.plot(lam)
    ax.plot(lrff)
    ax.set(xscale="log", yscale="log")
    ax.axhline(nv, ls="--", color="red")
    ax.legend(["$d_i$", "$\lambda_i$", "$\hat{\lambda_i}$"])
    ax.set_title("D=n")

#%% try CIQ method for comparison
# with gpytorch.settings.ciq_samples(state=True):
#     fciq = kernel(X).zero_mean_mvn_samples(1)
# %% plot KL as a function of D for given n
def test_KL(D):
    X = sampler.sample_covariates(n)
    K = numpify(kernel)(X)
    Krff = estimate_rff_kernel(X, D, ls, b)
    kl = kldiv(Krff, K)
    return kl

Drange = [2**i for i in range(int(0.5*np.log(n)/np.log(2)), int(2*np.log(n)/np.log(2))+1)]
kl = np.empty(len(Drange))
for i,Di in enumerate(Drange):
    kl[i]= test_KL(Di)
# %%
