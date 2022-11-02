#%%
from scipy import linalg
from numpy.testing._private.utils import assert_array_less
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
from gpytools.maths import safe_logdet, Tr, partial_svd_inv, low_rank_inv_approx, k_true, zrf, estimate_rff_kernel, estimate_rff_kernel_inv, f_gp_approx, am, msqrt, k_se
import gpybench.plotting as gplt
import gpytorch
from gpytorch.utils import contour_integral_quad
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
from pathlib import Path

from scipy import optimize

from gpsampler import gpsampler

#%%
if isnotebook():
    path = Path("..")
else:
    path = Path(".")
#%% define parameters
nv = 0.01
ls = 0.5
b = 1

rng = np.random.default_rng(1)
d = 2

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
    E = (-1)**np.random.randint(2)*1e-4 * K2
    Khat = K + E
    return K, Khat, E

def kldiv(K, Khat, lam=None, lhat=None):
    n = K.shape[0]
    if lam is not None:
        kl = 0.5 * (Tr(linalg.solve(K,Khat)) - n + np.log(lam).sum() - np.log(lhat).sum())
    else:
        kl = 0.5 * (Tr(linalg.solve(K,Khat)) - n + safe_logdet(K) - safe_logdet(Khat))
    if kl < 0:
        warnings.warn("Negative KL")
        kl = np.nan
    return kl

#%% check KL approximation
K, Khat, E = construct_kernels(n)

lam, U = linalg.eigh(K)
lhat, Uhat = linalg.eigh(Khat)

Kinv = U @ np.diag(1/lam) @ U.T

Delta = Kinv @ E

KL = kldiv(Khat, K)
KL_Delta = Tr(Delta @ Delta)/4
#%% - still gets worse with n :/
def test_KL_bounds(n: int) -> Tuple[float, float, float]:

    K, Khat, E = construct_kernels(n)

    M2 = linalg.norm(E, ord=2)
    lam, U = linalg.eigh(K)
    lhat, Uhat = linalg.eigh(Khat)

    Kinv = U @ np.diag(1/lam) @ U.T

    # KL_L = n/2 * (1/(1+M2/lam.min()) - 1 + np.log2(1 + M2/lam.max()))
    # KL_U = n/2 *(np.log2(1 + M2/lam.min()) - M2/lam.max())
    KL_L = 0
    KL_U = linalg.norm(Kinv)**2 * linalg.norm(E)**2/4
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

M2 = linalg.norm(E, ord=2)

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
lam, U = linalg.eigh(K)
lrff, Urff = linalg.eigh(Krff)
De = lrff-lam
V = Urff - U

rootK = U @ np.diag(np.sqrt(lam)) @ U.T
rootKrff = Urff @ np.diag(np.sqrt(lrff)) @ Urff.T

Delta = U @ np.diag(1/lam) @ U.T @ E

sqrt_err_rff = linalg.norm(rootK - rootKrff, ord=2)

#%% error in draws
f = rootK @ z
frff = rootKrff @ z
err_draw_rff = linalg.norm(f - frff)
# %% - theoretical bound on bias (?)
Elam1 = (n-1) * am(1) + b + (am(1)**2 - am(2))/am(1)
bias2 = Elam1 * (2+M2/reg + 2*np.sqrt(1+M2/reg))
#%% ciq draw comparison
J = int(np.sqrt(n) * np.log(n))
Q = int(np.log(n))
solves, weights, _, _ = contour_integral_quad(kernel(X).evaluate_kernel(), torch.tensor(z.reshape(-1,1)), max_lanczos_iter=J, num_contour_quadrature=Q)
fciq = (solves * weights).sum(0).detach().numpy().squeeze()
err_draw_ciq = linalg.norm(rootK @ z - fciq)
#%% plot error in draws
with gplt.LaTeX() as _:
    plt.plot(f-frff)
    plt.plot(f-fciq)
    plt.legend(["RFF err", "CIQ err"])
    plt.xlabel("n")
    plt.ylabel("$f_{true}-f_{approx}$")
# gplt.save_fig(path, "GP_draw_error_comp", suffix="jpg", show=True)
# %% calculate error bound
# sqrt_err_bound = 0.5 * np.max(De/lam) + 3*np.max(np.sqrt(lam + De))
sqrt_err_bound = np.sqrt(linalg.norm(K-Krff, ord=2))
print(sqrt_err_rff)
print(sqrt_err_bound)
# %%
with gplt.LaTeX() as _:
    fig,ax = plt.subplots()
    ax.plot(De)
    ax.plot(lam)
    ax.plot(lrff)
    ax.set(xscale="log", yscale="log")
    ax.axhline(nv, ls="--", color="red")
    ax.legend([r"$d_i$", r"$\lambda_i$", r"$\hat{\lambda_i}$", r"$\sigma_{\xi}^2$"])
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
# %% test ciq sqrt formation
rootKciq = gpsampler.estimate_ciq_kernel(X, J, Q, b, ls)
sqrt_err_ciq = linalg.norm(rootK - rootKciq, ord=2)
linalg.norm(K - rootKciq @ rootKciq)
# %%
data = gpsampler.generate_ciq_data(1000, np.ones(3), np.ones(3), 0.1, 0.9, 1.0, 10, 10)

#%% - plot J as a function of C and d for theoretical prediction based on
#preconditioning
from gpybench.plotting import contourplot
from gpytools.utils import round_ordermag
c = np.linspace(1e-3,10,100)
d = range(1,100)
fig, ax = plt.subplots(2,2)
for i,_ax in enumerate(ax.flatten()):
    n = int(10**(i+5))
    _ax.set_title(f"{n: 1.0e}")
    max_level = f(n,100,c.min())
    levels = np.unique(round_ordermag(np.linspace(1,max_level,6)))
    contourplot(c,d,lambda x,y: f(n,y,x), ax=_ax, fig=fig, opt_args={"levels":levels})
    _ax.set_xscale('log')
    _ax.set_yscale('log')
    _ax.set_xlabel("C")
    _ax.set_ylabel("d")
plt.suptitle("J")
# save_fig(Path("."), filename="J_contours_C_d",suffix="png")

#%% plot num. rank + cond. vs lengthscale for rbf
tol = 1e-6
_n = 100
_ls = np.logspace(np.log10(1e-3),np.log10(10),_n)
_ds = [1,2,3,4]
a = lambda d,l: (1+2/(d*l**2))**(-d/2)
shape = (_n,4,10)
cond = np.zeros(shape)
rank = np.zeros(shape)
approx_rank = np.zeros(shape)
expected_cond = np.zeros(shape)
for k in range(10):
    for i,l in enumerate(_ls):
        for j,d in enumerate(_ds):
            K = ds.sample_rbf_kernel(n=_n, ls=l, d=d)
            cond[i,j,k] = np.linalg.cond(K)
            rank[i,j,k] = np.linalg.matrix_rank(K, tol=tol)
            approx_rank[i,j,k] = _n - d + np.log(tol/_n)
            expected_cond[i,j,k] = (1+(_n-1)*a(d,l))/(1-a(d,l))
cond = cond.mean(axis=2)
rank = rank.mean(axis=2)
approx_rank = approx_rank.mean(axis=2)
expected_cond = expected_cond.mean(axis=2)
# %%
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(_ls, cond[:,0]/1e8,
        color="red")
ax.plot(_ls, cond[:,1]/1e8,
        color="red", linestyle="--")
ax.axhline(_n/1e2, color="black", linestyle="-.")
# set y-axis label
ax.set_ylabel("Condition Number/1e8",
              color="red",
              fontsize=14)
ax.set_xlabel("Lengthscale")              
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(_ls, rank[:,0], color="blue")
ax2.plot(_ls, rank[:,1], color="blue", linestyle="--")
# ax2.plot(_ls, approx_rank[:,0], color="green")
# eps = 1e-6
# ax2.axhline(9*np.log(_n)/(eps**2-eps**3))
with gplt.LaTeX():
    ax2.set_ylabel("Numerical. rank; $\epsilon=1e-6$",color="blue",fontsize=14)
plt.title("RBF-Kernel properties")
plt.legend(["d=1","d=2"])
leg = ax2.get_legend()
[lgd.set_color('black') for lgd in leg.legendHandles]
# plt.show()
# gplt.save_fig(path, f"rbf_k_cond-rank_by_l", suffix="eps", show=True, dpi=600, overwrite=True)
# %%
def crossing_pt(cond, rank):
    cross = np.where(cond/cond.max() * rank.max() - rank > 0)[0]
    if cross.size > 0:
        return np.min(cross)
    else:
        return np.nan
# %%
cross = np.zeros(4)
for i,(c,r) in enumerate(zip(cond.T,rank.T)):
    cross[i] = crossing_pt(c,r)
valid_elems = np.logical_not(np.isnan(cross))
idxd = np.where(valid_elems)[0]
idxl = [int(_c) for _c in cross[valid_elems]]
plt.plot(np.asarray(_ds)[idxd], _ls[idxl])
################################################################################
#%%
n = 200
d = 2
l = 0.1
sigma = 1.0
nv = 1e-3

x = np.random.randn(n,d)/np.sqrt(d)
K = k_se(x, ls=l,sigma=sigma)
Ke = K + np.eye(n) * nv

rootK = msqrt(Ke)
u = np.random.randn(n)

y = rootK @ u
# %%
def conj_grad(A,x0,b,u=None, loss=None, fprime=None):
    n = x0.shape[0]
    r0 = b - A @ x0
    p = np.copy(r0)
    r = np.column_stack((r0,r0))
    d = p.T @ A @ p
    x = np.copy(x0)
    y = np.copy(x)
    k = 1
    max_iter = np.min((n,1000))
    eps = 1e-20

    P = p
    D = [d]

    if loss is None:
        loss = lambda x: 0.5 * (A @ x - b).T @ (A @ x - b)
    if fprime is None:
        fprime = lambda x: A.T @ (b - A @ x)

    while linalg.norm(r[:,-1]) > eps and k < max_iter:
        g = r[:,0].T @ r[:,0] / d
        x = x + g * p
        if u is not None:
            z = u[k-1]
        else:
            z = np.random.randn(1)
        y = y + z/np.sqrt(d) * A @ p
        # r[:,-1] = -fprime(x)
        r[:,-1] = r[:,0] - g * A @ p
        beta = -(r[:,-1].T @ r[:,-1]) / (r[:,0].T @ r[:,0])
        p = r[:,-1] - beta * p
        d = p.T @ A @ p
        r[:,0] = r[:,-1]

        P = np.column_stack((P,p))
        D += [d]

        k += 1
    print(f"Terminated after {k} iterations.")
    return x, y, P, D
        
# %% check usual CG works
tmp, smple, _, _ = conj_grad(rootK, np.random.randn(n), y)
linalg.norm(tmp - u)

# out = optimize.fmin_cg(lambda x: 0.5*np.inner(rootK @ x - y, rootK @ x - y),
# np.random.randn(n), fprime=lambda x: rootK.T @ (rootK @ x - y))
# z = inv(rootK) @ y
# %% check sampling
x0 = np.zeros(n)
b = u/2#np.sign(np.random.uniform(-1,1,n))

_, ycg, P, D = conj_grad(Ke, x0, b, u/2)
print(linalg.norm(y-ycg))
 
ycg_test = Ke @ P @ np.sqrt(np.asarray(D))**(-1) * u/2
print(linalg.norm(y-ycg_test))
# %% RFF benchmark
Krff = estimate_rff_kernel(x, int(n**2*np.log(n)), l, sigma)
Krffe = Krff + nv * np.eye(n)
yrff = msqrt(Krffe) @ u
rff_err = linalg.norm(yrff - y)
print(rff_err)
 
# %% chol benchmark
chol_err = linalg.norm(linalg.cholesky(Ke, lower=True).T @ u - y)
print(chol_err)
# %% random var-matched benchmark
linalg.norm(u*np.sqrt(y.var()) - y)