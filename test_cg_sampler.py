#%%
from scipy import linalg
import gpytorch
import torch
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpytools.maths import estimate_rff_kernel, estimate_rff_kernel_inv, f_gp_approx, low_rank_inv_approx, msqrt,invmsqrt, k_se, k_mat
import gpybench.plotting as gplt
import gpsampler
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.sparse import diags
from contextlib import ExitStack
#%%
n = 500
d = 3
l = 1.0
sigma = 1.0
nv = 1e-1

x = np.random.randn(n,d)/np.sqrt(d)
K = k_mat(x, x, ls=l,sigma=sigma,nu=0.5)
K = k_se(x,x,sigma=sigma,ls=l)
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

# %% check usual CG works w rootK @ u - y
uhat, smple, P0, D0 = conj_grad(rootK, np.random.randn(n), y)
print(f"CG error: {linalg.norm(uhat - u)}")
linalg.norm(rootK - P0 @ np.diag(D0) @ P0.T)

#%%
out = optimize.fmin_cg(lambda x: 0.5*np.inner(rootK @ x - y, rootK @ x - y),
np.random.randn(n), fprime=lambda x: rootK.T @ (rootK @ x - y), retall=True)
# %% check sampling
x0 = np.zeros(n)
b = np.random.standard_normal(n)#np.sign(np.random.uniform(-1,1,n))

# Ke @ x - b
_, ycg, P, D = conj_grad(Ke, x0, b, u)
print(f"CG sample error (1): {linalg.norm(y-ycg)}")
 
ycg_test = P @ np.sqrt(np.asarray(D)) * u
print(f"CG sample error (2): {linalg.norm(y-ycg_test)}")
# %% RFF benchmark
# interestingly still does ok even with Matern 0.5 when only designed for SE
Krff = estimate_rff_kernel(x, int(n*np.log(n)), l, sigma)
Krffe = Krff + nv * np.eye(n)
yrff = msqrt(Krffe) @ u
rff_err = linalg.norm(yrff - y)
print(f"RFF sample error: {rff_err}")
 
 #%% CIQ benchmark - check details, should be beating RFF?? maybe just because
 #SE kernel?
J = int(n*np.log(n))
Q = int(np.log(n))
with ExitStack() as stack:
    checkpoint_size = stack.enter_context(
        gpytorch.beta_features.checkpoint_kernel(1500))
    max_preconditioner_size = stack.enter_context(
        gpytorch.settings.max_preconditioner_size(10000))
    min_preconditioning_size = stack.enter_context(
        gpytorch.settings.min_preconditioning_size(10))
    minres_tol = stack.enter_context(gpytorch.settings.minres_tolerance(1e-10))
    solves, weights, _, _ = gpsampler.samplers.contour_integral_quad(
                gpytorch.lazify(torch.as_tensor(K)),
                torch.as_tensor(u[:,np.newaxis]),
                max_lanczos_iter=J, num_contour_quadrature=Q)
yciq = (solves * weights).sum(0).squeeze()
ciq_err = linalg.norm(yciq - y)
print(f"CIQ sample error: {ciq_err}")
# %% chol benchmark - note: not that good for sampling...good at reproducing Ke
L = linalg.cholesky(Ke, lower=True)
chol_err = linalg.norm(L @ u - y)
print(f"Chol sample error: {chol_err}")
#%% nystrom benchmark
m = int(np.sqrt(n))
uind = np.random.choice(n,m)
Knyst = Ke[:,uind] @ invmsqrt(Ke[np.ix_(uind,uind)]) @ np.eye(m,n)
nyst_err = linalg.norm(Knyst @ u - y)
print(f"Nystrom sample error: {nyst_err}")
# %% random var-matched benchmark
print(f"Random sample error: {linalg.norm(u*np.sqrt(y.var()) - y)}")
# %%
def cg_sampler(Q, m, z):
    # z = np.random.randn(m)
    x0 = np.zeros(n)
    r0 = z - Q @ x0

    beta = np.zeros(m)
    alpha = np.zeros(m)

    V = np.zeros((n,m))

    beta[0] = linalg.norm(r0)
    V[:,0] = r0/beta[0]

    e1 = np.concatenate(([1],np.zeros(m-1, dtype=int)))

    for j in range(0, m-1):
        wj = Q @ V[:,j] - beta[j] * V[:,j-1]
        alpha[j] = wj.T @ V[:,j]
        beta[j+1] = linalg.norm(wj)
        V[:,j+1] = wj/beta[j+1]

    offset = [-1,0,1]
    # not sure if we should exclude the first or last beta...
    T = diags([beta[:-1],alpha,beta[:-1]],offset).toarray()
    # wT, vT = linalg.eigh_tridiagonal(alpha,beta[:-1])
    # Tinvsqrt = vT @ np.diag(wT**(-0.5)) @ vT.T
    # np.testing.assert_allclose(Tinvsqrt - invmsqrt(T), np.zeros((n,n)))
    return x0 + beta[0] * V @ invmsqrt(T) @ e1
#%%
# ycg2 = cg_sampler(Ke, int(np.sqrt(n)), u)
ycg2 = cg_sampler(Ke, n, u)
print(f"CG sample error (3): {linalg.norm(y-ycg2)}")
# %% testing theory that m >= O(sqrt(n)) might be sufficient
from scipy.optimize import fmin
_eta=1
_nv=0.001
_n=10000
eps=1e-6
Am = lambda n,nv,eta: (4*np.sqrt(n)*eta + 4*eta**2*nv/np.sqrt(n))
Bm = lambda n,nv,eta: (2*eta**(5/2)*nv**(3/2)/n - 2*eta**(3/2)*np.sqrt(nv) - 4*np.sqrt(n)*eta - 4*eta**2*nv/np.sqrt(n) - 4*np.sqrt(eta)*n/np.sqrt(nv))
Cm = lambda n,nv,eta,eps: (2*n**(3/2)/nv + 2*eta*np.sqrt(n) - eps)
def m_quadratic(m, n, nv, eta, eps):
    return Am(n,nv,eta) * m**2 + Bm(n,nv,eta) * m + Cm(n,nv,eta,eps)

solns = np.zeros(9)
ns = np.zeros(9)
for i in range(9):
    ns[i] = 10**i
    moptim = lambda m: m_quadratic(m, 10**i,_nv,_eta,eps)
    solns[i] = fmin(moptim, np.sqrt(10**i))
# %%
plt.loglog(ns,solns)
plt.loglog(ns, 1/(2*np.sqrt(_nv)) * np.sqrt(ns))
# %% wtf it gets worse as m increases?
cg_err = np.zeros(100-1)
ms = np.linspace(2,n,99,dtype=int)
for i in range(99):
    cg_err[i] = linalg.norm(y - cg_sampler(Ke, ms[i], u))

plt.plot(np.log(ms)/np.log(n), cg_err)
with gplt.LaTeX():
    plt.xlabel("$n^x$")
    plt.ylabel("$||y_{CG}-y||$")
# %% test cheating low-rank approx
# lam, U = np.linalg.eigh(Ke)
# y_ = np.zeros(n)
# for i in range(5):
#     lami = np.zeros(n)
#     lami[(100*i):(100*(i+1))] = lam[(100*i):(100*(i+1))] ** (-0.5)
#     y_ += Ke @ (U @ np.diag(lami) @ U.T) @ u

# #%% try semi-non-cheating... terrible!
# y_ = np.zeros(n)
# Kres = Ke
# for i in range(5):
#     lami = np.zeros(n)
#     if i == 0:
#         lami, Ui = partial_svd(Kres, 100)
#     else:
#         lami, _ = partial_svd(Kres, 100)
#     Li = Ui.T @ np.diag(np.sqrt(np.sqrt(lami)))
#     y_ += Ke @ np.real(low_rank_inv_approx(Li)) @ u
#     Kres -= Li @ Li.T @ Li @ Li.T



# %% divide by norm(y)?
print(f"CG sample error (1): {linalg.norm(y-ycg)/linalg.norm(y)}")
print(f"CG sample error (2): {linalg.norm(y-ycg_test)/linalg.norm(y)}")
print(f"CG sample error (3): {linalg.norm(y-ycg2)/linalg.norm(y)}")
print(f"Chol sample error: {chol_err/linalg.norm(y)}")
print(f"RFF sample error: {rff_err/linalg.norm(y)}")
print(f"CIQ sample error: {ciq_err/linalg.norm(y)}")
print(f"Random sample error: {linalg.norm(u*np.sqrt(y.var()) - y)/linalg.norm(y)}")
# %%
