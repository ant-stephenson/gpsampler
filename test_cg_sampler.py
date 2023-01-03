#%%
from scipy import linalg
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpytools.maths import estimate_rff_kernel, estimate_rff_kernel_inv, f_gp_approx, msqrt,invmsqrt, k_se
import gpybench.plotting as gplt
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.sparse import diags
#%%
n = 200
d = 2
l = 0.1
sigma = 1.0
nv = 1e-3

x = np.random.randn(n,d)/np.sqrt(d)
K = k_se(x, x, ls=l,sigma=sigma)
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
_, ycg, P, D = conj_grad(Ke, x0, b, u/2)
print(f"CG sample error (1): {linalg.norm(y-ycg)}")
 
ycg_test = P @ np.sqrt(np.asarray(D)) * u/2
print(f"CG sample error (2): {linalg.norm(y-ycg_test)}")
# %% RFF benchmark
Krff = estimate_rff_kernel(x, int(n**2*np.log(n)), l, sigma)
Krffe = Krff + nv * np.eye(n)
yrff = msqrt(Krffe) @ u
rff_err = linalg.norm(yrff - y)
print(f"RFF sample error: {rff_err}")
 
# %% chol benchmark - note: not that good...
L = linalg.cholesky(Ke, lower=True)
chol_err = linalg.norm(L @ u - y)
print(f"Chol sample error: {chol_err}")
# %% random var-matched benchmark
print(f"Random sample error: {linalg.norm(u*np.sqrt(y.var()) - y)}")
# %%
def cg_sampler(Q, m, z):
    # z = np.random.randn(m)
    x0 = np.zeros(m)
    r0 = z - Q @ x0

    beta = np.zeros(m)
    alpha = np.zeros(m)

    V = np.zeros((n,m))

    beta[0] = linalg.norm(r0)
    V[:,1] = r0/beta[0]

    e1 = np.concatenate(([1],np.zeros(m-1, dtype=int)))

    for j in range(1, m-1):
        wj = Q @ V[:,j] - beta[j] * V[:,j-1]
        alpha[j] = wj.T @ V[:,j]
        beta[j+1] = linalg.norm(wj)
        V[:,j+1] = wj/beta[j+1]

    offset = [-1,0,1]
    # not sure if we should exclude the first or last beta...
    T = diags([beta[:-1],alpha,beta[:-1]],offset).toarray()
    return x0 + beta[0] * V @ invmsqrt(T) @ e1
#%%
ycg2 = cg_sampler(Ke, n, u)
print(f"CG sample error (3): {linalg.norm(y-Ke @ ycg2)}")
# %%
