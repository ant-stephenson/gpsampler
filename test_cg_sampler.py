#%%
from scipy import linalg
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpytools.maths import estimate_rff_kernel, estimate_rff_kernel_inv, f_gp_approx, msqrt, k_se
import gpybench.plotting as gplt
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
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
print(f"CG error (1): {linalg.norm(y-ycg)}")
 
ycg_test = Ke @ P @ np.sqrt(np.asarray(D))**(-1) * u/2
print(f"CG error (2): {linalg.norm(y-ycg_test)}")
# %% RFF benchmark
Krff = estimate_rff_kernel(x, int(n**2*np.log(n)), l, sigma)
Krffe = Krff + nv * np.eye(n)
yrff = msqrt(Krffe) @ u
rff_err = linalg.norm(yrff - y)
print(f"RFF error: {rff_err}")
 
# %% chol benchmark
L = linalg.cholesky(Ke, lower=True)
chol_err = linalg.norm(L @ u - y)
print(f"Chol error: {chol_err}")
# %% random var-matched benchmark
print(f"Random error: {linalg.norm(u*np.sqrt(y.var()) - y)}")
# %%
