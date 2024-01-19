#%%
import numpy as np
from scipy.optimize import fsolve, brentq, minimize_scalar, Bounds

from gpsampler.utils import compute_rbf_eigenvalue as eig

#%% define params
n=1.1e5
delta =1e-3
J=44e3
l=0.5
eta=0.8
d=10

#%% define functions for computing eigenvalue according to GPML (4.41) p98
# a = lambda d: d/(4)
# b = lambda l: 1/(2*l**2)
# c = lambda d, l: np.sqrt(a(d)**2 + 2*a(d)*b(l))
# A = lambda d, l: a(d) + b(l) + c(d,l)
# B = lambda d,l: b(l)/A(d,l)

# eig = lambda n, d, l, k: n * 2*(a(d)/A(d,l))**(d/2)*B(d,l)**(k-1)

#%% define function to solve based on expression for J with n -> n*(2a/A)^(d/2)*B
C = 10
deltaQ = lambda nv: delta * np.sqrt(nv*(1-eta))/2
cond = lambda n, d, l, nv: eig(n,d,l,1)/(eta*nv)+1
Jfunkappa = lambda n,d,l,nv: 1 + 0.5 * np.sqrt(cond(n,d,l,nv)) * (np.log(cond(n,d,l,nv)*np.sqrt(nv*n)) + 2*np.log(np.log(cond(n,d,l,nv))) - np.log(delta*np.sqrt(nv*(1-eta))-deltaQ(nv)) + C)

Jfun = lambda nv: np.max((1,Jfunkappa(n,d,l,nv)))

func = lambda nv: J - Jfun(nv)

#%% numerically solve
min_nv = fsolve(func, 0.01)**2
print(min_nv)

################################################################################
#%% find n s.t. lambda_n ~ nv
# roughly scales as: 
# nv=10^-a, a->a+1, n->n+O(1)
# l=10^-b, b->b+1, n->O(10)*n
# d: (max at d=2 (?))
# nmineigs = [fsolve(lambda n: np.log(eig(n,d,l,n)) - np.log(nv), d*l**2) for d in np.arange(1,100)]
# plt.plot(np.arange(1,100),nmineigs)
d, l, nv = 1, 1e-3, 1e-3
n_min_eign_nv = fsolve(lambda n: np.log(eig(n,d,l,n)) - np.log(nv), d*l**2)
print(n_min_eign_nv)
#%% find n s.t. preconditioning ineffective 
l = 1e-3
bounds = Bounds(1,np.inf)
mineig = lambda logn: np.min([nv, eig(np.exp(logn),d,l,np.exp(logn))])
fun2 = lambda logn: np.abs(np.sqrt(eig(np.exp(logn),d,l,1)/mineig(logn)) - np.sqrt((1+4/mineig(logn)*np.exp(logn)**(5/4)*(2*a(d)/A(d,l))**(d/2)*B(d,l)**(np.sqrt(np.exp(logn))))))
min_out = minimize_scalar(fun2)
n_min = np.exp(min_out.x)
print(n_min)
# %%
fun3 = lambda logn: 3/4 * logn - np.sqrt(np.exp(logn)) * (1-B(d,l)) - np.log(mineig(logn)/4)
np.exp(fsolve(fun3, np.log(1000)))
# %% test preconditioning at large l
from gpytools.maths import id_inv
from gpybench.datasets import sample_rbf_kernel

d = 10
ms = np.arange(250,2000,250)
ls = np.asarray([1e-2,1e-1,1,2])
rcond = np.zeros((ms.shape[0],ls.shape[0]))
nv = 1e-3
# E = 1e-3 * np.abs(np.random.randn(m,m))
# np.fill_diagonal(E,0)
# A = np.ones((m,m)) + np.eye(m) * nv
# K = A - E
for i,m in enumerate(ms):
    for j,l in enumerate(ls):
        K = sample_rbf_kernel(n=m,d=d,ls=l)
        testeigs = np.linalg.eigvalsh(K + nv * np.eye(m))
        cond0 = testeigs.max()/testeigs.min()
        P = id_inv(K, nv, k=int(np.sqrt(m)))
        # P = incomplete_chol_kernel_inv(K, k=int(np.sqrt(m)))

        peigs = np.linalg.eigvalsh(P @ K)
        cond1 = peigs.max()/peigs.min()

        rcond[i,j] = np.abs(cond0/cond1)

#%%
import matplotlib.pyplot as plt
import gpybench.plotting as gplt
from pathlib import Path

# np.set_printoptions(formatter={'float_kind':float_formatter})
plt.loglog(ms, np.sqrt(rcond))
with gplt.LaTeX():
    float_formatter = "$l=$ {:.2g}".format
    plt.legend([float_formatter(x) for x in ls])
    plt.xlabel("n")
    plt.ylabel(r"$\sqrt{\frac{\kappa_0}{\kappa_1}}$")
    plt.title(rf"$d={d},\sigma_n^2={nv: .2g}$")
gplt.save_fig(Path(), f"cond_ratio_nl_d={d}","jpg", show=True)
# %%
