#%% 
import numpy as np
from scipy.special import gamma
from numpy.polynomial.hermite import hermval, hermmul, hermgrid2d, hermval2d
from sklearn.metrics import pairwise_distances
# define constants
n = 100
d = 1
ls = 1.1
b = 1
nv = 1e-4

# use eigenfunction expansion of gaussian kernel to approximately sample at
# low-rank

eig_n = lambda ls, n: (2*ls**2)**(1-n) / (0.5*(1+np.sqrt(1+4/(2*ls**2))) +
1/(2*ls**2))**(n-0.5)

# pick minimum eigenvalue to truncate eigenbasis expansion (should be related to
# error requirements)
min_eig = 1e-10
def get_N(delta, ls):
    logA = np.log((0.5*(1+np.sqrt(1+4/(2*ls**2))) +
1/(2*ls**2)))
    minN = (0.5*logA + np.log(2*ls**2) - np.log(delta)) / (np.log(2*ls**2) + logA)
    N = int(np.ceil(minN))
    return N

N = get_N(min_eig, ls)

#%% - truncated eigenbasis w sqrt(eig) weighting
prefactor = lambda ls, n: np.sqrt((1+4/(2*ls**2))**(1/4)/2**(n-1)/gamma(n))
exp = lambda x, ls: np.exp(-2*x**2/(2*ls**2)/(1+np.sqrt(1+4/(2*ls**2))))

def coef(ls, n):
    c = np.sqrt(eig_n(ls, n)) * prefactor(ls,n)
    return c

def eig_f(x: float, ls: float, N: int):
    c = coef(ls, np.asarray(range(1,N+1)))
    x_ = (1 + 4/(2*ls**2))**(1/4) * x
    return hermval(x_, c) * exp(x, ls)

def k_eig1(x, ls, N):
    c = coef(ls, np.asarray(range(1,N+1)))
    c = np.outer(c,c)
    x_ = (1 + 4/(2*ls**2))**(1/4) * x
    K = hermgrid2d(x_, x_, c).reshape(n,n)
    K = np.outer(exp(x, ls), exp(x, ls)) * K
    return K
    
def k_eig2(x, ls, N):
    x_ = (1 + 4/(2*ls**2))**(1/4) * x
    k = eig_f(x_, ls, N)
    K = np.outer(k, k)
    return K

def k_eig3(x, ls, N):
    n = x.shape[0]
    c = coef(ls, np.asarray(range(1,N+1)))
    x_ = (1 + 4/(2*ls**2))**(1/4) * x
    Z = np.empty((n,N))
    for i in range(N):
        _c = np.zeros(i+1)
        _c[-1] = c[i]
        Z[:,i] = hermval(x_, _c).squeeze()
    return Z @ Z.T
    
#%% check against real kernel
# diagonal terms != 1 for k_. why? how do I fix this?
x = np.random.randn(n,d)
k = np.exp(-pairwise_distances(x)**2/(2*ls**2))

k1 = k_eig1(x, ls, N)
k2 = k_eig2(x, ls, N)
k3 = k_eig3(x, ls, N)

print(np.linalg.norm(k-k1))
print(np.linalg.norm(k-k2))
print(np.linalg.norm(k-k3))
# %%
