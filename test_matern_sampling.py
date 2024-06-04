#%%
import numpy as np
from gpsampler.samplers import sample_se_rff_from_x, zrf, f_rf, sample_rff_from_x, _sample_se_rff_from_x, estimate_rff_kernel
from gpytools.maths import k_mat, msqrt
import matplotlib.pyplot as plt

n,d = 100,5
sigma = 1.0
l=3.0
nu=0.5
noise_var = 0.1
rng = np.random.default_rng(1)

#%%
x = np.random.randn(n,d)/np.sqrt(d)
Kmat = k_mat(x,x, sigma,l,nu)
u = rng.standard_normal(n)
y_true = msqrt(Kmat) @ u

C = int(n**2*np.log(n))
D = int(n/noise_var)
G = 100

#%%
def test_sample(x, sigma, noise_var, l, rng, D, G, nu):
    n,d = x.shape
    ym = sample_rff_from_x(x,sigma, noise_var, l, rng, D, "matern", G=G, nu=nu)
    y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
    # y_approx = ym[0
    err = np.linalg.norm(y_true-y_approx)
    return err

err_sqrt = test_sample(x, sigma, noise_var, l, rng, D, G, nu)
#%%
Ds = [int(round(I/2)*2) for I in np.logspace(1,np.log2(C)*2/3,15,base=2, endpoint=False)]
reps = 5
err = np.zeros((len(Ds),reps))
for i,D in enumerate(Ds):
    G = C // D
    for j in range(reps):
        err[i,j] = test_sample(x, sigma, noise_var, l, rng, D, G, nu)

#%%
plt.errorbar(Ds, err.mean(axis=1), 2*np.sqrt(err.var(axis=1)), label="_nolegend_")
plt.axhline(err_sqrt, color="r", linestyle="--", label="err(G=C)")
plt.axvline(np.sqrt(C), color="k", label="G=C")
plt.xlabel("$N_{RFF}$")
plt.ylabel("$||y-\hat{y}||$")
plt.title(rf"Error as ratio of samples swaps from RFF to G, $\nu=${nu}")
plt.legend()

#%%
from scipy.special import genlaguerre, roots_genlaguerre, gamma

def compute_ls_and_sigmas(n, J, nu):
    roots_j = roots_genlaguerre(J, nu-1)[0]
    w_j = gamma(J+nu)/gamma(J+1) * roots_j / ((n+1)**2 * genlaguerre(J+1, nu-1)(roots_j)**2)
    l_j = np.sqrt(roots_j/nu) * l 
    sigma_j =np.sqrt(w_j / gamma(nu))
    return l_j, sigma_j

#%%
def sample_mat_rff_from_x2(x, sigma: float, noise_var: float, l:
                          float, rng: np.random.Generator, D: int, J: int,
                          nu: float):
    n, d = x.shape
    w = rng.standard_normal((D, ))
    N = int(1e6)
    y, C = np.zeros(n,), np.nan

    l_J, sigma_J = compute_ls_and_sigmas(n, J, nu)

    for j in range(J):
        omega = rng.standard_normal((D//2, d))
        if n > N:
            ys, Cs = np.zeros(n,), np.nan
            parts = int(np.ceil(n/N))
            for p in range(parts):
                idx = np.s_[(p*N):((p+1)*N)]
                ys[idx], Cp = _sample_se_rff_from_x(
                    x[idx, :], sigma, omega[:, :]/l_J[j], w)
        else:
            ys, Cs = _sample_se_rff_from_x(x, sigma, omega[:, :]/l_J[j], w)
        y += sigma_J[j] * ys
        C += Cs

    y /= np.sqrt(J)
    C /= J
    noise = rng.normal(scale=np.sqrt(noise_var), size=n)
    y_noise = y + noise
    return y_noise, C
# %%
def test_sample2(x, sigma, noise_var, l, rng, D, J, nu):
    n,d = x.shape
    ym = sample_mat_rff_from_x2(x,sigma, noise_var, l, rng, D,J, nu)
    y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
    # y_approx = ym[0]
    err = np.linalg.norm(y_true-y_approx)
    return err

err_sqrt2 = test_sample2(x, sigma, noise_var, l, rng, D, 10, nu)
# %%
def sample_mat_rff_from_x3(x, sigma: float, noise_var: float, l:
                          float, rng: np.random.Generator, D: int, J: int,
                          nu: float):
    n, d = x.shape
    w = rng.standard_normal((D, ))
    y, C = np.zeros(n,), np.nan

    omega_y = rng.standard_normal((D//2, d)) * np.sqrt(2)/l
    omega_u = rng.chisquare(2*nu, size=(D//2,))
    omega = np.sqrt(2*nu/np.tile(omega_u,(d,1)).T) * omega_y
    y, approx_cov = _sample_se_rff_from_x(x, sigma, omega, w)
    noise = rng.normal(scale=np.sqrt(noise_var), size=(n, ))
    y_noise = y + noise
    return y_noise, approx_cov


def estimate_matern_rff_kernel(
        X, D: int, ks: float, l: float, nu: float):
    N, d = X.shape
    omega_y = rng.standard_normal((D//2, d)) * np.sqrt(2)/l
    omega_u = rng.chisquare(2*nu, size=(D//2,))
    omega = np.sqrt(2*nu/np.tile(omega_u,(d,1)).T) * omega_y
    Z = zrf(omega, D, X).T*np.sqrt(ks)
    approx_cov = np.inner(Z, Z)
    return approx_cov
#%%
def test_sample3(x, sigma, noise_var, l, rng, D, J, nu):
    n,d = x.shape
    ym = sample_mat_rff_from_x3(x,sigma, noise_var, l, rng, D,J, nu)
    y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
    # y_approx = ym[0]
    err = np.linalg.norm(y_true-y_approx)
    return err

err_sqrt3 = test_sample3(x, sigma, noise_var, l, rng, D, 10, nu)
# %%
