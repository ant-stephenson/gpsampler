import numpy as np
from gpsampler.samplers import sample_se_rff_from_x, zrf, f_rf, sample_rff_from_x, _sample_se_rff_from_x
from gpytools.maths import k_mat, msqrt
import matplotlib.pyplot as plt

n,d = 200,2
sigma = 1.0
l=0.1
nu=1.5
noise_var = 0.01
rng = np.random.default_rng()

x = np.random.randn(n,d)/np.sqrt(d)
Kmat = k_mat(x,x, sigma,l,2.5)
u = rng.standard_normal(n)
y_true = msqrt(Kmat) @ u

C = int(n**2*np.log(n))

def test_sample(x, sigma, noise_var, l, rng, D, G, nu):
    n,d = x.shape
    ym = sample_rff_from_x(x,sigma, noise_var, l, rng, D, "matern", G=G, nu=2.5)
    y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
    err = np.linalg.norm(y_true-y_approx)
    return err

err_sqrt = test_sample(x, sigma, noise_var, l, rng, int(np.sqrt(C)), C // int(np.sqrt(C)), nu)

Ds = [int(round(I/2)*2) for I in np.logspace(1,np.log2(C)*2/3,15,base=2, endpoint=False)]
reps = 5
err = np.zeros((len(Ds),reps))
for i,D in enumerate(Ds):
    G = C // D
    for j in range(reps):
        err[i,j] = test_sample(x, sigma, noise_var, l, rng, D, G, nu)

plt.errorbar(Ds, err.mean(axis=1), 2*np.sqrt(err.var(axis=1)), label="_nolegend_")
plt.axhline(err_sqrt, color="r", linestyle="--", label="err(G=C)")
plt.axvline(np.sqrt(C), color="k", label="G=C")
plt.xlabel("$N_{RFF}$")
plt.ylabel("$||y-\hat{y}||$")
plt.title(rf"Error as ratio of samples swaps from RFF to G, $\nu=${nu}")
plt.legend()