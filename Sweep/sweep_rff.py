import csv
from itertools import product
import numpy as np
from scipy import linalg, stats
from functools import partial
from joblib import Parallel, delayed

import rff

rng = np.random.default_rng()

#no. of fourier features, can depend on other params
Ds = lambda d, l, sigma, noise_var, N : [2**i for i in range(15)] 
min_l = 1e-3
max_l = 2.0
size_l = 10
default_param_set = {"ds" :[2], #input (x) dimensionality
"ls" : np.linspace(min_l, max_l, size_l), #length scale
"sigmas" : [1.0], #kernel scale
"noise_vars" : [1e-3], #noise_variance
"Ns" : [2**i for i in range(7,13)], #no. of data points
}

generate_param_list = lambda d, l, sigma, noise_var, Ns: [[d], [l], [sigma], [noise_var], Ns]

# def generate_param_set(NO_RUNS):
#     for _ in NO_RUNS:
#         d = get_d()
#         l = get_l()
#         sigma = get_sigma()
#         noise_var = get_noise_var()
#         Ns = get_Ns()
#         yield generate_param_list(d, l, sigma, noise_var, Ns)

param_sets = {0: default_param_set.values(), 1: [], 2: []}

def sweep_fun(tup, writer, fieldnames, NO_TRIALS, verbose, benchmark, significance_threshold):
    d, l, sigma, noise_var, N = tup
    noise_sd = np.sqrt(noise_var)

    cov_omega = np.eye(d)/l**2
    my_k_true = partial(rff.k_true, sigma, l)

    x = rng.standard_normal(size = (N,d))
    theory_cov = np.array([[my_k_true(xp,xq) for xp in x] for xq in x])
    theory_cov_noise = theory_cov + noise_var*np.eye(N)
    L = linalg.cholesky(theory_cov_noise, lower = True)

    errors = []
    if verbose:
        print("***d = %d, l = %.2f, sigma = %.2f, noise_var = %.2f, N = %d***" % tup)
    for D in Ds(*tup):
        avg_approx_cov = np.zeros_like(theory_cov)
        reject = 0
        for j in range(NO_TRIALS):
            omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)

            if benchmark:
                y = rng.multivariate_normal(0, theory_cov, N)
            else:
                w = rng.standard_normal(D)
                my_f = partial(rff.f, omega, D, w)
                y = np.array([my_f(xx) for xx in x])*np.sqrt(sigma)
            noise = rng.normal(scale=noise_sd, size=N)
            y_noise = y + noise

            spherical_y = linalg.solve_triangular(L, y_noise, lower = True)
            res = stats.cramervonmises(spherical_y, 'norm', args=(0,1))
            statistic = res.statistic
            pvalue = res.pvalue
            # pvalue unreliable (see doc) if estimating params
            reject += int(pvalue < significance_threshold)

            my_z = partial(rff.z, omega, D)
            Z = np.array([my_z(xx) for xx in x])*np.sqrt(sigma)
            avg_approx_cov += np.inner(Z, Z)

        # record variance as well as mean?
        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        err = np.linalg.norm(theory_cov - avg_approx_cov)
        errors.append(err)

        if verbose:
            print("D = %d" % D)
            print("Norm difference between average approximate and exact K: %.6f" % err)
            print("%.2f%% rejected" % (reject*100))
        
        writer.writerow(dict(zip(fieldnames, tup + (D, err, reject))))

def run_sweep(ds, ls, sigmas, noise_vars, Ns, verbose=True, NO_TRIALS=1, significance_threshold=0.1, param_index=0, benchmark=False, ncpus=2):
    if benchmark:
        filename = f"output_sweep_{param_index}_bench.csv"
    else:
        filename = f"output_sweep_{param_index}.csv"

    with open(filename, 'w', newline ='') as csvfile:
        fieldnames = ["d", "l", "sigma", "noise_var", "N", "D", "err", "reject"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        if ncpus > 1:
            Parallel(n_jobs=ncpus, require="sharedmem")(
                delayed(sweep_fun)(tup, writer, fieldnames, NO_TRIALS, verbose, benchmark, significance_threshold)
                for tup in product(ds, ls, sigmas, noise_vars, Ns)
            )
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(tup, writer, fieldnames, NO_TRIALS, verbose, benchmark, significance_threshold)

if __name__ == "__main__":
    run_sweep(**default_param_set)