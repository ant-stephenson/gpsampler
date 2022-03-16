from itertools import product
import numpy as np
from typing import Tuple, TextIO
from scipy import linalg, stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from gpybench.utils import check_exists
import pathlib

import rff.Sweep.rff as rff

rng = np.random.default_rng()

#no. of fourier features, can depend on other params
def Ds(d, l, sigma, noise_var, N):
    max_D = int(np.log2(N**2))
    _Ds = [2**i for i in range(max_D)]
    return _Ds
def Js(d, l, sigma, noise_var, N):
    # leave Q as default for now
    max_J = int(np.log2(N))
    _Js = [2**i for i in range(max_J)]
    return _Js

min_l = 1e-3
max_l = 2.0
size_l = 2
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

def sweep_fun(tup: Tuple, method: str, csvfile: TextIO, NO_TRIALS: int, verbose: bool, benchmark: bool, significance_threshold: float):
    d, l, sigma, noise_var, N = tup

    x = rng.standard_normal(size = (N,d))
    theory_cov = sigma * np.exp(-pairwise_distances(x)**2/(2*l**2))
    theory_cov_noise = theory_cov + noise_var*np.eye(N)
    L = linalg.cholesky(theory_cov_noise, lower = True)

    if method == "rff":
        _Ds = Ds
        sampling_function = rff.sample_rff_from_x
    elif method == "ciq":
        _Ds = Js
        sampling_function = rff.sample_ciq_from_x
    else:
        raise ValueError("Options supported are `rff` or `ciq`")

    errors = []
    if verbose:
        print("***d = %d, l = %.2f, sigma = %.2f, noise_var = %.2f, N = %d***" % tup)
    for D in _Ds(*tup):
        avg_approx_cov = np.zeros_like(theory_cov)
        reject = 0.0
        for j in range(NO_TRIALS):
            if benchmark:
                y_noise = rng.multivariate_normal(0, theory_cov, N)
                approx_cov = theory_cov_noise
            else:
                y_noise, approx_cov = sampling_function(x, sigma,noise_var, l, rng, D)

            spherical_y = linalg.solve_triangular(L, y_noise, lower = True)
            res = stats.cramervonmises(spherical_y, 'norm', args=(0,1))
            statistic = res.statistic
            pvalue = res.pvalue
            # pvalue unreliable (see doc) if estimating params
            reject += int(pvalue < significance_threshold)

            avg_approx_cov += approx_cov

        # record variance as well as mean?
        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        err = np.linalg.norm(theory_cov - avg_approx_cov)
        errors.append(err)

        if verbose:
            print("D = %d" % D)
            print("Norm difference between average approximate and exact K: %.6f" % err)
            print("%.2f%% rejected" % (reject*100))
        
        row_str = str(tup + (D, err, reject))[1:-1]
        print(row_str, file=csvfile, flush = True)

def run_sweep(ds, ls, sigmas, noise_vars, Ns, verbose=True, NO_TRIALS=1, significance_threshold=0.1, param_index=0, benchmark=False, ncpus=2, method="rff"):
    if benchmark:
        filename = f"output_sweep_{method}_{param_index}_bench.csv"
    else:
        filename = f"output_sweep_{method}_{param_index}.csv"
        
    filename = check_exists(pathlib.Path(".").joinpath(filename), ".csv")[0]

    with open(filename, 'w', newline ='') as csvfile:
        fieldnames = ["d", "l", "sigma", "noise_var", "N", "D", "err", "reject"]
        print(",".join(fieldnames), file=csvfile, flush=True)
        if ncpus > 1:
            Parallel(n_jobs=ncpus, require="sharedmem")(
                delayed(sweep_fun)(tup, method, csvfile, NO_TRIALS, verbose, benchmark, significance_threshold)
                for tup in product(ds, ls, sigmas, noise_vars, Ns)
            )
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(tup, csvfile, NO_TRIALS, verbose, benchmark, significance_threshold)

if __name__ == "__main__":
    run_sweep(**default_param_set, method="ciq")