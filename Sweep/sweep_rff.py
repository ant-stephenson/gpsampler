from itertools import product
from functools import partial
import numpy as np
from typing import Tuple, TextIO
from scipy import linalg, stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from gpybench.utils import check_exists
import pathlib

import rff

rng = np.random.default_rng()

# no. of fourier features, can depend on other params


def Ds(d, l, sigma, noise_var, N):
    """ creates array of #rff to use for different experiments, based on the
    input size N. Maxes out at N^2

    Args:
        d (_type_): _description_
        l (_type_): _description_
        sigma (_type_): _description_
        noise_var (_type_): _description_
        N (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_D = int(np.log2(N**2))
    _Ds = [2**i for i in range(max_D)]
    return _Ds


def Js(d, l, sigma, noise_var, N):
    """ creates array of #lanczsos iter to use for different experiments based
    on the input size N. Maxes out at N.

    Args:
        d (_type_): _description_
        l (_type_): _description_
        sigma (_type_): _description_
        noise_var (_type_): _description_
        N (_type_): _description_

    Returns:
        _type_: _description_
    """
    # leave Q as default for now
    max_J = int(np.log2(N))
    _Js = [2**i for i in range(max_J)]
    return _Js


min_l = 1e-3
max_l = 2.0
size_l = 2
default_param_set = {"ds": [2],  # input (x) dimensionality
                     "ls": np.linspace(min_l, max_l, size_l),  # length scale
                     "sigmas": [1.0],  # kernel scale
                     "noise_vars": [1e-3],  # noise_variance
                      "Ns": [2**i for i in range(7, 13)],  # no. of data points
                     }
param_set_1 = {"ds": [2],  # input (x) dimensionality
               "ls": np.linspace(min_l, max_l, 1),  # length scale
               "sigmas": [1.0],  # kernel scale
               "noise_vars": [1e-3],  # noise_variance
               "Ns": [int(100e3)],  # no. of data points
                     }


def generate_param_list(d, l, sigma, noise_var, Ns): return [
    [d], [l], [sigma], [noise_var], Ns]

# def generate_param_set(NO_RUNS):
#     for _ in NO_RUNS:
#         d = get_d()
#         l = get_l()
#         sigma = get_sigma()
#         noise_var = get_noise_var()
#         Ns = get_Ns()
#         yield generate_param_list(d, l, sigma, noise_var, Ns)


param_sets = {0: default_param_set.values(), 1: param_set_1.values(), 2: []}


def sweep_fun(
        tup: Tuple, method: str, csvfile: TextIO, NO_TRIALS: int, verbose: bool,
        benchmark: bool, significance_threshold: float) -> None:
    """ Run experiment over a tuple of parameters NO_TRIALS times, writing to a
    csvfile. Supports RFF and CIQ methods.

    Args:
        tup (Tuple): (d, l, sigma, noise_var, N)
        method (str): "rff" or "ciq"
        csvfile (TextIO): path to an open csvfile to write to
        NO_TRIALS (int): #repeat experiments
        verbose (bool): Print to console option
        benchmark (bool): deprecated
        significance_threshold (float): alpha

    Raises:
        ValueError: If method other than "rff" or "ciq" used
    """
    d, l, sigma, noise_var, N = tup

    x = rng.standard_normal(size=(N, d)) / np.sqrt(d)

    if method == "rff":
        _Ds = Ds
        sampling_function = rff.sample_rff_from_x
    elif method == "ciq":
        _Ds = Js
        sampling_function = partial(
    else:
        raise ValueError("Options supported are `rff` or `ciq`")

    errors=[]
    if verbose:
        print(
            "***d = %d, l = %.2f, sigma = %.2f, noise_var = %.2f, N = %d***" %
            tup, flush=True)
    for D in _Ds(*tup):
        avg_approx_cov=theory_cov_noise * 0
        reject=0.0
        for j in range(NO_TRIALS):
            if benchmark:
                y_noise=rng.multivariate_normal(0, theory_cov_noise, N)
                approx_cov=theory_cov_noise
            else:
                y_noise, approx_cov=sampling_function(
                    x, sigma, noise_var, l, rng, D)

            spherical_y=linalg.solve_triangular(L, y_noise, lower=True)
            res=stats.cramervonmises(spherical_y, 'norm', args=(0, 1))
            statistic=res.statistic
            pvalue=res.pvalue
            # pvalue unreliable (see doc) if estimating params
            reject += int(pvalue < significance_threshold)

            avg_approx_cov += approx_cov

        # record variance as well as mean?
        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        err=linalg.norm(theory_cov - avg_approx_cov)
        errors.append(err)

        if verbose:
            print("D = %d" % D, flush=True)
            print("Norm difference between average approximate and exact K: %.6f" %
                  err, flush=True)
            print("%.2f%% rejected" % (reject*100), flush=True)

        row_str=str(tup + (D, err, reject))[1:-1]
        print(row_str, file=csvfile, flush=True)


def run_sweep(
        ds, ls, sigmas, noise_vars, Ns, verbose=True, NO_TRIALS=1,
        significance_threshold=0.1, param_index=0, benchmark=False, ncpus=2,
        method="rff", job_id=0) -> None:
    """ Runs experiments over all sets of parameters. Runs in parallel if
    specified. Calls sweep_fun() for each parameter set.

    Args:
        ds (_type_): Array of dimensions to test over
        ls (_type_): Array of lengthscales to test over
        sigmas (_type_): Array of kernelscales to test over
        noise_vars (_type_): Array of noise variances to test over
        Ns (_type_): Array of sample sizes to test over
        verbose (bool, optional): Print to console?. Defaults to True.
        NO_TRIALS (int, optional): #Repeats. Defaults to 1.
        significance_threshold (float, optional): alpha. Defaults to 0.1.
        param_index (int, optional): Experiment label - currently not used effectively. Defaults to 0.
        benchmark (bool, optional): deprecated. Defaults to False.
        ncpus (int, optional): Number of CPUs to use. Defaults to 2.
        method (str, optional): "rff" or "ciq". Defaults to "rff".
    """
    if benchmark:
        filename=f"output_sweep_{method}_{param_index}_{job_id}_bench.csv"
    else:
        filename=f"output_sweep_{method}_{param_index}_{job_id}.csv"

    filename=check_exists(pathlib.Path(".").joinpath(filename), ".csv")[0]

    with open(filename, 'w', newline='') as csvfile:
        fieldnames=["d", "l", "sigma", "noise_var", "N", "D", "err", "reject"]
        print(",".join(fieldnames), file=csvfile, flush=True)
        if ncpus > 1:
            Parallel(n_jobs=ncpus, require="sharedmem")(
                delayed(sweep_fun)(tup, method, csvfile, NO_TRIALS,
                                   verbose, benchmark, significance_threshold)
                for tup in product(ds, ls, sigmas, noise_vars, Ns)
            )
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(tup, method, csvfile, NO_TRIALS, verbose,
                          benchmark, significance_threshold)


if __name__ == "__main__":
    run_sweep(**default_param_set, method="ciq")
