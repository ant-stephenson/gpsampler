from itertools import product
from functools import partial
import numpy as np
from typing import Tuple, TextIO, Iterable
from scipy import linalg, stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from gpybench.utils import check_exists
import pathlib

import gpsampler

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
    max_D = int(np.log2(N**2)) + 1
    _Ds = [2**i for i in range(16, max_D)]
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
    max_J = int(np.log2(np.sqrt(N / noise_var) * np.log(N))) + 1
    _Js = [2**i for i in range(4, max_J)]
    return _Js


min_l = 1e-2
max_l = 1.0

default_param_set = {"ds": [2],  # input (x) dimensionality
                     # np.linspace(min_l, max_l, size_l),  # length scale
                     "ls": [1e-3, 2],
                     "sigmas": [1.0],  # kernel scale
                     "noise_vars": [1e-3],  # noise_variance
                     "Ns": [2**i for i in range(8, 13)],  # no. of data points
                     }
problem_param_set = {"ds": [2],  # input (x) dimensionality
                     # np.linspace(min_l, max_l, size_l),  # length scale
                     "ls": [0.1, 1, 2],
                     "sigmas": [1.0],  # kernel scale
                     "noise_vars": [1e-3],  # noise_variance
                     "Ns": [2**i for i in range(8, 13)],  # no. of data points
                     }
paper_param_set = {"ds": [10],  # input (x) dimensionality
                   # np.linspace(min_l, max_l, size_l),  # length scale
                   "ls": [1e-1, 0.5, 1, 2],
                   "sigmas": [1.0],  # kernel scale
                   "noise_vars": [1e-2],  # noise_variance
                   "Ns": [2**i for i in range(8, 13)],  # no. of data points
                   }


param_sets = {
    0: default_param_set.values(),
    1: problem_param_set.values(),
    2: paper_param_set.values()}


def sweep_fun(
        tup: Tuple, method: str, csvfile: TextIO, NO_TRIALS: int, verbose: bool,
        benchmark: bool, significance_threshold: float, with_pre: bool) -> None:
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
    if with_pre:
        max_preconditioner_size = int(np.sqrt(N))
    else:
        max_preconditioner_size = 0
    # max_preconditioner_size = 0

    x = rng.standard_normal(size=(N, d)) / np.sqrt(d)
    theory_cov = sigma * np.exp(-pairwise_distances(x)**2/(2*l**2))
    theory_cov_noise = theory_cov + noise_var*np.eye(N)
    L = linalg.cholesky(theory_cov_noise, lower=True)

    if method == "rff":
        _Ds = Ds
        sampling_function = gpsampler.sample_rff_from_x
    elif method == "ciq":
        _Ds = Js
        sampling_function = partial(
            gpsampler.sample_ciq_from_x,
            Q=int(np.log(N)),
            max_preconditioner_size=max_preconditioner_size)
    elif method == "chol":
        _Ds = lambda *args: [L]
        sampling_function = gpsampler.sample_chol_from_x
    else:
        raise ValueError("Options supported are `rff` or `ciq`")

    errors = []
    if verbose:
        print(
            "***d = %d, l = %.2e, sigma = %.2e, noise_var = %.2e, N = %d***" %
            tup, flush=True)
    for D in _Ds(*tup):
        avg_approx_cov = theory_cov_noise * 0
        reject = 0.0
        for j in range(NO_TRIALS):
            if benchmark:
                y_noise = rng.multivariate_normal(np.zeros(N), theory_cov_noise)
                approx_cov = theory_cov_noise
            else:
                y_noise, approx_cov = sampling_function(
                    x, sigma, noise_var, l, rng, D)

            spherical_y = linalg.solve_triangular(L, y_noise, lower=True)
            res = stats.cramervonmises(spherical_y, 'norm', args=(0, 1))
            statistic = res.statistic
            pvalue = res.pvalue
            # pvalue unreliable (see doc) if estimating params
            reject += int(pvalue < significance_threshold)

            if np.isnan(approx_cov).any():
                approx_cov = approx_cov * avg_approx_cov
            avg_approx_cov += approx_cov

        # record variance as well as mean?
        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        if np.isnan(avg_approx_cov).any() or np.isnan(theory_cov_noise).any():
            err = np.nan
        else:
            err = linalg.norm(theory_cov_noise - avg_approx_cov)
        errors.append(err)

        if method == "chol":
            D = -999

        if verbose:
            print("D = %d" % D, flush=True)
            print(
                f"max_preconditioner_size={max_preconditioner_size}",
                flush=True)
            print("Norm difference between average approximate and exact K: %.6f" %
                  err, flush=True)
            print("%.2f%% rejected" % (reject*100), flush=True)

        row_str = str(tup + (D, err, reject))[1:-1]
        print(row_str, file=csvfile, flush=True)


def run_sweep(ds: Iterable, ls: Iterable, sigmas: Iterable,
              noise_vars: Iterable, Ns: Iterable, verbose: bool = True,
              NO_TRIALS: int = 10, significance_threshold: float = 0.1,
              param_index: int = 0, benchmark: bool = False, ncpus: int = 2,
              method: str = "ciq", job_id: int = 0, with_pre: bool = False) -> None:
    """ Runs experiments over all sets of parameters. Runs in parallel if
    specified. Calls sweep_fun() for each parameter set.

    Args:
        ds (Iterable): Array of dimensions to test over
        ls (Iterable): Array of lengthscales to test over
        sigmas (Iterable): Array of kernelscales to test over
        noise_vars (Iterable): Array of noise variances to test over
        Ns (Iterable): Array of sample sizes to test over
        verbose (bool, optional): Print to console?. Defaults to True.
        NO_TRIALS (int, optional): #Repeats. Defaults to 1.
        significance_threshold (float, optional): alpha. Defaults to 0.1.
        param_index (int, optional): Experiment label - currently not used effectively. Defaults to 0.
        benchmark (bool, optional): deprecated. Defaults to False.
        ncpus (int, optional): Number of CPUs to use. Defaults to 2.
        method (str, optional): "rff" or "ciq". Defaults to "ciq".
    """
    if __name__ == "__main__":
        filename = f"output_sweep_{method}_{param_index}_{job_id}_TEST.csv"
        overwrite = True
    else:
        if benchmark:
            filename = f"output_sweep_{method}_{param_index}_{job_id}_bench.csv"
        else:
            filename = f"output_sweep_{method}_{param_index}_{job_id}.csv"
        overwrite = False

    filepath = check_exists(
        pathlib.Path(".").joinpath(filename),
        ".csv", overwrite=overwrite)[0]

    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ["d", "l", "sigma", "noise_var", "N", "D", "err", "reject"]
        print(",".join(fieldnames), file=csvfile, flush=True)
        if ncpus > 1:
            Parallel(
                n_jobs=ncpus, require="sharedmem")(
                delayed(sweep_fun)
                (tup, method, csvfile, NO_TRIALS, verbose, benchmark,
                 significance_threshold, with_pre)
                for tup in product(ds, ls, sigmas, noise_vars, Ns))
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(tup, method, csvfile, NO_TRIALS, verbose,
                          benchmark, significance_threshold, with_pre)


if __name__ == "__main__":
    run_sweep(**default_param_set, method="chol")  # type: ignore
