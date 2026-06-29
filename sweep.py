from itertools import product
from functools import partial
import numpy as np
from typing import Tuple, TextIO, Iterable
from scipy import linalg, stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from gpsampler.utils import check_exists
import pathlib

import gpsampler


# ---------------------------------------------------------------------------
# Bayesian-validation helper  (reuses the Cholesky factor already held by
# sweep_fun, so we don't pay for a second O(N³) factorisation per trial)
# ---------------------------------------------------------------------------

def _bv_tv(L_xi: np.ndarray, Khat_xi: np.ndarray) -> float:
    """Total variation between N(0, K_ξ) and N(0, K̂_ξ) via Imhof (1961).

    Parameters
    ----------
    L_xi   : lower-triangular Cholesky factor of K_ξ  (already computed by caller)
    Khat_xi: realised observation covariance K̂_ξ

    Returns
    -------
    float in [0, 1]
    """
    from gpsampler.bayes_validation import imhof_sf
    from scipy.linalg import solve_triangular as _stri, eigvalsh as _eigh
    Linv_Khat = _stri(L_xi, Khat_xi, lower=True)
    A = _stri(L_xi, Linv_Khat.T, lower=True).T
    lambdas = np.maximum(_eigh(A), 1e-300)
    a = 0.5 * (1.0 - 1.0 / lambdas)
    b = 0.5 * float(np.sum(np.log(lambdas)))
    p1, _ = imhof_sf(a, b)
    p2, _ = imhof_sf(a * lambdas, b)
    return float(np.clip(p2 - p1, 0.0, 1.0))

rng = np.random.default_rng()

# no. of fourier features, can depend on other params


def Ds(d, l, sigma, noise_var, N):
    """creates array of #rff to use for different experiments, based on the
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
    """creates array of #lanczsos iter to use for different experiments based
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

default_param_set = {
    "ds": [2, 3],  # input (x) dimensionality
    # np.linspace(min_l, max_l, size_l),  # length scale
    "ls": [0.5, 2],
    "sigmas": [1.0],  # kernel scale
    "noise_vars": [1e-2],  # noise_variance
    "Ns": [2**i for i in range(8, 12)],  # no. of data points
}
problem_param_set = {
    "ds": [2],  # input (x) dimensionality
    # np.linspace(min_l, max_l, size_l),  # length scale
    "ls": [0.1, 1, 2],
    "sigmas": [1.0],  # kernel scale
    "noise_vars": [1e-3],  # noise_variance
    "Ns": [2**i for i in range(8, 13)],  # no. of data points
}
paper_param_set = {
    "ds": [10],  # input (x) dimensionality
    # np.linspace(min_l, max_l, size_l),  # length scale
    "ls": [1e-1, 0.5, 1, 2],
    "sigmas": [1.0],  # kernel scale
    "noise_vars": [1e-2],  # noise_variance
    "Ns": [2**i for i in range(8, 13)],  # no. of data points
}


param_sets = {
    0: default_param_set.values(),
    1: problem_param_set.values(),
    2: paper_param_set.values(),
}


def sweep_fun(
    tup: Tuple,
    method: str,
    csvfile: TextIO,
    NO_TRIALS: int,
    verbose: bool,
    benchmark: bool,
    significance_threshold: float,
    with_pre: bool,
    bv: bool = False,
    bv_delta: float = 0.05,
) -> None:
    """Run experiment over a tuple of parameters NO_TRIALS times, writing to a
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
    theory_cov = sigma * np.exp(-pairwise_distances(x) ** 2 / (2 * l**2))
    theory_cov_noise = theory_cov + noise_var * np.eye(N)
    L = linalg.cholesky(theory_cov_noise, lower=True)

    # For lrff: pre-build the Nystrom sketch (K, S, B, alpha_fn) once per
    # (N, l, noise_var) — reused across every D value and every trial.
    # This avoids recomputing the O(N^2) kernel matrix 1000x per D value.
    # Also compute n_eff via Hutchinson, reusing L already formed above.
    if method == "lrff":
        from gpsampler.leverage_reweighted_rff import (
            kernel_matrix as _km,
            recursive_rls as _rrls,
            nystrom_factor as _nf,
            ApproxLeverage as _AL,
        )

        _K_unit = _km(x, kind="rbf", ell=l)
        _S = _rrls(_K_unit, lam=noise_var, rng=np.random.default_rng(99))
        _B = _nf(_K_unit, _S)
        _lrff_alpha_fn = _AL(x, _B, noise_var)

        _rng_neff = np.random.default_rng(12345)
        _neff_probes = 30
        _neff_sum = 0.0
        for _ in range(_neff_probes):
            v = _rng_neff.standard_normal(N)
            # Hutchinson: E[v^T K K_xi^{-1} v] = Tr(K K_xi^{-1})
            # theory_cov = sigma * K_unit, so divide by sigma at the end
            _neff_sum += np.dot(theory_cov @ v, linalg.cho_solve((L, True), v))
        neff = _neff_sum / (_neff_probes * sigma)
    else:
        _lrff_alpha_fn = None
        neff = np.nan

    if method == "rff":
        _Ds = Ds
        sampling_function = gpsampler.samplers.sample_rff_from_x
    elif method == "ciq":
        _Ds = Js
        sampling_function = partial(
            gpsampler.samplers.sample_ciq_from_x,
            Q=int(np.log(N)),
            max_preconditioner_size=max_preconditioner_size,
        )
    elif method == "lrff":
        _Ds = Ds
        sampling_function = partial(
            gpsampler.samplers.sample_lrff_from_x, alpha_fn=_lrff_alpha_fn
        )
    elif method == "chol":
        _Ds = lambda *args: [L]
        sampling_function = gpsampler.samplers.sample_chol_from_x
    elif method == "cg":
        _Ds = lambda *args: [
            2**i for i in range(4, int(np.log2(np.sqrt(args[-1]))) + 1)
        ]
        if with_pre and max_preconditioner_size > 0:
            _pre = gpsampler.NystromPreconditioner(
                theory_cov, eta=0.8, noise_var=noise_var,
                rank=max_preconditioner_size, rng=rng)
            sampling_function = partial(
                gpsampler.samplers.sample_lanczos_from_x,
                preconditioner=_pre)
        else:
            sampling_function = gpsampler.samplers.sample_cg_from_x
    elif method == "sparse":
        _Ds = lambda *args: [
            2**i for i in range(4, int(np.log2(np.sqrt(args[-1]))) + 1)
        ]
        sampling_function = gpsampler.samplers.sample_sparse_from_x
    else:
        raise ValueError("Options supported are `rff` or `ciq`")

    errors = []
    if verbose:
        print(
            "***d = %d, l = %.2e, sigma = %.2e, noise_var = %.2e, N = %d***"
            % tup,
            flush=True,
        )
    for D in _Ds(*tup):
        # For lrff: build the SIR pool once per D (O(n·r·P)), then each of the
        # NO_TRIALS trials just does a cheap rng.choice() resample from it.
        if method == "lrff":
            from gpsampler.leverage_reweighted_rff import compute_sir_pool

            _pool_cache = compute_sir_pool(
                D // 2,
                d,
                "rbf",
                l,
                1.5,
                _lrff_alpha_fn,
                np.random.default_rng(D + 1_000_000),
                pool_factor=5,
                pool_min=4000,
            )
            _cur_sf = partial(
                gpsampler.samplers.sample_lrff_from_x,
                alpha_fn=_lrff_alpha_fn,
                pool_cache=_pool_cache,
            )
        else:
            _cur_sf = sampling_function

        avg_approx_cov = theory_cov_noise * 0
        reject = 0.0
        tv_values: list = []
        for j in range(NO_TRIALS):
            Khat_xi = None  # set below for BV-supported methods

            if benchmark:
                y_noise = rng.multivariate_normal(np.zeros(N), theory_cov_noise)
                approx_cov = theory_cov_noise
                if bv:
                    Khat_xi = theory_cov_noise
            elif bv and method == "rff":
                # Inline RFF sampling: capture Phi to build K̂_ξ = ΦΦᵀ + σ²I
                omega = rng.multivariate_normal(np.zeros(d), np.eye(d) / l**2, D // 2)
                v = x @ omega.T
                Z = np.sqrt(2.0 / D) * np.concatenate(
                    [np.cos(v), np.sin(v)], axis=1)
                Phi = np.sqrt(sigma) * Z                     # (N, D)
                w = rng.standard_normal(D)
                y_noise = Phi @ w + rng.standard_normal(N) * np.sqrt(noise_var)
                approx_cov = np.nan
                Khat_xi = Phi @ Phi.T + noise_var * np.eye(N)
            elif bv and method == "lrff":
                # Inline lrff sampling: capture Phi for BV
                from gpsampler.leverage_reweighted_rff import (
                    reweighted_rff_sampler as _rrff,
                )
                Phi32 = np.asarray(_rrff(
                    X=x, kind="rbf", ell=l, nu=1.5, sigma2=noise_var,
                    n_freq=D // 2, rng=rng,
                    alpha_fn=_lrff_alpha_fn,
                    pool_cache=_pool_cache,
                ), dtype=np.float32) * np.float32(np.sqrt(sigma))
                z32 = rng.standard_normal(Phi32.shape[1]).astype(np.float32)
                y_noise = (Phi32 @ z32).astype(np.float64) + \
                          rng.standard_normal(N) * np.sqrt(noise_var)
                approx_cov = np.nan
                Phi64 = np.asarray(Phi32, dtype=np.float64)
                Khat_xi = Phi64 @ Phi64.T + noise_var * np.eye(N)
            else:
                y_noise, approx_cov = _cur_sf(x, sigma, noise_var, l, rng, D)

            spherical_y = linalg.solve_triangular(L, y_noise, lower=True)
            res = stats.cramervonmises(spherical_y, "norm", args=(0, 1))
            statistic = res.statistic
            pvalue = res.pvalue
            # pvalue unreliable (see doc) if estimating params
            reject += int(pvalue < significance_threshold)

            if np.isnan(approx_cov).any():
                approx_cov = approx_cov * avg_approx_cov
            avg_approx_cov += approx_cov

            # Bayes validation: compute TV from realised covariance
            if Khat_xi is not None:
                tv_values.append(_bv_tv(L, Khat_xi))

        # record variance as well as mean?
        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        if np.isnan(avg_approx_cov).any() or np.isnan(theory_cov_noise).any():
            err = np.nan
        else:
            err = linalg.norm(theory_cov_noise - avg_approx_cov)
        errors.append(err)

        # BV aggregate statistics
        tv_mean = float(np.mean(tv_values)) if tv_values else np.nan
        tv_q = float(np.quantile(tv_values, 1.0 - bv_delta)) if tv_values else np.nan

        if method == "chol":
            D = -999

        if verbose:
            print("D = %d" % D, flush=True)
            print(
                f"max_preconditioner_size={max_preconditioner_size}", flush=True
            )
            print(
                "Norm difference between average approximate and exact K: %.6f"
                % err,
                flush=True,
            )
            print("%.2f%% rejected" % (reject * 100), flush=True)
            if bv and not np.isnan(tv_mean):
                print(
                    f"TV mean={tv_mean:.4f}  TV q{int((1-bv_delta)*100)}={tv_q:.4f}",
                    flush=True,
                )

        base = tup + (D, err, reject)
        if method == "lrff":
            base = tup + (D, err, reject, neff)
        if bv:
            base = base + (tv_mean, tv_q)
        row_str = str(base)[1:-1]
        print(row_str, file=csvfile, flush=True)


def run_sweep(
    ds: Iterable,
    ls: Iterable,
    sigmas: Iterable,
    noise_vars: Iterable,
    Ns: Iterable,
    verbose: bool = True,
    NO_TRIALS: int = 10,
    significance_threshold: float = 0.1,
    param_index: int = 0,
    benchmark: bool = False,
    ncpus: int = 2,
    method: str = "ciq",
    job_id: int = 0,
    with_pre: bool = False,
    bv: bool = False,
    bv_delta: float = 0.05,
) -> None:
    """Runs experiments over all sets of parameters. Runs in parallel if
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
    bv_suffix = "_bv" if bv else ""
    if __name__ == "__main__":
        filename = f"output_sweep_{method}_{param_index}_{job_id}_TEST{bv_suffix}.csv"
        overwrite = True
    else:
        if benchmark:
            filename = f"output_sweep_{method}_{param_index}_{job_id}_bench{bv_suffix}.csv"
        else:
            filename = f"output_sweep_{method}_{param_index}_{job_id}{bv_suffix}.csv"
        overwrite = False

    filepath = check_exists(
        pathlib.Path(".").joinpath(filename), ".csv", overwrite=overwrite
    )[0]

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = ["d", "l", "sigma", "noise_var", "N", "D", "err", "reject"]
        if method == "lrff":
            fieldnames.append("neff")
        if bv:
            tv_pct = int((1.0 - bv_delta) * 100)
            fieldnames += ["tv_mean", f"tv_q{tv_pct}"]
        print(",".join(fieldnames), file=csvfile, flush=True)
        if ncpus > 1:
            Parallel(n_jobs=ncpus, require="sharedmem")(
                delayed(sweep_fun)(
                    tup,
                    method,
                    csvfile,
                    NO_TRIALS,
                    verbose,
                    benchmark,
                    significance_threshold,
                    with_pre,
                    bv,
                    bv_delta,
                )
                for tup in product(ds, ls, sigmas, noise_vars, Ns)
            )
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(
                    tup,
                    method,
                    csvfile,
                    NO_TRIALS,
                    verbose,
                    benchmark,
                    significance_threshold,
                    with_pre,
                    bv,
                    bv_delta,
                )


if __name__ == "__main__":
    run_sweep(**default_param_set, method="cg")  # type: ignore
