"""CvM hypothesis-testing sweep for GP samplers.

Validates GP samplers via Cramér–von Mises goodness-of-fit on the
sphericalised sample L⁻¹ y vs N(0, I).  Works for *all* samplers,
including non-Gaussian ones (CG, Lanczos, sparse) that cannot use
the Bayesian-decision framework in sweeps.matern_bayes.

Optionally computes analytic total variation (Imhof) for Gaussian
samplers when ``bv=True``.

Usage
-----
    python -m sweeps.cvm_hyp.run_sweep          # default param set, CG
    python -m sweeps.cvm_hyp.multi_sweep --method rff --bv
"""

from itertools import product
from functools import partial
from typing import Tuple, TextIO, Iterable

import numpy as np
from scipy import linalg, stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import pathlib

import gpsampler
from gpsampler.utils import check_exists
from gpsampler.bayes_validation import gaussian_bayes_error
from gpsampler.samplers.lrff import compute_sir_pool

from sweeps._shared import lrff_setup, neff_hutchinson
from .config import default_param_set, param_sets


rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# Fidelity grids
# ---------------------------------------------------------------------------

def Ds(d, l, sigma, noise_var, N):
    """Array of RFF feature counts (powers of 2, up to N²)."""
    max_D = int(np.log2(N**2)) + 1
    return [2**i for i in range(16, max_D)]


def Js(d, l, sigma, noise_var, N):
    """Array of Lanczos iterations (powers of 2, up to √(N/σ²)·log N)."""
    max_J = int(np.log2(np.sqrt(N / noise_var) * np.log(N))) + 1
    return [2**i for i in range(4, max_J)]


# ---------------------------------------------------------------------------
# Main per-config experiment
# ---------------------------------------------------------------------------

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
    """Run experiment over a tuple of parameters NO_TRIALS times."""
    d, l, sigma, noise_var, N = tup
    if with_pre:
        max_preconditioner_size = int(np.sqrt(N))
    else:
        max_preconditioner_size = 0

    x = rng.standard_normal(size=(N, d)) / np.sqrt(d)
    theory_cov = sigma * np.exp(-pairwise_distances(x) ** 2 / (2 * l**2))
    theory_cov_noise = theory_cov + noise_var * np.eye(N)
    L = linalg.cholesky(theory_cov_noise, lower=True)

    # For lrff: pre-build the Nyström sketch once per config.
    if method == "lrff":
        # nu=inf → RBF (sweep.py was RBF-only)
        _, _lrff_alpha_fn, _ = lrff_setup(x, nu=float("inf"), ell=l, noise_var=noise_var)
        neff = neff_hutchinson(theory_cov / sigma, L, n_probes=30,
                               rng=np.random.default_rng(12345)) / sigma
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
        raise ValueError(f"Unknown method {method!r}")

    errors = []
    if verbose:
        print(
            "***d = %d, l = %.2e, sigma = %.2e, noise_var = %.2e, N = %d***"
            % tup,
            flush=True,
        )
    for D in _Ds(*tup):
        # For lrff: build SIR pool once per D value.
        if method == "lrff":
            _pool_cache = compute_sir_pool(
                D // 2, d, "rbf", l, 1.5, _lrff_alpha_fn,
                np.random.default_rng(D + 1_000_000),
                pool_factor=5, pool_min=4000,
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
            Khat_xi = None

            if benchmark:
                y_noise = rng.multivariate_normal(np.zeros(N), theory_cov_noise)
                approx_cov = theory_cov_noise
                if bv:
                    Khat_xi = theory_cov_noise
            elif bv and method == "rff":
                # Inline RFF sampling: capture Phi to build K̂_ξ
                omega = rng.multivariate_normal(np.zeros(d), np.eye(d) / l**2, D // 2)
                v = x @ omega.T
                Z = np.sqrt(2.0 / D) * np.concatenate(
                    [np.cos(v), np.sin(v)], axis=1)
                Phi = np.sqrt(sigma) * Z
                w = rng.standard_normal(D)
                y_noise = Phi @ w + rng.standard_normal(N) * np.sqrt(noise_var)
                approx_cov = np.nan
                Khat_xi = Phi @ Phi.T + noise_var * np.eye(N)
            elif bv and method == "lrff":
                # Inline lrff sampling: capture Phi for BV
                from gpsampler.samplers.lrff import reweighted_rff_sampler
                Phi32 = np.asarray(reweighted_rff_sampler(
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
            reject += int(res.pvalue < significance_threshold)

            if np.isnan(approx_cov).any():
                approx_cov = approx_cov * avg_approx_cov
            avg_approx_cov += approx_cov

            # Bayes validation: compute TV from realised covariance
            if Khat_xi is not None:
                bv_res = gaussian_bayes_error(theory_cov_noise, Khat_xi)
                tv_values.append(bv_res["tv"])

        reject /= NO_TRIALS
        avg_approx_cov /= NO_TRIALS
        if np.isnan(avg_approx_cov).any() or np.isnan(theory_cov_noise).any():
            err = np.nan
        else:
            err = linalg.norm(theory_cov_noise - avg_approx_cov)
        errors.append(err)

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
    """Run CvM sweep over all parameter combinations."""
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
                    tup, method, csvfile, NO_TRIALS, verbose, benchmark,
                    significance_threshold, with_pre, bv, bv_delta,
                )
                for tup in product(ds, ls, sigmas, noise_vars, Ns)
            )
        else:
            for tup in product(ds, ls, sigmas, noise_vars, Ns):
                sweep_fun(
                    tup, method, csvfile, NO_TRIALS, verbose, benchmark,
                    significance_threshold, with_pre, bv, bv_delta,
                )


if __name__ == "__main__":
    run_sweep(**default_param_set, method="cg")  # type: ignore
