# %%
import os
import numpy as np
from scipy import linalg, stats
import argparse
from gpprediction.utils import k_se, k_mat_half as k_exp, k_lap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1100)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--out", type=str, default="dataset_test_output.csv")
    parser.add_argument(
        "--filepath", type=str,
        default="synthetic-datasets/RFF/output_kt_exp_dim_10_ls_3.0_1.npy")
    parser.add_argument("--kernel_type", type=str, default="exp")
    parser.add_argument("--method", type=str, default="rff")
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--significance", type=float, default=0.1)

    args = parser.parse_args()

    id = args.id  # 1,3 for Nick datasets, 0 for Sam (CIQ)
    kernel_type = args.kernel_type
    method = args.method
    d = args.dimension
    ls = args.lengthscale
    nv = {"rff": 0.1, "ciq": 0.008}[method]
    m = args.m
    significance_threshold = args.significance

    filename = args.filepath

    # sub_dirs = {"ciq": "CIQ_GENERATION_RESULTS", "rff": "RFF"}

    # data_path = f"../synthetic-datasets/{sub_dirs[method]}/"
    # if id != 0:
    #     filename = f"output_kt_{kernel_type}_dim_{d}_ls_{ls}_{id}.npy"
    # else:
    #     filename = f"output_kt_{kernel_type}_dim_{d}_ls_{ls}.npy"

    # # Nick data
    # if id in (1, 3) and method == "ciq":
    #     data_path = f"../synthetic-datasets/ciq_synthetic_datasets/noise_var=0.008/"
    #     filename = f"DIM{{{d}}}_LENSCALE{{{ls}}}_{{{id}}}/data.npy"

    out_file_name = args.out

# %%
    data = np.load(filename)
# %%

    kernel = {"rbf": k_se, "exp": k_exp, "laplacian": k_lap}[kernel_type]

    def test_dataset(data, kernel, ls):
        subset = np.random.choice(data.shape[0], m)
        x = data[subset, :-1]
        y_noise = data[subset, -1]

        theory_cov_noise = kernel(x, x, 1-nv, ls) + np.eye(m) * nv

        L = linalg.cholesky(theory_cov_noise, lower=True)
        spherical_y = linalg.solve_triangular(L, y_noise, lower=True)
        res = stats.cramervonmises(spherical_y, 'norm', args=(0, 1))
        statistic = res.statistic
        pvalue = res.pvalue
        # pvalue unreliable (see doc) if estimating params
        reject = int(pvalue < significance_threshold)
        return reject, statistic, pvalue
# %%
    nreps = 10
    rejects = np.zeros(nreps)
    statistics = np.zeros(nreps)
    pvalues = np.zeros(nreps)
    for j in range(nreps):
        r, s, p = test_dataset(data, kernel, ls)
        rejects[j], statistics[j], pvalues[j] = r, s, p
# print(f"Rejection: {bool(r)},", f"Statistic: {s},", f"p-value: {p}," )
# %% should maybe be doing pvalues.prod()**(1/nreps): geometric mean?
    print(
        f"Rejection rate: {rejects.mean()},",
        f"Statistic: {statistics.mean()},", f"p-value: {pvalues.mean()}")
# %% Meta test: uniformity of p-values (needs theory check)
    from scipy.stats import kstest
    ks_res = kstest(pvalues, 'uniform')
    print(ks_res)
# plt.hist(pvalues)
# %%
    file_exists = os.path.isfile(out_file_name)
    with open(out_file_name, "a+") as out_file:
        if not file_exists:
            header = "method,kernel,d,ls,m,nreps,reject,stat,pval,id,ksstat,kspval"
            print(header, file=out_file, flush=True)

        line = f"{method},{kernel_type},{d},{ls},{m},{nreps},{rejects.mean()},{statistics.mean()},{pvalues.mean()},{id},{ks_res.statistic},{ks_res.pvalue}"
        print(line, file=out_file, flush=True)

# %%
if 0:
    from scipy.linalg import cholesky, solve_triangular
    from gpprediction.utils import inv

    def recursive_block_cholesky(data, m=1000):

        # compute no. blocks
        n = data.shape[0]
        B = n // m

        # get this (first) segment of data
        x0 = data[:m, :-1]
        y0 = data[:m, -1]

        # compute submatrix for these data and it's cholesky decomp. and inverse
        K00 = kernel(x0, x0, 1-nv, ls) + 1e-6 * np.eye(m)
        L00 = cholesky(K00)
        invL00 = inv(L00)
        del K00, L00

        # get the transformed observations (N(0,I) marginally under H0) for segment
        z0 = invL00 @ y0

        xn0 = data[m:, :-1]
        K0n = kernel(x0, xn0, 1-nv, ls)
        Ln0 = K0n @ invL00.T  # unstable?
        T = Ln0 @ z0
        del xn0, K0n, Ln0
