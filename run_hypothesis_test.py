#%%
import numpy as np
from scipy import linalg, stats
from gpprediction.utils import k_se, k_mat_half as k_exp

data_path = "../synthetic-datasets/CIQ_GENERATION_RESULTS/"

kernel_type = "rbf"
d = 10
ls = 3.0

filename = f"output_kt_{kernel_type}_dim_{d}_ls_{ls}.npy"

#%%
data = np.load(data_path + filename)
#%%
significance_threshold = 0.1

kernel = {"rbf": k_se, "exp": k_exp}[kernel_type]

n = 10000

def test_dataset(data, kernel, ls):
    subset = np.random.choice(data.shape[0], n)
    x = data[subset,:-1]
    y_noise = data[subset, -1]

    theory_cov_noise = kernel(x,x,0.992,ls) + np.eye(n) * 0.008

    L = linalg.cholesky(theory_cov_noise, lower=True)
    spherical_y = linalg.solve_triangular(L, y_noise, lower=True)
    res = stats.cramervonmises(spherical_y, 'norm', args=(0, 1))
    statistic = res.statistic
    pvalue = res.pvalue
    # pvalue unreliable (see doc) if estimating params
    reject = int(pvalue < significance_threshold)
    return reject, statistic, pvalue
#%%
nreps = 10
rejects = np.zeros(nreps)
statistics = np.zeros(nreps)
pvalues = np.zeros(nreps)
for j in range(nreps):
    r,s,p = test_dataset(data, kernel, ls)
    rejects[j], statistics[j], pvalues[j] = r,s,p
# print(f"Rejection: {bool(r)},", f"Statistic: {s},", f"p-value: {p}," )
# %%
print(f"Rejection rate: {rejects.mean()},", f"Statistic: {statistics.mean()},", f"p-value: {pvalues.mean()}," )
# %%
