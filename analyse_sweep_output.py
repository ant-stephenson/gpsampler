#%%
import numpy as np
import pandas as pd
import seaborn as sns
from gpybench.plotting import plot_yeqx, save_fig, LaTeX
from gpybench.utils import isnotebook, check_exists
from typing import Annotated, Final
from nptyping import NDArray
from pathlib import Path

import re
import inspect
 
#%%
if isnotebook():
    path = Path("..")
else:
    path = Path(".")
#%%
method = "ciq"
job_id = 2504261%2547443#2504261#2479515
param_idx = 0
sig_thresh = 0.1
with_pre = False
sweep = pd.read_csv(path.joinpath(f"output_sweep_{method}_{param_idx}_{job_id}.csv"))
# sweep = pd.read_csv(path.joinpath(f"output_sweep_{param_idx}_{job_id}.csv"))
sweep = sweep.sort_values(by=["N","D","l"])

if method == "ciq":
    sweep = sweep.rename({"D":"J"}, axis=1)
    fidel_param = "J"
else:
    fidel_param = "D"

#%% compute approximate confidence interval on the rejection rate, by using the
#following argument:
# E[r] = q, rhat ~ q; var(rhat) = q(1-q)/N; var(rhat)_hat ~ rhat(1-rhat)/N
# q = 1/2*(alpha + (1-beta)), rhat = 1/2(alpha + (1-betahat))
# betahat = 1 - 2rhat + alpha
# varrhat = rhat(betahat-alpha)/N = rhat(1-2rhat)/N
# assume N = 1000 (check file)
sweep.loc[:, "rsigma"] = 2 * np.sqrt(sweep.reject * (1-2*sweep.reject) / 1000)

#%% group by parameter values
sweep_grp = sweep.groupby(["d", "l", "sigma", "noise_var"])

grp = list(sweep_grp.groups.keys())[0]
title = [f"{k}:{v}" for k,v in zip(sweep_grp.keys, grp)]
#%% - plot error as function of no. RFF (logscales). err = ||K-Krff||_F
# ax1 = sns.lineplot(x="D", y="err", data=sweep, hue="N")
# ax1.set(xscale="log", yscale="log")
# ax1.set_title(title)
# save_fig(path, f"logerr-logD_byN_{method}", suffix="jpg", show=True)

#%% empirically estimate the convergence of r wrt D? e.g. D ~ nlogn or n^2 or
#... i.e. find the first D for each N s.t. the observed rejection rate is within
#some small distance (1e-2 for now; arbitrary need fluctuation analysis) of the
#significance level, taking the average over lengthscales:
crossing_Ds = dict()
for idx, grp_df in sweep.groupby(["N"]):
    avg_r_by_D = grp_df.groupby(fidel_param).reject.mean()
    ci_by_D = grp_df.groupby(fidel_param).rsigma.mean()
    crossing_Ds[idx] = (np.abs(avg_r_by_D - 0.1) > 1e-2).idxmin()
#%% - plot rejection rate as function of no. RFF (logscales)
import matplotlib.pyplot as plt
palette = sns.color_palette(palette="flare", n_colors=sweep.N.nunique())

def plot_reject_by_D(df: pd.DataFrame, ax=None, title=None):
    # palette = sns.color_palette(palette="flare", n_colors=df.N.nunique())
    if ax is None:
        _f, ax = plt.subplots()
    sns.lineplot(x=fidel_param, y="reject", data=df, hue="N", palette=palette, ax=ax)
    ax.set(xscale="log", yscale="log")
    # Put the legend out of the figure
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    maxy = df.reject.max()
    
    order_f = lambda n, nv: (n/nv) ** (1/2) * np.log10(n)
    ax.vlines(order_f(df.N.unique(), df.noise_var.unique()), ymin=0, ymax=maxy, colors=palette)
    ax.vlines(crossing_Ds.values(), ymin=0, ymax=maxy, colors=palette, ls="dotted")
    sig_thresh = 0.1
    ax.axhline(sig_thresh, ls='--')
    
    # grp_D = df.groupby("D")
    # upper = grp_D.reject.max() + grp_D.rsigma.mean()
    # lower = df.groupby("D").reject.min() - df.groupby("D").rsigma.mean()
    # ax2.plot(df.D.unique(), upper, ls='--', color="green")
    # ax2.plot(df.D.unique(), lower, ls='--', color="green")
    ax.axhline(sig_thresh + 1e-2, ls='--', color="green")
    ax.axhline(sig_thresh - 1e-2, ls='--', color="green")

    # eps = np.exp(-np.sqrt(sweep.noise_var)/np.sqrt(sweep.N) * (sweep.D-1) +
    # np.log(sweep.N)+5)
    # eps = np.log10(sweep.N)*np.log10(25*sweep.N/sweep.noise_var)*sweep.N/np.sqrt(sweep.noise_var)/np.pi * ((np.sqrt(sweep.N/sweep.noise_var) - 1) / (np.sqrt(sweep.N/sweep.noise_var) + 1)) ** (sweep.D-1)
    # ax2.plot(sweep.D, eps+0.1)

    if title is None:
        title = re.findall(r"(?<=\:).*", inspect.getsource(order_f))[0]
    ax.set_title(title)
    # save_fig(path, f"logreject-logD_byN_{method}_{job_id}", suffix="eps", show=True, dpi=600, overwrite=True)

sweep_sub = sweep_grp.get_group(grp)
# sns.relplot(
#     data=sweep,
#     x="D", y="reject",
#     hue="N", col="l",
#     kind="line", palette=palette,
#     height=5, aspect=.75, facet_kws=dict(sharex=False),
# )
#%%
# plot_reject_by_D(sweep.loc[sweep.l == 2.0, :])
plot_reject_by_D(sweep)
#%% poster plots

ax = sns.relplot(
    data=sweep.loc[sweep.l != 1.0005, :],
    x=fidel_param, y="reject",
    hue="N", col="l",
    kind="line", palette=palette, col_wrap=2,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
ax.set(xscale="log", yscale="log")

for _ax in ax.axes_dict.values():
    _ax.axhline(sig_thresh, ls='--')
    _ax.axhline(sig_thresh + 1e-2, ls='--', color="green")
    _ax.axhline(sig_thresh - 1e-2, ls='--', color="green")

fig = plt.gcf()
plt.ylim(0,1)
title = method.upper()
if with_pre:
    title = "P" + title
fig.suptitle(title)

save_fig(path, f"logreject-logD_byN_{method}_{param_idx}_{job_id}", suffix="eps", show=True, dpi=600, overwrite=True)

# ls = sweep.l.unique()
# nrows, ncols = 1, np.max(ls.shape)

# fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
# for i,(l,ax) in enumerate(zip(ls, axs.ravel())):
#     plot_reject_by_D(sweep.loc[sweep.l == l, :], ax, title=f"l={l}")
#     fig.suptitle("Experimental reject rate for CvM tests with CIQ")
#     save_fig(path, f"logreject-logD_byN_{method}_{job_id}", suffix="eps",
#     show=True, dpi=600, overwrite=True)
#%%
raise Exception("End script")
# %%- plot err vs reject (for particular value of l for now)
ax3 = sns.lineplot(x="err", y="reject", hue="N", marker="D", data=sweep_sub)
ax3.set(xscale="log", yscale="log")
save_fig(path, f"logreject-logerr_byN_{method}", suffix="jpg", show=True)
#%% not actually 1...want a placeholder as we want to just state that the
#function is shape-preserving
M: int = 1 #!
N: int = 1
def transform_cols(df: NDArray[(M,N)], transforms: dict) -> NDArray[(M,N)]:
    """Transforms a subset of columns of dataframe according to usual df.transform syntax
    i.e. dict, whilst retaining the other columns by transforming them according to an identity function

    Returns:
        pd.DataFrame: ...
    """
    id = lambda x: x
    _transforms = {**{k:id for k in df.columns}, **transforms}
    return df.transform(_transforms)
# %% fit linear reg. models: log(reject) ~ log(D | N)
transforms = {"reject": np.log}
ax4 = sns.lmplot(x=fidel_param, y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax4.set(xscale="log")
save_fig(path, f"logreject-logD_byN_reg_{method}", suffix="jpg", show=True)
# %% fit linear reg. models: log(err) ~ log(D | N)
transforms = {"err": np.log}
ax5 = sns.lmplot(x=fidel_param, y="err", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax5.set(xscale="log")
save_fig(path, f"logerr-logD_byN_reg_{method}", suffix="jpg", show=True)
# %% fit linear reg. models: log(reject) ~ log(err | N)
transforms = {"reject": np.log}
ax6 = sns.lmplot(x="err", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax6.set(xscale="log")
save_fig(path, f"logreject-logerr_byN_reg_{method}", suffix="jpg", show=True)
# %%
