#%%
import numpy as np
import pandas as pd
import seaborn as sns
from gpybench.plotting import save_fig, LaTeX
from gpybench.utils import isnotebook
from gpytools.utils import ordermag
from nptyping import NDArray
from pathlib import Path

from typing import Annotated, Final
 
#%%
if isnotebook():
    path = Path("../..")
else:
    path = Path(".")
#%% script params
method = "rff"
job_id = 2580211#2703849#2580211#2686645#
param_idx = 1
sig_thresh = 0.1
with_pre = False
#compute approximate confidence interval on the rejection rate, by using the
#following argument:
# E[r] = q, rhat ~ q; var(rhat) = q(1-q)/N; var(rhat)_hat ~ rhat(1-rhat)/N
# q = 1/2*(alpha + (1-beta)), rhat = 1/2(alpha + (1-betahat))
# betahat = 1 - 2rhat + alpha
# varrhat = rhat(betahat-alpha)/N = rhat(1-2rhat)/N
# assume N = 1000 (check file)
# sweep.loc[:, "rsigma"] = 2 * np.sqrt(sweep.reject * (1-sweep.reject) / 1000)

#BETTER, just use
ci95 = 1.96*np.sqrt(0.1*0.9/1000)
# on the grounds that for N=1000 => CLT, so this is ~95%ci

# get data
sweep = pd.read_csv(path.joinpath(f"output_sweep_{method}_{param_idx}_{job_id}.csv"))

# sort out data
sweep = sweep.rename({"N":"n"}, axis=1)
sweep = sweep.sort_values(by=["n","D","l"])
# if there are duplicates, average over them
sweep_grp = sweep.groupby(sweep.columns.difference(["reject","err"]).tolist())
sweep = sweep_grp.mean()
# RIP index
sweep = sweep.reset_index()

# only keep 0.1 and 2.0 for now?
sweep = sweep.query("l in [0.1,1.0]")

if method == "ciq":
    sweep = sweep.rename({"D":"J"}, axis=1)
    fidel_param = "J"
    if with_pre:
        order_f = lambda n: n**(3/8) * np.log(n)  
    else:
        order_f = lambda n: np.sqrt(n) * np.log(n)
else:
    fidel_param = "D"
    order_f = lambda n: n**(3/2) * np.log(n)

#%% empirically estimate the convergence of r wrt D? e.g. D ~ nlogn or n^2 or
#... i.e. find the first D for each N s.t. the observed rejection rate is within
#some small distance (1e-2 for now; arbitrary need fluctuation analysis) of the
#significance level, taking the average over lengthscales:
crossing_Ds = dict()
for l in sweep.l.unique():
    crossing_Ds[l] = dict()
    for idx, grp_df in sweep.loc[sweep.l == l,:].groupby(["n"]):
        avg_r_by_D = grp_df.groupby(fidel_param).reject.mean()
        crossing_Ds[l][idx] = (np.abs(avg_r_by_D - 0.1) > 1e-2).idxmin()

import matplotlib.pyplot as plt
palette = sns.color_palette(palette="flare", n_colors=sweep.n.nunique())

#%% poster plots #1

def conv_plot(df, xlabel, ylabel="reject"):

    with LaTeX() as ltx:
        ax = sns.relplot(
        data=df,
        x=xlabel, y=ylabel,
        hue="n", col="l",
        kind="line", palette=palette, col_wrap=2,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
        )
        ax.set(xscale="log", yscale="log")
        ax.set(yticks=[0.1,0.5,1])
        ax.set_yticklabels([0.1,0.5,1])

        xticks = [ordermag(x) for x in np.geomspace(df.loc[:,xlabel].min(), df.loc[:,xlabel].max(), num=4)]

        for _l,_ax in ax.axes_dict.items():
            _ax.axhline(sig_thresh, ls='--')
            _ax.axhline(sig_thresh + ci95, ls='--', color="green")
            _ax.axhline(sig_thresh - ci95, ls='--', color="green")
            _ax.axvline(1, ls='--', color="black")
            _ax.patch.set_edgecolor('black')  
            _ax.patch.set_linewidth('1') 
            _ax.set_xticks(xticks)
            _ax.set_ylim(top=1)

    fig = plt.gcf()
    title = method.upper()
    if with_pre:
        title = "P" + title
    fig.suptitle(title)

#%%
conv_plot(sweep, fidel_param)
save_fig(path, f"logreject-logD_byN_{method}_{param_idx}_{job_id}", suffix="pdf", show=True, dpi=600, overwrite=True)


#%% Repeat plots but re-scale x-axis by sqrt(n)log(n) to reflect scaling
#expectation
rescaled_fidel = "$"+fidel_param+"/"+"\\bar{"+f"{fidel_param}"+"}"+"(n)$"
sweep.loc[:, rescaled_fidel] = sweep.loc[:, fidel_param]/order_f(sweep.n)

#%%
conv_plot(sweep, rescaled_fidel)
save_fig(path, f"logreject-logD_byN_{method}_{param_idx}_{job_id}_rescaled", suffix="pdf", show=True, dpi=600, overwrite=True)
#%%
raise Exception("End script")
#group by parameter values
# sweep_grp = sweep.groupby(["d", "l", "sigma", "noise_var"])

# grp = list(sweep_grp.groups.keys())[0]
# title = [f"{k}:{v}" for k,v in zip(sweep_grp.keys, grp)]
#%% - plot error as function of no. RFF (logscales). err = ||K-Krff||_F
# ax1 = sns.lineplot(x="D", y="err", data=sweep, hue="N")
# ax1.set(xscale="log", yscale="log")
# ax1.set_title(title)
# save_fig(path, f"logerr-logD_byN_{method}", suffix="jpg", show=True)
# %- plot err vs reject (for particular value of l for now)
ax3 = sns.lineplot(x="err", y="reject", hue="N", marker="D", data=sweep_sub)
ax3.set(xscale="log", yscale="log")
save_fig(path, f"logreject-logerr_byN_{method}", suffix="jpg", show=True)
#% not actually 1...want a placeholder as we want to just state that the
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
# % fit linear reg. models: log(reject) ~ log(D | N)
transforms = {"reject": np.log}
ax4 = sns.lmplot(x=fidel_param, y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax4.set(xscale="log")
save_fig(path, f"logreject-logD_byN_reg_{method}", suffix="jpg", show=True)
# % fit linear reg. models: log(err) ~ log(D | N)
transforms = {"err": np.log}
ax5 = sns.lmplot(x=fidel_param, y="err", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax5.set(xscale="log")
save_fig(path, f"logerr-logD_byN_reg_{method}", suffix="jpg", show=True)
# % fit linear reg. models: log(reject) ~ log(err | N)
transforms = {"reject": np.log}
ax6 = sns.lmplot(x="err", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax6.set(xscale="log")
save_fig(path, f"logreject-logerr_byN_reg_{method}", suffix="jpg", show=True)
# %%
