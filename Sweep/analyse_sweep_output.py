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
    path = Path(".")
else:
    path = Path(".")
#%%
method = "ciq"
sweep = pd.read_csv(path.joinpath(f"output_sweep_{method}_1.csv"))
sweep = sweep.sort_values(by=["N","D","l"])
sweep_grp = sweep.groupby(["d", "l", "sigma", "noise_var"])

grp = list(sweep_grp.groups.keys())[0]
title = [f"{k}:{v}" for k,v in zip(sweep_grp.keys, grp)]
#%% - plot error as function of no. RFF (logscales). err = ||K-Krff||_F
ax1 = sns.lineplot(x="D", y="err", data=sweep, hue="N")
ax1.set(xscale="log", yscale="log")
ax1.set_title(title)
# save_fig(path, f"logerr-logD_byN_{method}", suffix="jpg", show=True)
#%% empirically estimate the convergence of r wrt D? e.g. D ~ nlogn or n^2 or ... i.e. find the first D for each N s.t. the observed rejection rate is within some small distance (1e-2 for now; arbitrary need fluctuation analysis) of the significance level, taking the average over lengthscales:
crossing_Ds = dict()
for idx, grp_df in sweep.groupby(["N"]):
    avg_r_by_D = grp_df.groupby("D").reject.mean()
    crossing_Ds[idx] = (np.abs(avg_r_by_D - 0.1) > 1e-2).idxmin()
#%% - plot rejection rate as function of no. RFF (logscales)

def plot_reject_by_D(df: pd.DataFrame):
    palette = sns.color_palette(palette="flare", n_colors=df.N.nunique())
    ax2 = sns.lineplot(x="D", y="reject", data=df, hue="N", palette=palette)
    ax2.set(xscale="log", yscale="log")
    # Put the legend out of the figure
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    maxy = df.reject.max()
    
    order_f = lambda n: n ** (1/2) * np.log10(n)
    ax2.vlines(order_f(df.N.unique()), ymin=0, ymax=maxy, colors=palette)
    ax2.vlines(crossing_Ds.values(), ymin=0, ymax=maxy, colors=palette, ls="dotted")
    sig_thresh = 0.1
    ax2.axhline(sig_thresh, ls='--')
    ax2.axhline(sig_thresh + 1e-2, ls='--', color="green")
    ax2.axhline(sig_thresh - 1e-2, ls='--', color="green")

    # eps = eps = np.exp(-sweep.noise_var/np.sqrt(sweep.N) * (sweep.D-1) - np.log(sweep.N)-5)
    # ax2.plot(sweep.D, eps+0.1)

    str_f = re.findall(r"(?<=\:).*", inspect.getsource(order_f))[0]
    ax2.set_title(str_f)
    # save_fig(path, f"logreject-logD_byN_{method}", suffix="jpg", show=True, size_inches=(12,8))

sweep_sub = sweep_grp.get_group(grp)
plot_reject_by_D(sweep)
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
ax4 = sns.lmplot(x="D", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax4.set(xscale="log")
save_fig(path, f"logreject-logD_byN_reg_{method}", suffix="jpg", show=True)
# %% fit linear reg. models: log(err) ~ log(D | N)
transforms = {"err": np.log}
ax5 = sns.lmplot(x="D", y="err", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax5.set(xscale="log")
save_fig(path, f"logerr-logD_byN_reg_{method}", suffix="jpg", show=True)
# %% fit linear reg. models: log(reject) ~ log(err | N)
transforms = {"reject": np.log}
ax6 = sns.lmplot(x="err", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=3)
ax6.set(xscale="log")
save_fig(path, f"logreject-logerr_byN_reg_{method}", suffix="jpg", show=True)
# %%
