#%%
import numpy as np
import pandas as pd
import seaborn as sns
from gpybench.plotting import plot_yeqx, save_fig
from gpybench.utils import isnotebook
from typing import Annotated, Final
from nptyping import NDArray
from pathlib import Path

#%%
if isnotebook():
    path = Path("../..")
else:
    path = Path("../.")
#%%
sweep = pd.read_csv("output_sweep.csv")
sweep_grp = sweep.groupby(["d", "l", "sigma", "noise_var"])

title = [f"{k}:{v}" for k,v in zip(sweep_grp.keys, *sweep_grp.groups.keys())]
#%% - plot error as function of no. RFF (logscales). err = ||K-Krff||_F
ax1 = sns.lineplot(x="D", y="err", data=sweep, hue="N")
ax1.set(xscale="log", yscale="log")
ax1.set_title(title)
save_fig(path, "logerr-logD_byN", suffix="png", show=True)
#%% - plot rejection rate as function of no. RFF (logscales)
ax2 = sns.lineplot(x="D", y="reject", data=sweep, hue="N")
ax2.set(xscale="log", yscale="log")
ax2.set_title(title)
save_fig(path, "logreject-logD_byN", suffix="png", show=True)
# %%- plot err vs reject
ax3 = sns.lineplot(x="err", y="reject", hue="N", marker="D", data=sweep)
ax3.set(xscale="log", yscale="log")
save_fig(path, "logreject-logerr_byN", suffix="png", show=True)
#%%
# not actually 1...want a placeholder as we want to just state that the function is
# shape-preserving
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
ax4 = sns.lmplot(x="D", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=2)
ax4.set(xscale="log")
save_fig(path, "logreject-logD_byN_reg", suffix="png", show=True)
# %% fit linear reg. models: log(err) ~ log(D | N)
transforms = {"err": np.log}
ax5 = sns.lmplot(x="D", y="err", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=2)
ax5.set(xscale="log")
save_fig(path, "logerr-logD_byN_reg", suffix="png", show=True)
# %% fit linear reg. models: log(reject) ~ log(err | N)
transforms = {"reject": np.log}
ax6 = sns.lmplot(x="err", y="reject", data=transform_cols(sweep, transforms), col="N", logx=True, col_wrap=2)
ax6.set(xscale="log")
save_fig(path, "logreject-logerr_byN_reg", suffix="png", show=True)
# %%
