#%%
import numpy as np
import pandas as pd
import seaborn as sns
from gpybench.plotting import save_fig, LaTeX
from gpybench.utils import isnotebook
from gpytools.utils import ordermag
from pathlib import Path
import warnings
 
#%%
if isnotebook():
    path = Path("..")
else:
    path = Path(".")
#%%
ci95 = 1.96*np.sqrt(0.1*0.9/1000)
# on the grounds that for N=1000 => CLT, so this is ~95%ci
sig_thresh = 0.1
#%% script params
def import_data(method, job_id, param_idx, with_pre):

    expt_path = path.parent.absolute().parent.joinpath("experiments")
    # get data
    sweep = pd.read_csv(expt_path.joinpath(f"output_sweep_{method}_{param_idx}_{job_id}.csv"))

    # sort out data
    sweep = sweep.rename({"N":"n"}, axis=1)
    sweep = sweep.sort_values(by=["n","D","l"])
    # if there are duplicates, average over them
    sweep_grp = sweep.groupby(sweep.columns.difference(["reject","err"]).tolist())
    sweep = sweep_grp.mean()
    # RIP index
    sweep = sweep.reset_index()

    # only keep 0.1 and 2.0 for now?
    # sweep = sweep.query("l in [0.1,1.0,2.0,5.0]")

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

    return sweep, order_f, fidel_param

#%% get data
method = "ciq"
job_id = 2703849#2580211#2703849#2580211#2686645#
param_idx = 1
with_pre = False
chol_sweep, _, _ = import_data("chol", 3057212, 1, None)
sweep, order_f, fidel_param = import_data(method, job_id, param_idx, with_pre)
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

#%% chol summary
chol_grp = chol_sweep.groupby("l")
chol_max = chol_grp.reject.max()
chol_min = chol_grp.reject.min()
#%% poster plots #1
import matplotlib.pyplot as plt
palette = sns.color_palette(palette="flare", n_colors=sweep.n.nunique())

def conv_plot(df, xlabel, ylabel="reject"):

    with LaTeX() as ltx:
        ax = sns.relplot(
        data=df,
        x=xlabel, y=ylabel,
        hue="n", col="l",
        kind="line", palette=palette, col_wrap=sweep.l.nunique(),
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
            try:
                _ax.fill_between(range(df.shape[0]), np.tile(chol_max[_l], df.shape[0]), np.tile(chol_min[_l], df.shape[0]), alpha=0.2, color='gray', interpolate=True)
            except KeyError as ke:
                warnings.warn(str(ke))

    # normal
    plt.setp(ax._legend.get_title(), fontsize=30)
    # for thumbnail:
        # leg = ax._legend
        # leg.set_bbox_to_anchor([0.65, 1.0])  # coordinates of lower left of bounding box
        # leg._loc = 2

    fig = plt.gcf()
    title = method.upper()
    if with_pre:
        title = "P" + title
    fig.suptitle(title)

#%% Repeat plots but re-scale x-axis by sqrt(n)log(n) to reflect scaling
#expectation
rescaled_fidel = "$"+fidel_param+"/"+"\\bar{"+f"{fidel_param}"+"}"+"(n)$"
sweep.loc[:, rescaled_fidel] = sweep.loc[:, fidel_param]/order_f(sweep.n)

#%%
conv_plot(sweep, rescaled_fidel)
save_fig(path, f"logreject-logD_byN_{method}_{param_idx}_{job_id}_rescaled", suffix="pdf", show=True, dpi=600, overwrite=True)
# with Thumbnail() as _:
#     conv_plot(sweep, rescaled_fidel)
#     save_fig(path, f"logreject-logD_byN_{method}_{param_idx}_{job_id}_rescaled", suffix="png", show=True, dpi=600, size_inches=(2.66667, 2.13333), overwrite=True)
# %%
