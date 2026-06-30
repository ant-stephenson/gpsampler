"""Figure F2 — Smoothness: RFF vs LRFF across Matérn orders (n=2048, ℓ=0.2).

Reads: Stage-1 sweep CSV.
Writes: figures/matern_bayes/figs/f2_smoothness_rff_vs_lrff.pdf

Layout: 1 row × 2 columns.
  (a) RFF: one curve per ν; x = D/n_eff², y = p_star_lowq.
      Curves spread with ν (smoother kernels need larger D).
  (b) LRFF: same but curves collapse onto a single track (n_eff² rescaling
      accounts for the smoothness-dependent effective dimension).

Both panels share y-axis.  Annotate the common n_eff² threshold (x=1).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sweeps.matern_bayes.config import NUS
from ._common import (
    apply_theme, load_sweep, method_label, nu_label, n_palette, save,
)

_FIGURE_N = 2048
_FIGURE_ELL_TARGET = 0.2


def _nearest_ell(df: pd.DataFrame, target: float) -> float:
    ells = sorted(df["ell"].unique())
    return min(ells, key=lambda e: abs(e - target))


def render(
    sweep_csv: pathlib.Path | str,
    n: int = _FIGURE_N,
    ell_target: float = _FIGURE_ELL_TARGET,
    d: int = 1,
    outname: str = "f2_smoothness_rff_vs_lrff",
    show: bool = False,
) -> pathlib.Path:
    apply_theme()
    df = load_sweep(sweep_csv)

    ell = _nearest_ell(df, ell_target)

    # Use largest available n if requested n not present
    available_ns = sorted(df["n"].unique())
    if n not in available_ns:
        n = max(available_ns)

    nu_vals = sorted(df["nu"].unique())
    nu_palette = list(sns.color_palette("viridis", n_colors=len(nu_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), sharey=True)

    for ax, method in zip(axes, ("rff", "lrff")):
        sub = df[(df["method"] == method) & (df["n"] == n) &
                 (df["ell"] == ell) & (df["d"] == d)]
        if sub.empty:
            ax.set_title(f"{method_label(method)} (no data)")
            continue

        for colour, nu in zip(nu_palette, nu_vals):
            ndf = sub[sub["nu"] == nu].sort_values("fidelity_rescaled")
            if ndf.empty:
                continue
            ax.plot(
                ndf["fidelity_rescaled"],
                ndf["p_star_lowq"],
                color=colour,
                label=nu_label(nu),
                linewidth=1.5,
            )

        ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
        ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$D / n_\mathrm{eff}^2$")
        ax.set_title(f"({['a','b'][axes.tolist().index(ax)]}) {method_label(method)}",
                     fontsize=9, fontweight="bold")
        ax.set_ylim(0.0, 0.52)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel(r"$p^*_\delta$ (lower quantile)")
    axes[-1].legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.suptitle(
        rf"F2: smoothness — $n={n}$, $\ell \approx {ell}$", fontsize=9, y=1.01
    )
    plt.tight_layout()
    base = pathlib.Path(__file__).parent
    save(outname, base_path=base, show=show, overwrite=True)
    return base / "figs" / f"{outname}.pdf"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sweep_csv", type=pathlib.Path)
    p.add_argument("--n", type=int, default=_FIGURE_N)
    p.add_argument("--ell", type=float, default=_FIGURE_ELL_TARGET)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = render(args.sweep_csv, n=args.n, ell_target=args.ell, show=args.show)
    print(f"Saved: {out}")
