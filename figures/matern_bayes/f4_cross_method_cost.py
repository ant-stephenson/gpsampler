"""Figure F4 — Cross-method cost comparison (FLOPs vs p*).

Reads: Stage-1 sweep CSV.
Writes: figures/matern_bayes/figs/f4_cross_method_cost.pdf

Layout: 2 rows × 2 columns  (ℓ ∈ {0.1, 1.0} × ν ∈ {1.5, ∞}).
  Each panel: n = 2048, all four methods overlaid.
  x = FLOPs (log scale)
  y = p_star_lowq

Vertical reference line at Cholesky FLOPs = n³/3 (exact sampler cost).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sweeps.matern_bayes.config import METHODS
from sweeps.matern_bayes.flops import flops_cholesky
from ._common import (
    apply_theme, load_sweep, method_label, save,
)

_PANEL_NUS = (1.5, float("inf"))
_PANEL_ELLS = (0.1, 1.0)

_METHOD_COLORS = {
    "rff":  "#e07b39",
    "lrff": "#8c5e2a",
    "ciq":  "#2670a8",
    "pciq": "#1a4a7a",
}
_METHOD_LS = {
    "rff":  "-",
    "lrff": "--",
    "ciq":  "-.",
    "pciq": ":",
}


def render(
    sweep_csv: pathlib.Path | str,
    n: int = 2048,
    d: int = 1,
    outname: str = "f4_cross_method_cost",
    show: bool = False,
) -> pathlib.Path:
    apply_theme()
    df = load_sweep(sweep_csv)

    # Use largest available n if requested n absent
    if n not in df["n"].unique():
        n = int(df["n"].max())

    chol_flops = flops_cholesky(n)

    panel_configs = [(nu, ell) for ell in _PANEL_ELLS for nu in _PANEL_NUS]
    n_panels = len(panel_configs)
    n_rows = 2
    n_cols = n_panels // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    panel_labels = list("abcdefgh")

    for idx, (ax, (nu, ell)) in enumerate(zip(axes_flat, panel_configs)):
        sub = df[(df["n"] == n) & (df["nu"] == nu) & (df["ell"] == ell) & (df["d"] == d)]
        if sub.empty:
            ax.set_title(f"ν={nu} ℓ={ell} (no data)")
            continue

        for method in METHODS:
            mdf = sub[sub["method"] == method].sort_values("flops")
            if mdf.empty:
                continue
            ax.plot(
                mdf["flops"],
                mdf["p_star_lowq"],
                color=_METHOD_COLORS[method],
                ls=_METHOD_LS[method],
                linewidth=1.5,
                label=method_label(method),
            )

        ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
        ax.axvline(chol_flops, ls=":", color="gray", linewidth=0.9, alpha=0.8,
                   label=f"Cholesky ($n^3/3$)")

        nu_str = r"\infty" if nu >= 1000 else str(nu)
        ax.set_title(
            f"({panel_labels[idx]}) $\\nu={nu_str}$, $\\ell={ell}$",
            fontsize=8, fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.52)
        ax.set_xlabel("FLOPs")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p^*_\delta$ (lower quantile)")

    # Shared legend in upper-right panel
    axes_flat[n_cols - 1].legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.suptitle(rf"F4: cross-method FLOPs — $n={n}$", fontsize=9, y=1.01)
    plt.tight_layout()
    base = pathlib.Path(__file__).parent
    save(outname, base_path=base, show=show, overwrite=True)
    return base / "figs" / f"{outname}.pdf"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sweep_csv", type=pathlib.Path)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = render(args.sweep_csv, n=args.n, show=args.show)
    print(f"Saved: {out}")
