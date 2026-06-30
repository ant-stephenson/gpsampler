"""Figure F3 — CIQ vs PCIQ across lengthscales.

Reads: Stage-1 sweep CSV.
Writes: figures/matern_bayes/figs/f3_ciq_vs_pciq_lengthscale.pdf

Layout: 1 row × 2 columns.
  (a) ℓ = 1.0  — CIQ and PCIQ curves overlaid, one line per n.
  (b) ℓ = 0.1  — same; PCIQ requires fewer J steps (better κ̃).

x = J (raw fidelity, NOT rescaled here — to show absolute cost difference).
y = p_star_lowq.  Separate line styles for CIQ (solid) and PCIQ (dashed).
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

from sweeps.matern_bayes.config import NUS
from ._common import (
    apply_theme, load_sweep, method_label, n_palette, save,
)


def render(
    sweep_csv: pathlib.Path | str,
    ells: tuple[float, ...] = (1.0, 0.1),
    nu: float = 1.5,
    d: int = 1,
    outname: str = "f3_ciq_vs_pciq_lengthscale",
    show: bool = False,
) -> pathlib.Path:
    apply_theme()
    df = load_sweep(sweep_csv)

    n_vals = sorted(df["n"].unique())
    palette = n_palette(n_vals)

    n_panels = len(ells)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 2.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    panel_labels = ["a", "b", "c", "d"]
    method_style = {"ciq": "-", "pciq": "--"}

    for panel_idx, (ax, ell) in enumerate(zip(axes, ells)):
        sub = df[(df["nu"] == nu) & (df["ell"] == ell) & (df["d"] == d) &
                 (df["method"].isin(["ciq", "pciq"]))]
        if sub.empty:
            ax.set_title(f"ℓ={ell} (no data)")
            continue

        for colour, n in zip(palette, n_vals):
            for method, ls in method_style.items():
                ndf = sub[(sub["n"] == n) & (sub["method"] == method)].sort_values("fidelity")
                if ndf.empty:
                    continue
                ax.plot(
                    ndf["fidelity"],
                    ndf["p_star_lowq"],
                    color=colour,
                    ls=ls,
                    linewidth=1.4,
                )

        ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$J$ (Lanczos / quadrature steps)")
        ax.set_title(
            f"({panel_labels[panel_idx]}) $\\ell = {ell}$",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylim(0.0, 0.52)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel(r"$p^*_\delta$ (lower quantile)")

    # Legend: n values + method linestyle
    legend_handles = []
    for colour, n in zip(palette, n_vals):
        legend_handles.append(
            mlines.Line2D([], [], color=colour, linewidth=1.4, label=f"$n={n}$")
        )
    legend_handles.append(
        mlines.Line2D([], [], color="gray", ls="-", linewidth=1.4, label="CIQ")
    )
    legend_handles.append(
        mlines.Line2D([], [], color="gray", ls="--", linewidth=1.4, label="PCIQ")
    )
    axes[-1].legend(handles=legend_handles, fontsize=7, loc="upper right", framealpha=0.8)

    fig.suptitle(rf"F3: CIQ vs PCIQ — Matérn-{nu}", fontsize=9, y=1.01)
    plt.tight_layout()
    base = pathlib.Path(__file__).parent
    save(outname, base_path=base, show=show, overwrite=True)
    return base / "figs" / f"{outname}.pdf"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sweep_csv", type=pathlib.Path)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = render(args.sweep_csv, show=args.show)
    print(f"Saved: {out}")
