"""Figure F1 — Bound sufficiency (4 subplots: RFF / LRFF / CIQ / PCIQ).

Reads: one Stage-1 sweep CSV (produced by sweeps/matern_bayes/run_sweep.py).
Writes: figures/matern_bayes/figs/f1_bound_sufficiency.pdf

Layout
------
1 row × 4 columns.  Each column = one method.
x-axis : fidelity_rescaled  (D/n_eff² for RFF/LRFF; J/√n·log n for CIQ;
                              J/n^{3/8}·log n for PCIQ)
y-axis : p_star_lowq  (δ-lower quantile of p*)

One line per n value (flare palette, matching existing rejection-rate figures).
Shaded ±p_star_err band (Guard G5: conservative Imhof error bound).
Reference lines:
  — horizontal dashed at p* = ½  (random-guess baseline)
  — vertical dashed at x = 1     (derived convergence bound)

Usage
-----
    python -m figures.matern_bayes.f1_bound_sufficiency \\
        sweeps/matern_bayes/output/matern_bayes_d1_<hash>.csv

Or call render(sweep_csv) from another script or test.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sweeps.matern_bayes.config import METHODS, NUS, ELLS
from ._common import (
    apply_theme,
    load_sweep,
    method_label,
    nu_label,
    ell_label,
    n_palette,
    plot_p_star_curves,
    save,
)

# Figure configuration for F1
_FIGURE_NU: float = 1.5   # Matérn-3/2, as per spec
_FIGURE_ELL: float = 0.2  # NOTE: if 0.2 absent, fall back to nearest ell


def _nearest_ell(df: pd.DataFrame, target: float = 0.2) -> float:
    """Return the ell value in df closest to target."""
    ells = sorted(df["ell"].unique())
    return min(ells, key=lambda e: abs(e - target))


def render(
    sweep_csv: pathlib.Path | str,
    nu: float = _FIGURE_NU,
    ell_target: float = _FIGURE_ELL,
    d: int = 1,
    y_col: str = "p_star_lowq",
    outname: str = "f1_bound_sufficiency",
    show: bool = False,
) -> pathlib.Path:
    """Render F1 and return the path to the saved PDF.

    Parameters
    ----------
    sweep_csv  : path to the Stage-1 CSV
    nu         : Matérn smoothness to plot (default 1.5 = Matérn-3/2)
    ell_target : lengthscale (0.2 preferred; nearest available used if absent)
    d          : input dimension (filter)
    y_col      : "p_star_lowq" (default) or "tv_uppq" for the TV variant
    outname    : output PDF stem
    show       : display figure interactively
    """
    apply_theme()

    df = load_sweep(sweep_csv)

    # --- Filter to the F1 config ---
    ell = _nearest_ell(df, ell_target)
    df_f1 = df[(df["nu"] == nu) & (df["ell"] == ell) & (df["d"] == d)].copy()

    if df_f1.empty:
        raise ValueError(
            f"No rows match nu={nu}, ell={ell}, d={d} in {sweep_csv}.\n"
            f"Available: nu={sorted(df['nu'].unique())}, "
            f"ell={sorted(df['ell'].unique())}, d={sorted(df['d'].unique())}"
        )

    n_vals = sorted(df_f1["n"].unique())
    palette = n_palette(n_vals)

    methods_in_data = [m for m in METHODS if m in df_f1["method"].unique()]
    n_methods = len(methods_in_data)
    if n_methods == 0:
        raise ValueError(f"No methods found in {sweep_csv} for the F1 filter.")

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(3.0 * n_methods, 3.0),
        sharey=True,
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_in_data):
        sub = df_f1[df_f1["method"] == method]
        plot_p_star_curves(
            ax, sub, n_vals, palette,
            y_col=y_col,
            show_band=True,
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel(_x_label(method))
        ax.set_title(method_label(method), fontsize=10, fontweight="bold")
        ax.set_xlim(left=min(sub["fidelity_rescaled"].min() * 0.8, 0.05))
        ax.set_ylim(0.0, 0.52)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    # y-label on leftmost axis
    y_axis_label = (
        r"$p^*_\delta$ (lower quantile)"
        if y_col == "p_star_lowq"
        else r"TV upper quantile"
    )
    axes[0].set_ylabel(y_axis_label)

    # Legend on rightmost axis
    axes[-1].legend(
        fontsize=7, loc="upper right",
        framealpha=0.8, handlelength=1.2,
    )

    fig.suptitle(
        rf"F1: bound sufficiency — Matérn-{nu}, $\ell={ell}$, $d={d}$",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()

    base = pathlib.Path(__file__).parent
    save(outname, base_path=base, show=show, overwrite=True)
    return base / "figs" / f"{outname}.pdf"


def _x_label(method: str) -> str:
    label_map = {
        "rff":  r"$D / n_\mathrm{eff}^2$",
        "lrff": r"$D / n_\mathrm{eff}^2$",
        "ciq":  r"$J / (\sqrt{n}\,\log n)$",
        "pciq": r"$J / (n^{3/8}\,\log n)$",
    }
    return label_map.get(method.lower(), "fidelity (rescaled)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render Figure F1 (bound sufficiency) from a Stage-1 CSV."
    )
    parser.add_argument("sweep_csv", type=pathlib.Path,
                        help="Path to the Stage-1 sweep CSV.")
    parser.add_argument("--nu", type=float, default=_FIGURE_NU)
    parser.add_argument("--ell", type=float, default=_FIGURE_ELL)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--tv", action="store_true",
                        help="Plot TV upper quantile instead of p* lower quantile.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    y_col = "tv_uppq" if args.tv else "p_star_lowq"
    suffix = "_tv" if args.tv else ""
    out = render(
        args.sweep_csv,
        nu=args.nu,
        ell_target=args.ell,
        d=args.d,
        y_col=y_col,
        outname=f"f1_bound_sufficiency{suffix}",
        show=args.show,
    )
    print(f"Saved: {out}")
