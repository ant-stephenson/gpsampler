"""Shared plot theme for Matérn Bayes-decision comparison figures (F1–F5, FA).

Matches the existing rejection-rate figure style from analyse_sweep_output.py:
  - seaborn relplot with col="ell" or col="nu", hue="n"
  - log-log axes
  - flare palette (n_colors = number of n values)
  - horizontal reference lines (p* = ½, TV = 0)
  - vertical reference line at fidelity_rescaled = 1
  - LaTeX text renderer (falls back to matplotlib default if LaTeX absent)
  - PDF output via save_fig

Public API
----------
apply_theme()           : call once per figure script at import time
method_label(method)    : "RFF", "LRFF", "CIQ", "PCIQ"
nu_label(nu)            : "$\\nu = 0.5$", "$\\nu = \\infty$ (RBF)", etc.
load_sweep(path)        : load + validate a Stage-1 sweep CSV
plot_p_star_curves(ax, sub_df, n_vals, palette, delta)
save(base_path, name)   : save as PDF via save_fig
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from gpsampler.plotting import LaTeX, save_fig
from sweeps.matern_bayes.config import METHODS, DELTA


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_APPLIED = False


def apply_theme() -> None:
    """Set rcParams for the paper figure style.  Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    _APPLIED = True


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_METHOD_LABELS: dict[str, str] = {
    "rff": "RFF",
    "lrff": "LRFF",
    "ciq": "CIQ",
    "pciq": "PCIQ",
}


def method_label(method: str) -> str:
    return _METHOD_LABELS.get(method.lower(), method.upper())


def nu_label(nu: float) -> str:
    if nu >= 1000.0:
        return r"$\nu = \infty$ (RBF)"
    return rf"$\nu = {nu}$"


def ell_label(ell: float) -> str:
    return rf"$\ell = {ell}$"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {
    "method", "n", "nu", "ell", "d",
    "fidelity", "fidelity_rescaled",
    "p_star", "tv", "p_star_lowq", "tv_uppq", "p_star_err",
    "n_eff", "kappa_eta", "flops", "R", "seed",
}


def load_sweep(path: pathlib.Path | str) -> pd.DataFrame:
    """Load a Stage-1 CSV and assert the full column schema is present."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Sweep file {path} is missing columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

def n_palette(n_vals: Sequence[int]) -> list:
    """Flare palette matching n values — same as existing rejection-rate figures."""
    return list(sns.color_palette("flare", n_colors=len(n_vals)))


# ---------------------------------------------------------------------------
# Core curve-plotting helper
# ---------------------------------------------------------------------------

def plot_p_star_curves(
    ax: plt.Axes,
    sub_df: pd.DataFrame,
    n_vals: Sequence[int],
    palette: list,
    delta: float = DELTA,
    y_col: str = "p_star_lowq",  # or "tv_uppq"
    x_col: str = "fidelity_rescaled",
    show_band: bool = True,
    chol_tv: Optional[float] = None,
) -> None:
    """Plot one p* (or TV) convergence curve per n value on a given Axes.

    Parameters
    ----------
    ax        : matplotlib Axes (log-log expected)
    sub_df    : DataFrame filtered to a single (method, nu, ell, d) slice
    n_vals    : ordered list of n values for legend ordering
    palette   : colour list aligned with n_vals
    delta     : certificate level (for label)
    y_col     : "p_star_lowq" or "tv_uppq"
    x_col     : "fidelity_rescaled"
    show_band : if True, draw a shaded ±p_star_err band
    chol_tv   : if provided, draw horizontal Cholesky baseline TV=chol_tv
    """
    for colour, n in zip(palette, n_vals):
        ndf = sub_df[sub_df["n"] == n].sort_values(x_col)
        if ndf.empty:
            continue
        ax.plot(ndf[x_col], ndf[y_col], color=colour, label=f"$n={n}$", linewidth=1.5)
        if show_band and "p_star_err" in ndf.columns and y_col == "p_star_lowq":
            lo = np.clip(ndf[y_col] - ndf["p_star_err"], 0.0, 0.5)
            hi = np.clip(ndf[y_col] + ndf["p_star_err"], 0.0, 0.5)
            ax.fill_between(ndf[x_col], lo, hi, color=colour, alpha=0.15)

    # Reference lines
    if y_col in ("p_star", "p_star_lowq"):
        ax.axhline(0.5, ls="--", color="black", linewidth=0.8, label=r"$p^*=\frac{1}{2}$")
    elif y_col in ("tv", "tv_uppq"):
        ax.axhline(0.0, ls="--", color="black", linewidth=0.8, label="TV $= 0$")

    if chol_tv is not None:
        ax.axhline(chol_tv, ls=":", color="gray", linewidth=0.8, label="Cholesky")

    # Vertical bound line at fidelity_rescaled = 1
    ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7, label="bound $= 1$")


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

_FIGS_DIR = pathlib.Path(__file__).parent / "figs"


def save(
    name: str,
    base_path: Optional[pathlib.Path] = None,
    show: bool = False,
    overwrite: bool = True,
) -> None:
    """Save current figure as a PDF.

    Parameters
    ----------
    name      : filename stem (no extension)
    base_path : directory that CONTAINS a 'figs/' subfolder (default: figures/matern_bayes/)
    """
    if base_path is None:
        base_path = pathlib.Path(__file__).parent
    _FIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_fig(
        base_path,
        name,
        suffix="pdf",
        show=show,
        overwrite=overwrite,
    )
