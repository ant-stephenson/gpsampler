"""Figure FA — Robustness panel (3 subplots).

Reads: one or two Stage-1 CSVs (core d=1 + robustness d=2 or alt σ_ξ²).
Writes: figures/matern_bayes/figs/fa_robustness.pdf

Layout: 1 row × 3 columns.
  (a) F1 replicated at d=2 (RFF only for conciseness).
  (b) F1 at alternative σ_ξ² = 1e-3 (from a second sweep CSV, or same CSV
      if σ_ξ² column varies).
  (c) CIQ across ν, collapsing under √n·log n rescaling.

Usage
-----
    python -m figures.matern_bayes.fa_robustness \\
        core.csv [--robust_csv robust.csv]
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


def _panel_a_d2(ax: plt.Axes, df: pd.DataFrame, nu: float = 1.5, ell: float = 0.5) -> None:
    """F1-style curves at d=2 (RFF only)."""
    sub = df[(df["method"] == "rff") & (df["nu"] == nu) &
             (df["ell"] == ell) & (df["d"] == 2)]
    if sub.empty:
        available_ells = sorted(df[(df["d"] == 2)]["ell"].unique()) if not df[(df["d"]==2)].empty else []
        ax.set_title(f"(a) d=2 (no data; available ells: {available_ells})", fontsize=8)
        return

    n_vals = sorted(sub["n"].unique())
    palette = n_palette(n_vals)

    for colour, n in zip(palette, n_vals):
        ndf = sub[sub["n"] == n].sort_values("fidelity_rescaled")
        ax.plot(ndf["fidelity_rescaled"], ndf["p_star_lowq"], color=colour,
                linewidth=1.5, label=f"$n={n}$")

    ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
    ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D / n_\mathrm{eff}^2$  ($d=2$)")
    ax.set_ylabel(r"$p^*_\delta$")
    ax.set_ylim(0.0, 0.52)
    ax.set_title(f"(a) RFF, $d=2$, $\\ell={ell}$", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)


def _panel_b_alt_noise(ax: plt.Axes, df_alt: pd.DataFrame, nu: float = 1.5,
                       ell: float = 0.5) -> None:
    """F1-style curves at alternative σ_ξ² (from a second sweep CSV)."""
    sub = df_alt[(df_alt["method"] == "rff") & (df_alt["nu"] == nu) &
                 (df_alt["d"] == 1)]
    if sub.empty:
        ax.set_title("(b) alt σ_ξ² (no data)", fontsize=8)
        return

    # Use nearest ell
    available_ells = sorted(sub["ell"].unique())
    ell_use = min(available_ells, key=lambda e: abs(e - ell))
    sub = sub[sub["ell"] == ell_use]

    n_vals = sorted(sub["n"].unique())
    palette = n_palette(n_vals)

    for colour, n in zip(palette, n_vals):
        ndf = sub[sub["n"] == n].sort_values("fidelity_rescaled")
        ax.plot(ndf["fidelity_rescaled"], ndf["p_star_lowq"], color=colour,
                linewidth=1.5, label=f"$n={n}$")

    ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
    ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D / n_\mathrm{eff}^2$")
    ax.set_ylim(0.0, 0.52)
    ax.set_title(r"(b) RFF, alt $\sigma_\xi^2$", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)


def _panel_c_ciq_nu_collapse(ax: plt.Axes, df: pd.DataFrame, ell: float = 0.5,
                              d: int = 1) -> None:
    """CIQ curves across ν, all collapsing under √n·log n rescaling."""
    sub = df[(df["method"] == "ciq") & (df["ell"] == ell) & (df["d"] == d)]
    if sub.empty:
        ax.set_title("(c) CIQ collapse (no data)", fontsize=8)
        return

    nu_vals = sorted(sub["nu"].unique())
    palette = list(sns.color_palette("viridis", n_colors=len(nu_vals)))

    n_vals = sorted(sub["n"].unique())
    n_max = max(n_vals)
    sub_n = sub[sub["n"] == n_max]

    for colour, nu in zip(palette, nu_vals):
        ndf = sub_n[sub_n["nu"] == nu].sort_values("fidelity_rescaled")
        ax.plot(ndf["fidelity_rescaled"], ndf["p_star_lowq"], color=colour,
                linewidth=1.5, label=nu_label(nu))

    ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
    ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$J / (\sqrt{n}\,\log n)$")
    ax.set_ylim(0.0, 0.52)
    ax.set_title(
        rf"(c) CIQ, $n={n_max}$, $\ell={ell}$: collapse under $\sqrt{{n}}\log n$",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Full figure
# ---------------------------------------------------------------------------

def render(
    sweep_csv: pathlib.Path | str,
    robust_csv: pathlib.Path | str | None = None,
    nu: float = 1.5,
    ell: float = 0.5,
    outname: str = "fa_robustness",
    show: bool = False,
) -> pathlib.Path:
    apply_theme()
    df = load_sweep(sweep_csv)

    # For panel (b): use robust_csv if supplied, else same df (data may be absent)
    df_alt = load_sweep(robust_csv) if robust_csv is not None else df

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))

    _panel_a_d2(axes[0], df, nu=nu, ell=ell)
    _panel_b_alt_noise(axes[1], df_alt, nu=nu, ell=ell)
    _panel_c_ciq_nu_collapse(axes[2], df, ell=ell)

    fig.suptitle("FA: robustness", fontsize=9, y=1.01)
    plt.tight_layout()

    base = pathlib.Path(__file__).parent
    save(outname, base_path=base, show=show, overwrite=True)
    return base / "figs" / f"{outname}.pdf"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sweep_csv", type=pathlib.Path)
    p.add_argument("--robust_csv", type=pathlib.Path, default=None)
    p.add_argument("--nu", type=float, default=1.5)
    p.add_argument("--ell", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = render(
        args.sweep_csv,
        robust_csv=args.robust_csv,
        nu=args.nu,
        ell=args.ell,
        show=args.show,
    )
    print(f"Saved: {out}")
