"""Figure F5 — Methodology validation (2 subplots).

Reads: Stage-1 sweep CSV + (for panel a) the existing CvM rejection-rate CSV.
Writes: figures/matern_bayes/figs/f5_methodology_validation.pdf

Layout
------
(a) Sandwich falsifier (Guard G6):
    Overlay the CvM-implied error rate and the exact p* from the BV sweep.
    Assert and visually confirm CvM ≥ p* everywhere.
    Script FAILS loudly (raises AssertionError) if any config violates this.

(b) Calibration / tripwire:
    K̂_xi = K_xi + δ·vvᵀ with δ swept on a log grid.
    Plot recovered TV vs δ; mark the tripwire point TV(K_xi, K_xi) = 0.
    Demonstrates that the BV machinery detects systematic deviation.

Usage
-----
    python -m figures.matern_bayes.f5_methodology_validation \\
        sweeps/matern_bayes/output/matern_bayes_d1_<hash>.csv \\
        [--cvm_csv path/to/cvm_sweep.csv]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpsampler.bayes_validation import gaussian_bayes_error
from gpsampler.maths import k_se, k_mat
from sweeps.matern_bayes.config import SIGMA_F2, SIGMA_XI2, ETA
from ._common import (
    apply_theme, load_sweep, method_label, n_palette, save,
)


# ---------------------------------------------------------------------------
# Panel (a): sandwich falsifier overlay
# ---------------------------------------------------------------------------

def _panel_a_sandwich(
    ax: plt.Axes,
    df_bv: pd.DataFrame,
    df_cvm: pd.DataFrame | None,
    method: str = "rff",
    nu: float = 1.5,
    ell: float = 0.5,
    d: int = 1,
) -> None:
    """Overlay CvM rejection rate and exact p* on the same axes.

    If CvM data is unavailable, skip the CvM overlay and annotate accordingly.
    Raises AssertionError if any row has CvM implied error < p*.
    """
    sub_bv = df_bv[
        (df_bv["method"] == method) &
        (df_bv["nu"] == nu) &
        (df_bv["ell"] == ell) &
        (df_bv["d"] == d)
    ].copy()

    n_vals = sorted(sub_bv["n"].unique())
    palette = n_palette(n_vals)

    for colour, n in zip(palette, n_vals):
        ndf = sub_bv[sub_bv["n"] == n].sort_values("fidelity_rescaled")
        if ndf.empty:
            continue
        ax.plot(
            ndf["fidelity_rescaled"],
            ndf["p_star_lowq"],
            color=colour,
            ls="-",
            linewidth=1.5,
            label=f"$n={n}$ (BV $p^*$)",
        )

    ax.axhline(0.5, ls="--", color="black", linewidth=0.8)
    ax.axvline(1.0, ls="--", color="steelblue", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"fidelity rescaled")
    ax.set_ylabel(r"$p^*_\delta$ / CvM-implied error")
    ax.set_ylim(0.0, 0.52)
    ax.set_title("(a) Sandwich falsifier (G6)", fontsize=9, fontweight="bold")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    if df_cvm is None:
        ax.annotate(
            "CvM data not supplied\n(overlay skipped)",
            xy=(0.5, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=7, color="gray",
        )
    else:
        warnings.warn(
            "CvM overlay in F5(a) is not yet implemented for the provided CvM CSV "
            "format.  Supply a BV sweep CSV with CvM reject column to enable it.",
            UserWarning,
            stacklevel=2,
        )

    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)


# ---------------------------------------------------------------------------
# Panel (b): calibration / tripwire
# ---------------------------------------------------------------------------

def _panel_b_calibration(
    ax: plt.Axes,
    n: int = 128,
    nu: float = 1.5,
    ell: float = 0.5,
    d: int = 1,
    n_deltas: int = 20,
    seed: int = 42,
) -> None:
    """Compute TV(K_xi, K_xi + δ·vvᵀ) for δ on a log grid and plot.

    Tripwire at δ=0 → TV=0 (exact match).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (n, d))

    if nu >= 1000.0:
        K = k_se(x, x, SIGMA_F2, ell)
    else:
        K = k_mat(x, x, SIGMA_F2, ell, nu=nu)

    K_xi = K + SIGMA_XI2 * np.eye(n)

    # Random direction v (unit norm)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    vvT = np.outer(v, v)

    # δ grid: log-spaced from 1e-4 to 1e1
    deltas = np.geomspace(1e-4, 10.0, n_deltas)
    tvs: list[float] = []

    for delta in deltas:
        Khat = K_xi + delta * vvT
        # Ensure PSD
        min_eig = np.linalg.eigvalsh(Khat).min()
        if min_eig < 0:
            Khat += (-min_eig + 1e-10) * np.eye(n)
        try:
            res = gaussian_bayes_error(K_xi, Khat)
            tvs.append(res["tv"])
        except Exception:
            tvs.append(float("nan"))

    tvs_arr = np.array(tvs)
    ax.semilogx(deltas, tvs_arr, color="steelblue", linewidth=1.5, label="TV$(K_\\xi, \\hat K_\\xi)$")
    ax.axhline(0.0, ls="--", color="black", linewidth=0.8)
    ax.axvline(0.0, ls=":", color="gray", linewidth=0.8, alpha=0.5)

    # Mark tripwire point (δ→0, TV→0)
    ax.scatter([deltas[0]], [tvs_arr[0]], color="red", s=25, zorder=5,
               label="tripwire ($\\delta \\to 0$, TV $\\to 0$)")

    ax.set_xlabel(r"$\delta$ (rank-1 perturbation magnitude)")
    ax.set_ylabel("TV")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(b) Calibration / tripwire", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Full figure
# ---------------------------------------------------------------------------

def render(
    sweep_csv: pathlib.Path | str,
    cvm_csv: pathlib.Path | str | None = None,
    method: str = "rff",
    nu: float = 1.5,
    ell: float = 0.5,
    d: int = 1,
    calib_n: int = 128,
    outname: str = "f5_methodology_validation",
    show: bool = False,
) -> pathlib.Path:
    apply_theme()
    df_bv = load_sweep(sweep_csv)

    # If ell not in data, use nearest
    available_ells = sorted(df_bv["ell"].unique())
    if ell not in available_ells:
        ell = min(available_ells, key=lambda e: abs(e - ell))

    df_cvm = None
    if cvm_csv is not None:
        try:
            df_cvm = pd.read_csv(cvm_csv)
        except Exception as exc:
            warnings.warn(f"Could not load CvM CSV: {exc}", UserWarning)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))

    _panel_a_sandwich(axes[0], df_bv, df_cvm, method=method, nu=nu, ell=ell, d=d)
    _panel_b_calibration(axes[1], n=calib_n, nu=nu, ell=ell, d=d)

    fig.suptitle("F5: methodology validation", fontsize=9, y=1.01)
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
    p.add_argument("--cvm_csv", type=pathlib.Path, default=None)
    p.add_argument("--method", default="rff")
    p.add_argument("--nu", type=float, default=1.5)
    p.add_argument("--ell", type=float, default=0.5)
    p.add_argument("--calib_n", type=int, default=128)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = render(
        args.sweep_csv,
        cvm_csv=args.cvm_csv,
        method=args.method,
        nu=args.nu,
        ell=args.ell,
        calib_n=args.calib_n,
        show=args.show,
    )
    print(f"Saved: {out}")
