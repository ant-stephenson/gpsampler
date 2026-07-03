"""Stage 1 — Matérn Bayes-decision comparison sweep.

Iterates over (method, n, ν, ℓ, d, fidelity) and writes one tidy long-format
CSV per run, plus a companion manifest JSON.  Figure scripts (Stage 2) consume
these files without recomputing Bayes errors.

Usage
-----
    python -m sweeps.matern_bayes.run_sweep [--smoke] [--seed SEED]
                                             [--methods rff lrff ciq pciq]
                                             [--d 1] [--outdir PATH]

Flags
-----
--smoke    Use SMOKE_* constants from config.py (small n, few fidelities, R=4)
           for quick end-to-end tests.
--seed     Integer random seed (default 42).
--methods  Subset of {rff, lrff, ciq, pciq} to run.
--d        Input dimension (default 1; use 2 for robustness panel FA).
--outdir   Output directory (default: sweeps/matern_bayes/output/).

Expected runtime (indicative, single CPU)
-----------------------------------------
Smoke sweep (n ∈ {64,128}, R=4, 2 fidelities):          < 2 minutes
Core sweep  (n ∈ {256,…,2048}, R=50/1, 10 fidelities): ≈ 6–12 hours
  — dominated by CIQ/PCIQ at n=2048 (matsqrt is O(n³ J))

Output columns
--------------
method, n, nu, ell, d, fidelity, fidelity_rescaled,
p_star, tv, p_star_lowq, tv_uppq, p_star_err,
n_eff, kappa_eta, flops, R, seed
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.linalg as linalg

# ---------------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------------
# Add repo root to sys.path so the package is importable when this script is
# run directly from any working directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpsampler.maths import k_se, k_mat
from gpsampler.bayes_validation import (
    gaussian_bayes_error,
    realised_cov_ciq,
)
from gpsampler.samplers import matsqrt, NystromPreconditioner
from gpsampler.leverage_reweighted_rff import (
    kernel_matrix as _km,
    recursive_rls as _rrls,
    nystrom_factor as _nf,
    ApproxLeverage as _AL,
    compute_sir_pool,
    resample_from_pool,
)

from .config import (
    SIGMA_F2,
    SIGMA_XI2,
    ETA,
    METHODS,
    R_RAND,
    R_DET,
    DET_METHODS,
    DELTA,
    NS,
    NUS,
    ELLS,
    D_CORE,
    SMOKE_NS,
    SMOKE_N_FIDELITY,
    SMOKE_R_RAND,
    SMOKE_R_DET,
    N_FIDELITY,
    fidelity_grid,
    fidelity_bound,
)
from .flops import flops as compute_flops
from .manifest import build_manifest, write_manifest


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_DEFAULT_OUTDIR = pathlib.Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _build_K(x: np.ndarray, nu: float, ell: float, sigma: float = 1.0) -> np.ndarray:
    """Stationary kernel matrix with k(0) = sigma."""
    if nu >= 1000.0:  # RBF / squared-exponential
        return k_se(x, x, sigma, ell)
    return k_mat(x, x, sigma, ell, nu=nu)


def _kernel_kind(nu: float) -> tuple[str, float]:
    """Return (kind, nu_effective) for leverage_reweighted_rff API."""
    if nu >= 1000.0:
        return "rbf", 1.5   # nu unused for rbf
    return "matern", float(nu)


# ---------------------------------------------------------------------------
# Effective dimension and condition number
# ---------------------------------------------------------------------------

def _neff_hutchinson(
    K: np.ndarray,
    L_xi: np.ndarray,
    n_probes: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate Tr(K K_ξ^{-1}) via Hutchinson trace estimator.

    Reuses the Cholesky factor L of K_ξ already held by the caller.
    """
    rng = rng or np.random.default_rng()
    n = K.shape[0]
    total = 0.0
    for _ in range(n_probes):
        v = rng.standard_normal(n)
        total += float(np.dot(K @ v, linalg.cho_solve((L_xi, True), v)))
    return total / n_probes


def _kappa_eta(K: np.ndarray, eta: float, noise_var: float) -> float:
    """Condition number κ(K_ηξ) = λ_max / λ_min of K + η σ² I."""
    n = K.shape[0]
    K_etaxi = K + eta * noise_var * np.eye(n)
    eigs = np.linalg.eigvalsh(K_etaxi)
    return float(eigs[-1] / max(eigs[0], 1e-300))


# ---------------------------------------------------------------------------
# Guard G2 check
# ---------------------------------------------------------------------------

def _assert_not_identical(K_xi: np.ndarray, Khat_xi: np.ndarray) -> None:
    """Guard G2: K̂_ξ must not be the *same object* or bitwise-identical copy of K_ξ.

    We use exact array equality (not np.allclose) because a near-perfect sampler
    legitimately produces K̂_ξ ≈ K_ξ: for CIQ at large J, K̂_ξ → K_ξ exactly.
    The guard is intended to catch bugs where K_ξ is passed *instead of* K̂_ξ
    (same object, or trivially copied without computing the realised covariance).

    We also reject NaN covariances (Guard G2 in bayes_validation.py).
    """
    if np.any(np.isnan(Khat_xi)):
        raise ValueError(
            "G2 violation: K̂_ξ contains NaN values.  "
            "Use realised_cov_rff / realised_cov_ciq to compute the analytic "
            "realised covariance from the sampler's actual features."
        )
    if Khat_xi is K_xi or np.array_equal(K_xi, Khat_xi):
        raise ValueError(
            "G2 violation: K̂_ξ is bitwise-identical to K_ξ.  "
            "The realised covariance must be built from the sampler's actual "
            "realised features, not the true kernel matrix."
        )


# ---------------------------------------------------------------------------
# CIQ / PCIQ realised covariance builders (deterministic given K)
# ---------------------------------------------------------------------------

def _khat_ciq(K: np.ndarray, eta: float, noise_var: float, J: int) -> np.ndarray:
    """K̂_ξ = MMᵀ + (1-η)σ²I,  M = matsqrt(K + ησ²I, Q=J)."""
    n = K.shape[0]
    K_etaxi = K + eta * noise_var * np.eye(n)
    M = matsqrt(K_etaxi, J, J, eta * noise_var)
    return realised_cov_ciq(M, eta, noise_var)


def _khat_pciq(K: np.ndarray, eta: float, noise_var: float, J: int,
               rng: np.random.Generator) -> np.ndarray:
    """K̂_ξ via preconditioned matsqrt: M_pre = P^{1/2} matsqrt(W, J), W = P^{-1/2} K_ηξ P^{-1/2}."""
    n = K.shape[0]
    K_etaxi = K + eta * noise_var * np.eye(n)
    pre = NystromPreconditioner(K, eta=eta, noise_var=noise_var,
                                rank=max(1, int(np.sqrt(n))), rng=rng)

    Pinvsqrt = np.column_stack([pre.apply_inv_sqrt(np.eye(n)[:, j]) for j in range(n)])
    W = 0.5 * (Pinvsqrt.T @ K_etaxi @ Pinvsqrt)
    W += W.T; W *= 0.5  # symmetrise
    w_min = float(max(np.linalg.eigvalsh(W).min(), 1e-10))
    M_W = matsqrt(W, J, J, w_min)

    Psqrt = np.column_stack([pre.apply_sqrt(np.eye(n)[:, j]) for j in range(n)])
    return realised_cov_ciq(Psqrt @ M_W, eta, noise_var)


# ---------------------------------------------------------------------------
# Per-config lrff setup (cached across fidelity values and trials)
# ---------------------------------------------------------------------------

def _lrff_setup(x: np.ndarray, nu: float, ell: float, noise_var: float):
    """Build ApproxLeverage callable and related objects for an (x,ν,ℓ) config.

    Returns (K_unit, alpha_fn).  K_unit is the unit-scale kernel matrix (k(0)=1).
    """
    kind, nu_eff = _kernel_kind(nu)
    K_unit = _km(x, kind=kind, ell=ell, nu=nu_eff)
    S = _rrls(K_unit, lam=noise_var, rng=np.random.default_rng(99))
    B = _nf(K_unit, S)
    alpha_fn = _AL(x, B, noise_var)
    return K_unit, alpha_fn


# ---------------------------------------------------------------------------
# Single-config sweep
# ---------------------------------------------------------------------------

def _sweep_config(
    method: str,
    n: int,
    nu: float,
    ell: float,
    d: int,
    seed: int,
    R: int,
    n_fidelity: int,
    verbose: bool = True,
    dtype: type = np.float64,
    chunk_size: int = 512,
) -> list[dict]:
    """Run BV sweep for one (method, n, ν, ℓ, d) configuration.

    Returns a list of row dicts (one per fidelity value).
    """
    rng = np.random.default_rng(seed)
    sigma = SIGMA_F2
    noise_var = SIGMA_XI2
    eta = ETA

    # ------------------------------------------------------------------
    # Input locations: uniform on [0,1]^d
    # ------------------------------------------------------------------
    x = rng.uniform(0.0, 1.0, (n, d))

    # ------------------------------------------------------------------
    # True kernel and observation covariance
    # ------------------------------------------------------------------
    K = _build_K(x, nu, ell, sigma)                         # K (no noise)
    K_xi = K + noise_var * np.eye(n)                        # K_ξ = K + σ²I
    L_xi = linalg.cholesky(K_xi, lower=True)               # L_xi for cho_solve

    # ------------------------------------------------------------------
    # Effective dimension and condition number (once per config)
    # ------------------------------------------------------------------
    n_eff = _neff_hutchinson(K, L_xi, n_probes=30, rng=np.random.default_rng(seed + 1))
    kappa = _kappa_eta(K, eta, noise_var)

    # kernel kind — constant per (nu, ell), used by rff/lrff branches
    kind, nu_eff = _kernel_kind(nu)

    # ------------------------------------------------------------------
    # LRFF-specific setup (skip for other methods — avoids O(n²) kernel)
    # ------------------------------------------------------------------
    lrff_alpha_fn = None
    if method == "lrff":
        _, lrff_alpha_fn = _lrff_setup(x, nu, ell, noise_var)

    # ------------------------------------------------------------------
    # Fidelity grid
    # ------------------------------------------------------------------
    grid = fidelity_grid(method, n, n_eff=n_eff, n_points=n_fidelity)

    rows: list[dict] = []

    for fid in grid:
        # --- LRFF pool cache (expensive, done once per fidelity) ---
        lrff_pool_cache = None
        if method == "lrff":
            lrff_pool_cache = compute_sir_pool(
                n_freq=fid // 2,
                d=d,
                kind=kind,
                ell=ell,
                nu=nu_eff,
                alpha_fn=lrff_alpha_fn,
                rng=np.random.default_rng(seed + fid * 1_000_000),
                pool_factor=5,
                pool_min=4000,
            )

        tv_vals: list[float] = []
        p_star_vals: list[float] = []
        p_star_err_vals: list[float] = []

        for r in range(R):
            trial_rng = np.random.default_rng(seed + fid * 10_000 + r)

            # --- build K̂_xi ---
            if method == "rff":
                # Chunked ΦΦᵀ: accumulate (2σ/D)·Σ_b [cos(vb)cos(vb)ᵀ + sin(vb)sin(vb)ᵀ]
                # without ever materialising the n×D feature matrix.
                half = fid // 2
                # K_acc always float64: n×n is cheap; float32 accumulation of
                # many outer products causes compounding rounding errors.
                # dtype only controls the large per-chunk (n×b) intermediates.
                K_acc = np.zeros((n, n), dtype=np.float64)
                for start in range(0, half, chunk_size):
                    b = min(chunk_size, half - start)
                    if kind == "rbf":
                        omega_b = trial_rng.multivariate_normal(
                            np.zeros(d), np.eye(d) / ell ** 2, b)
                    else:
                        g = trial_rng.standard_normal((b, d))
                        u = trial_rng.chisquare(2.0 * nu_eff, size=(b, 1))
                        omega_b = (g / ell) * np.sqrt(2.0 * nu_eff / u)
                    v = (x @ omega_b.T).astype(dtype)   # (n, b) — dtype controls memory
                    cv, sv = np.cos(v), np.sin(v)
                    K_acc += cv @ cv.T + sv @ sv.T      # numpy upcasts to float64
                Khat_xi = (2.0 * sigma / fid) * K_acc + noise_var * np.eye(n)

            elif method == "lrff":
                # Chunked weighted ΦΦᵀ: K̂_ξ = σ·Σ_b [(g_b⊙cos_b)(g_b⊙cos_b)ᵀ
                #                                    + (g_b⊙sin_b)(g_b⊙sin_b)ᵀ] + σ²I
                # g_j = sqrt(Z_hat / (n_freq · α_j)) are the importance weights.
                assert lrff_alpha_fn is not None and lrff_pool_cache is not None
                n_freq = fid // 2
                _pool, _a_pool, _Z_hat = lrff_pool_cache
                W, alpha_sel, Z_hat = resample_from_pool(
                    _pool, _a_pool, _Z_hat, n_freq, trial_rng)
                g = np.sqrt(Z_hat / (n_freq * alpha_sel))  # (n_freq,) importance weights
                K_acc = np.zeros((n, n), dtype=np.float64)
                for start in range(0, n_freq, chunk_size):
                    b = min(chunk_size, n_freq - start)
                    W_b = W[start:start + b]
                    g_b = g[start:start + b]
                    v = (x @ W_b.T).astype(dtype)          # (n, b)
                    cv_w = (np.cos(v) * g_b).astype(dtype)
                    sv_w = (np.sin(v) * g_b).astype(dtype)
                    K_acc += cv_w @ cv_w.T + sv_w @ sv_w.T  # upcasts to float64
                Khat_xi = sigma * K_acc + noise_var * np.eye(n)

            elif method == "ciq":
                # Deterministic: ignore trial_rng
                Khat_xi = _khat_ciq(K, eta, noise_var, fid)

            elif method == "pciq":
                # Deterministic: pass rng only for NystromPreconditioner landmarks
                Khat_xi = _khat_pciq(K, eta, noise_var, fid,
                                     rng=np.random.default_rng(seed + 777))
            else:
                raise ValueError(f"Unknown method {method!r}")

            # --- Guard G2: K̂_ξ ≠ K_ξ ---
            _assert_not_identical(K_xi, Khat_xi)

            # --- Bayes error (Guard G1: Imhof, not MC) ---
            res = gaussian_bayes_error(K_xi, Khat_xi)
            tv_vals.append(res["tv"])
            p_star_vals.append(res["p_star"])
            p_star_err_vals.append(res["p_star_err"])

        # --- Guard G4: report (1−δ) upper quantile, not mean ---
        tv_uppq = float(np.quantile(tv_vals, 1.0 - DELTA))
        p_star_lowq = float(np.quantile(p_star_vals, DELTA))
        tv_mean = float(np.mean(tv_vals))
        p_star_mean = float(np.mean(p_star_vals))
        p_star_err_mean = float(np.mean(p_star_err_vals))

        # --- fidelity rescaled ---
        bound = fidelity_bound(method, n, n_eff=n_eff)
        fidelity_rescaled = fid / bound if bound > 0 else float("nan")

        # --- FLOP count ---
        fl = compute_flops(method, n, fid, d=d, n_eff=n_eff)

        rows.append(
            {
                "method": method,
                "n": n,
                "nu": nu,
                "ell": ell,
                "d": d,
                "fidelity": fid,
                "fidelity_rescaled": fidelity_rescaled,
                "p_star": p_star_mean,
                "tv": tv_mean,
                "p_star_lowq": p_star_lowq,
                "tv_uppq": tv_uppq,
                "p_star_err": p_star_err_mean,
                "n_eff": n_eff,
                "kappa_eta": kappa,
                "flops": fl,
                "R": R,
                "seed": seed,
            }
        )

        if verbose:
            print(
                f"  [{method.upper():5s}] n={n:4d} nu={nu:4.1f} ell={ell} d={d}"
                f"  fid={fid:6d}  tv_uppq={tv_uppq:.4f}  p*={p_star_mean:.4f}"
                f"  flops={fl:.2e}",
                flush=True,
            )

    return rows


# ---------------------------------------------------------------------------
# Top-level sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    methods: tuple[str, ...] = METHODS,
    ns: tuple[int, ...] = NS,
    nus: tuple[float, ...] = NUS,
    ells: tuple[float, ...] = ELLS,
    d: int = D_CORE,
    seed: int = 42,
    n_fidelity: int = N_FIDELITY,
    R_rand: int = R_RAND,
    R_det: int = R_DET,
    outdir: pathlib.Path = _DEFAULT_OUTDIR,
    verbose: bool = True,
    tag: str = "",
    dtype: type = np.float64,
    chunk_size: int = 512,
) -> pathlib.Path:
    """Run the full BV comparison sweep and persist results.

    Parameters
    ----------
    methods    : subset of METHODS to run
    ns         : n values to sweep
    nus        : ν values to sweep
    ells       : ℓ values to sweep
    d          : input dimension
    seed       : global random seed
    n_fidelity : number of fidelity points per curve
    R_rand     : realisations for randomised methods (rff, lrff)
    R_det      : realisations for deterministic methods (ciq, pciq)
    outdir     : directory for output CSV and manifest
    verbose    : print progress
    tag        : optional string appended to the output filename stem

    Returns
    -------
    Path to the output CSV file.
    """
    # --- Guard G3: verify CG/Lanczos absent from method list ---
    for m in methods:
        if m not in METHODS:
            raise ValueError(
                f"G3 violation: method {m!r} is not in the allowed registry "
                f"{METHODS}.  CG/Lanczos are excluded because their outputs "
                "are non-Gaussian (Guard G3)."
            )

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "methods": list(methods),
        "ns": list(ns),
        "nus": [float(v) for v in nus],
        "ells": list(ells),
        "d": d,
        "n_fidelity": n_fidelity,
        "R_rand": R_rand,
        "R_det": R_det,
        "SIGMA_F2": SIGMA_F2,
        "SIGMA_XI2": SIGMA_XI2,
        "ETA": ETA,
        "DELTA": DELTA,
    }

    manifest = build_manifest(cfg, seed)
    cfg_hash = manifest["config_hash"]

    stem = f"matern_bayes_d{d}_{cfg_hash}"
    if tag:
        stem = f"{stem}_{tag}"
    csv_path = outdir / f"{stem}.csv"
    manifest_path = outdir / f"{stem}_manifest.json"
    write_manifest(manifest, manifest_path)

    all_rows: list[dict] = []

    t0 = time.time()
    for method in methods:
        R = R_det if method in DET_METHODS else R_rand
        for n in ns:
            for nu in nus:
                for ell in ells:
                    if verbose:
                        print(
                            f"\n=== {method.upper():<5}  n={n}  nu={nu}  ell={ell}  d={d}  R={R} ===",
                            flush=True,
                        )
                    rows = _sweep_config(
                        method=method,
                        n=n,
                        nu=nu,
                        ell=ell,
                        d=d,
                        seed=seed,
                        R=R,
                        n_fidelity=n_fidelity,
                        verbose=verbose,
                        dtype=dtype,
                        chunk_size=chunk_size,
                    )
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nSweep complete in {elapsed:.1f}s -> {csv_path}", flush=True)

    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Matérn Bayes-decision comparison sweep (Stage 1)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke-test overrides (small n, few fidelities, R=4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Methods to run (subset of rff lrff ciq pciq).",
    )
    parser.add_argument(
        "--d", type=int, default=D_CORE, help="Input dimension (default 1)."
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=_DEFAULT_OUTDIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Optional tag appended to filename."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-config progress output."
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64",
        help="Dtype for RFF feature computation (float32 halves per-chunk memory).",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512,
        help="Frequencies per chunk in RFF Khat accumulation (default 512).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.smoke:
        ns = SMOKE_NS
        n_fidelity = SMOKE_N_FIDELITY
        R_rand = SMOKE_R_RAND
        R_det = SMOKE_R_DET
        tag = args.tag or "smoke"
    else:
        ns = NS
        n_fidelity = N_FIDELITY
        R_rand = R_RAND
        R_det = R_DET
        tag = args.tag

    out = run_sweep(
        methods=tuple(args.methods),
        ns=ns,
        nus=NUS,
        ells=ELLS,
        d=args.d,
        seed=args.seed,
        n_fidelity=n_fidelity,
        R_rand=R_rand,
        R_det=R_det,
        outdir=args.outdir,
        verbose=not args.quiet,
        tag=tag,
        dtype=np.float32 if args.dtype == "float32" else np.float64,
        chunk_size=args.chunk_size,
    )
    print(f"Output: {out}")
