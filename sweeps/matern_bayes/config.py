"""Fixed experimental design for the Matérn Bayes-decision comparison.

All hyperparameters below are set once and held constant across the paper's
figures F1–F5 and FA.  Change nothing here without updating the paper.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixed kernel and noise hyperparameters
# ---------------------------------------------------------------------------

SIGMA_F2: float = 1.0    # kernel output scale σ_f²
SIGMA_XI2: float = 1e-2  # observation noise σ_ξ²
ETA: float = 0.8         # noise-split η — repo default for CIQ/PCIQ

# ---------------------------------------------------------------------------
# Method registry (Guard G3: CG / Lanczos deliberately absent)
# ---------------------------------------------------------------------------

METHODS: tuple[str, ...] = ("rff", "lrff", "ciq", "pciq")

# Realisations per method type
R_RAND: int = 50   # RFF, LRFF — randomised
R_DET: int = 1     # CIQ, PCIQ — deterministic given K

# Deterministic methods (realised covariance does not depend on random seed)
DET_METHODS: frozenset[str] = frozenset({"ciq", "pciq"})

# ---------------------------------------------------------------------------
# High-probability certificate level
# ---------------------------------------------------------------------------

DELTA: float = 0.05   # TV upper quantile at 1 − δ = 0.95; p_star lower at δ

# ---------------------------------------------------------------------------
# Core experimental grid
# ---------------------------------------------------------------------------

NS: tuple[int, ...] = (256, 512, 1024, 2048)
NUS: tuple[float, ...] = (0.5, 1.5, 2.5, float("inf"))   # inf → RBF
ELLS: tuple[float, ...] = (0.1, 1.0)
D_CORE: int = 1    # input dimension for core sweeps
D_ROBUST: int = 2  # input dimension for robustness panel FA

# Number of fidelity grid points per curve (~10 geometric spacing)
N_FIDELITY: int = 10

# Alternative σ_ξ² for FA robustness panel (b): alternative noise level
SIGMA_XI2_ALT: float = 1e-3

# ---------------------------------------------------------------------------
# Smoke-test (small) overrides — used in tests only
# ---------------------------------------------------------------------------

SMOKE_NS: tuple[int, ...] = (64, 128)
SMOKE_N_FIDELITY: int = 2
SMOKE_R_RAND: int = 4
SMOKE_R_DET: int = 1


def r_for_method(method: str) -> int:
    """Return R (number of realisations) appropriate for method."""
    return R_DET if method in DET_METHODS else R_RAND


def fidelity_bound(method: str, n: int, n_eff: float | None = None) -> float:
    """Return the derived convergence-bound fidelity for rescaling the x-axis.

    This is the denominator used in `fidelity_rescaled`:

        RFF / LRFF : n_eff²                (Proposition "Leverage bound")
        CIQ        : √n · log n            (Proposition "CIQ bound")
        PCIQ       : n^{3/8} · log n       (Proposition "PCIQ preconditioned bound")

    Parameters
    ----------
    method : one of METHODS
    n      : number of training points
    n_eff  : effective dimension Tr(K K_ξ^{-1}); required for rff/lrff
    """
    import math
    if method in ("rff", "lrff"):
        if n_eff is None:
            raise ValueError("n_eff required for rff/lrff fidelity_bound")
        return float(n_eff ** 2)
    if method == "ciq":
        return math.sqrt(n) * math.log(n)
    if method == "pciq":
        return n ** (3 / 8) * math.log(n)
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")


def fidelity_grid(
    method: str,
    n: int,
    n_eff: float | None = None,
    n_points: int = N_FIDELITY,
) -> list[int]:
    """Geometric fidelity grid of `n_points` values spanning bound/10 … bound*10.

    Values are rounded to positive integers; RFF/LRFF values are rounded to the
    nearest even integer (since D must be even for the cos/sin feature split).
    The grid always contains at least two distinct positive values.
    """
    import numpy as np

    bound = fidelity_bound(method, n, n_eff)
    lo = max(2, bound / 10)
    hi = bound * 10
    raw = np.geomspace(lo, hi, n_points)

    if method in ("rff", "lrff"):
        # D must be even
        grid = [int(round(v / 2) * 2) for v in raw]
    else:
        grid = [max(1, int(round(v))) for v in raw]

    # Deduplicate while preserving order
    seen: set[int] = set()
    result: list[int] = []
    for v in grid:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result
