"""FLOP accounting for the Matérn Bayes-decision comparison.

ONE source of truth for computational cost across all four methods.  Every
formula is annotated with the paper appendix section ("Implementation costs")
it implements.  Figure F4's cross-method comparison depends entirely on these
being consistent.

Public API
----------
flops(method, n, fidelity, *, d=1, n_eff=None) -> int

Formulae (appendix "Implementation costs")
------------------------------------------
RFF
    Feature build:  (D/2) × n × d  dot products  +  n × D  trig evaluations
                 =  n D (d/2 + 1)  total multiplications
    Sample draw:    n × D  (y = Φ w)
    Total:          n D (d/2 + 2)

LRFF
    Same as RFF, plus the leverage-score overhead:
      Nystrom construction:  O(n × n_eff)  (landmark column build + small eigh)
      SIR pool evaluation:   pool_size × n × r  ≈  n × n_eff²  (pool_size ~ n_eff)
    Overhead:  n × n_eff²
    Total:     n D (d/2 + 2) + n × n_eff²

CIQ  (appendix "CIQ convergence")
    Each Lanczos / MINRES step applies K_ηξ once:  O(n²)
    J steps total:  n² × J

PCIQ  (appendix "Nyström-preconditioned CIQ")
    Nyström construction (rank r = ⌊√n⌋):
      Distance matrix K[:,I]:  n × r × d  ≈  n^{3/2} d
      Small eigh + SVD:        O(r³) = O(n^{3/2})
    Total Nyström overhead:  n^{3/2}  (dominates for small d)
    Preconditioned CIQ:  n² × J  (same n² matvec per step, but
                                    κ̃(W) << κ(K_ηξ) so fewer J needed)
    Total:  n² J + n^{3/2}

Cholesky baseline (for figure F4 reference line)
    L = chol(K):  n³ / 3

Notes
-----
- All formulae count arithmetic operations (multiplications + additions),
  ignoring constant factors < 2.  This is consistent with standard FLOP counts
  in numerical linear algebra (Golub & Van Loan, 4th ed.).
- Memory bandwidth cost is excluded; FLOP counts are the primary scaling metric
  used in the paper.
- For n_eff, pass the Hutchinson estimate  Tr(K K_ξ^{-1})  computed by the
  sweep harness (gpsampler/sweep.py, _neff_hutchinson helper).
"""

from __future__ import annotations

import math


def flops_rff(n: int, D: int, d: int = 1) -> int:
    """FLOPs for one RFF sample at n points with D features and d-dim inputs.

    Formula: n D (d/2 + 2)
    """
    return n * D * (d // 2 + 2)


def flops_lrff(n: int, D: int, d: int = 1, n_eff: float | None = None) -> int:
    """FLOPs for one LRFF sample.

    Formula: n D (d/2 + 2)  +  n n_eff²

    The n n_eff² term accounts for the SIR pool evaluation (pool_size ~ n_eff
    frequencies evaluated against n points each at r = n_eff Nystrom features).

    Parameters
    ----------
    n_eff : Tr(K K_ξ^{-1}); defaults to √n when not supplied.
    """
    if n_eff is None:
        n_eff = math.sqrt(n)
    return flops_rff(n, D, d) + int(n * n_eff ** 2)


def flops_ciq(n: int, J: int) -> int:
    """FLOPs for one CIQ sample.

    Formula: n² J
    """
    return n * n * J


def flops_pciq(n: int, J: int) -> int:
    """FLOPs for one PCIQ sample (CIQ + Nyström construction).

    Formula: n² J  +  n^{3/2}
    """
    return flops_ciq(n, J) + int(n ** 1.5)


def flops_cholesky(n: int) -> int:
    """FLOPs for a dense Cholesky factorisation.

    Formula: n³ / 3  (standard FLOP count for the L = chol(K) factorisation).
    """
    return int(n ** 3 / 3)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def flops(
    method: str,
    n: int,
    fidelity: int,
    *,
    d: int = 1,
    n_eff: float | None = None,
) -> int:
    """Return the FLOP count for one sample from ``method`` at ``fidelity``.

    Parameters
    ----------
    method   : "rff", "lrff", "ciq", "pciq", or "chol"
    n        : number of training points
    fidelity : D (features) for rff/lrff; J (Lanczos steps) for ciq/pciq
    d        : input dimension (default 1)
    n_eff    : effective dimension Tr(K K_ξ^{-1}); only needed for lrff

    Returns
    -------
    int — FLOP count (always ≥ 1)
    """
    m = method.lower()
    if m == "rff":
        return flops_rff(n, fidelity, d)
    if m == "lrff":
        return flops_lrff(n, fidelity, d, n_eff)
    if m == "ciq":
        return flops_ciq(n, fidelity)
    if m == "pciq":
        return flops_pciq(n, fidelity)
    if m == "chol":
        return flops_cholesky(n)
    raise ValueError(
        f"Unknown method {method!r}. "
        "Expected one of 'rff', 'lrff', 'ciq', 'pciq', 'chol'."
    )
