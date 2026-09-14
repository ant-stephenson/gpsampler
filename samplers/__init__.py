"""gpsampler.samplers — GP sampling methods.

Re-exports every public symbol so that existing imports
    from gpsampler.samplers import sample_rff_from_x
continue to work unchanged after the samplers.py → samplers/ refactor.

CIQ imports are lazy (deferred until first access) so that pure-numpy code
paths never pay the ~1 GB torch/gpytorch import cost.
"""

# -- Shared utilities (numpy-only at import time) ----------------------------
from ._utils import (
    construct_kernels,
    k_true,
    approx_extreme_eigs,
    rng,
    T_TYPE,
    NPInputVec,
    NPInputMat,
    NPSample,
    NPKernel,
    kernel_matrix,
    spectral_sampler,
)

# -- Plain RFF / Cholesky ---------------------------------------------------
from .rff import (
    zrf,
    f_rf,
    estimate_rff_kernel,
    generate_rff_data,
    sample_rff_from_x,
    sample_chol_from_x,
    sample_mat_rff_from_x,
    sample_se_rff_from_x,
    sample_lap_rff_from_x,
    _sample_se_rff_from_x,
)

# -- Importance-weighted RFF ------------------------------------------------
from .iw_rff import (
    sample_iw_rff_from_x,
    _log_spectral_density,
    _build_iw_rff_features,
    _iw_rff_draw_frequencies,
)

# -- Stratified truncated-Taylor RFF ----------------------------------------
from .stratified_rff import (
    sample_stratified_rff_from_x,
    _enumerate_multi_indices,
    _monomial_design,
    _taylor_coeffs_batch,
    _choose_taylor_order,
    _raw_gaussian_moments,
    _truncated_normal_moments,
    _matern_box_prob,
    _build_H_matrix,
    _build_H_matrix_matern,
    _build_B_via_woodbury,
    _leverage_batch,
    _rejection_sample_vectorised,
    _stratified_rff_draw_frequencies,
    _build_stratified_rff_features,
)

# -- Leverage-reweighted RFF ------------------------------------------------
from .lrff import (
    sample_lrff_from_x,
    reweighted_rff_sampler,
    recursive_rls,
    nystrom_factor,
    ApproxLeverage,
    compute_sir_pool,
    resample_from_pool,
    sample_frequencies,
    draw_sample,
)

# -- Nystrom-preconditioned CG / Lanczos ------------------------------------
from .cg import (
    NystromPreconditioner,
    suggest_k,
    sample_lanczos_from_x,
    sample_cg_from_x,
    _lanczos_core,
    _tsqrt_times_e1,
)

# -- Contour-integral quadrature (CIQ) — LAZY --------------------------------
# These symbols are only resolved on first access to avoid importing
# torch/gpytorch for code paths that don't need them.

_CIQ_NAMES = {
    'matsqrt', 'contour_integral_quad', 'generate_ciq_data',
    'estimate_ciq_kernel', 'sample_ciq_from_x', 'sample_sparse_from_x',
    'ID_Preconditioner', 'SparseRBFKernel', 'SparseKernel',
}


def __getattr__(name):
    if name in _CIQ_NAMES:
        from . import ciq as _ciq
        # Hoist all CIQ symbols into this module's namespace so
        # __getattr__ is only called once per symbol.
        for _n in _CIQ_NAMES:
            globals()[_n] = getattr(_ciq, _n)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
