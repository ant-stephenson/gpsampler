from .samplers import construct_kernels, generate_ciq_data, generate_rff_data, sample_ciq_from_x, sample_rff_from_x, sample_chol_from_x, sample_lrff_from_x, sample_lanczos_from_x, NystromPreconditioner, suggest_k, sample_iw_rff_from_x, sample_stratified_rff_from_x
from .bayes_validation import imhof_sf, gaussian_bayes_error, realised_cov_rff, realised_cov_ciq, certify, NonGaussianSamplerError

__all__ = [
    # samplers
    "construct_kernels", "generate_ciq_data", "generate_rff_data",
    "sample_ciq_from_x", "sample_rff_from_x", "sample_chol_from_x",
    "sample_lrff_from_x", "sample_lanczos_from_x",
    "sample_iw_rff_from_x", "sample_stratified_rff_from_x",
    "NystromPreconditioner", "suggest_k",
    # bayes_validation
    "imhof_sf", "gaussian_bayes_error",
    "realised_cov_rff", "realised_cov_ciq",
    "certify", "NonGaussianSamplerError",
]

""" 
Install with `pip install -e .`
"""

__version__ = "0.1.0"
__author__ = "Anthony Stephenson"
