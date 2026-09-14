"""Parameter sets for the CvM hypothesis-testing sweep.

These param sets define (d, ℓ, σ, σ², N) grids for non-Gaussian samplers
(CG, Lanczos, sparse) and as a CvM cross-check for Gaussian samplers.
"""

default_param_set = {
    "ds": [2, 3],
    "ls": [0.5, 2],
    "sigmas": [1.0],
    "noise_vars": [1e-2],
    "Ns": [2**i for i in range(8, 12)],
}

problem_param_set = {
    "ds": [2],
    "ls": [0.1, 1, 2],
    "sigmas": [1.0],
    "noise_vars": [1e-3],
    "Ns": [2**i for i in range(8, 13)],
}

paper_param_set = {
    "ds": [10],
    "ls": [1e-1, 0.5, 1, 2],
    "sigmas": [1.0],
    "noise_vars": [1e-2],
    "Ns": [2**i for i in range(8, 13)],
}

param_sets = {
    0: default_param_set,
    1: problem_param_set,
    2: paper_param_set,
}
