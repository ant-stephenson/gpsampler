"""
Simple script to run CIQ generation code from command line.
"""
import argparse

import numpy as np

from gpsampler.samplers import generate_ciq_data
from gpsampler.utils import compute_max_eigval_UB, compute_J_noprecond, compute_J_precond

# On a 16GB GPU, we found empirically this is close to the largest feasible value (larger => more memory)
MAX_ITERATIONS = 42010
# Also found empirically on 16GB GPU. Increase if a bigger card is available.
CHECKPOINT_SIZE = 1500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=110000)
    parser.add_argument("--lengthscale", type=float, default=0.5)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--noise_variance", type=float, default=0.01)
    parser.add_argument("--kernel_type", type=str, default='exp')
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--out", type=str, default="data.npy")
    parser.add_argument("--precond", type=bool, default=True)
    parser.add_argument("--eta", type=float, default=0.8)

    args = parser.parse_args()
    mean = np.zeros(args.dimension)
    covs = np.ones(args.dimension) / args.dimension
    # if args.precond:
    #     max_preconditioner_size = int(np.sqrt(args.n))
    #     if args.kernel_type.lower() == 'rbf':
    #         eigk = compute_sqrtnth_rbf_eigval(args.n, args.dimension, args.lengthscale)
    #         _iterations = max(1 + int(np.sqrt(eigk) * args.n**(3/8))/np.sqrt(args.eta*args.noise_variance * 5/4 * np.log(args.n)), 100)
    #     if args.kernel_type.lower() == 'exp':
    #         eigk = compute_exp_eigenvalue(args.n, args.dimension, args.lengthscale, max_preconditioner_size)
    #         _iterations = max(1 + int(np.sqrt(eigk) * args.n**(3/8))/np.sqrt(args.noise_variance), 100)
    # else:
    #     max_preconditioner_size = 0
    #     _iterations = int(np.log(args.n) * np.sqrt(args.n) /
    #     np.sqrt(args.noise_variance))
    if args.precond:
        max_preconditioner_size = int(np.sqrt(args.n))
        _iterations = compute_J_precond(
            args.n, args.dimension, args.lengthscale, args.noise_variance, args.outputscale, args.nu)
    else:
        max_preconditioner_size = 0
        cond = compute_max_eigval_UB(
            args.n, args.dimension, args.lengthscale, args.nu, args.outputscale) / args.noise_variance
        _iterations = compute_J_noprecond(args.n, args.noise_variance, cond)

    print(f"Requested iterations = {_iterations}.", flush=True)
    iterations = min(MAX_ITERATIONS, _iterations)
    print(
        f"Using {iterations} iterations. Using/requested = {iterations/_iterations}",
        flush=True)

    quadrature_points = int(np.log(args.n))

    lengthscale = np.array(args.lengthscale, dtype=np.float64).item()
    outputscale = np.array(args.outputscale, dtype=np.float64).item()
    noise_variance = np.array(args.noise_variance, dtype=np.float64).item()
    kernel_type = args.kernel_type
    x, y = generate_ciq_data(
        args.n, mean, covs, noise_variance, outputscale, lengthscale,
        kernel_type, iterations, quadrature_points,
        checkpoint_size=CHECKPOINT_SIZE,
        max_preconditioner_size=max_preconditioner_size)
    data = np.hstack([x, y.reshape((-1, 1))])
    np.save(args.out, data)
