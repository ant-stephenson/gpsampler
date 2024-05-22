"""
Simple script to run CIQ generation code from command line.
"""
import argparse
import time
import os

import numpy as np

from gpsampler import generate_rff_data

MAX_ITERATIONS = int(1e8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1100)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--noise_variance", type=float, default=0.1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--out", type=str, default="data.npy")
    parser.add_argument("--kernel_type", type=str, default="rbf")
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--id", type=int, default=0)

    args = parser.parse_args()

    print("Generating synthetic data with parameters:", flush=True)
    print(f"kernel: {args.kernel_type}, n: {args.n}, dim: {args.dimension}, outputscale: {args.outputscale}, lengthscale: {args.lengthscale}, noise: {args.noise_variance}, nu: {args.nu}", flush=True)

    mean = np.zeros(args.dimension)
    covs = np.ones(args.dimension) / args.dimension

    if args.kernel_type == "matern" and args.id != 0:
        _iterations = int(args.n ** (3/2) / args.noise_variance)
    else:
        _iterations = int(3 * args.n / args.noise_variance)
    print(f"Requested iterations = {_iterations}.")
    iterations = min(MAX_ITERATIONS, _iterations)
    print(
        f"Using {iterations} iterations. Using/requested = {iterations/_iterations}")

    lengthscale = np.array(args.lengthscale, dtype=np.float64).item()
    outputscale = np.array(args.outputscale, dtype=np.float64).item()
    noise_variance = np.array(args.noise_variance, dtype=np.float64).item()

    tic = time.perf_counter()
    x, y = generate_rff_data(
        args.n, mean, covs, noise_variance, outputscale, lengthscale,
        iterations, args.kernel_type, nu=args.nu)
    data = np.hstack([x, y.reshape((-1, 1))])

    np.save(args.out, data)
    toc = time.perf_counter()
    print(f"Time taken: {toc-tic}", flush=True)
