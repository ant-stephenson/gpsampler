"""
Simple script to run CIQ generation code from command line.
"""
import argparse

import numpy as np

from sampling import generate_ciq_data

MAX_ITERATIONS = 42010 # On a 16GB GPU, we found empirically this is close to the largest feasible value (larger => more memory)
CHECKPOINT_SIZE = 1500 # Also found empirically on 16GB GPU. Increase if a bigger card is available.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1100)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--noise_variance", type=float, default=0.1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--out", type=str, default="data.npy")

    args = parser.parse_args()
    mean = np.zeros(args.dimension)
    covs = np.ones(args.dimension) / args.dimension
    _iterations = int(np.log(args.n) * np.sqrt(args.n) / np.sqrt(args.noise_variance))
    print(f"Requested iterations = {_iterations}.")
    iterations = min(MAX_ITERATIONS, _iterations)
    print(f"Using {iterations} iterations. Using/requested = {iterations/_iterations}")

    quadrature_points = int(np.log(args.n))

    lengthscale = np.array(args.lengthscale, dtype=np.float64).item()
    outputscale = np.array(args.outputscale, dtype=np.float64).item()
    noise_variance = np.array(args.noise_variance, dtype=np.float64).item()
    x, y = generate_ciq_data(args.n, mean, covs, noise_variance, outputscale,
                             lengthscale, iterations, quadrature_points, checkpoint_size=CHECKPOINT_SIZE)
    data = np.hstack([x, y.reshape((-1, 1))])
    np.save(args.out, data)
