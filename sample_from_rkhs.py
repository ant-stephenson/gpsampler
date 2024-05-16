from multiprocessing.sharedctypes import Value
import numpy as np
import argparse
from functools import partial

from gpprediction.utils import k_se, k_mat_half as k_exp, k_mat, k_mat_3half as k_mat32


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n', '--size', help='length of dataset',
        default=1000)

    parser.add_argument(
        "-d",
        "--dimension",
        help="Dimensionality of input data",
        default=10,
        type=int,
    )

    parser.add_argument(
        "-seed",
        "--random_seed",
        help="Seed for rng",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-ls",
        "--lenscale",
        help="Lengthscale of kernel",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "-nv",
        "--noise",
        help="Noise variance of output",
        default=0.00,
        type=float,
    )

    parser.add_argument(
        "-nu",
        "--matern_nu",
        help="Matern smoothness parameter",
        default=0.00,
        type=float,
    )

    parser.add_argument(
        "-ktpe",
        "--kernel_type",
        help="Kernel type",
        default="RBF",
        type=str,
        choices=["RBF", "Exp", "Matern", "Matern32"]
    )
    return parser.parse_args()


args = parse_args()
n = args.n
d = args.d
ls = args.ls
nv = args.nv
nu = args.nu
kernel_type = args.ktype
seed = args.seed

if kernel_type == "RBF":
    kernel = k_se
elif kernel_type == "Exp":
    kernel = k_exp
elif kernel_type == "Matern32":
    kernel = k_mat32
elif kernel_type == "Matern":
    kernel = partial(k_mat, nu=nu)
else:
    raise ValueError

rng = np.random.default_rng(seed)

x = rng.standard_normal((n, d)) / np.sqrt(d)
alpha = rng.standard_normal((n, 1))

f = np.zeros(n)
for i in range(n):
    f[i] = (alpha * k_se(x[i, np.newaxis], x, 1, ls)).sum()

xy_data_array = np.hstack([x, f.reshape(-1, 1)])

if kernel_type == "Matern":
    filename = f"/user/work/ll20823/mini-project/synthetic-datasets/rkhs/Output_kt_{kernel_type.lower()}_sz_{n}_dim_{d}_ls_{ls}_nu_{nu}_seed_{seed}"
else:
    filename = f"/user/work/ll20823/mini-project/synthetic-datasets/rkhs/Output_kt_{kernel_type.lower()}_sz_{n}_dim_{d}_ls_{ls}_seed_{seed}"

np.save(filename)
