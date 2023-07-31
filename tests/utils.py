import pytest
from pytest import MonkeyPatch
from unittest.mock import patch, MagicMock
from pytest_mock import mocker
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed

import gpytools.maths as gm
import gpytorch
import torch
import numpy as np


n = 200
ls = 0.5
ks = 1.0
d = 1
nv = 1e-2

rng = np.random.default_rng(1)

D = int(n**2)

def mse(y0, y1):
    return np.sqrt(np.sum((y1-y0)**2)/n)

@pytest.fixture
def X():
    return rng.standard_normal((n,d))/np.sqrt(d)

@pytest.fixture
def K(X):
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = ls
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    kernel.outputscale = ks

    K = kernel(torch.as_tensor(X)).add_jitter(nv)
    return K.evaluate().detach().numpy()

@pytest.fixture
def u():
    return rng.standard_normal((n,1))

@pytest.fixture
def y0(K, u):
    y0 = gm.msqrt(K) @ u
    return y0

@pytest.fixture
def chol_benchmark(K, y0, u):
    L = np.linalg.cholesky(K)
    y1 = L @ u
    return mse(y0,y1)

@pytest.fixture
def rand_benchmark(K, y0, u):
    y1 = u*np.sqrt(y0.var())
    return mse(y0,y1)

@pytest.fixture
def benchmarks(chol_benchmark, rand_benchmark):
    return np.asarray([chol_benchmark, rand_benchmark])