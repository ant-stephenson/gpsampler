import pytest
from pytest import MonkeyPatch
from unittest.mock import patch, MagicMock
# from pytest_mock import mocker
# from hypothesis import given, strategies as st
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed

import gpytools.maths as gm
import gpytorch
import torch
from itertools import product as cartesian_prod


# n = 200
# d = 50
# ls = 3.0
# nv = 0.008
# ks = 1-nv

n = 100
# all_ds = [1,5,10]
# all_ls = [0.5, 3.0]
d = 5
ls =0.5
nv = 0.008
ks = 1-nv
nu = 1.5


# pytestmark = pytest.mark.parametrize("d,ls", list(cartesian_prod([n],all_ds,all_ls,[nv],[ks])))

rng = np.random.default_rng(1)

def mse(y0, y1):
    # if y0.shape != y1.shape:
    return np.sqrt(np.sum((y1.flatten()-y0.flatten())**2)/n)

def am(m):
    return ks**m*(1+2*m/(d*ls**2))**(-d/2)

@pytest.fixture
def X():
    return rng.standard_normal((n,d))/np.sqrt(d)

@pytest.fixture
def K(X):
    # kernel = gpytorch.kernels.RBFKernel()
    # kernel.lengthscale = ls
    # kernel = gpytorch.kernels.ScaleKernel(kernel)
    # kernel.outputscale = ks

    # K = kernel(torch.as_tensor(X)).add_jitter(nv)
    # return K.evaluate().detach().numpy()
    return gm.k_se(X,X,ks,ls)

@pytest.fixture
def Kmat(X):
    # kernel = gpytorch.kernels.RBFKernel()
    # kernel.lengthscale = ls
    # kernel = gpytorch.kernels.ScaleKernel(kernel)
    # kernel.outputscale = ks

    # K = kernel(torch.as_tensor(X)).add_jitter(nv)
    # return K.evaluate().detach().numpy()
    return gm.k_mat(X,X,ks,ls,nu)

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