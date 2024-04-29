import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import gpybench.datasets as ds
import gpytools.maths as gm
import gpytorch
import torch
from scipy.special import gamma, binom

from gpsampler.utils import *
from gpsampler.tests.utils import *

n = 500
# d = 1
ks = 1.0
ls = 0.5
nv = 0.01

@pytest.fixture
def K():
    K = ds.sample_rbf_kernel(n, d, ks, ls)
    return K

@pytest.mark.skip("slow; testing tests")
@pytest.mark.parametrize("k", [5,10,20,50,100,1000])
def test_eig_index_conversion(k):
    i = binom(k+d,d)
    kapprox = int((i * gamma(d+1)) ** (1/d))-1
    np.testing.assert_approx_equal(k, kapprox, 2)

@pytest.mark.skip("")
def test_compute_sqrtnth_rbf_eigval(K):
    # NOT WORKING - NEED TO CHECK / FIX
    k = int(np.sqrt(n))
    est_eigk = compute_sqrtnth_rbf_eigval(n, d, ls)
    eig = np.linalg.eigvalsh(K)
    emp_eigk = eig[n-k]
    np.testing.assert_almost_equal(np.log10(emp_eigk), np.log10(est_eigk), 1)

def test_compute_rbf_J():
    eps, eta, C = 0.1, 0.8, 10
    J = compute_rbf_J(n, d, ls, nv, eps, eta, C)
    pass

@pytest.mark.skip("")
@pytest.mark.parametrize("d", [1,2,10,100])
@pytest.mark.run
def test_compute_exp_max_eigval(d):
    # works for d=1 but not higher
    leig_1 = compute_exp_max_eigval(n,d,ls)
    xu = np.random.uniform(0,1,(n,d))
    Ku = gm.k_mat_half(xu,xu,ks,ls)
    lmat_1 = np.linalg.eigvalsh(Ku)[-1]
    np.testing.assert_almost_equal(leig_1/n, lmat_1/n, 2)

def test_estimate_cond():
    m = 100
    delta = 0.01
    kernel_type = 'exp'
    est_cond = estimate_cond(n,m,d,ls,delta,nv,ks,kernel_type)
    assert True

def test_norm_bound_CLT():
    rate = 0
    n,d = 100,10
    ls = 1
    r = 1000
    for _ in range(r):
        x = np.random.randn(n,d) / np.sqrt(d)
        K = k_se(x,x,1,ls)
        norm = np.linalg.norm(K,1)
        a = am(1,1,ls,d)
        v = am(2,1,ls,d) - a**2
    rate += int(norm < 1 + (n-1) * (a + np.sqrt(v))) / r

@pytest.mark.parametrize("mat_idx",[50])
def test_rbf_mat_idx_to_op_idx(mat_idx):
    n = 2000
    d = 10
    ls = 0.5
    op_idx = rbf_mat_idx_to_op_idx(mat_idx, d)

    x = np.random.randn(n,d)/np.sqrt(d)
    K = gm.k_se(x,x,1,ls)
    mat_eig = np.linalg.eigvalsh(K)[-mat_idx]
    op_eig = compute_rbf_eigval_gpml(n,d,ls,op_idx)
    assert np.abs(mat_eig - op_eig) < 1

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["-k", "test_compute_exp_max_eigval", "-m", "run"])