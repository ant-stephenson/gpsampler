import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import gpybench.datasets as ds
from gpybench.utils import temp_seed
import gpytools.maths as gm
import gpytorch
import torch
from scipy.special import gamma, binom

from gpsampler.utils import compute_sqrtnth_rbf_eigval, compute_exp_eigenvalue
from gpsampler.tests.utils import *

n = 1000
d = 5
ks = 1.0
ls = 0.5
nv = 0.01

@pytest.fixture
def K():
    K = ds.sample_rbf_kernel(n, d, ks, ls)
    return K

@pytest.mark.parametrize("k", [5,10,20,50,100,1000])
def test_eig_index_conversion(k):
    i = binom(k+d,d)
    kapprox = int((i * gamma(d+1)) ** (1/d))-1
    np.testing.assert_approx_equal(k, kapprox, 2)

def test_compute_sqrtnth_rbf_eigval(K):
    # NOT WORKING - NEED TO CHECK / FIX
    k = int(np.sqrt(n))
    est_eigk = compute_sqrtnth_rbf_eigval(n, d, ls)
    eig = np.linalg.eigvalsh(K)
    emp_eigk = eig[n-k]
    np.testing.assert_almost_equal(np.log10(emp_eigk), np.log10(est_eigk), 1)


if __name__ == "__main__":
    pytest.main([__file__])