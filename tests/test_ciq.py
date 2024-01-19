import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed
import gpytools.maths as gm
import gpytorch
import torch

from gpsampler.samplers import contour_integral_quad, matsqrt, sample_ciq_from_x, generate_ciq_data
from gpsampler.tests.utils import *
import gpsampler

J = int(np.log(n)*np.sqrt(n/nv))
Q = int(np.log(n))

@pytest.fixture
def mat():
    with temp_seed(1):
        M = sample_with_correlation(n)
    return M

class TestMatSqrt:
    def test_Kciq(self, K):
        Q = int(np.log(n))
        rootX = matsqrt(K, None, Q, None)
        err = np.linalg.norm(K - rootX @ rootX)
        np.testing.assert_almost_equal(rootX @ rootX, K, decimal=1e-6)
        assert err < 1.0

# @pytest.mark.parametrize("kernel_type", ['rbf','exp'])
@pytest.fixture
def y1_rbf(X, u):
    mocked_rng = MagicMock()
    mocked_rng.standard_normal.return_value = u
    kernel_type = 'rbf'
    y1,_ = sample_ciq_from_x(X, ks, nv, ls, kernel_type, mocked_rng, J, Q, max_preconditioner_size=0)
    return y1
class TestCIQ:

    def test_ciq_rbf(self, y0, y1_rbf, benchmarks):
        err = mse(y0,y1_rbf)
        np.testing.assert_array_less(err,benchmarks)

    def test_y_stats(self, y1_rbf):
        # law of total var says that marginal var > cond. var
        # EVarHat(y) ~= Var(y) + a - 2a
        approx_margin = np.sqrt((ks+nv)/n)
        Evar = (ks+nv) - am(1)
        # assert np.abs(y1_rbf.mean()) < approx_margin
        assert np.abs(y1_rbf.var() - Evar) < 2*approx_margin

def test_generate_ciq_data():
    xmean = np.zeros(d)
    xcov_diag = np.ones(d) / d
    kernel_type = 'rbf'
    checkpoint_size = 100
    max_preconditioner_size = 100
    xsample, ysample = generate_ciq_data(n, xmean, xcov_diag, nv ,ks, ls, kernel_type, J, Q, checkpoint_size, max_preconditioner_size)
    assert True

def test_mem_usage():
    # import sqlite3
    # con = sqlite3.connect(".pymon")
    # cur = con.cursor()
    # res = cur.execute("select ITEM, MEM_USAGE from TEST_METRICS ORDER BY MEM_USAGE DESC LIMIT 10;")

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMatSqrt()])