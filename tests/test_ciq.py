import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed
import gpytools.maths as gm
import gpytorch
import torch

from gpsampler.samplers import contour_integral_quad, matsqrt, sample_ciq_from_x
from gpsampler.tests.utils import *

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

@pytest.fixture
def y1(X, u):
    mocked_rng = MagicMock()
    mocked_rng.standard_normal.return_value = u
    y1,_ = sample_ciq_from_x(X, ks, nv, ls, mocked_rng, J, Q, max_preconditioner_size=0)
    return y1
class TestCIQ:

    def test_ciq(self, y0, y1, benchmarks):
        err = mse(y0,y1)
        np.testing.assert_array_less(err,benchmarks)

    def test_y_stats(self, y1):
        # law of total var says that marginal var > cond. var
        # EVarHat(y) ~= Var(y) + a - 2a
        approx_margin = np.sqrt((ks+nv)/n)
        Evar = (ks+nv) - am(1)
        assert np.abs(y1.mean()) < approx_margin
        assert np.abs(y1.var() - Evar) < 2*approx_margin

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMatSqrt()])