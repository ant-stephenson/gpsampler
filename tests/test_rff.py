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

import gpsampler.samplers 
from gpsampler.samplers import sample_rff_from_x, zrf, estimate_rff_kernel

from gpsampler.tests.utils import *

class TestRFF:
    def test_zrf(self):
        pass

    def test_rff(self, X, K, benchmarks):
        w = rng.standard_normal((D,1))
        cov_omega = np.eye(d)/ls**2
        omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)
        noise = rng.normal(scale=np.sqrt(nv),size=n)

        mocked_rng = MagicMock()
        mocked_rng.standard_normal.return_value = w
        mocked_rng.multivariate_normal.return_value = omega
        mocked_rng.normal.return_value = noise

        Z = zrf(omega, D, X)*np.sqrt(ks)
        with patch('samplers.zrf') as mock_zrf:
            mock_zrf.return_value = Z
            f1,C1 = sample_rff_from_x(X, ks, nv, ls, mocked_rng, D)
        # transform w to u to make comparable
        f0 = (gm.msqrt(K) @ gm.invmsqrt(C1) @ Z @ w).flatten()
        err = mse(f0,f1)
        Kerr = np.linalg.norm(K - C1)
        assert Kerr < 1.0
        np.testing.assert_array_less(err,benchmarks, verbose=True)

    def test_Krff(self, X, K):
        Krff = estimate_rff_kernel(X, D, ks, ls)
        Krffe = Krff + nv * np.eye(n)
        err = np.linalg.norm(K-Krffe)
        assert err < 1.0

    def test_Krff_sample(self, X, K, u, benchmarks):
        Krff = estimate_rff_kernel(X, D, ks, ls)
        Krffe = Krff + nv * np.eye(n)
        y1 = gm.msqrt(Krffe) @ u
        y0 = gm.msqrt(K) @ u
        err = mse(y0,y1)
        np.testing.assert_array_less(err,benchmarks)

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMatSqrt()])