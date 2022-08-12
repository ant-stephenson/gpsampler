import pytest
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed
import rff.Sweep.sampling as sampling

n = 100

@pytest.fixture
def mat():
    with temp_seed(1):
        X = sample_with_correlation(n)
    return X

class TestMatSqrt:
    def test_matsqrt(self, mat):
        Q = int(np.log(n))
        rootX = sampling.matsqrt(mat, None, Q, None)
        err = np.linalg.norm(mat - rootX @ rootX)
        np.testing.assert_almost_equal(rootX @ rootX, mat, decimal=1e-6)
        assert err < 0.1

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMatSqrt()])