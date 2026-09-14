import pytest
import numpy as np

import gpsampler.maths as gm
from gpsampler.samplers import estimate_rff_kernel

# Local constants — smaller n so D doesn't need to be huge.
# ||K - K_hat||_F ~ n/sqrt(D), so n=30, D=2000 gives err ~ 0.67.
n = 30
d = 2
ls = 0.5
nv = 0.008
ks = 1 - nv
D = 2000

rng = np.random.default_rng(1)

def mse(y0, y1):
    return np.sqrt(np.sum((y1.flatten() - y0.flatten())**2) / n)


@pytest.fixture
def X():
    return rng.standard_normal((n, d)) / np.sqrt(d)


@pytest.fixture
def K(X):
    import gpytorch, torch
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = ls
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    kernel.outputscale = ks
    K = kernel(torch.as_tensor(X)).add_jitter(nv)
    return K.evaluate().detach().numpy()


@pytest.fixture
def u():
    return rng.standard_normal((n, 1))


@pytest.fixture
def y0(K, u):
    return gm.msqrt(K) @ u


@pytest.fixture
def benchmarks(K, y0, u):
    L = np.linalg.cholesky(K)
    chol_bench = mse(y0, L @ u)
    rand_bench = mse(y0, u * np.sqrt(y0.var()))
    return np.asarray([chol_bench, rand_bench])


class TestRFF:
    # Note that kernel errors ||K-Khat|| ~ sum_ij |E_ij|^2 ~ n^2/D ≤ 1
    def test_zrf(self):
        pass

    def test_Krff(self, X, K):
        Krff = estimate_rff_kernel(X, D, ks, ls)
        Krffe = Krff + nv * np.eye(n)
        err = np.linalg.norm(K - Krffe)
        assert err < 1.0

    def test_Krff_sample(self, X, K, u, benchmarks):
        Krff = estimate_rff_kernel(X, D, ks, ls)
        Krffe = Krff + nv * np.eye(n)
        y1 = gm.msqrt(Krffe) @ u
        y0 = gm.msqrt(K) @ u
        err = mse(y0, y1)
        np.testing.assert_array_less(err, benchmarks)


if __name__ == "__main__":
    pytest.main([__file__])
