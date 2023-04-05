import pytest
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed
from gpytools.maths import k_mat, msqrt
from gpsampler.samplers import sample_mat_rff_from_x

n,d = 200,2
sigma = 0.99
l=0.2
nu=1.5
noise_var = 0.01
rng = np.random.default_rng(1)

C = int(n**2*np.log(n))
D = int(C**0.5)
D = int(round(D/2)*2)
G = C//D

@pytest.fixture
def x():
    return rng.standard_normal((n,d))/np.sqrt(d)

@pytest.fixture
def u():
    return rng.standard_normal(n)

@pytest.fixture
def Kmat(x):
    K = k_mat(x,x, sigma,l,nu)
    return K

@pytest.fixture
def y_true(Kmat, u):
    return msqrt(Kmat) @ u

class TestMaternSampler:
    def test_sample(self, x, u, y_true):
        n,d = x.shape
        ym = sample_mat_rff_from_x(x,sigma, noise_var, l, rng, D, G=G, nu=nu)
        y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
        err = np.linalg.norm(y_true-y_approx)
        return err

    def test_sample_mat_rff_from_x(self, x, u, y_true):
        # would be nice to have some theory to give a bound for err, rather than
        # a hand-picked no.
        err = self.test_sample(x, u, y_true)
        assert err < 0.5

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMatSqrt()])