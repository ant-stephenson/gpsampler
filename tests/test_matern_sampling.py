import pytest
import numpy as np
from gpybench.datasets import sample_with_correlation
from gpybench.utils import temp_seed
from gpytools.maths import k_mat, msqrt
from gpsampler.samplers import sample_mat_rff_from_x, estimate_mat_rff_kernel, _sample_se_rff_from_x, compute_ls_and_sigmas

from gpsampler.tests.utils import *

n,d = 200,2
sigma = 0.9
l=3.0
nu=1.5
noise_var = 0.1
rng = np.random.default_rng(1)

C = int(n**2*np.log(n))
D = int(n/nv)
G = 1000

@pytest.fixture
def x():
    return rng.standard_normal((n,d))/np.sqrt(d)

@pytest.fixture
def u():
    return rng.standard_normal(n)

@pytest.fixture
def Kmat(x):
    K = k_mat(x,x,sigma,l,nu)
    return K

@pytest.fixture
def y_true(Kmat, u):
    return msqrt(Kmat) @ u

class TestMaternSampler:
    @pytest.mark.skip()
    def test_sample(self, x, u, y_true):
        n,d = x.shape
        ym = sample_mat_rff_from_x(x,sigma, noise_var, l, rng, D, G=G, nu=nu)
        y_approx = msqrt((ym[1] + np.eye(n)*noise_var)) @ u
        err = np.linalg.norm(y_true-y_approx)
        return err
    
    def _test_sample12(self, K1, K2, u):
        y1 = msqrt(K1) @ u
        y2 = msqrt(K2) @ u
        err = np.linalg.norm(y1-y2)
        assert err < 0.5
        return err

    def _test_sample_chol(self, K1, Kmat, u):
        # doesn't work..(check with K1=Kmat)
        y1 = msqrt(K1) @ u
        y2 =  np.linalg.cholesky(Kmat) @ u
        err = np.linalg.norm(y1-y2)
        assert err < 0.5
        return err

    @pytest.mark.skip()
    def test_sample_mat_rff_from_x(self, x, u, y_true):
        # would be nice to have some theory to give a bound for err, rather than
        # a hand-picked no.
        err = self.test_sample(x, u, y_true)
        assert err < 0.5

    def test_mat_Krff1(self, x, Kmat):
        s = rng.gamma(shape=nu, scale=l**2/nu, size=G)
        w = rng.standard_normal((D,))
        C = np.zeros((n,n))
        for i, ss in enumerate(s):
            omega = rng.standard_normal((D//2, d))
            _, Cs = _sample_se_rff_from_x(x, sigma, omega[:, :]/np.sqrt(ss), w, compute_cov=True)
            C += Cs / G
        err = np.linalg.norm(Kmat-C)
        assert err < 1.0

    def test_mat_Krff2(self, x, Kmat, u):
        J = 12
        l_J, sigma_J = compute_ls_and_sigmas(n, J, nu, l)
        w = rng.standard_normal((D,))
        C = np.zeros((n,n))
        for j in range(J):
            omega = rng.multivariate_normal(np.zeros(d), np.eye(d)/l_J[j]**2, D//2)
            _, Cs = _sample_se_rff_from_x(x, sigma, omega, w, compute_cov=True)
            C += sigma_J[j] * Cs
        C /= sigma_J.sum()
        err = np.linalg.norm(Kmat-C)
        y_err = self._test_sample12(C, Kmat, u)
        assert err < 1.0

    def test_mat_Krff3(self, x, Kmat, u):
        Krff = estimate_mat_rff_kernel(x, D, sigma, l, nu)
        err = np.linalg.norm(Kmat-Krff)
        y_err = self._test_sample12(Krff, Kmat, u)
        assert err < 1.0

if __name__ == "__main__":
    pytest.main([__file__])
    # pytest.main(["--trace"], plugins=[TestMaternSampler()])