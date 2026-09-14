"""Tests for leverage-reweighted RFF (LRFF) family.

Covers every function in leverage_reweighted_rff.py plus sample_lrff_from_x
from samplers.py.  These tests serve as a regression safety net before
the samplers.py → samplers/ package refactor.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gpsampler.samplers._utils import kernel_matrix, spectral_sampler
from gpsampler.samplers.lrff import (
    recursive_rls,
    nystrom_factor,
    ApproxLeverage,
    compute_sir_pool,
    resample_from_pool,
    reweighted_rff_sampler,
    draw_sample,
    sample_lrff_from_x,
)

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

n = 80
d = 1
ell = 0.3
sigma2 = 0.01
rng_seed = 42


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(rng_seed).standard_normal((n, d))


@pytest.fixture(scope="module")
def K_rbf(X):
    return kernel_matrix(X, kind="rbf", ell=ell)


# ---------------------------------------------------------------------------
# kernel_matrix
# ---------------------------------------------------------------------------

class TestKernelMatrix:

    def test_rbf_shape_and_symmetry(self, X, K_rbf):
        assert K_rbf.shape == (n, n)
        np.testing.assert_allclose(K_rbf, K_rbf.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(K_rbf), 1.0, atol=1e-14)

    def test_rbf_matches_scipy(self, X):
        D_mat = cdist(X, X)
        K_ref = np.exp(-(D_mat ** 2) / (2 * ell ** 2))
        K = kernel_matrix(X, kind="rbf", ell=ell)
        np.testing.assert_allclose(K, K_ref, atol=1e-12)

    def test_matern_shape_and_unit_diag(self, X):
        K = kernel_matrix(X, kind="matern", ell=ell, nu=1.5)
        assert K.shape == (n, n)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# spectral_sampler
# ---------------------------------------------------------------------------

class TestSpectralSampler:

    def test_rbf_shape(self):
        rng = np.random.default_rng(0)
        W = spectral_sampler(200, d, kind="rbf", ell=ell, rng=rng)
        assert W.shape == (200, d)

    def test_matern_shape(self):
        rng = np.random.default_rng(0)
        W = spectral_sampler(200, d, kind="matern", ell=ell, nu=1.5, rng=rng)
        assert W.shape == (200, d)

    def test_rbf_scale(self):
        rng = np.random.default_rng(1)
        W = spectral_sampler(50000, d, kind="rbf", ell=ell, rng=rng)
        # RBF spectral density: omega ~ N(0, 1/ell^2 I)
        empirical_var = np.var(W)
        expected_var = 1.0 / ell ** 2
        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.05)


# ---------------------------------------------------------------------------
# recursive_rls + nystrom_factor
# ---------------------------------------------------------------------------

class TestNystromSketch:

    def test_recursive_rls_returns_valid_indices(self, K_rbf):
        rng = np.random.default_rng(0)
        S = recursive_rls(K_rbf, lam=sigma2, rng=rng)
        # S is a sorted array of unique indices in [0, n)
        assert S.ndim == 1
        assert len(S) == len(np.unique(S))
        assert np.all(S >= 0) and np.all(S < n)
        assert np.all(np.diff(S) > 0)  # sorted

    def test_nystrom_factor_shape(self, K_rbf):
        rng = np.random.default_rng(0)
        S = recursive_rls(K_rbf, lam=sigma2, rng=rng)
        B = nystrom_factor(K_rbf, S)
        assert B.shape == (n, len(S))

    def test_nystrom_full_landmarks_exact(self, K_rbf):
        """When S = all indices, B @ B.T should equal K."""
        S = np.arange(n)
        B = nystrom_factor(K_rbf, S)
        Khat = B @ B.T
        np.testing.assert_allclose(Khat, K_rbf, atol=1e-8)


# ---------------------------------------------------------------------------
# ApproxLeverage
# ---------------------------------------------------------------------------

class TestApproxLeverage:

    @pytest.fixture(scope="class")
    def alpha_fn(self, X, K_rbf):
        S = recursive_rls(K_rbf, lam=sigma2, rng=np.random.default_rng(0))
        B = nystrom_factor(K_rbf, S)
        return ApproxLeverage(X, B, sigma2)

    def test_output_shape_and_positivity(self, alpha_fn):
        rng = np.random.default_rng(1)
        W = spectral_sampler(50, d, kind="rbf", ell=ell, rng=rng)
        alpha = alpha_fn(W)
        assert alpha.shape == (50,)
        assert np.all(alpha > 0)

    def test_chunking_invariant(self, alpha_fn):
        rng = np.random.default_rng(2)
        W = spectral_sampler(100, d, kind="rbf", ell=ell, rng=rng)
        a_big = alpha_fn(W, chunk=2000)
        a_small = alpha_fn(W, chunk=10)
        np.testing.assert_allclose(a_big, a_small, atol=1e-10)

    def test_exact_leverage_small_n(self):
        """For n <= base, recursive_rls returns all indices,
        so ApproxLeverage should match exact ridge leverage."""
        rng = np.random.default_rng(3)
        n_small = 30
        X_small = rng.standard_normal((n_small, d))
        K = kernel_matrix(X_small, kind="rbf", ell=ell)
        S = np.arange(n_small)
        B = nystrom_factor(K, S)
        al = ApproxLeverage(X_small, B, sigma2)
        # Evaluate at a few test frequencies
        W_test = spectral_sampler(20, d, kind="rbf", ell=ell, rng=rng)
        alpha_approx = al(W_test)
        # Exact ridge leverage: alpha(w) = u^* K_xi^{-1} u
        # (ApproxLeverage returns (n - v^* M^{-1} v) / sigma2
        #  which equals u^* K_xi^{-1} u by Woodbury identity)
        K_xi = K + sigma2 * np.eye(n_small)
        K_xi_inv = np.linalg.inv(K_xi)
        alpha_exact = np.empty(20)
        for j in range(20):
            u = np.exp(1j * (X_small @ W_test[j:j+1].T)).ravel()
            alpha_exact[j] = np.real(u.conj() @ K_xi_inv @ u)
        np.testing.assert_allclose(alpha_approx, alpha_exact, rtol=0.05)


# ---------------------------------------------------------------------------
# SIR sampling
# ---------------------------------------------------------------------------

class TestSIRSampling:

    @pytest.fixture(scope="class")
    def alpha_fn(self, X, K_rbf):
        S = recursive_rls(K_rbf, lam=sigma2, rng=np.random.default_rng(0))
        B = nystrom_factor(K_rbf, S)
        return ApproxLeverage(X, B, sigma2)

    def test_compute_sir_pool_shapes(self, alpha_fn):
        rng = np.random.default_rng(10)
        n_freq = 100
        pool, a_pool, Z_hat = compute_sir_pool(
            n_freq, d, "rbf", ell, 1.5, alpha_fn, rng, pool_factor=5)
        P = max(5 * n_freq, 4000)
        assert pool.shape == (P, d)
        assert a_pool.shape == (P,)
        assert Z_hat > 0
        assert np.all(a_pool > 0)

    def test_resample_from_pool_shapes(self, alpha_fn):
        rng = np.random.default_rng(11)
        n_freq = 100
        pool, a_pool, Z_hat = compute_sir_pool(
            n_freq, d, "rbf", ell, 1.5, alpha_fn, rng, pool_factor=5)
        W, alpha_sel, Z_hat2 = resample_from_pool(pool, a_pool, Z_hat, n_freq, rng)
        assert W.shape == (n_freq, d)
        assert alpha_sel.shape == (n_freq,)
        assert Z_hat2 == Z_hat

    def test_resample_subset_of_pool(self, alpha_fn):
        rng = np.random.default_rng(12)
        n_freq = 50
        pool, a_pool, Z_hat = compute_sir_pool(
            n_freq, d, "rbf", ell, 1.5, alpha_fn, rng, pool_factor=5)
        W, _, _ = resample_from_pool(pool, a_pool, Z_hat, n_freq, rng)
        # Every resampled row must appear in the original pool
        pool_set = set(map(tuple, pool))
        for row in W:
            assert tuple(row) in pool_set


# ---------------------------------------------------------------------------
# reweighted_rff_sampler
# ---------------------------------------------------------------------------

class TestReweightedRFFSampler:

    def test_phi_shape(self, X):
        rng = np.random.default_rng(20)
        n_freq = 64
        Phi = reweighted_rff_sampler(X, kind="rbf", ell=ell, sigma2=sigma2,
                                      n_freq=n_freq, rng=rng)
        assert Phi.shape == (n, 2 * n_freq)
        assert Phi.dtype == np.float32

    def test_diagnostics_keys(self, X):
        rng = np.random.default_rng(21)
        Phi, diag = reweighted_rff_sampler(
            X, kind="rbf", ell=ell, sigma2=sigma2,
            n_freq=64, rng=rng, return_diagnostics=True)
        assert "Z_hat" in diag
        assert "alpha_min" in diag
        assert "alpha_max" in diag
        assert "n_landmarks" in diag
        assert "K" in diag

    def test_covariance_close_to_K(self, X, K_rbf):
        rng = np.random.default_rng(22)
        n_freq = 2000
        Phi = reweighted_rff_sampler(X, kind="rbf", ell=ell, sigma2=sigma2,
                                      n_freq=n_freq, rng=rng)
        Khat = Phi.astype(np.float64) @ Phi.astype(np.float64).T
        rel_err = np.linalg.norm(Khat - K_rbf) / np.linalg.norm(K_rbf)
        assert rel_err < 0.25, f"relative Frobenius error {rel_err:.3f} too large"

    def test_with_alpha_fn_prebuilt(self, X, K_rbf):
        rng = np.random.default_rng(23)
        S = recursive_rls(K_rbf, lam=sigma2, rng=np.random.default_rng(0))
        B = nystrom_factor(K_rbf, S)
        afn = ApproxLeverage(X, B, sigma2)
        Phi = reweighted_rff_sampler(X, kind="rbf", ell=ell, sigma2=sigma2,
                                      n_freq=64, rng=rng, alpha_fn=afn)
        assert Phi.shape == (n, 128)
        assert np.all(np.isfinite(Phi))

    def test_with_pool_cache(self, X, K_rbf):
        rng = np.random.default_rng(24)
        S = recursive_rls(K_rbf, lam=sigma2, rng=np.random.default_rng(0))
        B = nystrom_factor(K_rbf, S)
        afn = ApproxLeverage(X, B, sigma2)
        n_freq = 64
        cache = compute_sir_pool(n_freq, d, "rbf", ell, 1.5, afn, rng)
        Phi = reweighted_rff_sampler(
            X, kind="rbf", ell=ell, sigma2=sigma2,
            n_freq=n_freq, rng=rng, alpha_fn=afn, pool_cache=cache)
        assert Phi.shape == (n, 2 * n_freq)
        assert np.all(np.isfinite(Phi))


# ---------------------------------------------------------------------------
# draw_sample
# ---------------------------------------------------------------------------

class TestDrawSample:

    @pytest.fixture(scope="class")
    def Phi(self, X):
        rng = np.random.default_rng(30)
        return reweighted_rff_sampler(X, kind="rbf", ell=ell, sigma2=sigma2,
                                       n_freq=128, rng=rng)

    def test_single_sample_shape(self, Phi):
        rng = np.random.default_rng(31)
        f = draw_sample(Phi, n_samples=1, rng=rng)
        assert f.shape == (n,)

    def test_multi_sample_shape(self, Phi):
        rng = np.random.default_rng(32)
        f = draw_sample(Phi, n_samples=5, rng=rng)
        assert f.shape == (n, 5)

    def test_noise_increases_variance(self, Phi):
        rng1 = np.random.default_rng(33)
        rng2 = np.random.default_rng(33)
        f_clean = draw_sample(Phi, n_samples=200, sigma_obs=0.0, rng=rng1)
        f_noisy = draw_sample(Phi, n_samples=200, sigma_obs=0.1, rng=rng2)
        assert f_noisy.var() > f_clean.var()


# ---------------------------------------------------------------------------
# sample_lrff_from_x
# ---------------------------------------------------------------------------

class TestSampleLrffFromX:

    def test_output_shape_and_finite(self, X):
        rng = np.random.default_rng(40)
        y, cov = sample_lrff_from_x(X, sigma=1.0, noise_var=sigma2, l=ell,
                                      rng=rng, D=128)
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))

    def test_matern_runs(self, X):
        rng = np.random.default_rng(41)
        y, _ = sample_lrff_from_x(X, sigma=1.0, noise_var=sigma2, l=ell,
                                    rng=rng, D=128, kernel_type="matern", nu=1.5)
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))

    def test_covariance_sanity(self, X):
        """Over many draws, E[yy^T] should approximate sigma*K + noise_var*I."""
        sigma = 1.0
        n_freq = 1000
        D_feat = 2 * n_freq
        n_draws = 300
        K = kernel_matrix(X, kind="rbf", ell=ell)
        K_target = sigma * K + sigma2 * np.eye(n)

        yyT = np.zeros((n, n))
        for trial in range(n_draws):
            rng = np.random.default_rng(1000 + trial)
            y, _ = sample_lrff_from_x(X, sigma=sigma, noise_var=sigma2, l=ell,
                                        rng=rng, D=D_feat)
            yyT += np.outer(y, y)
        yyT /= n_draws

        rel_err = np.linalg.norm(yyT - K_target) / np.linalg.norm(K_target)
        assert rel_err < 0.30, f"relative Frobenius error {rel_err:.3f} too large"
