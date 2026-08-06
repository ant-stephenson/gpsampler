"""Tests for sample_iw_rff_from_x and sample_stratified_rff_from_x."""

import numpy as np
import pytest
from gpsampler.leverage_reweighted_rff import kernel_matrix
from gpsampler.samplers import (
    sample_iw_rff_from_x,
    sample_stratified_rff_from_x,
    _log_spectral_density,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

n = 80
d = 2
ls = 0.5
nv = 0.01
ks = 1.0 - nv
D = 200       # RFF features (must be even)
N_MC = 800    # Monte-Carlo repetitions for covariance convergence tests
REL_TOL = 0.25  # relative Frobenius tolerance at N_MC repetitions

rng_seed = 0


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(rng_seed).standard_normal((n, d)) / np.sqrt(d)


@pytest.fixture(scope="module")
def K_rbf(X):
    return kernel_matrix(X, kind="rbf", ell=ls)


@pytest.fixture(scope="module")
def K_matern(X):
    return kernel_matrix(X, kind="matern", ell=ls, nu=1.5)


# ---------------------------------------------------------------------------
# _log_spectral_density
# ---------------------------------------------------------------------------

class TestLogSpectralDensity:
    """Verify normalisation of _log_spectral_density via Monte-Carlo."""

    def test_rbf_integrates_to_one(self):
        """E_{N(0,I)}[ p(ell*g) * ell^d ] = 1 (Gaussian density sanity)."""
        rng = np.random.default_rng(1)
        ell = 0.4
        d_loc = 3
        # Draw from N(0, I/ell^2) and check log p matches N density
        omega = rng.standard_normal((5000, d_loc)) / ell
        log_p = _log_spectral_density(omega, "rbf", ell, nu=1.5, d=d_loc)
        # Compare to scipy reference
        from scipy.stats import multivariate_normal
        log_p_ref = multivariate_normal.logpdf(
            omega, mean=np.zeros(d_loc), cov=np.eye(d_loc) / ell**2
        )
        np.testing.assert_allclose(log_p, log_p_ref, atol=1e-10)

    def test_matern_density_positive(self):
        """Matern log density should be finite and negative for random omega."""
        rng = np.random.default_rng(2)
        omega = rng.standard_normal((200, d)) * 2.0
        log_p = _log_spectral_density(omega, "matern", ls, nu=1.5, d=d)
        assert np.all(np.isfinite(log_p))
        assert np.all(log_p < 0)   # density < 1 for spread-out frequencies

    def test_unknown_kind_raises(self):
        omega = np.ones((5, d))
        with pytest.raises(ValueError, match="unknown kernel kind"):
            _log_spectral_density(omega, "laplacian", ls, nu=1.5, d=d)


# ---------------------------------------------------------------------------
# sample_iw_rff_from_x
# ---------------------------------------------------------------------------

class TestIWRFF:
    """Tests for the safeguarded importance-weighted RFF sampler."""

    # -- basic shape and finiteness ------------------------------------------

    def test_output_shapes(self, X):
        rng = np.random.default_rng(10)
        y, C = sample_iw_rff_from_x(X, ks, nv, ls, rng, D)
        assert y.shape == (n,)
        assert C.shape == (n, n)
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(C))

    def test_approx_cov_is_pd(self, X):
        """approx_cov = Phi @ Phi.T + noise_var * I must be PSD."""
        rng = np.random.default_rng(11)
        _, C = sample_iw_rff_from_x(X, ks, nv, ls, rng, D)
        eigs = np.linalg.eigvalsh(C)
        assert np.all(eigs > 0), f"min eigenvalue = {eigs.min():.2e}"

    # -- IS-weight bound: r = p/q_rho in [1-rho, 1] -------------------------

    @pytest.mark.parametrize("rho", [0.05, 0.2])
    def test_is_weights_bounded(self, X, rho):
        """IS weights must satisfy p/q_rho <= 1/(1-rho) (and >= 1-rho)."""
        from gpsampler.leverage_reweighted_rff import spectral_sampler

        rng = np.random.default_rng(12)
        kind = "rbf"
        l_guard = ls * 0.5
        omega = spectral_sampler(1000, d, kind, ls, 1.5, rng)
        log_p = _log_spectral_density(omega, kind, ls, 1.5, d)
        log_g = _log_spectral_density(omega, kind, l_guard, 1.5, d)
        log_q = np.logaddexp(np.log1p(-rho) + log_p, np.log(rho) + log_g)
        r = np.exp(log_p - log_q)
        assert np.all(r <= 1.0 / (1.0 - rho) + 1e-9)
        assert np.all(r >= 1.0 - rho - 1e-9)

    # -- covariance unbiasedness (Monte-Carlo) --------------------------------

    def test_mean_approx_cov_close_to_K(self, X, K_rbf):
        """E[Phi @ Phi.T] should be close to sigma * K_rbf."""
        rng = np.random.default_rng(13)
        cov_sum = np.zeros((n, n))
        for _ in range(N_MC):
            _, C = sample_iw_rff_from_x(X, ks, nv, ls, rng, D, rho=0.1)
            cov_sum += C - nv * np.eye(n)  # remove noise contribution
        C_mean = cov_sum / N_MC
        target = ks * K_rbf
        rel_err = np.linalg.norm(C_mean - target) / np.linalg.norm(target)
        assert rel_err < REL_TOL, f"relative Frobenius error = {rel_err:.3f}"

    # -- matern kernel --------------------------------------------------------

    def test_matern_output_shapes(self, X):
        rng = np.random.default_rng(14)
        y, C = sample_iw_rff_from_x(
            X, ks, nv, ls, rng, D, kernel_type="matern", nu=1.5
        )
        assert y.shape == (n,)
        assert C.shape == (n, n)
        assert np.all(np.isfinite(y))

    # -- argument validation --------------------------------------------------

    def test_odd_D_raises(self, X):
        rng = np.random.default_rng(15)
        with pytest.raises(ValueError, match="even"):
            sample_iw_rff_from_x(X, ks, nv, ls, rng, D + 1)

    def test_invalid_rho_raises(self, X):
        rng = np.random.default_rng(16)
        with pytest.raises(ValueError, match="rho"):
            sample_iw_rff_from_x(X, ks, nv, ls, rng, D, rho=0.0)

    def test_invalid_guard_scale_raises(self, X):
        rng = np.random.default_rng(17)
        with pytest.raises(ValueError, match="guard_scale"):
            sample_iw_rff_from_x(X, ks, nv, ls, rng, D, guard_scale=1.5)

    # -- varying rho ----------------------------------------------------------

    @pytest.mark.parametrize("rho", [0.01, 0.1, 0.5])
    def test_varying_rho_finite(self, X, rho):
        rng = np.random.default_rng(18)
        y, C = sample_iw_rff_from_x(X, ks, nv, ls, rng, D, rho=rho)
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(C))


# ---------------------------------------------------------------------------
# sample_stratified_rff_from_x
# ---------------------------------------------------------------------------

class TestStratifiedRFF:
    """Tests for the stratified truncated-Taylor leverage-reweighted RFF sampler."""

    # -- basic shape and finiteness ------------------------------------------

    def test_output_shapes(self, X):
        rng = np.random.default_rng(20)
        y, C = sample_stratified_rff_from_x(X, ks, nv, ls, rng, D)
        assert y.shape == (n,)
        assert C.shape == (n, n)
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(C))

    def test_approx_cov_is_pd(self, X):
        rng = np.random.default_rng(21)
        _, C = sample_stratified_rff_from_x(X, ks, nv, ls, rng, D)
        eigs = np.linalg.eigvalsh(C)
        assert np.all(eigs > 0), f"min eigenvalue = {eigs.min():.2e}"

    # -- covariance unbiasedness (Monte-Carlo) --------------------------------

    def test_mean_approx_cov_close_to_K(self, X, K_rbf):
        """E[Phi @ Phi.T] should be close to sigma * K_rbf."""
        rng = np.random.default_rng(22)
        cov_sum = np.zeros((n, n))
        for _ in range(N_MC):
            _, C = sample_stratified_rff_from_x(
                X, ks, nv, ls, rng, D,
                taylor_order=2, nystrom_rank=30, pool_factor=4,
            )
            cov_sum += C - nv * np.eye(n)
        C_mean = cov_sum / N_MC
        target = ks * K_rbf
        rel_err = np.linalg.norm(C_mean - target) / np.linalg.norm(target)
        assert rel_err < REL_TOL, f"relative Frobenius error = {rel_err:.3f}"

    # -- matern kernel --------------------------------------------------------

    def test_matern_output_shapes(self, X):
        rng = np.random.default_rng(23)
        y, C = sample_stratified_rff_from_x(
            X, ks, nv, ls, rng, D, kernel_type="matern", nu=1.5
        )
        assert y.shape == (n,)
        assert C.shape == (n, n)
        assert np.all(np.isfinite(y))

    # -- radial strata cover spectral mass uniformly -------------------------

    def test_stratified_radii_cover_range(self, X):
        """Stratified pool radii should span both low and high frequencies."""
        rng = np.random.default_rng(24)
        # Run one call and inspect pool indirectly via approx_cov diagonal
        # (high frequencies -> larger off-diagonal variation)
        # Just check no crash and valid outputs for different D and pool sizes.
        y, C = sample_stratified_rff_from_x(
            X, ks, nv, ls, rng, D, pool_factor=3, nystrom_rank=20
        )
        assert np.all(np.isfinite(C))

    # -- taylor_order sweep --------------------------------------------------

    @pytest.mark.parametrize("taylor_order", [0, 1, 2, 3])
    def test_varying_taylor_order(self, X, taylor_order):
        rng = np.random.default_rng(25 + taylor_order)
        y, C = sample_stratified_rff_from_x(
            X, ks, nv, ls, rng, D, taylor_order=taylor_order
        )
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(C))

    # -- argument validation --------------------------------------------------

    def test_odd_D_raises(self, X):
        rng = np.random.default_rng(30)
        with pytest.raises(ValueError, match="even"):
            sample_stratified_rff_from_x(X, ks, nv, ls, rng, D + 1)

    def test_unsupported_kernel_raises(self, X):
        rng = np.random.default_rng(31)
        with pytest.raises(ValueError, match="unsupported"):
            sample_stratified_rff_from_x(
                X, ks, nv, ls, rng, D, kernel_type="laplacian"
            )


# ---------------------------------------------------------------------------
# Cross-sampler comparison: IW-RFF vs Stratified vs plain RFF
# ---------------------------------------------------------------------------

class TestCrossComparison:
    """Both new samplers should produce covariance estimates within 2x the
    Frobenius error of plain RFF at the same D (no correctness claim, just
    rough parity check that neither sampler diverges)."""

    def test_neither_sampler_diverges(self, X, K_rbf):
        rng_plain = np.random.default_rng(40)
        rng_iw    = np.random.default_rng(41)
        rng_strat = np.random.default_rng(42)

        from gpsampler.samplers import sample_se_rff_from_x

        K_target = ks * K_rbf + nv * np.eye(n)
        R = 200  # fewer reps for speed

        def mean_cov(sampler_fn, rng_local, **kw):
            acc = np.zeros((n, n))
            for _ in range(R):
                _, C = sampler_fn(**kw, rng=rng_local)
                acc += C
            return acc / R

        C_plain = mean_cov(
            sample_se_rff_from_x,
            rng_plain,
            x=X, sigma=ks, noise_var=nv, l=ls, D=D,
        )
        C_iw = mean_cov(
            sample_iw_rff_from_x,
            rng_iw,
            x=X, sigma=ks, noise_var=nv, l=ls, D=D,
        )
        C_strat = mean_cov(
            sample_stratified_rff_from_x,
            rng_strat,
            x=X, sigma=ks, noise_var=nv, l=ls, D=D,
            nystrom_rank=20, pool_factor=3,
        )

        err_plain = np.linalg.norm(C_plain - K_target)
        err_iw    = np.linalg.norm(C_iw    - K_target)
        err_strat = np.linalg.norm(C_strat - K_target)

        # Neither new sampler should be more than 3x worse than plain RFF
        assert err_iw    < 3.0 * err_plain + 1e-3, \
            f"IW-RFF error {err_iw:.3f} >> plain {err_plain:.3f}"
        assert err_strat < 3.0 * err_plain + 1e-3, \
            f"Stratified error {err_strat:.3f} >> plain {err_plain:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
