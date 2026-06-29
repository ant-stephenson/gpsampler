"""Tests for gpsampler.bayes_validation.

Validation tests
----------------
Test 1 : imhof_sf agrees with Monte-Carlo estimate to ≤ 5σ.
Test 2 : gaussian_bayes_error TV → 0 as RFF feature count D → ∞.
Test 3 : Sandwich consistency — TV (1−δ)-quantile ≥ MC TV lower bound − CI.
Test 4 : certify end-to-end — perfect sampler is certified; bad sampler is not.

Guard tests
-----------
G1  compute-don't-estimate    : p* inside certify is Imhof-exact, not MC.
G2  realised-analytic-cov     : certify raises ValueError for NaN covariance.
G3  gaussianity-precondition  : certify raises NonGaussianSamplerError when is_gaussian=False.
G4  high-probability-framing  : certified flag based on quantile, not mean TV.
G5  clopper-pearson-intervals : cp_lo / cp_hi present and form a valid interval.
G6  sandwich-falsifier        : g6_ok is True in the report for a well-behaved sampler.
G7  adversarial-corroboration : g7_accuracy ≈ 0.5 for a perfect sampler.
"""

import numpy as np
import pytest
from scipy import stats

from gpsampler.bayes_validation import (
    NonGaussianSamplerError,
    _clopper_pearson,
    certify,
    gaussian_bayes_error,
    imhof_sf,
    realised_cov_ciq,
    realised_cov_rff,
)
from gpsampler.maths import k_se


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _g7_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("sklearn") is not None


# Shared constants
N_SMALL = 20
SIGMA = 1.0
NOISE_VAR = 0.1
LS = 1.0
SEED = 2025


@pytest.fixture(scope="module")
def small_K_xi():
    """SPD true observation covariance for n = N_SMALL."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N_SMALL, 2)) / np.sqrt(2)
    K = k_se(x, x, sigma=SIGMA, ls=LS)
    return K + NOISE_VAR * np.eye(N_SMALL)


def _make_rff_phi(n: int, d: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n, D) RFF feature matrix Φ scaled by √σ_f."""
    omega = rng.multivariate_normal(np.zeros(d), np.eye(d) / LS ** 2, D // 2)
    rng_x = np.random.default_rng(SEED)
    x = rng_x.standard_normal((n, d)) / np.sqrt(d)
    v = x @ omega.T                                            # (n, D//2)
    Z = np.sqrt(2.0 / D) * np.concatenate([np.cos(v), np.sin(v)], axis=1)  # (n, D)
    return np.sqrt(SIGMA) * Z


def _chol_sampler(K_xi: np.ndarray, rng: np.random.Generator):
    """Factory: returns a sampler whose realised cov equals K_xi exactly."""
    L = np.linalg.cholesky(K_xi)
    n = K_xi.shape[0]
    K_copy = K_xi.copy()

    def sampler(**_):
        return L @ rng.standard_normal(n), K_copy.copy()

    return sampler


# ---------------------------------------------------------------------------
# TestImhof — Validation Test 1
# ---------------------------------------------------------------------------

class TestImhof:
    """Test 1: imhof_sf matches Monte-Carlo to within 5σ."""

    def test_vs_mc_positive_coeffs(self):
        rng = np.random.default_rng(42)
        coeffs = np.array([1.5, 0.5, 2.0])
        x = 3.0
        n_mc = 200_000
        Q = (rng.standard_normal((n_mc, len(coeffs))) ** 2) @ coeffs
        mc_sf = float(np.mean(Q > x))
        p, err = imhof_sf(coeffs, x)
        ci = 5.0 * np.sqrt(max(mc_sf * (1.0 - mc_sf), 1e-10) / n_mc)
        assert abs(p - mc_sf) < ci + err + 1e-6, (
            f"imhof={p:.6f}  MC={mc_sf:.6f}  5σ-CI={ci:.6f}"
        )

    def test_vs_mc_mixed_coeffs(self):
        """Mixed-sign coefficients: Σ cⱼ χ²₁ can be negative."""
        rng = np.random.default_rng(123)
        coeffs = np.array([2.0, -1.0, 0.5, -0.3])
        x = 0.5
        n_mc = 200_000
        Q = (rng.standard_normal((n_mc, len(coeffs))) ** 2) @ coeffs
        mc_sf = float(np.mean(Q > x))
        p, err = imhof_sf(coeffs, x)
        ci = 5.0 * np.sqrt(max(mc_sf * (1.0 - mc_sf), 1e-10) / n_mc)
        assert abs(p - mc_sf) < ci + err + 1e-6

    def test_vs_chi2_scalar(self):
        """Single positive coefficient: imhof_sf([c], x) = chi2.sf(x/c, 1)."""
        c, x = 2.5, 3.0
        p, _ = imhof_sf([c], x)
        expected = float(stats.chi2.sf(x / c, df=1))
        assert abs(p - expected) < 1e-7, f"{p} vs {expected}"

    def test_vs_chi2_uniform_coeffs(self):
        """All-equal coefficients: Σ c χ²₁ = c χ²_n; compare to chi2.sf."""
        n, c, x = 5, 1.5, 4.0
        p, _ = imhof_sf([c] * n, x)
        expected = float(stats.chi2.sf(x / c, df=n))
        assert abs(p - expected) < 1e-7

    def test_all_zero_coeffs_above_zero(self):
        """Q = 0 a.s. → Pr(Q > x) = 0 for x > 0."""
        p, err = imhof_sf([], 1.0)
        assert p == 0.0 and err == 0.0

    def test_all_zero_coeffs_below_zero(self):
        """Q = 0 a.s. → Pr(Q > x) = 1 for x < 0."""
        p, err = imhof_sf([], -1.0)
        assert p == 1.0 and err == 0.0

    def test_all_zero_coeffs_at_zero(self):
        """Convention Pr(Q > 0) = ½ ensures TV = 0 when K̂ = K (all λᵢ = 1)."""
        p, err = imhof_sf([], 0.0)
        assert p == 0.5 and err == 0.0

    def test_probability_in_unit_interval(self):
        """Output is always clipped to [0, 1]."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            coeffs = rng.standard_normal(6)
            x = float(rng.standard_normal())
            p, err = imhof_sf(coeffs, x)
            assert 0.0 <= p <= 1.0
            assert err >= 0.0


# ---------------------------------------------------------------------------
# TestGaussianBayesError — Validation Test 2
# ---------------------------------------------------------------------------

class TestGaussianBayesError:

    def test_identical_distributions_tv_zero(self, small_K_xi):
        """TV = 0, p* = ½ when K̂_ξ = K_ξ exactly."""
        res = gaussian_bayes_error(small_K_xi, small_K_xi)
        assert res["tv"] < 1e-6, f"TV={res['tv']}"
        assert abs(res["p_star"] - 0.5) < 1e-6

    def test_analytic_n2_uniform_lambda(self):
        """n=2, K̂=2I, K=I: analytic TV = ¼.

        λ = 2 for all i, so aᵢ = ¼, b = log 2.
        p₁ = Pr(χ²₂ > 4 log 2) = exp(−2 log 2) = ¼
        p₂ = Pr(χ²₂ > 2 log 2) = exp(−log 2)  = ½
        p* = ½(¼ + 1 − ½) = ⅜   →   TV = ¼
        """
        K_xi = np.eye(2)
        Khat_xi = 2.0 * np.eye(2)
        res = gaussian_bayes_error(K_xi, Khat_xi)
        assert abs(res["tv"] - 0.25) < 1e-7, f"TV={res['tv']}"
        assert abs(res["p_star"] - 0.375) < 1e-7

    def test_tv_in_unit_interval(self, small_K_xi):
        rng = np.random.default_rng(SEED + 1)
        A = rng.standard_normal((N_SMALL, N_SMALL // 2))
        Khat = A @ A.T + 0.05 * np.eye(N_SMALL)
        res = gaussian_bayes_error(small_K_xi, Khat)
        assert 0.0 <= res["tv"] <= 1.0
        assert 0.0 <= res["p_star"] <= 0.5

    def test_output_keys(self, small_K_xi):
        res = gaussian_bayes_error(small_K_xi, small_K_xi)
        for key in ("p_star", "tv", "p_star_err", "lambdas"):
            assert key in res

    def test_lambdas_all_one_for_identical(self, small_K_xi):
        res = gaussian_bayes_error(small_K_xi, small_K_xi)
        np.testing.assert_allclose(res["lambdas"], 1.0, atol=1e-8)

    def test_tv_decreases_with_D(self):
        """Validation Test 2: TV → 0 as D increases."""
        rng = np.random.default_rng(SEED + 2)
        n, d = N_SMALL, 2
        x = np.random.default_rng(SEED).standard_normal((n, d)) / np.sqrt(d)
        K = k_se(x, x, sigma=SIGMA, ls=LS)
        K_xi = K + NOISE_VAR * np.eye(n)

        tvs = {}
        for D in [20, 100, 1000]:
            Phi = _make_rff_phi(n, d, D, rng)
            Khat_xi = realised_cov_rff(Phi, NOISE_VAR)
            tvs[D] = gaussian_bayes_error(K_xi, Khat_xi)["tv"]

        assert tvs[1000] < tvs[20], (
            f"TV should decrease with D: TV(20)={tvs[20]:.4f}, TV(1000)={tvs[1000]:.4f}"
        )


# ---------------------------------------------------------------------------
# TestRealisedCov
# ---------------------------------------------------------------------------

class TestRealisedCov:

    def test_rff_shape(self):
        n, D = 10, 30
        Phi = np.random.randn(n, D)
        assert realised_cov_rff(Phi, 0.1).shape == (n, n)

    def test_rff_is_spd(self):
        rng = np.random.default_rng(1)
        Phi = rng.standard_normal((12, 40))
        K = realised_cov_rff(Phi, 0.05)
        assert np.all(np.linalg.eigvalsh(K) > 0)

    def test_rff_noise_floor(self):
        """Diagonal of K̂_ξ = ‖Φᵢ‖² + σ²."""
        rng = np.random.default_rng(2)
        n, D, sigma2 = 8, 20, 0.3
        Phi = rng.standard_normal((n, D))
        K = realised_cov_rff(Phi, sigma2)
        np.testing.assert_allclose(
            np.diag(K), np.sum(Phi ** 2, axis=1) + sigma2, rtol=1e-12
        )

    def test_ciq_shape(self):
        n = 15
        M = np.random.randn(n, n)
        assert realised_cov_ciq(M, 0.8, 0.1).shape == (n, n)

    def test_ciq_is_psd(self):
        rng = np.random.default_rng(3)
        M = rng.standard_normal((10, 10)) / np.sqrt(10)
        K = realised_cov_ciq(M, 0.8, 0.1)
        assert np.all(np.linalg.eigvalsh(K) > 0)

    def test_ciq_eta_one_no_noise_term(self):
        """η = 1 → additive noise term vanishes; K̂ = M Mᵀ."""
        rng = np.random.default_rng(4)
        M = rng.standard_normal((8, 8))
        np.testing.assert_allclose(
            realised_cov_ciq(M, eta=1.0, sigma_xi2=5.0), M @ M.T, rtol=1e-12
        )

    def test_ciq_eta_zero_full_noise(self):
        """η = 0 → additive term is σ²I; K̂ = M Mᵀ + σ²I."""
        rng = np.random.default_rng(5)
        n, sigma2 = 8, 0.5
        M = rng.standard_normal((n, n))
        np.testing.assert_allclose(
            realised_cov_ciq(M, eta=0.0, sigma_xi2=sigma2),
            M @ M.T + sigma2 * np.eye(n),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# TestClopperPearson
# ---------------------------------------------------------------------------

class TestClopperPearson:

    def test_all_pass(self):
        lo, hi = _clopper_pearson(100, 100, 0.05)
        assert lo > 0.9 and hi == 1.0

    def test_no_pass(self):
        lo, hi = _clopper_pearson(0, 100, 0.05)
        assert lo == 0.0 and hi < 0.1

    def test_half_pass(self):
        lo, hi = _clopper_pearson(50, 100, 0.05)
        assert lo < 0.5 < hi

    def test_valid_interval(self):
        lo, hi = _clopper_pearson(37, 80, 0.10)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Guard tests G1–G7
# ---------------------------------------------------------------------------

class TestGuards:

    def test_g3_non_gaussian_raises(self, small_K_xi):
        """G3: NonGaussianSamplerError when is_gaussian=False."""
        rng = np.random.default_rng(SEED)
        K_xi = small_K_xi

        def dummy(**_):
            return rng.standard_normal(N_SMALL), K_xi.copy()

        with pytest.raises(NonGaussianSamplerError):
            certify(dummy, {}, K_xi, R=3, eps=0.1, delta=0.05, is_gaussian=False)

    def test_g2_nan_cov_raises(self, small_K_xi):
        """G2: ValueError when sampler returns NaN covariance."""
        def nan_sampler(**_):
            return np.zeros(N_SMALL), np.full((N_SMALL, N_SMALL), np.nan)

        with pytest.raises(ValueError, match="G2 violation"):
            certify(nan_sampler, {}, small_K_xi, R=2, eps=0.1, delta=0.05)

    def test_g1_p_star_exact(self, small_K_xi):
        """G1: p* values equal gaussian_bayes_error with identical K̂ = K."""
        rng = np.random.default_rng(SEED)
        K_xi = small_K_xi
        expected = gaussian_bayes_error(K_xi, K_xi)["p_star"]

        _L = np.linalg.cholesky(K_xi)

        def perfect_sampler(**_):
            return _L @ rng.standard_normal(N_SMALL), K_xi.copy()

        report = certify(perfect_sampler, {}, K_xi, R=5, eps=0.5, delta=0.05)
        np.testing.assert_allclose(report["p_stars"], expected, atol=1e-10)

    def test_g4_quantile_used(self, small_K_xi):
        """G4: tv_quantile is the empirical (1-δ)-quantile, not mean."""
        rng = np.random.default_rng(SEED)
        K_xi = small_K_xi
        _i = [0]

        _L_good = np.linalg.cholesky(K_xi)
        _L_bad = np.linalg.cholesky(8.0 * K_xi)
        _bad_K = 8.0 * K_xi

        def mixed_sampler(**_):
            _i[0] += 1
            if _i[0] % 3 == 0:
                return _L_bad @ rng.standard_normal(N_SMALL), _bad_K
            return _L_good @ rng.standard_normal(N_SMALL), K_xi.copy()

        report = certify(mixed_sampler, {}, K_xi, R=30, eps=0.5, delta=0.05)
        tvs = report["tvs"]
        expected_q = float(np.quantile(tvs, 0.95))
        assert abs(report["tv_quantile"] - expected_q) < 1e-12

    def test_g5_cp_bounds_valid(self, small_K_xi):
        """G5: Clopper–Pearson bounds are present and valid."""
        rng = np.random.default_rng(SEED)
        K_xi = small_K_xi
        sampler = _chol_sampler(K_xi, rng)
        report = certify(sampler, {}, K_xi, R=20, eps=0.5, delta=0.05)
        assert 0.0 <= report["cp_lo"] <= report["cp_hi"] <= 1.0

    def test_g6_sandwich_ok_for_perfect_sampler(self, small_K_xi):
        """G6: sandwich falsifier satisfied for a perfect sampler."""
        rng = np.random.default_rng(SEED)
        K_xi = small_K_xi
        sampler = _chol_sampler(K_xi, rng)
        report = certify(sampler, {}, K_xi, R=20, eps=0.5, delta=0.05)
        assert report["g6_ok"]


# ---------------------------------------------------------------------------
# TestCertify — Validation Tests 3 and 4
# ---------------------------------------------------------------------------

class TestCertify:

    @pytest.fixture
    def env(self, small_K_xi):
        rng = np.random.default_rng(SEED + 10)
        return small_K_xi, rng

    def _rff_sampler(self, n, d, D, rng):
        """Wrapped RFF sampler returning (y, K̂_ξ) analytically (G2 compliant)."""
        rng_x = np.random.default_rng(SEED)
        x = rng_x.standard_normal((n, d)) / np.sqrt(d)

        def sampler(**_):
            omega = rng.multivariate_normal(np.zeros(d), np.eye(d) / LS ** 2, D // 2)
            v = x @ omega.T
            Z = np.sqrt(2.0 / D) * np.concatenate([np.cos(v), np.sin(v)], axis=1)
            Phi = np.sqrt(SIGMA) * Z
            w = rng.standard_normal(D)
            y = Phi @ w + rng.standard_normal(n) * np.sqrt(NOISE_VAR)
            return y, realised_cov_rff(Phi, NOISE_VAR)

        return sampler

    def test_perfect_sampler_certified(self, env):
        """Test 4a: draws from the true GP (K̂ = K) must be certified."""
        K_xi, rng = env
        sampler = _chol_sampler(K_xi, rng)
        report = certify(sampler, {}, K_xi, R=30, eps=0.01, delta=0.05)
        assert report["certified"], (
            f"Perfect sampler not certified: TV_quantile={report['tv_quantile']:.2e}"
        )
        assert report["tv_quantile"] < 1e-4
        assert report["g6_ok"]

    def test_bad_sampler_not_certified(self, env):
        """Test 4b: draws from 5×K_xi must not be certified at eps=0.01."""
        K_xi, rng = env

        _bad_K = 5.0 * K_xi
        _L_bad = np.linalg.cholesky(_bad_K)
        _n = K_xi.shape[0]

        def bad_sampler(**_):
            return _L_bad @ rng.standard_normal(_n), _bad_K

        report = certify(bad_sampler, {}, K_xi, R=10, eps=0.01, delta=0.05)
        assert not report["certified"]
        assert report["tv_quantile"] > 0.05

    def test_sandwich_consistency(self, env):
        """Test 3: G6 sandwich holds for an RFF sampler at D=200."""
        K_xi, rng = env
        n, d, D = N_SMALL, 2, 200
        sampler = self._rff_sampler(n, d, D, rng)
        report = certify(sampler, {}, K_xi, R=20, eps=0.5, delta=0.10)
        assert report["g6_ok"], (
            f"G6 failed: tv_quantile={report['tv_quantile']:.4f}, "
            f"mean_p*={np.mean(report['p_stars']):.4f}"
        )

    def test_report_keys(self, env):
        """Report contains all documented keys."""
        K_xi, rng = env
        sampler = _chol_sampler(K_xi, rng)
        report = certify(sampler, {}, K_xi, R=5, eps=0.5, delta=0.05)
        for key in ("certified", "tv_quantile", "tv_mean", "tvs", "p_stars",
                    "cp_lo", "cp_hi", "g6_ok", "g7_accuracy", "eps", "delta", "R"):
            assert key in report

    def test_quantile_gte_mean_for_skewed(self, env):
        """G4: (1−δ)-quantile ≥ mean when TV distribution is right-skewed."""
        K_xi, rng = env
        _i = [0]
        _L_good = np.linalg.cholesky(K_xi)
        _bad_K2 = 4.0 * K_xi
        _L_bad2 = np.linalg.cholesky(_bad_K2)

        def skewed(**_):
            _i[0] += 1
            if _i[0] % 5 == 0:
                return _L_bad2 @ rng.standard_normal(N_SMALL), _bad_K2
            return _L_good @ rng.standard_normal(N_SMALL), K_xi.copy()

        report = certify(skewed, {}, K_xi, R=30, eps=0.5, delta=0.05)
        assert report["tv_quantile"] >= report["tv_mean"] - 1e-9

    @pytest.mark.skipif(not _g7_available(), reason="scikit-learn not installed")
    def test_g7_accuracy_near_half_for_perfect(self, env):
        """G7: classifier accuracy ≈ 0.5 when sampler matches the true distribution."""
        K_xi, rng = env
        sampler = _chol_sampler(K_xi, rng)
        report = certify(
            sampler, {}, K_xi,
            R=100, eps=0.5, delta=0.05,
            include_g7=True, rng=rng,
        )
        assert report["g7_accuracy"] is not None
        # A perfect sampler should be indistinguishable; accuracy < 0.7 is a loose bound
        assert report["g7_accuracy"] < 0.70, (
            f"G7 accuracy={report['g7_accuracy']:.3f} unexpectedly high"
        )
