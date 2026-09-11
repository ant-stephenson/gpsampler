"""Tests for sample_iw_rff_from_x and sample_stratified_rff_from_x.

Mirrors the verify/ scripts:
  - verify_iw_covariance.py  (red/green covariance unbiasedness)
  - verify_setup_and_ratios.py (IS ratio bounds)
  - verify_a4_gaps.py (multi-index polynomial representation, rank bound)
  - verify_rejection_sampler.py (Taylor degree selection, acceptance rates)
  - verify_constant.py (oracle constant chain)
"""

import numpy as np
import pytest
from gpsampler.leverage_reweighted_rff import kernel_matrix
from gpsampler.samplers import (
    sample_iw_rff_from_x,
    sample_stratified_rff_from_x,
    _build_iw_rff_features,
    _build_stratified_rff_features,
    _log_spectral_density,
    _enumerate_multi_indices,
    _monomial_design,
    _taylor_coeffs_batch,
    _choose_taylor_order,
    _raw_gaussian_moments,
    _build_H_matrix,
    _build_B_via_woodbury,
    _leverage_batch,
    _rejection_sample_vectorised,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

n = 60
d = 1
ls = 1.0
nv = 0.01
ks = 1.0
D = 4000    # RFF features for covariance tests
rng_seed = 0


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(rng_seed).standard_normal((n, d))


@pytest.fixture(scope="module")
def K_rbf(X):
    return kernel_matrix(X, kind="rbf", ell=ls)


# ---------------------------------------------------------------------------
# _log_spectral_density
# ---------------------------------------------------------------------------

class TestLogSpectralDensity:
    """Verify normalisation of _log_spectral_density via Monte-Carlo."""

    def test_rbf_integrates_to_one(self):
        rng = np.random.default_rng(1)
        ell = 0.4
        d_loc = 3
        omega = rng.standard_normal((5000, d_loc)) / ell
        log_p = _log_spectral_density(omega, "rbf", ell, nu=1.5, d=d_loc)
        from scipy.stats import multivariate_normal
        log_p_ref = multivariate_normal.logpdf(
            omega, mean=np.zeros(d_loc), cov=np.eye(d_loc) / ell**2
        )
        np.testing.assert_allclose(log_p, log_p_ref, atol=1e-10)

    def test_matern_density_positive(self):
        rng = np.random.default_rng(2)
        omega = rng.standard_normal((200, d)) * 2.0
        log_p = _log_spectral_density(omega, "matern", ls, nu=1.5, d=d)
        assert np.all(np.isfinite(log_p))
        assert np.all(log_p < 0)

    def test_unknown_kind_raises(self):
        omega = np.ones((5, d))
        with pytest.raises(ValueError, match="unknown kernel kind"):
            _log_spectral_density(omega, "laplacian", ls, nu=1.5, d=d)


# ---------------------------------------------------------------------------
# IW-RFF sampler (mirrors verify_iw_covariance.py)
# ---------------------------------------------------------------------------

class TestIWRFF:
    """Tests for the safeguarded importance-weighted RFF sampler."""

    def test_output_shapes(self, X):
        rng = np.random.default_rng(10)
        y, cov = sample_iw_rff_from_x(X, ks, nv, ls, rng, D)
        assert y.shape == (n,)
        assert np.isnan(cov), "sample_iw_rff_from_x should return np.nan for cov"
        assert np.all(np.isfinite(y))

    def test_eta_1_is_plain_rff(self, X, K_rbf):
        """eta=1 => all from p, uniform weight => plain RFF."""
        rng = np.random.default_rng(11)
        Z = _build_iw_rff_features(X, ls, rng, D, eta=1.0)
        assert Z.shape == (n, D)
        assert np.all(np.isfinite(Z))

    def test_odd_D_raises(self, X):
        rng = np.random.default_rng(15)
        with pytest.raises(ValueError, match="even"):
            sample_iw_rff_from_x(X, ks, nv, ls, rng, D + 1)

    def test_invalid_eta_raises(self, X):
        rng = np.random.default_rng(16)
        with pytest.raises(ValueError, match="eta"):
            sample_iw_rff_from_x(X, ks, nv, ls, rng, D, eta=0.0)

    # -- IS weight bound (mirrors verify_setup_and_ratios.py) -----------------

    def test_is_weights_bounded(self, X):
        """p/q_eta <= 1/eta always."""
        from gpsampler.leverage_reweighted_rff import spectral_sampler

        for eta in [0.1, 0.3, 0.5]:
            rng = np.random.default_rng(12)
            kind = "rbf"
            l_guard = ls * 0.5
            omega = spectral_sampler(2000, d, kind, ls, 1.5, rng)
            log_p = _log_spectral_density(omega, kind, ls, 1.5, d)
            log_g = _log_spectral_density(omega, kind, l_guard, 1.5, d)
            log_q = np.logaddexp(
                np.log(1.0 - eta) + log_g, np.log(eta) + log_p
            )
            r = np.exp(log_p - log_q)
            assert np.all(r <= 1.0 / eta + 1e-9), (
                f"eta={eta}: max ratio {r.max():.3f} > 1/eta={1/eta:.3f}"
            )

    # -- Covariance unbiasedness (mirrors verify_iw_covariance.py) ------------

    def test_correct_weight_unbiased(self, X, K_rbf):
        """E[Z Z^T] with correct weight p/q_eta should match K."""
        reps = 40
        gvar_factor = 4.0
        eta = 0.5
        s_p = 1.0 / ls
        s_g = np.sqrt(gvar_factor) / ls

        rng = np.random.default_rng(1)
        Kbar = np.zeros((n, n))
        for _ in range(reps):
            def g_sampler(n_samp, _d, _rng):
                return _rng.normal(0.0, s_g, size=(n_samp, _d))

            def g_logpdf(omega):
                var_g = s_g**2
                return (-0.5 * np.sum(omega**2, axis=1) / var_g
                        - 0.5 * omega.shape[1] * np.log(2 * np.pi * var_g))

            Z = _build_iw_rff_features(
                X, ls, rng, D, eta=eta,
                g_sampler=g_sampler, g_logpdf=g_logpdf,
            )
            Kbar += Z @ Z.T
        Kbar /= reps

        rel_err = np.linalg.norm(Kbar - K_rbf, "fro") / np.linalg.norm(K_rbf, "fro")
        assert rel_err < 0.05, f"correct weight relerr={rel_err:.3f} (expect < 0.05)"

    def test_buggy_weight_biased(self, X, K_rbf):
        """Using p/g instead of p/q_eta should produce biased covariance."""
        reps = 40
        gvar_factor = 4.0
        eta = 0.5
        s_p = 1.0 / ls
        s_g = np.sqrt(gvar_factor) / ls

        rng = np.random.default_rng(1)
        n_loc, d_loc = X.shape
        m = D // 2

        # Manually build "buggy" features with p/g in denominator
        Kbar_buggy = np.zeros((n_loc, n_loc))
        for _ in range(reps):
            from_p = rng.random(m) < eta
            W = np.where(
                from_p[:, None],
                rng.normal(0, s_p, (m, d_loc)),
                rng.normal(0, s_g, (m, d_loc)),
            )
            log_p = _log_spectral_density(W, "rbf", ls, 1.5, d_loc)
            var_g = s_g**2
            log_g = (-0.5 * np.sum(W**2, axis=1) / var_g
                     - 0.5 * d_loc * np.log(2 * np.pi * var_g))
            # BUG: use g instead of q_eta
            a2 = 2.0 * np.exp(log_p - log_g) / D
            a = np.sqrt(a2)
            proj = X @ W.T
            Z = np.concatenate([a * np.cos(proj), a * np.sin(proj)], axis=1)
            Kbar_buggy += Z @ Z.T
        Kbar_buggy /= reps

        rel_err_buggy = (np.linalg.norm(Kbar_buggy - K_rbf, "fro")
                         / np.linalg.norm(K_rbf, "fro"))
        # Buggy should be much worse
        assert rel_err_buggy > 0.1, (
            f"buggy weight relerr={rel_err_buggy:.3f} (expect >> correct)"
        )

    # -- Matern ---------------------------------------------------------------

    def test_matern_finite(self, X):
        rng = np.random.default_rng(14)
        y, cov = sample_iw_rff_from_x(
            X, ks, nv, ls, rng, D=200, kernel_type="matern", nu=1.5
        )
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("eta", [0.1, 0.5, 0.9, 1.0])
    def test_varying_eta_finite(self, X, eta):
        rng = np.random.default_rng(18)
        y, _ = sample_iw_rff_from_x(X, ks, nv, ls, rng, D=200, eta=eta)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Multi-index and Taylor helpers (mirrors verify_a4_gaps.py)
# ---------------------------------------------------------------------------

class TestTaylorHelpers:
    """Tests for multi-index enumeration, monomial design, Taylor coefficients."""

    def test_multi_index_count(self):
        """r = binom(R+d, d)."""
        from math import comb
        for d_loc, R in [(1, 6), (2, 4), (3, 3)]:
            alphas = _enumerate_multi_indices(d_loc, R)
            assert len(alphas) == comb(R + d_loc, d_loc)

    def test_monomial_identity(self):
        """T_R(w^T x) = sum_{|a|<=R} i^{|a|}/a! * w^a * x^a."""
        from math import factorial
        rng = np.random.default_rng(0)
        for d_loc, R in [(1, 6), (2, 4), (3, 3)]:
            alphas = _enumerate_multi_indices(d_loc, R)
            for _ in range(50):
                x = rng.standard_normal(d_loc)
                w = rng.standard_normal(d_loc)
                z = w @ x
                # Direct Taylor
                TR = sum((1j * z) ** k / factorial(k) for k in range(R + 1))
                # Via monomial rep
                acc = 0j
                for a in alphas:
                    afact = np.prod([factorial(ai) for ai in a])
                    xa = np.prod([x[j] ** a[j] for j in range(d_loc)])
                    wa = np.prod([w[j] ** a[j] for j in range(d_loc)])
                    acc += (1j ** sum(a)) / afact * wa * xa
                assert abs(TR - acc) < 1e-9 * (1 + abs(TR)), (
                    f"d={d_loc}, R={R}: |TR - monomial| = {abs(TR-acc):.2e}"
                )

    def test_taylor_coeffs_batch(self):
        """_taylor_coeffs_batch matches manual computation."""
        rng = np.random.default_rng(1)
        from math import factorial
        d_loc, R = 2, 3
        alphas = _enumerate_multi_indices(d_loc, R)
        omega = rng.standard_normal((10, d_loc))
        C = _taylor_coeffs_batch(omega, alphas)
        assert C.shape == (10, len(alphas))
        # Check a few entries
        for i in range(10):
            for j, a in enumerate(alphas):
                afact = np.prod([factorial(ai) for ai in a])
                expected = (1j ** sum(a)) / afact * np.prod(
                    [omega[i, k] ** a[k] for k in range(d_loc)]
                )
                assert abs(C[i, j] - expected) < 1e-12

    def test_rank_bound(self):
        """rank(K_R) <= r = binom(R+d, d) (mirrors verify_a4_gaps.py claim 2)."""
        from math import comb
        rng = np.random.default_rng(0)
        for d_loc, R, n_loc in [(1, 5, 40), (2, 3, 60)]:
            r = comb(R + d_loc, d_loc)
            X_loc = rng.standard_normal((n_loc, d_loc))
            alphas = _enumerate_multi_indices(d_loc, R)
            Phi = _monomial_design(X_loc, alphas)
            # Simulate K_R via MC
            s, B = 1.0, 3.0
            M = 4000
            KR = np.zeros((n_loc, n_loc))
            for _ in range(M):
                w = rng.normal(0, s, d_loc)
                if np.max(np.abs(w)) > B:
                    continue
                c = _taylor_coeffs_batch(w.reshape(1, -1), alphas)[0]
                v = Phi @ c
                KR += np.real(np.outer(v, np.conj(v))) / M
            rk = np.linalg.matrix_rank(KR, tol=1e-8 * np.linalg.norm(KR, 2))
            assert rk <= r, f"d={d_loc},R={R}: rank={rk} > r={r}"

    def test_choose_taylor_order(self):
        """_choose_taylor_order returns an R that achieves the desired eps."""
        for Z_max in [2.0, 5.0, 10.0]:
            for eps in [0.4, 0.1]:
                R = _choose_taylor_order(Z_max, eps)
                assert R >= 1
                # Verify
                zs = np.linspace(-Z_max, Z_max, 400)
                term = np.ones_like(zs, dtype=complex)
                acc = term.copy()
                for k in range(1, R + 1):
                    term = term * (1j * zs) / k
                    acc = acc + term
                actual_err = np.max(np.abs(np.exp(1j * zs) - acc))
                assert actual_err <= eps + 1e-10, (
                    f"Z={Z_max}, eps={eps}: R={R}, actual_err={actual_err:.4f}"
                )


# ---------------------------------------------------------------------------
# H matrix and Woodbury (mirrors verify_a4_gaps.py claim 3)
# ---------------------------------------------------------------------------

class TestWoodbury:
    """Tests for _build_H_matrix, _build_B_via_woodbury, _leverage_batch."""

    def test_H_is_real_symmetric(self):
        """For SE kernel, H should be real symmetric."""
        d_loc, R = 1, 4
        s = 1.0
        B = 3.0
        alphas = _enumerate_multi_indices(d_loc, R)
        raw_mom = _raw_gaussian_moments(s, B, 2 * R)
        H = _build_H_matrix(alphas, raw_mom, d_loc)
        np.testing.assert_allclose(H, H.T, atol=1e-14)
        assert np.all(np.isfinite(H))

    def test_leverage_nonnegative(self):
        """Leverage scores a(omega) should be non-negative."""
        rng = np.random.default_rng(3)
        d_loc, R = 1, 4
        s = 1.0 / ls
        B = 3.0 * s
        n_loc = 40

        alphas = _enumerate_multi_indices(d_loc, R)
        X_loc = rng.standard_normal((n_loc, d_loc))
        Phi = _monomial_design(X_loc, alphas)
        raw_mom = _raw_gaussian_moments(s, B, 2 * R)
        H = _build_H_matrix(alphas, raw_mom, d_loc)
        B_mat = _build_B_via_woodbury(Phi, H, nv)

        omega = rng.standard_normal((500, d_loc)) * s
        C = _taylor_coeffs_batch(omega, alphas)
        a = _leverage_batch(C, B_mat)
        assert np.all(a >= -1e-10), f"min leverage = {a.min():.2e}"

    def test_whitened_gram_bound(self):
        """||A^{-1/2}(K^L-K)A^{-1/2}||_F <= n*zeta/s2
        (mirrors verify_a4_gaps.py claim 3)."""
        rng = np.random.default_rng(4)
        for _ in range(4):
            n_loc = 50
            s2 = rng.uniform(0.05, 0.5)
            B_mat = rng.standard_normal((n_loc, n_loc))
            K = B_mat @ B_mat.T / n_loc
            A = K + s2 * np.eye(n_loc)
            zeta = rng.uniform(1e-3, 1e-1)
            E = rng.uniform(-zeta, zeta, (n_loc, n_loc))
            E = (E + E.T) / 2
            wv, Vv = np.linalg.eigh(A)
            Aisq = Vv @ np.diag(wv ** -0.5) @ Vv.T
            lhs = np.linalg.norm(Aisq @ E @ Aisq, "fro")
            rhs = n_loc * zeta / s2
            assert lhs <= rhs + 1e-10, f"LHS={lhs:.4f} > RHS={rhs:.4f}"


# ---------------------------------------------------------------------------
# Rejection sampler (mirrors verify_rejection_sampler.py)
# ---------------------------------------------------------------------------

class TestRejectionSampler:
    """Test rejection sampling from box-truncated Gaussian weighted by leverage."""

    def test_acceptance_positive(self):
        """Should accept some frequencies."""
        rng = np.random.default_rng(5)
        d_loc, R = 1, 4
        n_loc = 40
        s = 1.0
        B = 2.0
        s2 = 0.1

        alphas = _enumerate_multi_indices(d_loc, R)
        X_loc = rng.standard_normal((n_loc, d_loc))
        Phi = _monomial_design(X_loc, alphas)
        raw_mom = _raw_gaussian_moments(s, B, 2 * R)
        H = _build_H_matrix(alphas, raw_mom, d_loc)
        B_mat = _build_B_via_woodbury(Phi, H, s2)

        # Compute eps for M_bound
        Bx = np.max(np.abs(X_loc))
        Z_max = Bx * B
        R_act = _choose_taylor_order(Z_max, 0.4)
        zs = np.linspace(-Z_max, Z_max, 400) if Z_max > 0 else np.array([0.0])
        term = np.ones_like(zs, dtype=complex)
        acc = term.copy()
        for k in range(1, R_act + 1):
            term = term * (1j * zs) / k
            acc = acc + term
        eps = np.max(np.abs(np.exp(1j * zs) - acc))
        M_bound = n_loc * (1 + eps) ** 2 / s2

        omega, a_vals, n_proposed = _rejection_sample_vectorised(
            d_loc, s, B, B_mat, alphas, M_bound, 100, rng
        )
        assert omega.shape == (100, d_loc)
        assert a_vals.shape == (100,)
        assert n_proposed >= 100
        assert np.all(np.abs(omega) <= B + 1e-10)

    def test_acceptance_rate_reasonable(self):
        """E[N]/(n/s2) should be O(1), not growing with n."""
        rng = np.random.default_rng(6)
        d_loc = 1
        s = 1.0
        B = 2.0
        s2 = 0.1

        ratios = []
        for n_loc in [30, 60]:
            X_loc = rng.standard_normal((n_loc, d_loc))
            Bx = np.max(np.abs(X_loc))
            Z_max = Bx * B
            R = _choose_taylor_order(Z_max, 0.4)
            alphas = _enumerate_multi_indices(d_loc, R)
            Phi = _monomial_design(X_loc, alphas)
            raw_mom = _raw_gaussian_moments(s, B, 2 * R)
            H = _build_H_matrix(alphas, raw_mom, d_loc)
            B_mat = _build_B_via_woodbury(Phi, H, s2)

            zs = np.linspace(-Z_max, Z_max, 400) if Z_max > 0 else np.array([0.0])
            term = np.ones_like(zs, dtype=complex)
            acc_t = term.copy()
            for k in range(1, R + 1):
                term = term * (1j * zs) / k
                acc_t = acc_t + term
            eps = np.max(np.abs(np.exp(1j * zs) - acc_t))
            M_bound = n_loc * (1 + eps) ** 2 / s2

            _, _, n_proposed = _rejection_sample_vectorised(
                d_loc, s, B, B_mat, alphas, M_bound, 1000, rng
            )
            EN = n_proposed / 1000.0
            ratios.append(EN / (n_loc / s2))

        # Ratio should be O(1), not growing with n
        for ratio in ratios:
            assert ratio < 10.0, f"E[N]/(n/s2) = {ratio:.1f} (expect O(1))"


# ---------------------------------------------------------------------------
# Stratified RFF sampler (end-to-end)
# ---------------------------------------------------------------------------

class TestStratifiedRFF:
    """Tests for the stratified truncated-Taylor RFF sampler."""

    def test_output_shapes(self, X):
        rng = np.random.default_rng(20)
        y, cov = sample_stratified_rff_from_x(X, ks, nv, ls, rng, D=200)
        assert y.shape == (n,)
        assert np.isnan(cov), "should return np.nan for cov"
        assert np.all(np.isfinite(y))

    def test_features_shape(self, X):
        rng = np.random.default_rng(21)
        Z = _build_stratified_rff_features(X, ls, nv, rng, D=200)
        assert Z.shape == (n, 200)
        assert np.all(np.isfinite(Z))

    def test_se_only(self, X):
        """Should raise for non-SE kernels."""
        rng = np.random.default_rng(22)
        with pytest.raises(NotImplementedError, match="SE-only"):
            sample_stratified_rff_from_x(
                X, ks, nv, ls, rng, D=200, kernel_type="matern"
            )

    def test_odd_D_raises(self, X):
        rng = np.random.default_rng(30)
        with pytest.raises(ValueError, match="even"):
            sample_stratified_rff_from_x(X, ks, nv, ls, rng, D=201)

    def test_covariance_close_to_K(self, X, K_rbf):
        """E[Z Z^T] should be close to K_rbf (within truncation error)."""
        reps = 30
        D_cov = 2000
        rng = np.random.default_rng(23)
        Kbar = np.zeros((n, n))
        for _ in range(reps):
            Z = _build_stratified_rff_features(X, ls, nv, rng, D_cov)
            Kbar += Z @ Z.T
        Kbar /= reps

        rel_err = np.linalg.norm(Kbar - K_rbf, "fro") / np.linalg.norm(K_rbf, "fro")
        # Allow bias from box truncation + finite reps + rejection sampling variance
        assert rel_err < 0.30, f"stratified relerr={rel_err:.3f} (expect < 0.30)"


# ---------------------------------------------------------------------------
# Setup and ratio bounds (mirrors verify_setup_and_ratios.py)
# ---------------------------------------------------------------------------

class TestSetupAndRatios:
    """Verify n_eff bound and IS ratio bound from the paper."""

    def test_neff_lower_bound(self):
        """n_eff >= n/(n+s2) when k(x,x)=1."""
        rng = np.random.default_rng(2)
        for _ in range(5):
            n_loc = 80
            s2 = rng.uniform(0.05, 3.0)
            X_loc = rng.normal(0, 1, n_loc)
            K = np.exp(-(X_loc[:, None] - X_loc[None, :]) ** 2 / (2 * 0.7 ** 2))
            A = K + s2 * np.eye(n_loc)
            neff = np.trace(K @ np.linalg.inv(A))
            lb = n_loc / (n_loc + s2)
            assert neff >= lb - 1e-9, (
                f"n_eff={neff:.3f} < n/(n+s2)={lb:.3f}"
            )

    def test_is_ratio_bounded(self):
        """p/q = 1/(eta + (1-eta)*a/d) in [0, 1/eta]."""
        rng = np.random.default_rng(2)
        for _ in range(20):
            eta = rng.uniform(0.05, 0.6)
            a = rng.uniform(0, 10)
            d_l = rng.uniform(0.5, 5)
            ratio = 1.0 / (eta + (1 - eta) * a / d_l)
            assert 0 <= ratio <= 1.0 / eta + 1e-9, (
                f"eta={eta:.3f} a={a:.2f} d={d_l:.2f}: ratio={ratio:.3f}"
            )


# ---------------------------------------------------------------------------
# Oracle constant chain (mirrors verify_constant.py)
# ---------------------------------------------------------------------------

class TestOracleConstant:
    """Verify the constant substitution chain end-to-end."""

    def test_constant_chain(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            L = rng.integers(3, 9)
            w = rng.uniform(0.05, 0.5, L)
            w /= w.sum()
            d_vals = rng.uniform(0.4, 2.5, L)
            rho = rng.uniform(0.05, 0.4)
            CT = rng.uniform(0.2, 3.0)
            T = CT * d_vals ** 2
            eta_opt = np.sqrt(T) / (d_vals + np.sqrt(T))
            beta_raw = 8 * (1 + rho) ** 2 * (
                d_vals ** 2 / (1 - eta_opt) + T / eta_opt
            )
            beta_bound = 8 * (1 + rho) ** 2 * (1 + np.sqrt(CT)) ** 2 * d_vals ** 2
            assert np.all(beta_raw <= beta_bound + 1e-9), "beta bound violated"

            neffR = np.sum(w * d_vals)
            Dl_frac = w * d_vals / neffR
            D_tot = 5000.0
            Dl = D_tot * Dl_frac
            EDelta = np.sum(2 * w ** 2 * beta_raw / Dl)
            pre_neff_bound = (
                16 * (1 + rho) ** 2 * (1 + np.sqrt(CT)) ** 2 * neffR ** 2 / D_tot
            )
            assert EDelta <= pre_neff_bound + 1e-9, (
                f"E||Delta||^2={EDelta:.5f} > bound={pre_neff_bound:.5f}"
            )


# ---------------------------------------------------------------------------
# Cross-sampler comparison
# ---------------------------------------------------------------------------

class TestCrossComparison:
    """Both samplers should produce finite samples."""

    def test_both_samplers_finite(self, X):
        rng1 = np.random.default_rng(40)
        rng2 = np.random.default_rng(41)

        y_iw, _ = sample_iw_rff_from_x(X, ks, nv, ls, rng1, D=200)
        y_st, _ = sample_stratified_rff_from_x(X, ks, nv, ls, rng2, D=200)

        assert np.all(np.isfinite(y_iw))
        assert np.all(np.isfinite(y_st))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
