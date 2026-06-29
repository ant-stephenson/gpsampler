"""
Numerical correctness and statistical tests for the Lanczos GP prior sampler.

Tests
-----
1. P^{±1/2} round-trips  (P^{1/2} P^{1/2} ≈ P;  P^{-1/2}P^{1/2} ≈ I;  P^{1/2}P^{-1/2} ≈ I).
2. Factor exactness:  ‖(P^{1/2}W^{1/2})(P^{1/2}W^{1/2})^T − K_ηξ‖_F/‖K_ηξ‖_F < 1e-10.
3. Condition-number identity:  κ(W) from eigh(W) equals max/min generalised eigenvalue
   of (K_ηξ, P).  Documents (does not assert) that np.linalg.cond(P^{-1}K_ηξ) returns
   the singular-value ratio of a non-symmetric product — a different quantity.
4. Covariance convergence:  empirical Cov(ŷ) → K_ξ in Frobenius norm, for both variants.
5. Error decay:  ‖K_ηξ^{1/2}u − f̂_k‖₂ is non-increasing in k and stays below the
   Chebyshev bound  2√λ_max ‖u‖ ρ^k.
6. Preconditioning speedup:  κ̃ < κ_η on smooth RBF; fewer iters to fixed target error.
7. Cramér–von Mises indistinguishability:  rejection rate ≤ 2 × nominal α = 0.1.
"""

import pytest
import numpy as np
from scipy.linalg import eigvalsh as sp_eigvalsh, solve_triangular
from scipy.stats import cramervonmises

from gpsampler.maths import k_se, k_mat, msqrt
from gpsampler.samplers import (
    NystromPreconditioner,
    suggest_k,
    sample_lanczos_from_x,
    _lanczos_core,
    _tsqrt_times_e1,
)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N = 64          # number of data points for most tests
D_IN = 3        # input dimension
SIGMA = 1.0
LS = 0.5        # moderate RBF lengthscale → rapidly decaying eigenvalues
NOISE_VAR = 0.01
ETA = 0.8
RNG_SEED = 42

N_MC = 5_000    # samples for empirical covariance (MC floor ≈ √(n/N_MC) ≈ 0.11)
N_TRIALS = 200  # Cramér–von Mises trials


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rbf_problem(rng, n=N, ls=LS):
    x = rng.standard_normal((n, D_IN)) / np.sqrt(D_IN)
    K = k_se(x, x, SIGMA, ls)
    K_etaxi = K + ETA * NOISE_VAR * np.eye(n)
    K_xi = K + NOISE_VAR * np.eye(n)
    return x, K, K_etaxi, K_xi


def _build_mat32_problem(rng, n=N):
    x = rng.standard_normal((n, D_IN)) / np.sqrt(D_IN)
    K = k_mat(x, x, SIGMA, LS, nu=1.5)
    K_etaxi = K + ETA * NOISE_VAR * np.eye(n)
    K_xi = K + NOISE_VAR * np.eye(n)
    return x, K, K_etaxi, K_xi


def _dense_W(pre, K_etaxi):
    """W = P^{-1/2} K_ηξ P^{-1/2} as a dense matrix — O(n³), for testing only."""
    n = K_etaxi.shape[0]
    I = np.eye(n)
    P_invsqrt = np.column_stack([pre.apply_inv_sqrt(I[:, j]) for j in range(n)])
    return P_invsqrt.T @ K_etaxi @ P_invsqrt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rbf_data():
    rng = np.random.default_rng(RNG_SEED)
    return _build_rbf_problem(rng)


@pytest.fixture(scope="module")
def mat32_data():
    rng = np.random.default_rng(RNG_SEED + 1)
    return _build_mat32_problem(rng)


@pytest.fixture(scope="module")
def rbf_preconditioner(rbf_data):
    _, K, _, _ = rbf_data
    return NystromPreconditioner(K, ETA, NOISE_VAR, rng=np.random.default_rng(7))


# ---------------------------------------------------------------------------
# Test 1 — P^{±1/2} round-trips
# ---------------------------------------------------------------------------

class TestPreconditionerFormulae:
    def test_sqrt_squared_equals_P(self, rbf_preconditioner, rbf_data):
        """P^{1/2} P^{1/2} v ≈ P v  (atol 1e-10)."""
        _, _, _, _ = rbf_data
        pre = rbf_preconditioner
        n = len(pre.V)
        rng_t = np.random.default_rng(1)
        P_dense = pre.dense_P
        for _ in range(8):
            v = rng_t.standard_normal(n)
            np.testing.assert_allclose(
                pre.apply_sqrt(pre.apply_sqrt(v)), P_dense @ v, atol=1e-10)

    def test_inv_sqrt_then_sqrt(self, rbf_preconditioner):
        """P^{-1/2} P^{1/2} v ≈ v  (atol 1e-10)."""
        pre = rbf_preconditioner
        n = len(pre.V)
        rng_t = np.random.default_rng(2)
        for _ in range(8):
            v = rng_t.standard_normal(n)
            np.testing.assert_allclose(
                pre.apply_inv_sqrt(pre.apply_sqrt(v)), v, atol=1e-10)

    def test_sqrt_then_inv_sqrt(self, rbf_preconditioner):
        """P^{1/2} P^{-1/2} v ≈ v  (atol 1e-10)."""
        pre = rbf_preconditioner
        n = len(pre.V)
        rng_t = np.random.default_rng(3)
        for _ in range(8):
            v = rng_t.standard_normal(n)
            np.testing.assert_allclose(
                pre.apply_sqrt(pre.apply_inv_sqrt(v)), v, atol=1e-10)

    def test_mat32_round_trips(self, mat32_data):
        """Same round-trips for Matérn-3/2 kernel."""
        _, K, _, _ = mat32_data
        pre = NystromPreconditioner(K, ETA, NOISE_VAR, rng=np.random.default_rng(9))
        n = len(pre.V)
        rng_t = np.random.default_rng(10)
        for _ in range(5):
            v = rng_t.standard_normal(n)
            np.testing.assert_allclose(
                pre.apply_inv_sqrt(pre.apply_sqrt(v)), v, atol=1e-10)
            np.testing.assert_allclose(
                pre.apply_sqrt(pre.apply_inv_sqrt(v)), v, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 2 — Factor exactness
# ---------------------------------------------------------------------------

class TestFactorExactness:
    def _check(self, pre, K_etaxi, tol=1e-10):
        W = _dense_W(pre, K_etaxi)
        w_eig, S_eig = np.linalg.eigh(W)
        w_eig = np.maximum(w_eig, 0.0)
        W_sqrt = S_eig @ np.diag(np.sqrt(w_eig)) @ S_eig.T
        n = W_sqrt.shape[0]
        Factor = np.column_stack([pre.apply_sqrt(W_sqrt[:, j]) for j in range(n)])
        rel_err = (np.linalg.norm(Factor @ Factor.T - K_etaxi, "fro")
                   / np.linalg.norm(K_etaxi, "fro"))
        assert rel_err < tol, f"Factor rel F-error = {rel_err:.2e}"

    def test_rbf(self, rbf_preconditioner, rbf_data):
        """(P^{1/2}W^{1/2})(P^{1/2}W^{1/2})^T ≈ K_ηξ  (RBF, rel F-err < 1e-10)."""
        _, _, K_etaxi, _ = rbf_data
        self._check(rbf_preconditioner, K_etaxi)

    def test_mat32(self, mat32_data):
        """Same identity for Matérn-3/2."""
        _, K, K_etaxi, _ = mat32_data
        pre = NystromPreconditioner(K, ETA, NOISE_VAR, rng=np.random.default_rng(55))
        self._check(pre, K_etaxi)


# ---------------------------------------------------------------------------
# Test 3 — Condition-number identity + SVD trap
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_kappa_W_equals_gen_eigenvalue_ratio(self, rbf_preconditioner, rbf_data):
        """
        κ(W) from eigh(W)  ==  max/min generalised eigenvalue of (K_ηξ, P).

        TRAP — documented here:  np.linalg.cond(P^{-1} K_ηξ) computes the
        *singular-value* ratio of the non-symmetric product P^{-1} K_ηξ.
        This does NOT equal κ(W) and does NOT govern Krylov convergence.
        """
        _, _, K_etaxi, _ = rbf_data
        pre = rbf_preconditioner

        # κ(W) via direct eigendecomposition of the whitened operator
        W = _dense_W(pre, K_etaxi)
        w_W = np.linalg.eigvalsh(W)
        kappa_W = w_W[-1] / w_W[0]

        # Identical quantity via the generalised eigenvalue problem (K_ηξ, P)
        P = pre.dense_P
        w_gen = sp_eigvalsh(K_etaxi, P)
        kappa_gen = w_gen[-1] / w_gen[0]

        np.testing.assert_allclose(kappa_W, kappa_gen, rtol=1e-8,
                                    err_msg="κ(W) ≠ generalised eigenvalue ratio")

        # Document the SVD trap (not asserted — would be a design test, not a math test)
        # P_inv_K = linalg.solve(P, K_etaxi, assume_a='pos')
        # kappa_svd = np.linalg.cond(P_inv_K)  # singular-value ratio; ≠ kappa_W in general


# ---------------------------------------------------------------------------
# Test 4 — Covariance convergence (Monte-Carlo)
# ---------------------------------------------------------------------------

class TestCovarianceConvergence:
    def _emp_cov_rel_err(self, x, K_xi, sigma, noise_var, l, k,
                         kernel_type, pre=None, seed=0):
        rng_s = np.random.default_rng(seed)
        samples = np.array([
            sample_lanczos_from_x(x, sigma, noise_var, l, rng_s, k,
                                   kernel_type=kernel_type, eta=ETA,
                                   preconditioner=pre)[0]
            for _ in range(N_MC)
        ])  # (N_MC, n)
        emp_cov = np.cov(samples.T)
        return np.linalg.norm(emp_cov - K_xi, "fro") / np.linalg.norm(K_xi, "fro")

    @pytest.mark.parametrize("k", [2, 12])
    def test_unpreconditioned_rbf(self, rbf_data, k):
        x, K, _, K_xi = rbf_data
        err = self._emp_cov_rel_err(x, K_xi, SIGMA, NOISE_VAR, LS, k, "rbf",
                                     seed=100 + k)
        if k >= 12:
            assert err < 0.3, f"k={k}: rel cov error = {err:.3f}"

    @pytest.mark.parametrize("k", [2, 8])
    def test_preconditioned_rbf(self, rbf_data, rbf_preconditioner, k):
        x, K, _, K_xi = rbf_data
        err = self._emp_cov_rel_err(x, K_xi, SIGMA, NOISE_VAR, LS, k, "rbf",
                                     pre=rbf_preconditioner, seed=200 + k)
        if k >= 8:
            assert err < 0.3, f"k={k}: rel cov error = {err:.3f}"


# ---------------------------------------------------------------------------
# Test 5 — Geometric error decay
# ---------------------------------------------------------------------------

class TestErrorDecay:
    def _run_lanczos_errors(self, K_etaxi, u, k_list, matvec=None):
        if matvec is None:
            matvec = lambda v: K_etaxi @ v
        exact = msqrt(K_etaxi) @ u
        norm_u = np.linalg.norm(u)
        errs = []
        for k in k_list:
            Q, alpha, beta, _ = _lanczos_core(matvec, u, k)
            f_hat = norm_u * (Q @ _tsqrt_times_e1(alpha, beta))
            errs.append(float(np.linalg.norm(exact - f_hat)))
        return np.array(errs)

    def test_error_non_increasing(self, rbf_data):
        """‖K_ηξ^{1/2}u − f̂_k‖₂ is non-increasing in k."""
        _, _, K_etaxi, _ = rbf_data
        u = np.random.default_rng(77).standard_normal(N)
        errs = self._run_lanczos_errors(K_etaxi, u, [1, 3, 8, 16, 32])
        diffs = np.diff(errs)
        assert np.all(diffs <= 1e-8 * errs[:-1] + 1e-14), (
            f"Errors not non-increasing: {errs}")

    def test_chebyshev_bound(self, rbf_data):
        """Error stays below  2√λ_max ‖u‖ ρ^k,  ρ = (√κ−1)/(√κ+1)."""
        _, _, K_etaxi, _ = rbf_data
        u = np.random.default_rng(88).standard_normal(N)
        w = np.linalg.eigvalsh(K_etaxi)
        kappa = w[-1] / w[0]
        rho = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
        norm_u = np.linalg.norm(u)
        k_list = [2, 5, 10, 20]
        errs = self._run_lanczos_errors(K_etaxi, u, k_list)
        for k, err in zip(k_list, errs):
            bound = 2.0 * np.sqrt(w[-1]) * norm_u * rho ** k
            # Allow 10% slack to account for round-off and re-orthogonalisation
            assert err <= 1.1 * bound + 1e-12, (
                f"k={k}: err={err:.3e} > 1.1×bound={1.1*bound:.3e}")

    def test_preconditioned_error_non_increasing(self, rbf_data, rbf_preconditioner):
        """Preconditioned Lanczos error ‖W^{1/2}u − ĝ_k‖₂ is non-increasing."""
        _, _, K_etaxi, _ = rbf_data
        pre = rbf_preconditioner
        u = np.random.default_rng(99).standard_normal(N)

        # Exact W^{1/2} u via dense eigendecomposition
        W = _dense_W(pre, K_etaxi)
        w_eig, S_eig = np.linalg.eigh(W)
        w_eig = np.maximum(w_eig, 0.0)
        W_sqrt_u = (S_eig @ np.diag(np.sqrt(w_eig)) @ S_eig.T) @ u

        def mv_W(v):
            return pre.apply_inv_sqrt(K_etaxi @ pre.apply_inv_sqrt(v))

        norm_u = np.linalg.norm(u)
        errs = []
        for k in [1, 3, 8, 16]:
            Q, alpha, beta, _ = _lanczos_core(mv_W, u, k)
            g_hat = norm_u * (Q @ _tsqrt_times_e1(alpha, beta))
            errs.append(float(np.linalg.norm(W_sqrt_u - g_hat)))

        diffs = np.diff(errs)
        assert np.all(diffs <= 1e-8 * np.array(errs[:-1]) + 1e-14), (
            f"Preconditioned errors not non-increasing: {errs}")


# ---------------------------------------------------------------------------
# Test 6 — Preconditioning speedup
# ---------------------------------------------------------------------------

class TestPreconditioningSpeedup:
    def test_kappa_tilde_less_than_kappa_eta(self, rbf_data, rbf_preconditioner):
        """κ̃ = κ(W) < κ(K_ηξ) for smooth RBF at moderate lengthscale."""
        _, _, K_etaxi, _ = rbf_data
        pre = rbf_preconditioner

        kappa_un = np.linalg.cond(K_etaxi)      # κ(K_ηξ)
        W = _dense_W(pre, K_etaxi)
        w_W = np.linalg.eigvalsh(W)
        kappa_pre = w_W[-1] / w_W[0]            # κ̃ = κ(W)

        assert kappa_pre < kappa_un, (
            f"Expected κ̃={kappa_pre:.2f} < κ_η={kappa_un:.2f}")

    def test_fewer_iterations_to_target(self, rbf_data, rbf_preconditioner):
        """Preconditioned reaches a fixed sample-error target faster than unpreconditioned."""
        _, _, K_etaxi, _ = rbf_data
        pre = rbf_preconditioner
        u = np.random.default_rng(111).standard_normal(N)

        exact = msqrt(K_etaxi) @ u
        norm_u = np.linalg.norm(u)
        # Target: 10% of the initial residual  ‖K_ηξ^{1/2}u − 0‖
        target = 0.10 * np.linalg.norm(exact)

        k_range = list(range(1, N + 1))

        # Unpreconditioned
        mv_un = lambda v: K_etaxi @ v
        k_un = None
        for k in k_range:
            Q, alpha, beta, _ = _lanczos_core(mv_un, u, k)
            f_hat = norm_u * (Q @ _tsqrt_times_e1(alpha, beta))
            if np.linalg.norm(exact - f_hat) < target:
                k_un = k
                break

        # Preconditioned
        mv_W = lambda v: pre.apply_inv_sqrt(K_etaxi @ pre.apply_inv_sqrt(v))
        k_pre = None
        for k in k_range:
            Q, alpha, beta, _ = _lanczos_core(mv_W, u, k)
            g_hat = norm_u * (Q @ _tsqrt_times_e1(alpha, beta))
            f_hat = pre.apply_sqrt(g_hat)
            if np.linalg.norm(exact - f_hat) < target:
                k_pre = k
                break

        assert k_un is not None and k_pre is not None, (
            "Neither sampler reached the target — loosen target or increase k_range")
        assert k_pre <= k_un, (
            f"Preconditioned took {k_pre} iters; expected ≤ unpreconditioned {k_un}")


# ---------------------------------------------------------------------------
# Test 7 — Cramér–von Mises indistinguishability
# ---------------------------------------------------------------------------

class TestCvMIndistinguishability:
    @pytest.mark.parametrize("use_precond", [False, True])
    def test_rejection_rate_at_suggest_k(self, rbf_data, rbf_preconditioner,
                                          use_precond):
        """
        Whitening + CvM rejection rate ≤ 2 × α = 0.1 at suggest_k iterations.

        The whitened vector L^{-1} ŷ  (L = chol(K_ξ)) should be N(0, I_n);
        the CvM test checks whether the n-component marginal empirical distribution
        matches N(0,1).  We repeat for N_TRIALS independent samples and report
        the fraction of trials where pvalue < α.
        """
        x, K, _, K_xi = rbf_data
        alpha_nom = 0.10

        k = max(suggest_k(N, ETA, NOISE_VAR, eps=0.5), 3)
        pre = rbf_preconditioner if use_precond else None

        L = np.linalg.cholesky(K_xi)
        rng_s = np.random.default_rng(2025 + int(use_precond))
        rejections = 0
        for _ in range(N_TRIALS):
            y, _ = sample_lanczos_from_x(
                x, SIGMA, NOISE_VAR, LS, rng_s, k,
                kernel_type="rbf", eta=ETA, preconditioner=pre)
            spherical = solve_triangular(L, y, lower=True)
            res = cramervonmises(spherical, "norm", args=(0, 1))
            rejections += int(res.pvalue < alpha_nom)

        rate = rejections / N_TRIALS
        assert rate <= 2.0 * alpha_nom, (
            f"CvM rejection rate {rate:.3f} > 2α={2*alpha_nom:.3f} "
            f"(k={k}, precond={use_precond})")


# ---------------------------------------------------------------------------
# Test: suggest_k sanity
# ---------------------------------------------------------------------------

class TestSuggestK:
    def test_non_decreasing_with_n(self):
        """Larger n should need weakly more iterations."""
        ks = [suggest_k(n, ETA, NOISE_VAR, 0.1) for n in [32, 128, 512]]
        assert ks[0] <= ks[1] <= ks[2], f"Expected non-decreasing: {ks}"

    def test_non_increasing_with_eta(self):
        """Larger η (less noisy split) reduces effective κ_η → fewer steps."""
        ks = [suggest_k(256, eta, NOISE_VAR, 0.1) for eta in [0.2, 0.5, 0.9]]
        # Should be non-increasing: more eta → smaller κ_η
        assert ks[0] >= ks[-1], f"Expected k to decrease with η: {ks}"

    def test_positive_integer(self):
        k = suggest_k(100, ETA, NOISE_VAR, 0.1)
        assert isinstance(k, int) and k >= 1

    def test_tighter_lambda1_gives_fewer_iters(self):
        """Providing a tighter λ₁ (< trace bound = n) should give same or fewer iters."""
        k_trace = suggest_k(256, ETA, NOISE_VAR, 0.1, lambda1=None)
        k_tight = suggest_k(256, ETA, NOISE_VAR, 0.1, lambda1=5.0)
        assert k_tight <= k_trace
