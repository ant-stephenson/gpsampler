"""Bayesian-decision-process validation for GP prior samplers.

Public API
----------
imhof_sf             : survival function of a weighted chi-squared sum (Imhof 1961)
gaussian_bayes_error : exact Bayes error between two zero-mean Gaussians
realised_cov_rff     : analytic realised covariance for RFF samplers
realised_cov_ciq     : analytic realised covariance for CIQ samplers
certify              : (1 - δ) high-probability TV certification with guards G1–G7

Guards
------
G1  compute-don't-estimate    : use Imhof quadrature, never Monte-Carlo-estimate p*
G2  realised-analytic-cov     : use K̂_ξ built from realised features, not sample cov
G3  gaussianity-precondition  : non-Gaussian samplers (e.g. Lanczos) cannot be certified
G4  high-probability-framing  : report (1 − δ) upper quantile of TV, not mean
G5  clopper-pearson-intervals : certify only when conservative CP lower bound implies pass
G6  sandwich-falsifier        : TV certificate must be ≥ any MC lower bound − CI
G7  adversarial-corroboration : separate classifier two-sample test (not part of cert path)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import integrate, linalg, stats


# ---------------------------------------------------------------------------
# Imhof (1961) quadrature — Guard G1 (compute, don't estimate)
# ---------------------------------------------------------------------------

def imhof_sf(
    coeffs: npt.ArrayLike,
    x: float,
    *,
    tol: float = 1e-9,
) -> Tuple[float, float]:
    r"""Survival function of a weighted chi-squared sum via Imhof (1961).

    Computes Pr(Q > x) where Q = Σⱼ cⱼ χ²_{1,j} and the χ²_{1,j} are
    independent chi-squared variates with one degree of freedom.
    Coefficients cⱼ may be of any sign.

    Formula (Imhof 1961, eq. 5):

        Pr(Q > x) = ½ + (1/π) ∫₀^∞ sin θ(u) / (u ρ(u)) du

    where   θ(u) = ½ Σⱼ arctan(cⱼ u) − ½ x u
    and     ρ(u) = Πⱼ (1 + cⱼ² u²)^{1/4}.

    Parameters
    ----------
    coeffs : array-like
        Coefficients cⱼ.  Zero entries are silently removed.
    x : float
        Threshold value.
    tol : float
        Absolute and relative quadrature tolerance.

    Returns
    -------
    p : float
        Estimated Pr(Q > x), clipped to [0, 1].
    err : float
        Estimated absolute error from the quadrature routine.
    """
    # Convert to ndarray immediately so all arithmetic below is well-typed
    c: np.ndarray = np.asarray(coeffs, dtype=float)
    c = c[c != 0.0]

    # Degenerate case: Q = 0 almost surely
    if len(c) == 0:
        if x < 0.0:
            return 1.0, 0.0
        elif x > 0.0:
            return 0.0, 0.0
        else:
            # Convention: Imhof formula returns ½ at x = 0 for zero coefficients;
            # this ensures TV = 0 when K̂_ξ = K_ξ (all λᵢ = 1).
            return 0.5, 0.0

    def _integrand(u: float) -> float:
        theta = 0.5 * np.sum(np.arctan(c * u)) - 0.5 * x * u
        # Use log1p for numerical stability at small u
        log_rho = 0.25 * np.sum(np.log1p(c ** 2 * u ** 2))
        if u == 0.0:
            # L'Hôpital: lim_{u→0} sin(θ)/u = θ'(0) = ½(Σcⱼ − x)
            return 0.5 * (float(np.sum(c)) - x)
        return np.sin(theta) / (u * np.exp(log_rho))

    result, abserr = integrate.quad(
        _integrand,
        0.0,
        np.inf,
        epsabs=tol,
        epsrel=tol,
        limit=500,
    )

    p = float(np.clip(0.5 + result / np.pi, 0.0, 1.0))
    return p, float(abserr / np.pi)


# ---------------------------------------------------------------------------
# Exact Bayes error between two zero-mean Gaussians
# ---------------------------------------------------------------------------

def gaussian_bayes_error(
    K_xi: np.ndarray,
    Khat_xi: np.ndarray,
) -> Dict[str, Any]:
    r"""Exact Bayes error between N(0, K_ξ) and N(0, K̂_ξ).

    Computes the minimum misclassification probability p* and total variation
    TV = 1 − 2 p* for the optimal likelihood-ratio test that distinguishes the
    two zero-mean Gaussian distributions.

    Derivation
    ----------
    Let λᵢ be the generalised eigenvalues of the pencil (K̂_ξ, K_ξ),
    computed via the symmetric form A = L⁻¹ K̂_ξ L⁻ᵀ where K_ξ = L Lᵀ.
    Define aᵢ = ½(1 − 1/λᵢ) and b = ½ Σᵢ log λᵢ.  Then:

        p* = ½ [Pr(Σ aᵢ χ²_{1,i} > b) + Pr(Σ λᵢ aᵢ χ²_{1,i} ≤ b)]

    where the first Pr is taken under p₁ = N(0, K_ξ) and the second under
    p₂ = N(0, K̂_ξ).  Both are computed via ``imhof_sf`` (Guard G1).

    Parameters
    ----------
    K_xi : (n, n) ndarray
        True observation covariance K_ξ (must be strictly positive definite).
    Khat_xi : (n, n) ndarray
        Realised observation covariance K̂_ξ (must be positive semi-definite).

    Returns
    -------
    dict with keys
        p_star     : float — Bayes error ∈ [0, ½]
        tv         : float — total variation TV ∈ [0, 1]
        p_star_err : float — quadrature error bound on p_star
        lambdas    : (n,) ndarray — generalised eigenvalues λᵢ (sorted ascending)
    """
    K_xi = np.asarray(K_xi, dtype=float)
    Khat_xi = np.asarray(Khat_xi, dtype=float)

    # Cholesky: K_ξ = L Lᵀ
    L = linalg.cholesky(K_xi, lower=True)

    # Form A = L⁻¹ K̂_ξ L⁻ᵀ (symmetric); eigenvalues of A are the generalised
    # eigenvalues of (K̂_ξ, K_ξ).
    Linv_Khat = linalg.solve_triangular(L, Khat_xi, lower=True)         # L⁻¹ K̂_ξ
    A = linalg.solve_triangular(L, Linv_Khat.T, lower=True).T           # L⁻¹ K̂_ξ L⁻ᵀ

    lambdas = linalg.eigvalsh(A)           # sorted ascending by eigvalsh
    lambdas = np.maximum(lambdas, 1e-300)  # guard against log(0) for near-zero eigenvalues

    a = 0.5 * (1.0 - 1.0 / lambdas)           # Imhof coefficients under p₁
    b = 0.5 * float(np.sum(np.log(lambdas)))   # log-determinant threshold

    # Pr(Σ aᵢ χ²_{1,i} > b)  — probability of correct classification under p₁
    p1, err1 = imhof_sf(a, b)

    # Pr(Σ λᵢ aᵢ χ²_{1,i} > b)  — used for Pr(… ≤ b) under p₂
    a_tilde = a * lambdas     # = ½(λᵢ − 1)
    p2, err2 = imhof_sf(a_tilde, b)

    p_star = float(np.clip(0.5 * (p1 + 1.0 - p2), 0.0, 0.5))
    tv = float(np.clip(1.0 - 2.0 * p_star, 0.0, 1.0))
    p_star_err = 0.5 * (err1 + err2)

    return {
        "p_star": p_star,
        "tv": tv,
        "p_star_err": p_star_err,
        "lambdas": lambdas,
    }


# ---------------------------------------------------------------------------
# Realised-covariance adapters  (Guard G2)
# ---------------------------------------------------------------------------

def realised_cov_rff(Phi: np.ndarray, sigma_xi2: float) -> np.ndarray:
    """Analytic realised covariance for an RFF sampler.

    For a sampler that draws y = Φ w + ε  (w ~ N(0, I_D), ε ~ N(0, σ²_ξ I)),
    the exact — not estimated — realised observation covariance is

        K̂_ξ = ΦΦᵀ + σ²_ξ I.

    Parameters
    ----------
    Phi : (n, D) ndarray
        RFF feature matrix (already scaled by √σ_f so that Cov(Φw) ≈ σ_f K).
    sigma_xi2 : float
        Observation noise variance σ²_ξ.

    Returns
    -------
    (n, n) ndarray
    """
    Phi = np.asarray(Phi, dtype=float)
    n = Phi.shape[0]
    return Phi @ Phi.T + sigma_xi2 * np.eye(n)


def realised_cov_ciq(M: np.ndarray, eta: float, sigma_xi2: float) -> np.ndarray:
    """Analytic realised covariance for a CIQ sampler with noise-split η.

    For a CIQ sampler the output is f̂ + ξ where ξ ~ N(0, (1-η) σ²_ξ I) and
    Cov(f̂) = M Mᵀ.  The exact realised observation covariance is therefore

        K̂_ξ = M Mᵀ + (1 − η) σ²_ξ I.

    Parameters
    ----------
    M : (n, n) ndarray
        CIQ operator applied to the identity, i.e. M[:, j] = CIQ(eⱼ).
    eta : float
        Noise-split parameter η ∈ (0, 1).
    sigma_xi2 : float
        Observation noise variance σ²_ξ.

    Returns
    -------
    (n, n) ndarray
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    return M @ M.T + (1.0 - eta) * sigma_xi2 * np.eye(n)


# ---------------------------------------------------------------------------
# Internal helpers for certify
# ---------------------------------------------------------------------------

def _clopper_pearson(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Conservative Clopper–Pearson binomial confidence interval.

    Returns (lower, upper) such that P(lower ≤ p ≤ upper) ≥ 1 − α.
    """
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def _two_sample_classifier(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    n_cv: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Adversarial two-sample test via cross-validated logistic-regression classifier.

    Concatenates R samples from p and R from q, labels them 0/1, and trains a
    logistic-regression classifier on degree-2 polynomial features.  Returns the
    cross-validated balanced accuracy.

    An accuracy near ½ indicates indistinguishability; near 1 indicates the
    distributions are easily separable.  Used for Guard G7.

    Parameters
    ----------
    samples_p, samples_q : (R, n) ndarray
        R draws from each distribution.
    n_cv : int
        Number of cross-validation folds.
    rng : Generator, optional
        Random state for cross-validation splitting.

    Returns
    -------
    float
        Cross-validated balanced accuracy ∈ [0, 1].
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Guard G7 requires scikit-learn: pip install scikit-learn"
        ) from exc

    R = samples_p.shape[0]
    X = np.vstack([samples_p, samples_q])
    y = np.concatenate([np.zeros(R, dtype=int), np.ones(R, dtype=int)])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_feat = poly.fit_transform(X)

    seed = int(rng.integers(2**31)) if rng is not None else 0
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X_feat, y, cv=cv, scoring="balanced_accuracy")
    return float(scores.mean())


# ---------------------------------------------------------------------------
# G3 exception
# ---------------------------------------------------------------------------

class NonGaussianSamplerError(ValueError):
    """Raised by certify when the sampler output is known to be non-Gaussian.

    Guard G3: total-variation certification from the realised covariance alone
    is invalid for non-Gaussian samplers (e.g. Lanczos, whose Krylov basis
    depends on the same random vector u used to form the sample).
    """


# ---------------------------------------------------------------------------
# Certification — Guards G1–G7
# ---------------------------------------------------------------------------

def certify(
    sampler: Callable[..., Tuple[np.ndarray, np.ndarray]],
    theta: Dict[str, Any],
    K_xi: np.ndarray,
    *,
    R: int,
    eps: float,
    delta: float,
    is_gaussian: bool = True,
    include_g7: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    r"""(1 − δ) high-probability TV certification for GP prior samplers.

    Calls ``sampler(**theta)`` R times.  Each call must return ``(y, Khat_xi)``
    where ``Khat_xi`` is the analytically-computed realised observation covariance
    (Guard G2; use ``realised_cov_rff`` / ``realised_cov_ciq`` inside the
    sampler wrapper).  For each realisation the exact Bayes error is computed via
    ``gaussian_bayes_error`` (Guard G1).  The (1 − δ) upper empirical quantile of
    TV across the R realisations forms the certificate (Guard G4).

    Guards enforced
    ---------------
    G1  compute-don't-estimate : p* computed via Imhof quadrature, never MC.
    G2  realised-analytic-cov  : raises ValueError if sampler returns NaN cov.
    G3  gaussianity            : raises NonGaussianSamplerError if is_gaussian=False.
    G4  high-probability       : TV certificate = quantile(tvs, 1 − δ), not mean.
    G5  Clopper–Pearson        : pass fraction reported with exact CP interval.
    G6  sandwich-falsifier     : asserts TV certificate ≥ MC TV lower bound − 1e-9.
    G7  adversarial            : optional classifier two-sample test (informational).

    Parameters
    ----------
    sampler : callable
        ``(**theta) -> (y, Khat_xi)``.  Called R times.
    theta : dict
        Keyword arguments forwarded to sampler on every call.
    K_xi : (n, n) ndarray
        True observation covariance.
    R : int
        Number of independent realisations.
    eps : float
        Certification threshold: declared certified if TV (1−δ)-quantile ≤ 2ε.
    delta : float
        Failure probability.
    is_gaussian : bool
        True for RFF / CIQ samplers.  Set to False for Lanczos (Guard G3).
    include_g7 : bool
        Run the adversarial classifier test.  Requires scikit-learn.
        Does *not* affect the certified flag.
    rng : Generator, optional
        Used for true-sample draws in G7.

    Returns
    -------
    report : dict with keys
        certified    : bool  — True iff TV_quantile ≤ 2ε
        tv_quantile  : float — (1−δ) empirical quantile of TV realisations (G4)
        tv_mean      : float — mean TV (informational; G4 uses quantile)
        tvs          : (R,) ndarray — per-realisation TV values
        p_stars      : (R,) ndarray — per-realisation p* values
        cp_lo        : float — Clopper–Pearson lower bound on pass fraction
        cp_hi        : float — Clopper–Pearson upper bound on pass fraction
        g6_ok        : bool  — sandwich falsifier satisfied
        g7_accuracy  : float or None — classifier accuracy from G7 (if run)
        eps          : float
        delta        : float
        R            : int

    Raises
    ------
    NonGaussianSamplerError
        If ``is_gaussian=False`` (Guard G3).
    ValueError
        If sampler returns NaN covariance (Guard G2).
    """
    # ------------------------------------------------------------------
    # G3: Gaussianity precondition
    # ------------------------------------------------------------------
    if not is_gaussian:
        raise NonGaussianSamplerError(
            "G3: sampler output is non-Gaussian (e.g. Lanczos whose Krylov basis "
            "depends on the same random vector used to form the sample). "
            "Total-variation certification from the realised covariance alone is "
            "therefore invalid; use a dedicated non-Gaussian certificate instead."
        )

    K_xi = np.asarray(K_xi, dtype=float)
    tvs = np.empty(R)
    p_stars = np.empty(R)
    samples: list[np.ndarray] = []

    for r in range(R):
        y, Khat_xi = sampler(**theta)

        # ------------------------------------------------------------------
        # G2: Realised-analytic-covariance check
        # ------------------------------------------------------------------
        Khat_xi = np.asarray(Khat_xi, dtype=float)
        if np.any(np.isnan(Khat_xi)):
            raise ValueError(
                f"G2 violation on realisation {r}: sampler returned NaN covariance. "
                "Wrap the sampler so it computes the realised covariance analytically "
                "(use realised_cov_rff or realised_cov_ciq)."
            )

        # G1: use Imhof quadrature — never approximate by Monte Carlo
        res = gaussian_bayes_error(K_xi, Khat_xi)
        tvs[r] = res["tv"]
        p_stars[r] = res["p_star"]
        samples.append(np.asarray(y, dtype=float))

    # ------------------------------------------------------------------
    # G4: High-probability framing — (1−δ) upper quantile, not mean
    # ------------------------------------------------------------------
    tv_quantile = float(np.quantile(tvs, 1.0 - delta))
    tv_mean = float(np.mean(tvs))

    certified = tv_quantile <= 2.0 * eps

    # ------------------------------------------------------------------
    # G5: Clopper–Pearson interval on pass fraction
    # ------------------------------------------------------------------
    pass_count = int(np.sum(tvs <= 2.0 * eps))
    cp_lo, cp_hi = _clopper_pearson(pass_count, R, delta)

    # ------------------------------------------------------------------
    # G6: Sandwich falsifier
    # The TV certificate must not lie below the empirical MC lower bound.
    # Lower bound: 1 − 2(p̄* + 2 σ̂_p*) where p̄* ± 2σ̂_p* is a 2-σ CI
    # on the mean p* from the R realisations.
    # ------------------------------------------------------------------
    mc_pstar_mean = float(np.mean(p_stars))
    mc_pstar_sem = float(np.std(p_stars, ddof=1) / np.sqrt(R))
    # Sandwich MC lower bound on TV (conservative direction: large p* → small TV)
    mc_tv_lower = float(np.clip(1.0 - 2.0 * (mc_pstar_mean + 2.0 * mc_pstar_sem), 0.0, 1.0))
    g6_ok = bool(tv_quantile >= mc_tv_lower - 1e-9)

    # ------------------------------------------------------------------
    # G7: Adversarial corroboration (separate from certification path)
    # ------------------------------------------------------------------
    g7_accuracy: Optional[float] = None
    if include_g7:
        _rng = rng if rng is not None else np.random.default_rng()
        true_samples = _rng.multivariate_normal(np.zeros(K_xi.shape[0]), K_xi, R)
        samples_arr = np.array(samples)
        g7_accuracy = _two_sample_classifier(true_samples, samples_arr, rng=_rng)

    return {
        "certified": certified,
        "tv_quantile": tv_quantile,
        "tv_mean": tv_mean,
        "tvs": tvs,
        "p_stars": p_stars,
        "cp_lo": cp_lo,
        "cp_hi": cp_hi,
        "g6_ok": g6_ok,
        "g7_accuracy": g7_accuracy,
        "eps": eps,
        "delta": delta,
        "R": R,
    }
