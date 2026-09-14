"""Stratified truncated-Taylor RFF sampler."""

import numpy as np
from typing import Tuple

from ._utils import spectral_sampler, construct_kernels, NPInputMat, NPSample, NPKernel


# ---------------------------------------------------------------------------
# Multi-index and Taylor helpers
# ---------------------------------------------------------------------------

def _enumerate_multi_indices(d: int, R: int):
    """All d-tuples alpha with |alpha| <= R, sorted by total degree."""
    from itertools import product as _product
    return sorted(
        [a for a in _product(range(R + 1), repeat=d) if sum(a) <= R],
        key=lambda a: (sum(a), a),
    )


def _monomial_design(X: np.ndarray, alphas: list) -> np.ndarray:
    """Build monomial design matrix Phi (n, r) for multi-indices alphas."""
    n, d = X.shape
    r = len(alphas)
    Phi = np.ones((n, r), dtype=np.float64)
    for col, a in enumerate(alphas):
        for j in range(d):
            if a[j] > 0:
                Phi[:, col] *= X[:, j] ** a[j]
    return Phi


def _taylor_coeffs_batch(omega: np.ndarray, alphas: list) -> np.ndarray:
    """Taylor coefficients c_alpha(omega) = i^{|alpha|} / alpha! * omega^alpha.

    Parameters
    ----------
    omega  : (F, d) frequency array
    alphas : list of d-tuples

    Returns
    -------
    C : (F, r) complex array
    """
    from math import factorial as _fact
    F, d = omega.shape
    r = len(alphas)
    C = np.empty((F, r), dtype=np.complex128)
    for col, a in enumerate(alphas):
        afact = 1
        for ai in a:
            afact *= _fact(ai)
        coeff = (1j ** sum(a)) / afact
        col_val = np.ones(F, dtype=np.complex128)
        for j in range(d):
            if a[j] > 0:
                col_val *= omega[:, j] ** a[j]
        C[:, col] = coeff * col_val
    return C


def _choose_taylor_order(Z_max: float, eps: float = 0.4) -> int:
    """Smallest R such that sup_{|z|<=Z_max} |e^{iz} - T_R(iz)| <= eps."""
    zs = np.linspace(-Z_max, Z_max, 400)
    for R in range(1, 200):
        term = np.ones_like(zs, dtype=complex)
        acc = term.copy()
        for k in range(1, R + 1):
            term = term * (1j * zs) / k
            acc = acc + term
        if np.max(np.abs(np.exp(1j * zs) - acc)) <= eps:
            return R
    return 200


# ---------------------------------------------------------------------------
# Moment matrices
# ---------------------------------------------------------------------------

def _raw_gaussian_moments(s: float, B: float, max_deg: int) -> np.ndarray:
    """Compute M_k = int_{-B}^{B} w^k * N(w; 0, s^2) dw for k=0..max_deg.

    Uses scipy.integrate.quad for accuracy.  For SE kernels the spectral
    density is N(0, 1/l^2) so s = 1/l.
    """
    from scipy.integrate import quad
    from scipy.stats import norm as _norm
    moments = np.zeros(max_deg + 1)
    for k in range(max_deg + 1):
        integrand = lambda w, _k=k: w**_k * _norm.pdf(w, scale=s)
        moments[k], _ = quad(integrand, -B, B)
    return moments


def _truncated_normal_moments(s: float, B: float, max_deg: int) -> np.ndarray:
    """Fast O(max_deg) recurrence for M_k = int_{-B}^{B} w^k N(w;0,s²) dw.

    Recurrence (even k ≥ 2):
        M_k = s²(k−1)·M_{k−2} − 2s²·B^{k−1}·φ_s(B)
    Odd moments vanish by symmetry.
    """
    from scipy.stats import norm as _norm
    mom = np.zeros(max_deg + 1)
    mom[0] = 2.0 * _norm.cdf(B / s) - 1.0
    phi_B = _norm.pdf(B, scale=s)
    s2 = s * s
    for k in range(2, max_deg + 1, 2):
        mom[k] = s2 * (k - 1) * mom[k - 2] - 2.0 * s2 * B ** (k - 1) * phi_B
    return mom


def _matern_box_prob(l: float, B: float, nu: float, d: int) -> float:
    """Box probability π_box = P(|ω_j| ≤ B ∀j) for Matérn-ν spectral density.

    d=1: closed-form via t-distribution CDF.
    d>1: numerical integration over χ²(2ν) mixing variable.
    """
    from scipy.stats import t as _t
    box_scale_dimless = B * l  # = box_scale (dimensionless)
    if d == 1:
        return float(2.0 * _t.cdf(box_scale_dimless, df=2.0 * nu) - 1.0)
    from scipy.integrate import quad
    from scipy.stats import chi2 as _chi2, norm as _norm
    df = 2.0 * nu

    def integrand(u):
        # Conditional Gaussian scale: s(u) = √(2ν/u) / l
        # B / s(u) = B·l·√(u/(2ν))
        z = box_scale_dimless * np.sqrt(u / df)
        box_1d = 2.0 * _norm.cdf(z) - 1.0
        return box_1d ** d * _chi2.pdf(u, df=df)

    result, _ = quad(integrand, 0, np.inf, limit=100)
    return float(result)


def _build_H_matrix(alphas: list, raw_mom: np.ndarray, d: int) -> np.ndarray:
    """Build H (r, r) real matrix where H[a,b] = prod_j M_{a_j+b_j}.

    For SE kernels H is real because odd moments vanish and the complex
    pre-factor i^{|a|+|b|} / (a! b!) produces real entries when combined
    with the moment parity.
    """
    from math import factorial as _fact
    r = len(alphas)
    H = np.zeros((r, r), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            val = 1.0
            for dim in range(d):
                val *= raw_mom[a[dim] + b[dim]]
            # pre-factor: i^{|a|+|b|} / (a! b!)
            total_deg = sum(a) + sum(b)
            # i^k is real iff k is even — and M_k = 0 for odd k (symmetric),
            # so the product is always real for SE.
            i_pow = (1j ** total_deg)
            afact = 1
            bfact = 1
            for ai in a:
                afact *= _fact(ai)
            for bi in b:
                bfact *= _fact(bi)
            H[i, j] = np.real(i_pow * val / (afact * bfact))
    return H


def _build_H_matrix_matern(alphas: list, l: float, B: float,
                            nu: float, d: int) -> np.ndarray:
    """Build H (r, r) moment matrix for Matérn-ν spectral density.

    Uses the χ²(2ν) scale-mixture representation: conditioned on
    u ~ χ²(2ν), ω | u ~ N(0, (2ν/(u·l²))·I_d).  The joint box-truncated
    moments are computed via quadrature over u:

        E[∏_j ω_j^{k_j} · 1_box] = ∫ [∏_j M_{k_j}(s(u), B)] p_{χ²}(u) du

    H is real because odd moments vanish (same parity argument as SE).
    """
    from math import factorial as _fact
    from scipy.integrate import quad_vec
    from scipy.stats import chi2 as _chi2

    r = len(alphas)
    R = max(sum(a) for a in alphas)
    max_mom_deg = 2 * R
    df = 2.0 * nu

    # Precompute combined degrees (r, r, d) and complex prefactors (r, r)
    combined_degs = np.zeros((r, r, d), dtype=np.intp)
    prefactors = np.zeros((r, r), dtype=np.complex128)
    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            for dim in range(d):
                combined_degs[i, j, dim] = a[dim] + b[dim]
            total_deg = sum(a) + sum(b)
            afact = bfact = 1
            for ai in a:
                afact *= _fact(ai)
            for bi in b:
                bfact *= _fact(bi)
            prefactors[i, j] = (1j ** total_deg) / (afact * bfact)

    def integrand(u):
        s_u = np.sqrt(df / u) / l
        mom = _truncated_normal_moments(s_u, B, max_mom_deg)
        mom_vals = mom[combined_degs]           # (r, r, d)
        joint_mom = np.prod(mom_vals, axis=2)   # (r, r)
        return (joint_mom * _chi2.pdf(u, df=df)).ravel()

    result, _ = quad_vec(integrand, 1e-12, np.inf, limit=200)
    H = np.real(prefactors * result.reshape(r, r))
    return H


# ---------------------------------------------------------------------------
# Woodbury B matrix and leverage
# ---------------------------------------------------------------------------

def _build_B_via_woodbury(Phi: np.ndarray, H: np.ndarray,
                          s2: float) -> np.ndarray:
    """B = Phi^T (Phi H Phi^T + s2 I)^{-1} Phi via Woodbury.

    Cost: O(n r^2 + r^3) — never forms n×n matrices.

    Returns B (r, r) symmetric positive semi-definite.
    """
    r = Phi.shape[1]
    G = Phi.T @ Phi  # (r, r)

    # Regularise H for Cholesky
    H_reg = H + 1e-12 * np.eye(r)
    try:
        L = np.linalg.cholesky(H_reg)
    except np.linalg.LinAlgError:
        # H may be singular/near-singular; fall back to eigendecomposition
        evals, evecs = np.linalg.eigh(H_reg)
        evals = np.maximum(evals, 1e-12)
        L = evecs * np.sqrt(evals)  # (r, r) "pseudo-Cholesky" V = Phi @ L

    VtV = L.T @ G @ L  # (r, r)  = (Phi L)^T (Phi L)
    M = VtV + s2 * np.eye(r)
    # B = Phi^T inv(A_R) Phi = H - s2 L^T inv(V^T V + s2 I) L
    M_inv = np.linalg.solve(M, np.eye(r))
    B = H_reg - s2 * (L.T @ M_inv @ L)
    # Symmetrise
    B = 0.5 * (B + B.T)
    return B


def _leverage_batch(C: np.ndarray, B_mat: np.ndarray) -> np.ndarray:
    """Compute leverage a(omega) = Re(c)^T B Re(c) + Im(c)^T B Im(c).

    Parameters
    ----------
    C     : (F, r) complex Taylor coefficient matrix
    B_mat : (r, r) real symmetric Woodbury factor

    Returns
    -------
    a : (F,) leverage scores (non-negative)
    """
    Cr = C.real  # (F, r)
    Ci = C.imag  # (F, r)
    # a_j = Cr[j] @ B @ Cr[j] + Ci[j] @ B @ Ci[j]
    a = np.sum((Cr @ B_mat) * Cr, axis=1) + np.sum((Ci @ B_mat) * Ci, axis=1)
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------------

def _rejection_sample_vectorised(
    d: int, s: float, B: float,
    B_mat: np.ndarray, alphas: list, M_bound: float,
    n_accept: int, rng: np.random.Generator,
    batch: int = 20000,
    kernel_type: str = "rbf",
    nu: float = 1.5,
    l: float = 1.0,
    pi_box: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rejection-sample frequencies from box-truncated spectral density weighted by leverage.

    SE: proposes from N(0, s² I_d) | [-B,B]^d via truncnorm.
    Matérn: proposes from full multivariate t, filters to [-B,B]^d.

    Returns
    -------
    omega_acc : (n_accept, d) accepted frequencies
    a_acc     : (n_accept,) leverage at accepted frequencies
    n_proposed: total proposals made
    """
    is_se = kernel_type in ("rbf", "se")
    collected_w = []
    collected_a = []
    total_proposed = 0

    if is_se:
        from scipy.stats import truncnorm as _tn
        a_tn, b_tn = -B / s, B / s

    while sum(len(c) for c in collected_w) < n_accept:
        if is_se:
            omega = np.empty((batch, d), dtype=np.float64)
            for j in range(d):
                omega[:, j] = _tn.rvs(a_tn, b_tn, loc=0, scale=s, size=batch,
                                       random_state=rng)
            n_in_box = batch
        else:
            # Matérn: draw from full multivariate t, filter to box
            kind = "matern" if nu < 1000 else "rbf"
            batch_inflated = max(batch, int(np.ceil(batch / max(pi_box, 1e-6))))
            omega_full = spectral_sampler(batch_inflated, d, kind, l, nu, rng)
            in_box = np.all(np.abs(omega_full) <= B, axis=1)
            omega = omega_full[in_box]
            n_in_box = len(omega)
            total_proposed += batch_inflated
            if n_in_box == 0:
                continue

        C = _taylor_coeffs_batch(omega, alphas)
        a_vals = _leverage_batch(C, B_mat)
        accept = rng.uniform(size=n_in_box) * M_bound <= a_vals
        if is_se:
            total_proposed += batch
        if accept.any():
            collected_w.append(omega[accept])
            collected_a.append(a_vals[accept])

    omega_all = np.concatenate(collected_w, axis=0)[:n_accept]
    a_all = np.concatenate(collected_a)[:n_accept]
    return omega_all, a_all, total_proposed


# ---------------------------------------------------------------------------
# Main stratified RFF pipeline
# ---------------------------------------------------------------------------

def _stratified_rff_draw_frequencies(
    x: np.ndarray,
    l: float,
    noise_var: float,
    rng: np.random.Generator,
    m: int,
    eps: float = 0.4,
    box_scale: float = 3.0,
    eta: float = None,
    rank_cap: int = 5000,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw m frequencies and compute per-frequency stratified Taylor RFF amplitudes.

    Supports SE (rbf) and Matérn kernels.  Returns (omega, a) where omega is
    (m, d) and a is (m,).  The amplitudes encode sqrt(2 * ratio / D) with
    D = 2*m, so that K_acc = sum_j a_j^2 (cos cos^T + sin sin^T) satisfies
    E[K_acc] = K (unit output scale).
    """
    from scipy.stats import norm as _norm

    is_se = kernel_type in ("rbf", "se")
    kind = "rbf" if is_se else "matern"

    D = 2 * m
    n, d = x.shape
    s2 = noise_var

    # ---- Spectral scale and box -----------------------------------------
    s = 1.0 / l  # spectral scale (SE std; also sets box for Matérn)
    B = box_scale * s  # frequency box half-width

    # ---- Taylor order ---------------------------------------------------
    Bx = np.max(np.abs(x))
    Z_max = Bx * B * np.sqrt(d) if d > 1 else Bx * B
    R = _choose_taylor_order(Z_max, eps)
    alphas = _enumerate_multi_indices(d, R)
    r = len(alphas)
    if r > rank_cap:
        while r > rank_cap and R > 1:
            R -= 1
            alphas = _enumerate_multi_indices(d, R)
            r = len(alphas)

    # ---- Monomial design Phi (n, r) -------------------------------------
    Phi = _monomial_design(x, alphas)

    # ---- Moment matrix H (r, r) -----------------------------------------
    if is_se:
        max_deg = 2 * R
        raw_mom = _raw_gaussian_moments(s, B, max_deg)
        H = _build_H_matrix(alphas, raw_mom, d)
    else:
        H = _build_H_matrix_matern(alphas, l, B, nu, d)

    # ---- Woodbury B = Phi^T inv(A_R) Phi --------------------------------
    B_mat = _build_B_via_woodbury(Phi, H, s2)

    # ---- Leverage bound M -----------------------------------------------
    zs = np.linspace(-Z_max, Z_max, 400) if Z_max > 0 else np.array([0.0])
    term = np.ones_like(zs, dtype=complex)
    acc = term.copy()
    for k in range(1, R + 1):
        term = term * (1j * zs) / k
        acc = acc + term
    actual_eps = np.max(np.abs(np.exp(1j * zs) - acc))
    M_bound = n * (1.0 + actual_eps) ** 2 / s2

    # ---- Box probability π_box ------------------------------------------
    if is_se:
        pi_box = float((_norm.cdf(B, scale=s) - _norm.cdf(-B, scale=s)) ** d)
    else:
        pi_box = _matern_box_prob(l, B, nu, d)

    # ---- Tighten M_bound empirically ------------------------------------
    # The theoretical bound n·(1+ε)²/σ² can be orders of magnitude above
    # the true leverage maximum, especially for Matérn kernels, making
    # rejection sampling impractically slow.  We draw a probe batch from
    # the spectral density (filtered to the box), compute leverage scores,
    # and use 2× the observed maximum as a tighter bound.
    n_probe = 20_000
    probe_rng = np.random.default_rng(rng.integers(2**63))
    if is_se:
        from scipy.stats import truncnorm as _tn_probe
        a_tn_p, b_tn_p = -B / s, B / s
        omega_probe = np.empty((n_probe, d), dtype=np.float64)
        for j in range(d):
            omega_probe[:, j] = _tn_probe.rvs(
                a_tn_p, b_tn_p, loc=0, scale=s,
                size=n_probe, random_state=probe_rng)
    else:
        n_draw = int(np.ceil(n_probe / max(pi_box, 0.01)))
        omega_probe_full = spectral_sampler(n_draw, d, kind, l, nu, probe_rng)
        in_box_probe = np.all(np.abs(omega_probe_full) <= B, axis=1)
        omega_probe = omega_probe_full[in_box_probe][:n_probe]

    if len(omega_probe) >= 100:
        C_probe = _taylor_coeffs_batch(omega_probe, alphas)
        a_probe = _leverage_batch(C_probe, B_mat)
        a_max_obs = float(np.max(a_probe))
        if a_max_obs > 0:
            M_bound = min(M_bound, 2.0 * a_max_obs)

    # ---- Rejection-sample pilot for eta estimation ------------------------
    n_pilot = min(m, 5000)
    omega_pilot, a_pilot, _ = _rejection_sample_vectorised(
        d, s, B, B_mat, alphas, M_bound, n_pilot, rng,
        kernel_type=kernel_type, nu=nu, l=l, pi_box=pi_box,
    )

    # ---- d_l = E_p[a * 1_box]: average leverage under FULL p ------------
    d_l = float(np.mean(a_pilot)) * pi_box
    T_l = float(np.var(a_pilot)) * pi_box**2

    if eta is None:
        if T_l > 0 and d_l > 0:
            eta = np.sqrt(T_l) / (d_l + np.sqrt(T_l))
            eta = np.clip(eta, 0.01, 0.99)
        else:
            eta = 0.5

    # ---- Draw m frequencies from mixture q = (1-eta)*g + eta*p ----------
    from_p = rng.uniform(size=m) < eta
    n_from_p = int(from_p.sum())
    n_from_g = m - n_from_p

    omega_all = np.empty((m, d), dtype=np.float64)
    a_all = np.zeros(m, dtype=np.float64)

    # eta fraction: draw from full p
    if n_from_p > 0:
        omega_all[from_p] = spectral_sampler(n_from_p, d, kind, l, nu, rng)
        in_box_p = np.all(np.abs(omega_all[from_p]) <= B, axis=1)
        if in_box_p.any():
            C_p = _taylor_coeffs_batch(omega_all[from_p][in_box_p], alphas)
            a_p = _leverage_batch(C_p, B_mat)
            a_all_p = np.zeros(n_from_p)
            a_all_p[in_box_p] = a_p
            a_all[from_p] = a_all_p

    # (1-eta) fraction: draw from rejection sampler (in box)
    if n_from_g > 0:
        if n_from_g <= len(omega_pilot):
            omega_all[~from_p] = omega_pilot[:n_from_g]
            a_all[~from_p] = a_pilot[:n_from_g]
        else:
            omega_rej, a_rej, _ = _rejection_sample_vectorised(
                d, s, B, B_mat, alphas, M_bound, n_from_g, rng,
                kernel_type=kernel_type, nu=nu, l=l, pi_box=pi_box,
            )
            omega_all[~from_p] = omega_rej
            a_all[~from_p] = a_rej

    # ---- IS weights: w = p / q with NO pi_box factor --------------------
    in_box = np.all(np.abs(omega_all) <= B, axis=1)
    ratio = np.full(m, 1.0 / eta)
    ratio[in_box] = 1.0 / (eta + (1.0 - eta) * a_all[in_box] / d_l)
    a_feat = np.sqrt(2.0 * ratio / D)  # (m,)  NO pi_box here

    return omega_all, a_feat


def _build_stratified_rff_features(
    x: np.ndarray,
    l: float,
    noise_var: float,
    rng: np.random.Generator,
    D: int,
    eps: float = 0.4,
    box_scale: float = 3.0,
    eta: float = None,
    rank_cap: int = 5000,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> np.ndarray:
    """Build stratified truncated-Taylor RFF features.  Returns Z (n, D)."""
    if D % 2 != 0:
        raise ValueError("D must be even")
    n, d = x.shape
    m = D // 2

    omega, a = _stratified_rff_draw_frequencies(
        x, l, noise_var, rng, m,
        eps=eps, box_scale=box_scale, eta=eta, rank_cap=rank_cap,
        kernel_type=kernel_type, nu=nu,
    )

    proj = x.astype(np.float64) @ omega.T  # (n, m)
    Z = np.empty((n, D), dtype=np.float64)
    Z[:, :m] = a[None, :] * np.cos(proj)
    Z[:, m:] = a[None, :] * np.sin(proj)
    return Z


def sample_stratified_rff_from_x(
    x: np.ndarray,
    sigma: float,
    noise_var: float,
    l: float,
    rng: np.random.Generator,
    D: int,
    eps: float = 0.4,
    box_scale: float = 3.0,
    eta: float = None,
    rank_cap: int = 5000,
    kernel_type: str = "rbf",
    nu: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """Stratified truncated-Taylor RFF GP prior sampler.

    Supports SE (rbf) and Matérn kernels.  Implements the paper's
    Algorithm 1: Taylor polynomial feature map, Woodbury leverage scoring,
    rejection sampling from box-truncated spectral density, and safeguarded
    importance weighting.

    Returns (y_noise, np.nan) — no n×n matrix is formed.
    """
    n = x.shape[0]
    Z = _build_stratified_rff_features(
        x, l, noise_var, rng, D,
        eps=eps, box_scale=box_scale, eta=eta, rank_cap=rank_cap,
        kernel_type=kernel_type, nu=nu,
    )
    Z = float(np.sqrt(sigma)) * Z
    w = rng.standard_normal(D)
    y_noise = Z @ w + rng.normal(scale=float(np.sqrt(noise_var)), size=n)
    return y_noise, np.nan
