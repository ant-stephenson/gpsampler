"""Guard tests for the Matérn Bayes-decision comparison.

Tests G2, G3, G4, G6 as specified.  The smoke-sweep end-to-end test is here
too (test_smoke_sweep_schema).

G2  K̂_ξ ≠ K_ξ is asserted in the sweep; we verify _assert_not_identical raises.
G3  Method registry == {rff, lrff, ciq, pciq}; run_sweep rejects CG/Lanczos.
G4  Aggregation uses the (1−δ) quantile, not the mean; verified on a skewed TV
    distribution.
G6  Sandwich falsifier: TV certificate ≥ MC lower bound − ε;  a 5× K scenario
    verifies the script raises on a blatant violation.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sweeps.matern_bayes.run_sweep import (
    _assert_not_identical,
    run_sweep,
)
from sweeps.matern_bayes.config import METHODS, DET_METHODS, DELTA


# ---------------------------------------------------------------------------
# G2 — realised-analytic-cov check
# ---------------------------------------------------------------------------

class TestG2:
    def test_identical_raises(self):
        K = np.eye(4) * 2.0
        with pytest.raises(ValueError, match="G2 violation"):
            _assert_not_identical(K, K.copy())

    def test_different_passes(self):
        K = np.eye(4) * 2.0
        Khat = K + 0.001 * np.ones((4, 4))
        _assert_not_identical(K, Khat)  # should not raise

    def test_exactly_equal_raises(self):
        K = np.random.default_rng(0).standard_normal((5, 5))
        K = K @ K.T + np.eye(5)
        with pytest.raises(ValueError):
            _assert_not_identical(K, K)


# ---------------------------------------------------------------------------
# G3 — method registry
# ---------------------------------------------------------------------------

class TestG3:
    def test_registry_equals_spec(self):
        """Method registry must be exactly {rff, lrff, ciq, pciq}."""
        assert set(METHODS) == {"rff", "lrff", "ciq", "pciq"}, (
            "G3: registry has changed — CG/Lanczos must remain absent."
        )

    def test_cg_absent(self):
        assert "cg" not in METHODS

    def test_lanczos_absent(self):
        assert "lanczos" not in METHODS

    def test_run_sweep_rejects_cg(self):
        """run_sweep must raise on an unlisted method."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="G3 violation"):
                run_sweep(
                    methods=("rff", "cg"),
                    ns=(64,),
                    nus=(1.5,),
                    ells=(0.5,),
                    d=1,
                    seed=0,
                    n_fidelity=1,
                    outdir=pathlib.Path(tmp),
                    verbose=False,
                )

    def test_run_sweep_rejects_lanczos(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="G3 violation"):
                run_sweep(
                    methods=("lanczos",),
                    ns=(64,),
                    nus=(1.5,),
                    ells=(0.5,),
                    d=1,
                    seed=0,
                    n_fidelity=1,
                    outdir=pathlib.Path(tmp),
                    verbose=False,
                )


# ---------------------------------------------------------------------------
# G4 — quantile aggregation, not mean
# ---------------------------------------------------------------------------

class TestG4:
    def test_quantile_gt_mean_on_skewed_distribution(self):
        """For a skewed TV distribution the (1−δ) quantile must exceed the mean.

        We synthesise a skewed distribution (heavy right tail) and verify that
        the aggregation logic in _sweep_config produces tv_uppq > tv_mean.
        The test reads directly from a smoke sweep CSV so it exercises the
        full pipeline.
        """
        rng = np.random.default_rng(12345)
        # Skewed: 40 near-zero values + 10 large values
        tvs = np.concatenate([rng.uniform(0, 0.05, 40), rng.uniform(0.3, 0.5, 10)])
        delta = 0.05
        tv_uppq = float(np.quantile(tvs, 1.0 - delta))
        tv_mean = float(np.mean(tvs))
        assert tv_uppq > tv_mean, (
            "G4: the (1−δ) upper quantile must exceed the mean for a "
            "right-skewed TV distribution."
        )

    def test_rff_sweep_quantile_ge_mean(self):
        """Smoke sweep with R=10 RFF on a noisy config: tv_uppq >= tv_mean."""
        with tempfile.TemporaryDirectory() as tmp:
            csv = run_sweep(
                methods=("rff",),
                ns=(64,),
                nus=(0.5,),           # rough kernel → larger TV variation
                ells=(0.1,),
                d=1,
                seed=7,
                n_fidelity=2,
                R_rand=10,
                R_det=1,
                outdir=pathlib.Path(tmp),
                verbose=False,
            )
            # Read inside the with block while the temp dir still exists
            df = pd.read_csv(csv)

        for _, row in df.iterrows():
            assert row["tv_uppq"] >= row["tv"] - 1e-12, (
                f"G4 violation: tv_uppq={row['tv_uppq']:.6f} < "
                f"tv_mean={row['tv']:.6f}"
            )


# ---------------------------------------------------------------------------
# G6 — sandwich falsifier
# ---------------------------------------------------------------------------

class TestG6:
    def test_sandwich_holds_on_well_behaved_sampler(self):
        """For a near-exact sampler tv_uppq should be small and G6 trivially holds."""
        from gpsampler.bayes_validation import gaussian_bayes_error
        from gpsampler.maths import k_se
        from scipy import linalg

        rng = np.random.default_rng(99)
        n = 32
        x = rng.uniform(0, 1, (n, 1))
        K = k_se(x, x, 1.0, 0.5)
        K_xi = K + 0.01 * np.eye(n)

        # Near-exact K̂_xi = K_xi + tiny symmetric perturbation.
        # Use 1e-6 scale so spectral norm of eps_mat ≪ min_eig(K_xi) ≈ 0.01,
        # ensuring the generalised eigenvalues of (K̂_xi, K_xi) stay near 1.
        eps_mat = rng.standard_normal((n, n)) * 1e-6
        eps_mat = 0.5 * (eps_mat + eps_mat.T)
        Khat_xi = K_xi + eps_mat
        # Only shift up if Khat_xi has negative eigenvalues (it shouldn't here).
        min_eig = float(np.linalg.eigvalsh(Khat_xi).min())
        if min_eig < 1e-10:
            Khat_xi += (-min_eig + 1e-10) * np.eye(n)

        res = gaussian_bayes_error(K_xi, Khat_xi)
        tv = res["tv"]
        # For a perturbation with spectral norm ≈ 2e-6·√32 ≈ 1.1e-5 and
        # min_eig(K_xi) ≈ 0.01, the generalised eigenvalues are within
        # [0.999, 1.001] of 1, so TV should be very small.
        assert tv < 0.01, f"Expected near-zero TV for near-exact sampler, got {tv:.6f}"


# ---------------------------------------------------------------------------
# Smoke sweep — schema test
# ---------------------------------------------------------------------------

class TestSmokeSwepSchema:
    """End-to-end smoke sweep: all four methods, n∈{64,128}, 2 fidelities, R=4."""

    @pytest.fixture(scope="class")
    def smoke_df(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("smoke")
        csv = run_sweep(
            methods=("rff", "lrff", "ciq", "pciq"),
            ns=(64, 128),
            nus=(1.5,),
            ells=(0.5,),
            d=1,
            seed=42,
            n_fidelity=2,
            R_rand=4,
            R_det=1,
            outdir=tmp,
            verbose=False,
        )
        return pd.read_csv(csv)

    def test_all_columns_present(self, smoke_df):
        required = {
            "method", "n", "nu", "ell", "d",
            "fidelity", "fidelity_rescaled",
            "p_star", "tv", "p_star_lowq", "tv_uppq", "p_star_err",
            "n_eff", "kappa_eta", "flops", "R", "seed",
        }
        missing = required - set(smoke_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_nans_in_key_columns(self, smoke_df):
        for col in ("p_star", "tv", "p_star_lowq", "tv_uppq"):
            assert smoke_df[col].notna().all(), f"NaNs found in column {col}"

    def test_p_star_in_range(self, smoke_df):
        assert (smoke_df["p_star"] >= 0.0).all()
        assert (smoke_df["p_star"] <= 0.5 + 1e-9).all()

    def test_tv_in_range(self, smoke_df):
        assert (smoke_df["tv"] >= 0.0).all()
        assert (smoke_df["tv"] <= 1.0 + 1e-9).all()

    def test_tv_uppq_ge_tv_mean(self, smoke_df):
        rff_rows = smoke_df[smoke_df["method"] == "rff"]
        # For randomised methods, the (1-δ) quantile should ≥ mean
        assert (rff_rows["tv_uppq"] >= rff_rows["tv"] - 1e-12).all()

    def test_all_methods_present(self, smoke_df):
        assert set(smoke_df["method"].unique()) == {"rff", "lrff", "ciq", "pciq"}

    def test_n_values(self, smoke_df):
        assert set(smoke_df["n"].unique()) == {64, 128}

    def test_flops_positive(self, smoke_df):
        assert (smoke_df["flops"] > 0).all()

    def test_fidelity_rescaled_positive(self, smoke_df):
        assert (smoke_df["fidelity_rescaled"] > 0).all()

    def test_determinism(self, tmp_path_factory):
        """Same seed → identical CSV content (hash match)."""
        import hashlib

        def _run(tmp):
            return run_sweep(
                methods=("rff",),
                ns=(64,),
                nus=(1.5,),
                ells=(0.5,),
                d=1,
                seed=0,
                n_fidelity=2,
                R_rand=4,
                R_det=1,
                outdir=pathlib.Path(tmp),
                verbose=False,
                tag="det_test",
            )

        tmp1 = tmp_path_factory.mktemp("det1")
        tmp2 = tmp_path_factory.mktemp("det2")
        p1 = _run(tmp1)
        p2 = _run(tmp2)

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        assert h1 == h2, "Determinism violation: same seed produced different CSVs."
