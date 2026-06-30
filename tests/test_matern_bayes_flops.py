"""Tests for sweeps/matern_bayes/flops.py.

Covers:
1. Monotone-in-fidelity: flops increases (or stays equal) as fidelity increases.
2. Monotone-in-n: flops increases as n increases.
3. Golden-value checks: formulae match the appendix exactly.
4. lrff n_eff default (√n when not supplied).
5. Unknown method raises ValueError.
"""

import math
import sys
import pathlib

import pytest

# Make sure repo root is on path
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sweeps.matern_bayes.flops import (
    flops,
    flops_rff,
    flops_lrff,
    flops_ciq,
    flops_pciq,
    flops_cholesky,
)


# ---------------------------------------------------------------------------
# Golden values — must match the appendix formulae exactly
# ---------------------------------------------------------------------------

class TestGoldenValues:
    def test_rff_formula(self):
        # n=100, D=50, d=1: n*D*(d//2 + 2) = 100*50*(0+2) = 10_000
        assert flops_rff(100, 50, d=1) == 10_000
        # n=200, D=100, d=2: 200*100*(1+2) = 60_000
        assert flops_rff(200, 100, d=2) == 60_000

    def test_lrff_formula(self):
        # n=100, D=50, d=1, n_eff=10: rff + n*n_eff^2 = 10_000 + 100*100 = 20_000
        assert flops_lrff(100, 50, d=1, n_eff=10.0) == 20_000
        # n=256, D=128, d=1, n_eff=16: rff=256*128*2=65536; overhead=256*256=65536; total=131072
        assert flops_lrff(256, 128, d=1, n_eff=16.0) == 65_536 + 65_536

    def test_lrff_default_neff(self):
        # When n_eff is None, defaults to sqrt(n)
        n, D = 100, 50
        expected = flops_rff(n, D, 1) + int(n * math.sqrt(n) ** 2)
        assert flops_lrff(n, D, d=1, n_eff=None) == expected

    def test_ciq_formula(self):
        # n=64, J=8: n^2 * J = 4096 * 8 = 32_768
        assert flops_ciq(64, 8) == 32_768
        # n=256, J=16: 256^2 * 16 = 1_048_576
        assert flops_ciq(256, 16) == 256 * 256 * 16

    def test_pciq_formula(self):
        # n=64, J=8: ciq + n^{3/2} = 32_768 + 512 = 33_280
        assert flops_pciq(64, 8) == flops_ciq(64, 8) + int(64 ** 1.5)
        # n=256, J=16:
        assert flops_pciq(256, 16) == flops_ciq(256, 16) + int(256 ** 1.5)

    def test_cholesky_formula(self):
        # n=3: 3^3/3 = 9
        assert flops_cholesky(3) == 9
        # n=6: 6^3/3 = 72
        assert flops_cholesky(6) == 72
        # n=100: 100^3/3 = 333_333
        assert flops_cholesky(100) == 100 ** 3 // 3

    def test_dispatch(self):
        assert flops("rff",  100, 50, d=1)         == flops_rff(100, 50, 1)
        assert flops("lrff", 100, 50, d=1,
                     n_eff=10.0)                   == flops_lrff(100, 50, 1, 10.0)
        assert flops("ciq",  64, 8)                == flops_ciq(64, 8)
        assert flops("pciq", 64, 8)                == flops_pciq(64, 8)
        assert flops("chol", 64, 0)                == flops_cholesky(64)


# ---------------------------------------------------------------------------
# Monotone in fidelity
# ---------------------------------------------------------------------------

class TestMonotoneInFidelity:
    @pytest.mark.parametrize("method,fids", [
        ("rff",  [10, 50, 100, 500]),
        ("lrff", [10, 50, 100, 500]),
        ("ciq",  [4,  8,  16,   32]),
        ("pciq", [4,  8,  16,   32]),
    ])
    def test_monotone_fidelity(self, method, fids):
        prev = 0
        for f in fids:
            current = flops(method, n=128, fidelity=f, d=1, n_eff=10.0)
            assert current >= prev, (
                f"flops({method!r}, n=128, fidelity={f}) = {current} "
                f"< previous fidelity {prev}"
            )
            prev = current


# ---------------------------------------------------------------------------
# Monotone in n
# ---------------------------------------------------------------------------

class TestMonotoneInN:
    @pytest.mark.parametrize("method,fid", [
        ("rff",  50),
        ("lrff", 50),
        ("ciq",  10),
        ("pciq", 10),
    ])
    def test_monotone_n(self, method, fid):
        ns = [64, 128, 256, 512]
        prev = 0
        for n in ns:
            current = flops(method, n=n, fidelity=fid, d=1, n_eff=math.sqrt(n))
            assert current >= prev, (
                f"flops({method!r}, n={n}, fidelity={fid}) = {current} "
                f"< previous n cost {prev}"
            )
            prev = current


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            flops("lanczos", 64, 8)

    def test_cg_raises(self):
        with pytest.raises(ValueError):
            flops("cg", 64, 8)
