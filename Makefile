# Makefile for the gpsampler experiment pipeline.
#
# Primary target
# --------------
#   make matern_bayes   — run the full Matérn Bayes-decision comparison
#                          sweep (Stage 1) then generate all figures (Stage 2).
#
# Smoke test
# ----------
#   make smoke          — quick end-to-end check (n≤128, R=4, 2 fidelities).
#
# Expected runtime (single CPU, no GPU)
# --------------------------------------
#   Smoke:       < 2 minutes
#   Full grid:   6–12 hours  (dominated by CIQ/PCIQ matsqrt at n=2048,
#                             which is O(n³ J) per config)
#
# The n=2048 cap is intentional: gaussian_bayes_error is O(n³) and the
# realised-covariance adapters are dense.  Do not raise this cap without
# reading the Implementation appendix on memory and runtime.

PYTHON := python
SWEEP_OUT := sweeps/matern_bayes/output

# ---------------------------------------------------------------------------
# Stage 1: sweep
# ---------------------------------------------------------------------------

.PHONY: matern_bayes_sweep
matern_bayes_sweep:
	$(PYTHON) -m sweeps.matern_bayes.run_sweep --d 1

.PHONY: matern_bayes_sweep_robust
matern_bayes_sweep_robust:
	$(PYTHON) -m sweeps.matern_bayes.run_sweep --d 2

# ---------------------------------------------------------------------------
# Stage 2: figures (require a CSV from Stage 1)
# ---------------------------------------------------------------------------
# Pass SWEEP_CSV=<path> to point at a specific sweep file, e.g.:
#   make matern_bayes_figures SWEEP_CSV=sweeps/matern_bayes/output/matern_bayes_d1_abc123.csv

SWEEP_CSV ?= $(shell ls -t $(SWEEP_OUT)/matern_bayes_d1_*.csv 2>/dev/null | head -1)

.PHONY: matern_bayes_figures
matern_bayes_figures: $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.f1_bound_sufficiency $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.f2_smoothness_rff_vs_lrff $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.f3_ciq_vs_pciq_lengthscale $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.f4_cross_method_cost $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.f5_methodology_validation $(SWEEP_CSV)
	$(PYTHON) -m figures.matern_bayes.fa_robustness $(SWEEP_CSV)

# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

.PHONY: matern_bayes
matern_bayes: matern_bayes_sweep matern_bayes_figures

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

.PHONY: smoke
smoke:
	$(PYTHON) -m sweeps.matern_bayes.run_sweep --smoke
	$(eval SMOKE_CSV := $(shell ls -t $(SWEEP_OUT)/matern_bayes_d1_*smoke*.csv 2>/dev/null | head -1))
	$(PYTHON) -m figures.matern_bayes.f1_bound_sufficiency $(SMOKE_CSV)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test_matern_bayes
test_matern_bayes:
	pytest tests/test_matern_bayes_flops.py tests/test_matern_bayes_guards.py -v
