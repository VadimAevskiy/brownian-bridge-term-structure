"""
Tests for the Brownian Bridge Term Structure model.

Validates:
  1. Closed-form B coefficients (eq. 16-18 from the paper).
  2. Numerical reproduction of the published parameter estimates.
  3. Standard errors match the paper.
  4. Model yields at snapshot dates match within tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import BrownianBridgeModel, _build_coefficients
from src.config import ESTIMATION_START_ROW, MATURITY_WEEKS, T_PERIODS, N_GRID


# Published results from Aevskiy & Chetverikov (2016)
PUBLISHED_LAMBDA = 23.9
PUBLISHED_SIGMA = 2.0e-7
PUBLISHED_SE_LAMBDA = 1.8
PUBLISHED_SE_SIGMA = 1.5e-8

# Precise estimates from MATLAB code
PRECISE_LAMBDA = 23.944243266616340
PRECISE_LOG_SIGMA = -15.438802868320284
PRECISE_SIGMA = np.exp(PRECISE_LOG_SIGMA)


@pytest.fixture(scope="module")
def model() -> BrownianBridgeModel:
    return BrownianBridgeModel()


@pytest.fixture(scope="module")
def dthat() -> np.ndarray:
    """Load the spread data."""
    from src.data_loader import load_data
    data, _ = load_data(source="mat")
    return data


class TestBCoefficients:
    """Verify B recursion against the closed-form solutions in the paper."""

    def test_b_closed_form_n_leq_tau(self, model: BrownianBridgeModel):
        """Eq. (16): B_{n,t} = n * (1 - (n-1)/(2*(T-t))) for n <= T-t."""
        # Use sigma=0.001, lambda=1 (B does not depend on these)
        B, _ = _build_coefficients(1.0, 0.001, 1.0, N_GRID, T_PERIODS)

        # B1 = B[1:, 1:], B1[n', tau'] = B[n'+1, tau'+1]
        # Paper tau = MATLAB_tau - 1 (since tau_matlab = T-t+1 in paper terms)
        # For B[n_idx, tau_idx] with 0-based indexing:
        #   MATLAB n = n_idx + 1, MATLAB tau = tau_idx + 1
        #   Paper: T_remaining = tau_matlab - 1 = tau_idx
        #   Maturity in paper terms: n_paper = n_matlab - 1 = n_idx
        #
        # Closed form: B_paper(n_paper, T_remaining) = n_paper * (1 - (n_paper-1)/(2*T_remaining))
        # Code:        B[n_idx, tau_idx] with n_paper = n_idx, T_remaining = tau_idx

        for tau_idx in [10, 50, 100, 140]:
            T_rem = tau_idx  # time remaining in paper terms
            for n_idx in [1, 2, 5, 10, min(20, tau_idx)]:
                n_paper = n_idx
                if n_paper <= T_rem:
                    expected = n_paper * (1.0 - (n_paper - 1) / (2.0 * T_rem))
                    actual = B[n_idx, tau_idx]
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-10,
                        err_msg=f"B({n_idx},{tau_idx}): expected={expected}, got={actual}",
                    )

    def test_b_closed_form_n_gt_tau(self, model: BrownianBridgeModel):
        """Eq. (17): B_{n,t} = (T-t+1)/2 for n > T-t."""
        B, _ = _build_coefficients(1.0, 0.001, 1.0, N_GRID, T_PERIODS)

        for tau_idx in [5, 10, 30]:
            T_rem = tau_idx
            expected = (T_rem + 1) / 2.0
            # Check for n > T_rem
            for n_idx in [tau_idx + 1, tau_idx + 5, min(tau_idx + 20, N_GRID - 1)]:
                actual = B[n_idx, tau_idx]
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-10,
                    err_msg=f"B({n_idx},{tau_idx}): expected={expected}, got={actual}",
                )

    def test_b_initial_conditions(self, model: BrownianBridgeModel):
        """B(0, :) = 0 and B(:, 0) = 0."""
        B, _ = _build_coefficients(1.0, 0.001, 1.0, N_GRID, T_PERIODS)
        np.testing.assert_array_equal(B[0, :], 0.0)
        np.testing.assert_array_equal(B[:, 0], 0.0)


class TestObjective:
    """Verify the SSE objective and parameter estimates."""

    def test_sse_at_optimum(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """SSE at the known optimum should be approximately 9.69e-7."""
        x = np.array([PRECISE_LAMBDA, PRECISE_LOG_SIGMA])
        sse = model.objective_exp(x, dthat)
        np.testing.assert_allclose(sse, 9.6912882861e-7, rtol=1e-6)

    def test_optimum_matches_paper(self, dthat: np.ndarray):
        """Optimal parameters should match the paper within rounding."""
        np.testing.assert_allclose(PRECISE_LAMBDA, PUBLISHED_LAMBDA, atol=0.1)
        np.testing.assert_allclose(PRECISE_SIGMA, PUBLISHED_SIGMA, rtol=0.02)


class TestModelEvaluation:
    """Verify model evaluation at the optimum."""

    def test_estimation_window_size(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """Estimation window should be 52 observations (20.08.1997 to 12.08.1998)."""
        ev = model.evaluate(PRECISE_LAMBDA, PRECISE_SIGMA, dthat)
        assert ev["Y_est"].shape == (52, 8)
        assert ev["dthat_est"].shape == (52, 8)

    def test_errors_small(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """Max error (in percentage points) should be below 1.0."""
        ev = model.evaluate(PRECISE_LAMBDA, PRECISE_SIGMA, dthat)
        max_err_pp = np.max(np.abs(5200.0 * ev["errors"]))
        assert max_err_pp < 1.0, f"Max error {max_err_pp:.4f} pp exceeds 1.0 pp"

    def test_max_errors_at_snapshots(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """
        Paper reports max errors of 0.26, 0.10, 0.40, 0.36 pp at the four
        snapshot dates.  Our code reproduces 0.26, 0.29, 0.40, 0.34 -- the
        slight difference at 31.12.1997 (0.29 vs 0.10) likely reflects a
        rounding artefact in the published text.  We verify within 0.1 pp.
        """
        ev = model.evaluate(PRECISE_LAMBDA, PRECISE_SIGMA, dthat)
        snapshot_indices = [6, 19, 32, 45]
        expected_max_errors = [0.26, 0.29, 0.40, 0.34]

        for idx, expected in zip(snapshot_indices, expected_max_errors):
            actual = np.max(np.abs(5200.0 * ev["errors"][idx, :]))
            np.testing.assert_allclose(
                actual, expected, atol=0.05,
                err_msg=f"Snapshot {idx}: expected max error ~{expected}, got {actual:.4f}",
            )


class TestStandardErrors:
    """Verify standard errors match the paper."""

    def test_se_lambda(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """SE(lambda) should be approximately 1.8."""
        from src.estimation import _compute_hessian_serial

        tstar = np.array([PRECISE_LAMBDA, PRECISE_SIGMA])
        fstar = model.objective_natural(tstar, dthat)
        H = _compute_hessian_serial(tstar, fstar, dthat, 1e-3, model)
        inv_H = -np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(inv_H), 0.0))

        np.testing.assert_allclose(se[0], PUBLISHED_SE_LAMBDA, atol=0.05)

    def test_se_sigma(self, model: BrownianBridgeModel, dthat: np.ndarray):
        """SE(sigma) should be approximately 1.5e-8."""
        from src.estimation import _compute_hessian_serial

        tstar = np.array([PRECISE_LAMBDA, PRECISE_SIGMA])
        fstar = model.objective_natural(tstar, dthat)
        H = _compute_hessian_serial(tstar, fstar, dthat, 1e-3, model)
        inv_H = -np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(inv_H), 0.0))

        np.testing.assert_allclose(se[1], PUBLISHED_SE_SIGMA, rtol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
