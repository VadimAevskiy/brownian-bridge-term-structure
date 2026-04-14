"""
Core model: discrete-time Brownian bridge affine term structure.

Implements the double-recursive formulas (8)-(9) from Aevskiy & Chetverikov (2016)
for computing the affine coefficients A(n, tau) and B(n, tau), and the resulting
model-implied yield spreads.

Performance:
    The nested O(N * T) loops are JIT-compiled via Numba.  On a typical laptop
    the full 261 x 155 grid evaluates in < 2 ms after the first warm-up call.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .config import (
    ANNUAL_SCALE,
    B_COL_OFFSET,
    DATA_START_ROW,
    ESTIMATION_START_ROW,
    MATURITY_WEEKS,
    MATURITY_YEARS,
    N_GRID,
    T_PERIODS,
)


# ======================================================================
# JIT-compiled kernels
# ======================================================================

@njit(cache=True)
def _build_coefficients(
    lamda: float,
    sigma_eff: float,
    k: float,
    N: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the (N, T) coefficient matrices B and A using the double recursion.

    Parameters
    ----------
    lamda : float
        Price of risk (lambda).
    sigma_eff : float
        Effective volatility fed into the A recursion.  For the optimisation
        objective ``ll_k`` this is ``exp(sigma_param)``; for the standard-error
        objective ``ll_kse`` this is ``sigma`` itself.
    k : float
        Mean-reversion speed (fixed at 1 in the published model).
    N, T : int
        Grid dimensions.

    Returns
    -------
    B, A : ndarray of shape (N, T)
    """
    B = np.zeros((N, T))
    A = np.zeros((N, T))
    half_lam2 = lamda * lamda / 2.0

    for tau_idx in range(1, T):            # 0-based, maps to MATLAB tau = tau_idx + 1
        tau_m = tau_idx + 1                # MATLAB 1-based tau
        inv_tau_m1 = 1.0 / (tau_m - 1)    # 1 / (tau - 1) in MATLAB
        for n_idx in range(1, N):          # 0-based, maps to MATLAB n = n_idx + 1
            n_m = n_idx + 1
            b_prev = B[n_idx - 1, tau_idx - 1]
            a_prev = A[n_idx - 1, tau_idx - 1]

            if tau_m >= n_m:
                B[n_idx, tau_idx] = 1.0 + b_prev * k * (1.0 - inv_tau_m1)
            else:
                B[n_idx, tau_idx] = B[tau_idx, tau_idx]

            term = lamda + b_prev * sigma_eff
            A[n_idx, tau_idx] = half_lam2 + a_prev - term * term / 2.0

    return B, A


@njit(cache=True)
def _extract_yields(
    A2: np.ndarray,
    B2: np.ndarray,
    dthat: np.ndarray,
    mat_weeks: np.ndarray,
    capt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the unobservable short rate z_t from the 1-month (4-week) spread
    and compute model-implied yields for all maturities.

    Parameters
    ----------
    A2, B2 : ndarray of shape (8, capt)
        Sliced affine coefficients for the 8 observed maturities.
    dthat : ndarray of shape (capt, 8)
        Observed spread data divided by ANNUAL_SCALE.
    mat_weeks : ndarray of shape (8,)
        Maturity in weeks.
    capt : int
        Number of time observations.

    Returns
    -------
    Y : ndarray of shape (capt, 8)
        Model-implied yields.
    r : ndarray of shape (capt,)
        Extracted unobservable factor z_t.
    """
    r = np.empty(capt)
    Y = np.empty((capt, 8))

    for j in range(capt):
        r[j] = (4.0 * dthat[j, 0] - A2[0, j]) / B2[0, j]

    for j in range(capt):
        for m in range(8):
            Y[j, m] = (A2[m, j] + B2[m, j] * r[j]) / mat_weeks[m]

    return Y, r


# ======================================================================
# High-level interface
# ======================================================================

class BrownianBridgeModel:
    """
    Discrete-time Brownian bridge term structure model.

    Parameters
    ----------
    T : int
        Number of periods until monetary union.
    N : int
        Maximum maturity grid size in weeks.
    k : float
        Mean-reversion parameter (default 1).
    """

    __slots__ = ("T", "N", "k", "_mat_weeks", "_mat_idx")

    def __init__(
        self,
        T: int = T_PERIODS,
        N: int = N_GRID,
        k: float = 1.0,
    ) -> None:
        self.T = T
        self.N = N
        self.k = k
        self._mat_weeks = MATURITY_WEEKS.astype(np.float64)
        # 0-based indices into B1 = B[1:, 1:]  ->  MATLAB imat minus 1
        self._mat_idx = MATURITY_WEEKS - 1

    # ------------------------------------------------------------------
    # Coefficient computation
    # ------------------------------------------------------------------

    def build_coefficients(
        self,
        lamda: float,
        sigma_eff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return full (N, T) coefficient grids B, A."""
        return _build_coefficients(lamda, sigma_eff, self.k, self.N, self.T)

    # ------------------------------------------------------------------
    # Objective for optimisation (sigma parameterised as exp)
    # ------------------------------------------------------------------

    def objective_exp(
        self,
        bigtheto: np.ndarray,
        dthat: np.ndarray,
    ) -> float:
        """
        Sum-of-squared-errors objective (matches MATLAB ``ll_k.m``).

        ``bigtheto = [lambda, log_sigma]``; volatility enters A recursion
        as ``exp(log_sigma)`` to enforce positivity.
        """
        lamda, log_sigma = bigtheto[0], bigtheto[1]
        sigma_eff = np.exp(log_sigma)
        return self._sse(lamda, sigma_eff, dthat)

    # ------------------------------------------------------------------
    # Objective for standard errors (sigma in natural scale)
    # ------------------------------------------------------------------

    def objective_natural(
        self,
        tstar: np.ndarray,
        dthat: np.ndarray,
    ) -> float:
        """
        SSE objective with natural parameterisation (matches ``ll_kse.m``).

        ``tstar = [lambda, sigma]``.
        """
        return self._sse(tstar[0], tstar[1], dthat)

    # ------------------------------------------------------------------
    # Full model evaluation (for figures / diagnostics)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        lamda: float,
        sigma: float,
        dthat: np.ndarray,
    ) -> dict:
        """
        Run the model and return a dict with all intermediate quantities
        needed for plotting (matches MATLAB ``ll_k_test.m``).

        Parameters
        ----------
        lamda : float
        sigma : float
            Volatility in natural scale.
        dthat : ndarray of shape (capt, 8)
            Observed spreads / ANNUAL_SCALE.

        Returns
        -------
        dict with keys:
            Y_full, Y_est, dthat_est, errors, r, q_mat, sse
        """
        capt = dthat.shape[0]
        B, A = _build_coefficients(lamda, sigma, self.k, self.N, self.T)
        B2, A2 = self._slice_coefficients(B, A, capt)
        Y_full, r = _extract_yields(A2, B2, dthat, self._mat_weeks, capt)

        Y_est = Y_full[ESTIMATION_START_ROW:]
        dthat_est = dthat[ESTIMATION_START_ROW:]
        errors = dthat_est - Y_est
        sse = float(np.sum(errors ** 2))

        return {
            "Y_full": Y_full,
            "Y_est": Y_est,
            "dthat_est": dthat_est,
            "errors": errors,
            "r": r,
            "q_mat": MATURITY_YEARS.copy(),
            "sse": sse,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _slice_coefficients(
        self,
        B: np.ndarray,
        A: np.ndarray,
        capt: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice (N, T) grids to (8, capt) for the 8 observed maturities."""
        B1 = B[1:, 1:]
        A1 = A[1:, 1:]
        B2 = B1[self._mat_idx][:, B_COL_OFFSET : B_COL_OFFSET + capt]
        A2 = A1[self._mat_idx][:, B_COL_OFFSET : B_COL_OFFSET + capt]
        return B2, A2

    def _sse(
        self,
        lamda: float,
        sigma_eff: float,
        dthat: np.ndarray,
    ) -> float:
        capt = dthat.shape[0]
        B, A = _build_coefficients(lamda, sigma_eff, self.k, self.N, self.T)
        B2, A2 = self._slice_coefficients(B, A, capt)
        Y, _ = _extract_yields(A2, B2, dthat, self._mat_weeks, capt)
        Y_sub = Y[ESTIMATION_START_ROW:]
        dthat_sub = dthat[ESTIMATION_START_ROW:]
        return float(np.sum((dthat_sub - Y_sub) ** 2))
