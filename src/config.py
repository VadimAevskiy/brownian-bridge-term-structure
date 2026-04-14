"""
Configuration and constants for the Brownian Bridge Term Structure model.

References:
    Aevskiy, V. and Chetverikov, V. (2016), 'A discrete time model of convergence
    for the term structure of interest rates in the case of entering a monetary
    union', Applied Economics, 48:25, 2333-2340.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------
T_PERIODS: int = 155          # Total weeks until monetary union (01.01.1999)
N_GRID: int = 261             # Maximum maturity grid in weeks

# Maturity indices (weeks) used for the 8 observed yields
MATURITY_WEEKS: np.ndarray = np.array([4, 13, 26, 52, 104, 156, 208, 260], dtype=np.int64)

# Maturity in years (for plotting / labelling)
MATURITY_YEARS: np.ndarray = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 4, 5])

# Maturity labels for display
MATURITY_LABELS: list[str] = [
    "1-month", "3-month", "6-month", "1-year",
    "2-year", "3-year", "4-year", "5-year",
]

# Weekly-to-annual scaling factor (52 weeks x 100 for percentage points)
ANNUAL_SCALE: float = 5200.0

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
# MATLAB row 268 (1-based) = Python row 267 (0-based): start of spread data
DATA_START_ROW: int = 267

# MATLAB row 87 (1-based) within the sub-sample = Python row 86 (0-based):
# start of the estimation window (20.08.1997)
ESTIMATION_START_ROW: int = 86

# B1 column offset: columns 17:end in MATLAB (1-based) -> 16: in Python
B_COL_OFFSET: int = 16

# ---------------------------------------------------------------------------
# Optimizer defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizerConfig:
    """Settings for the L-BFGS-B / Nelder-Mead optimizer."""
    x0: np.ndarray = field(
        default_factory=lambda: np.array([23.944243266616340, -15.438802868320284])
    )
    method: str = "L-BFGS-B"
    tol_x: float = 1e-16
    tol_fun: float = 1e-16
    max_fun_evals: int = 25_000
    max_iter: int = 12_500
    hessian_eps: float = 1e-3

# ---------------------------------------------------------------------------
# Figure configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FigureConfig:
    """Dates (as row indices into the 52-row estimation window) for Figure 2."""
    # Date-matched indices into the 52-row estimation window (0-based)
    snapshot_indices: Sequence[int] = (6, 19, 32, 45)
    snapshot_labels: Sequence[str] = (
        "01.10.1997", "31.12.1997", "01.04.1998", "01.07.1998",
    )
    # Time-series maturity indices for Figure 3 (weeks)
    ts_maturity_weeks: Sequence[int] = (13, 26, 52, 104, 208, 260)
    dpi: int = 150
