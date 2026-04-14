"""
Data loading utilities for ITL-DEM spread data.

Supports two sources:
  1. Original MATLAB ``.mat`` file (``ITL.mat``)
  2. Excel workbook (``ITL_DEM_data.xlsx``)

Both produce the same (capt, 8) array of weekly spreads scaled by 1/5200.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANNUAL_SCALE, DATA_DIR, DATA_START_ROW


def load_mat(path: Path | str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Load spread data from the MATLAB ``.mat`` file.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``ITL.mat``.  Defaults to ``data/ITL.mat``.

    Returns
    -------
    dthat : ndarray of shape (capt, 8)
        Weekly spreads divided by 5200.
    dates : list[str]
        Date strings for each observation.
    """
    import scipy.io

    path = Path(path) if path is not None else DATA_DIR / "ITL.mat"
    mat = scipy.io.loadmat(str(path))
    emuweek: np.ndarray = mat["emuweek"]
    textdata: np.ndarray = mat["textdata"]

    sub = emuweek[DATA_START_ROW:]
    dthat = sub[:, :8] / ANNUAL_SCALE
    dates = [str(textdata[DATA_START_ROW + i, 0].item()) for i in range(len(dthat))]
    return dthat, dates


def load_excel(path: Path | str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Load spread data from the Excel workbook.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``ITL_DEM_data.xlsx``.  Defaults to ``data/ITL_DEM_data.xlsx``.

    Returns
    -------
    dthat : ndarray of shape (capt, 8)
        Weekly spreads divided by 5200.
    dates : list[str]
        Date strings for each observation.
    """
    path = Path(path) if path is not None else DATA_DIR / "ITL_DEM_data.xlsx"
    df = pd.read_excel(path, header=None)

    # Row 0 = column labels, row 1 = maturities header
    # Data starts from row 2 onward; dates are in column 0
    # The MAT subset starts at DATA_START_ROW (267) in the full emuweek,
    # which maps to Excel row = DATA_START_ROW + 1 (accounting for the extra
    # label row in the xlsx that is not in emuweek).
    excel_start = DATA_START_ROW + 1
    data_slice = df.iloc[excel_start:]
    dates_raw = data_slice.iloc[:, 0].astype(str).str.strip("'").tolist()
    spreads = data_slice.iloc[:, 1:9].values.astype(np.float64)
    dthat = spreads / ANNUAL_SCALE
    return dthat, dates_raw


def load_data(
    source: str = "mat",
    path: Path | str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Unified loader.

    Parameters
    ----------
    source : {"mat", "excel"}
    path : optional override

    Returns
    -------
    dthat, dates
    """
    if source == "mat":
        return load_mat(path)
    if source == "excel":
        return load_excel(path)
    raise ValueError(f"Unknown source: {source!r}. Use 'mat' or 'excel'.")
