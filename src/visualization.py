"""
Visualization: publication-quality figures replicating Aevskiy & Chetverikov (2016).

Figure 1: Historical spreads for 8 maturities over the estimation window.
Figure 2: Cross-sectional term structure snapshots at four dates (paper Fig. 2).
Figure 3: Time-series comparison of model vs. data for 6 maturities (paper Fig. 3).
Figure S1: SSE objective surface contour plot (supplementary).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .config import (
    ANNUAL_SCALE,
    ESTIMATION_START_ROW,
    MATURITY_LABELS,
    MATURITY_YEARS,
    FigureConfig,
    OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Global style -- academic / publication defaults
# ---------------------------------------------------------------------------
_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ======================================================================
# Figure 2: Cross-sectional term structure snapshots
# ======================================================================

def plot_figure2(
    dthat_est: np.ndarray,
    Y_est: np.ndarray,
    q_mat: np.ndarray,
    cfg: FigureConfig | None = None,
    save: bool = True,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Replicate paper Figure 2.

    Model (blue dashed) vs. data (red solid) yield-spread term structures
    at four snapshot dates before EMU introduction.
    """
    _apply_style()
    cfg = cfg or FigureConfig()
    out = output_dir or OUTPUT_DIR

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    for idx, (row_idx, label) in enumerate(
        zip(cfg.snapshot_indices, cfg.snapshot_labels)
    ):
        ax = axes.flat[idx]
        data_pp = ANNUAL_SCALE * dthat_est[row_idx, :]
        model_pp = ANNUAL_SCALE * Y_est[row_idx, :]

        ax.plot(q_mat, data_pp, "r-", linewidth=2.2, label="Data (observed)")
        ax.plot(q_mat, model_pp, "b--", linewidth=2.2, label="Model")

        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 5)
        ax.set_ylim(bottom=min(0, np.min(model_pp) - 0.1))
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Spread (% p.a.)")

        # Annotate max absolute error
        max_err_idx = np.argmax(np.abs(data_pp - model_pp))
        max_err = np.abs(data_pp[max_err_idx] - model_pp[max_err_idx])
        ax.annotate(
            f"max |err| = {max_err:.2f} pp",
            xy=(0.97, 0.03),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color="gray",
        )

        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Figure 2.  Model and data term structures of spreads\n"
        "Italian lira vs. German mark, selected dates before EMU",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save:
        fig.savefig(
            out / "figure2_term_structures.png",
            dpi=cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig


# ======================================================================
# Figure 3: Time-series comparison by maturity
# ======================================================================

def plot_figure3(
    dthat_est: np.ndarray,
    Y_est: np.ndarray,
    cfg: FigureConfig | None = None,
    save: bool = True,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Replicate paper Figure 3.

    Time-series of observed vs. model spreads for 6 maturities
    over the 52-week estimation window.
    """
    _apply_style()
    cfg = cfg or FigureConfig()
    out = output_dir or OUTPUT_DIR
    n_est = dthat_est.shape[0]
    t_axis = np.arange(1, n_est + 1)

    from .config import MATURITY_WEEKS

    mat_to_col = {int(w): i for i, w in enumerate(MATURITY_WEEKS)}
    n_panels = len(cfg.ts_maturity_weeks)
    nrows = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.5 * nrows))

    for idx, mat_w in enumerate(cfg.ts_maturity_weeks):
        ax = axes.flat[idx]
        col = mat_to_col[mat_w]
        data_ts = ANNUAL_SCALE * dthat_est[:, col]
        model_ts = ANNUAL_SCALE * Y_est[:, col]

        ax.plot(t_axis, data_ts, color="#2166ac", linewidth=1.3, label="Data")
        ax.plot(
            t_axis,
            model_ts,
            color="#b2182b",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label="Model",
        )
        ax.fill_between(
            t_axis,
            data_ts,
            model_ts,
            alpha=0.08,
            color="gray",
        )

        # Title with maturity
        years = mat_w / 52
        if years >= 1:
            title = f"{int(years)}-year ({mat_w} weeks)"
        else:
            months = int(round(years * 12))
            title = f"{months}-month ({mat_w} weeks)"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("$t$ (weeks from 20.08.1997)")
        ax.set_ylabel("Spread (% p.a.)")

        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    # Hide unused axes
    for idx in range(n_panels, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    fig.suptitle(
        "Figure 3.  Observable and model spreads, ITL vs. DEM\n"
        "Estimation window: 20.08.1997 -- 12.08.1998 (N = 52)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save:
        fig.savefig(
            out / "figure3_time_series.png",
            dpi=cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig


# ======================================================================
# Figure 1: Historical spreads
# ======================================================================

def plot_historical_spreads(
    dthat_est: np.ndarray,
    dates: list[str] | None = None,
    save: bool = True,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Replicate paper Figure 1.

    Historical spreads for all 8 maturities over the estimation window.
    """
    _apply_style()
    out = output_dir or OUTPUT_DIR
    n_est = dthat_est.shape[0]
    t_axis = np.arange(n_est)

    # Colour palette: short -> long maturities, warm -> cool
    colors = ["#d73027", "#f46d43", "#fdae61", "#fee08b",
              "#a6d96a", "#66bd63", "#1a9850", "#006837"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for col, (label, color) in enumerate(zip(MATURITY_LABELS, colors)):
        ax.plot(
            t_axis,
            ANNUAL_SCALE * dthat_est[:, col],
            linewidth=1.1,
            label=label,
            color=color,
        )

    ax.set_xlabel("Week")
    ax.set_ylabel("Spread (% p.a.)")
    ax.set_title(
        "Figure 1.  Interest rate spreads, Italian lira vs. German mark\n"
        "Estimation window: 20.08.1997 -- 12.08.1998",
        fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=4, loc="upper right", framealpha=0.9)

    if dates is not None and len(dates) >= n_est:
        step = max(1, n_est // 8)
        ax.set_xticks(t_axis[::step])
        ax.set_xticklabels(dates[::step], rotation=35, fontsize=9, ha="right")

    fig.tight_layout()
    if save:
        fig.savefig(
            out / "figure1_historical_spreads.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig


# ======================================================================
# Supplementary: SSE objective surface
# ======================================================================

def plot_error_surface(
    dthat: np.ndarray,
    lamda_range: tuple[float, float] = (20.0, 28.0),
    log_sigma_range: tuple[float, float] = (-17.0, -14.0),
    grid_size: int = 50,
    save: bool = True,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Supplementary figure: 2-D contour of log10(SSE) around the optimum.
    """
    _apply_style()
    from .model import BrownianBridgeModel

    out = output_dir or OUTPUT_DIR
    model = BrownianBridgeModel()
    lam_grid = np.linspace(*lamda_range, grid_size)
    ls_grid = np.linspace(*log_sigma_range, grid_size)
    Z = np.full((grid_size, grid_size), np.nan)

    for i, lam in enumerate(lam_grid):
        for j, ls in enumerate(ls_grid):
            Z[j, i] = model.objective_exp(np.array([lam, ls]), dthat)

    fig, ax = plt.subplots(figsize=(9, 7))
    cs = ax.contourf(lam_grid, ls_grid, np.log10(Z), levels=30, cmap="RdYlBu_r")
    cb = fig.colorbar(cs, ax=ax, shrink=0.85)
    cb.set_label("$\\log_{10}$(SSE)", fontsize=11)

    # Mark the optimum
    ax.plot(23.9442, -15.4388, "k*", markersize=14, label="Optimum")
    ax.legend(loc="upper left", fontsize=10)

    ax.set_xlabel("$\\lambda$ (price of risk)")
    ax.set_ylabel("$\\log(\\sigma)$")
    ax.set_title(
        "Figure S1.  SSE objective surface",
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        fig.savefig(
            out / "sse_surface.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig
