#!/usr/bin/env python3
"""
backtest_yahoo.py -- Brownian bridge trading on real convergence events.

The model ONLY applies when there is:
    1. A known convergence date T
    2. A known target value (spread -> 0, price -> offer, FX -> peg)

Without both, the Brownian bridge has no economic basis.

Usage:
    python fetch_data.py          # Download data first
    python backtest_yahoo.py      # Run this
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "yahoo"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2,
})


# ======================================================================
# Brownian bridge model for a single convergence series
# ======================================================================

def fit_bridge(spread: np.ndarray, T_total: int) -> dict:
    """
    Fit a 1-D Brownian bridge: spread should converge to 0 by period T_total.

    z_{t+1} = (1 - 1/(T-t)) * z_t + sigma * eps_{t+1}

    Returns model path (expected value), residuals, and estimated sigma.
    """
    n = len(spread)

    # Model-implied expected path (no shocks, just the deterministic drift)
    model_path = np.full(n, np.nan)
    model_path[0] = spread[0]
    for t in range(n - 1):
        tau = T_total - t
        if tau <= 1:
            model_path[t + 1] = 0.0
        else:
            model_path[t + 1] = (1.0 - 1.0 / tau) * model_path[t]

    # Residuals: actual vs bridge recursion applied to actual z_t
    residuals = np.full(n, np.nan)
    for t in range(n - 1):
        tau = T_total - t
        if tau <= 1:
            break
        expected = (1.0 - 1.0 / tau) * spread[t]
        residuals[t + 1] = spread[t + 1] - expected

    valid = residuals[~np.isnan(residuals)]
    sigma = np.std(valid) if len(valid) > 2 else 1e-10

    return {"model_path": model_path, "residuals": residuals, "sigma": sigma}


def backtest_bridge(
    spread: np.ndarray,
    model_path: np.ndarray,
    sigma: float,
    threshold: float = 0.5,
) -> dict:
    """
    Trade deviations from the bridge model path.

    If spread > model + threshold*sigma: spread is too high, short it
       (expect convergence to pull it back toward model)
    If spread < model - threshold*sigma: spread overshot, long it
       (expect mean reversion back up toward model)
    """
    n = len(spread)
    position = np.zeros(n)
    pnl = np.zeros(n)

    for t in range(1, n):
        if np.isnan(model_path[t - 1]) or sigma < 1e-12:
            continue
        dev = spread[t - 1] - model_path[t - 1]
        if dev > threshold * sigma:
            position[t] = -1.0    # short spread (bet on convergence)
        elif dev < -threshold * sigma:
            position[t] = 1.0     # long spread (bet on reversion)

        pnl[t] = position[t] * (spread[t] - spread[t - 1])

    cum_pnl = np.cumsum(pnl)
    pnl_clean = pnl[pnl != 0]
    sharpe = (np.mean(pnl_clean) / np.std(pnl_clean) * np.sqrt(52)
              if len(pnl_clean) > 2 and np.std(pnl_clean) > 0 else 0)
    win_rate = np.sum(pnl_clean > 0) / len(pnl_clean) * 100 if len(pnl_clean) > 0 else 0
    active = int(np.sum(position != 0))

    return {
        "pnl": pnl, "cum_pnl": cum_pnl, "position": position,
        "sharpe": sharpe, "win_rate": win_rate, "active_weeks": active,
        "total_pnl": cum_pnl[-1],
    }


# ======================================================================
# Generic event runner
# ======================================================================

def run_event(
    csv_name: str,
    title: str,
    spread_col: str,
    weeks_col: str,
    close_col: str | None = None,
    target_val: float | None = None,
    unit: str = "",
    fig_name: str = "",
):
    """Run Brownian bridge fit and backtest on one convergence event."""
    path = DATA_DIR / csv_name
    if not path.exists():
        print(f"  [!] {csv_name} not found. Run: python fetch_data.py")
        return None

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    spread = df[spread_col].dropna().values
    T_total = int(df[weeks_col].iloc[0]) + 1

    print(f"  Observations:      {len(spread)}")
    print(f"  T (weeks at start): {T_total}")
    print(f"  Initial spread:    {spread[0]:.4f} {unit}")
    print(f"  Final spread:      {spread[-1]:.4f} {unit}")

    bridge = fit_bridge(spread, T_total)
    print(f"  Estimated sigma:   {bridge['sigma']:.6f}")

    bt = backtest_bridge(spread, bridge["model_path"], bridge["sigma"])
    print(f"  Total P&L:         {bt['total_pnl']:.4f} {unit}")
    print(f"  Sharpe:            {bt['sharpe']:.2f}")
    print(f"  Win rate:          {bt['win_rate']:.0f}%")
    print(f"  Active weeks:      {bt['active_weeks']}/{len(spread)}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 0.8, 1.5]})

    ax = axes[0]
    if close_col and close_col in df.columns:
        close_vals = df[close_col].dropna().values[:len(spread)]
        ax.plot(close_vals, "b-", lw=1.5, label="Observed")
        if target_val is not None:
            ax.plot(bridge["model_path"] + target_val, "r--", lw=1.5, label="Bridge model path")
            ax.axhline(target_val, color="green", lw=1, ls=":", label=f"Target = {target_val}")
        ax.set_ylabel("Level")
    else:
        ax.plot(spread, "b-", lw=1.5, label="Observed spread")
        ax.plot(bridge["model_path"], "r--", lw=1.5, label="Bridge model path")
        ax.axhline(0, color="green", lw=1, ls=":")
        ax.set_ylabel(f"Spread ({unit})")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(fontsize=10)

    ax = axes[1]
    colors = ["#1a9850" if p > 0 else "#d73027" if p < 0 else "#eeeeee" for p in bt["position"]]
    ax.bar(range(len(bt["position"])), bt["position"], color=colors, width=1.0)
    ax.set_ylabel("Position")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Flat", "Long"], fontsize=9)

    ax = axes[2]
    cum = bt["cum_pnl"]
    ax.plot(cum, "k-", lw=2)
    ax.fill_between(range(len(cum)), cum, 0, where=cum >= 0, alpha=0.2, color="#1a9850")
    ax.fill_between(range(len(cum)), cum, 0, where=cum < 0, alpha=0.2, color="#d73027")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Week")
    ax.set_ylabel(f"Cumulative P&L ({unit})")
    ax.set_title(f"Strategy P&L  (Sharpe = {bt['sharpe']:.2f},  Win = {bt['win_rate']:.0f}%)", fontsize=12)

    fig.tight_layout()
    out_name = fig_name or csv_name.replace(".csv", ".png")
    fig.savefig(OUTPUT_DIR / f"yahoo_{out_name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  >>> Saved yahoo_{out_name}")
    return bt


def main():
    print("=" * 65)
    print("  Brownian Bridge Trading -- Out-of-Sample (Yahoo Finance)")
    print("  Only convergence events with known date T and target")
    print("=" * 65)

    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.csv"))) == 0:
        print("\n  No data found. Run first:  python fetch_data.py\n")
        sys.exit(0)

    run_event(
        csv_name="croatia_eurhrk.csv",
        title="Croatia Euro Adoption: EUR/HRK -> 7.5345 by 01.01.2023",
        spread_col="spread", weeks_col="weeks_to_T",
        close_col="close", target_val=7.5345, unit="HRK",
        fig_name="croatia_convergence.png",
    )

    run_event(
        csv_name="ma_broadcom_vmware.csv",
        title="Broadcom / VMware: VMW -> $142.50 by 22.11.2023",
        spread_col="spread", weeks_col="weeks_to_T",
        close_col="close", target_val=142.50, unit="USD",
        fig_name="ma_broadcom_vmware.png",
    )

    run_event(
        csv_name="ma_tmobile_sprint.csv",
        title="T-Mobile / Sprint: basis -> 0 by 01.04.2020",
        spread_col="spread", weeks_col="weeks_to_T", unit="USD",
        fig_name="ma_tmobile_sprint.png",
    )

    print(f"\n{'='*65}")
    print(f"  All output saved to {OUTPUT_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
