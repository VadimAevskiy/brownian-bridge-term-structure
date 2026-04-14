#!/usr/bin/env python3
"""
backtest.py -- Model-guided maturity rotation strategy on ITL-DEM convergence data.

The Brownian bridge model identifies which maturities are cheap (observed spread
above model fair value) and which are rich (below fair value).  The strategy
concentrates capital in cheap maturities, earning +495 bps vs +333 bps for the
equal-weight benchmark -- a 49% outperformance.

Usage:
    python backtest.py

Output (in output/):
    backtest_final.png       Cumulative P&L: strategy vs benchmark
    model_mechanism.png      4-panel explanation of the trading edge
    backtest_results.txt     Performance summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ANNUAL_SCALE, ESTIMATION_START_ROW, MATURITY_YEARS, MATURITY_WEEKS
from src.data_loader import load_data
from src.model import BrownianBridgeModel

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DURATIONS = MATURITY_YEARS.copy()
TRADEABLE = np.arange(1, 8)  # columns 1-7 (skip 1-month observable)
MAT_LABELS_SHORT = ["3m", "6m", "1y", "2y", "3y", "4y", "5y"]
MAT_LABELS_ALL = ["1m", "3m", "6m", "1y", "2y", "3y", "4y", "5y"]


def get_metrics(pnl: np.ndarray) -> dict:
    """Compute performance metrics from a weekly return series."""
    p = pnl[1:]
    cum = np.cumsum(p)
    total = cum[-1] * 10000
    mean_w = np.mean(p)
    std_w = np.std(p, ddof=1) if len(p) > 1 else 1e-10
    sharpe = mean_w / std_w * np.sqrt(52) if std_w > 1e-12 else 0
    rm = np.maximum.accumulate(cum)
    maxdd = np.min(cum - rm) * 10000
    win = np.sum(p > 0) / len(p) * 100
    return dict(total=total, sharpe=sharpe, maxdd=maxdd, win=win)


def run_backtest():
    """Run the maturity rotation strategy and benchmark."""
    # Load data and evaluate model
    dthat, dates = load_data(source="mat")
    est_dates = dates[ESTIMATION_START_ROW:]
    model = BrownianBridgeModel()
    ev = model.evaluate(23.944243266616340, np.exp(-15.438802868320284), dthat)

    data_pp = ANNUAL_SCALE * ev["dthat_est"]
    model_pp = ANNUAL_SCALE * ev["Y_est"]
    error_pp = ANNUAL_SCALE * ev["errors"]
    n_weeks = data_pp.shape[0]

    # === Benchmark: equal-weight long all 7 tradeable maturities ===
    bm_pnl = np.zeros(n_weeks)
    for t in range(1, n_weeks):
        for col in TRADEABLE:
            ds = data_pp[t, col] - data_pp[t - 1, col]
            bm_pnl[t] += -DURATIONS[col] * ds / 100 / len(TRADEABLE)

    # === Strategy: concentrate in cheap maturities (model error > 0) ===
    st_pnl = np.zeros(n_weeks)
    weight_history = np.zeros((n_weeks, len(TRADEABLE)))

    for t in range(1, n_weeks):
        signal = error_pp[t - 1, TRADEABLE]  # previous week (no look-ahead)
        weights = np.maximum(signal, 0.0)
        if weights.sum() < 0.01:
            weights = np.ones(len(TRADEABLE))
        weights = weights / weights.sum()
        weight_history[t] = weights

        for i, col in enumerate(TRADEABLE):
            ds = data_pp[t, col] - data_pp[t - 1, col]
            st_pnl[t] += -DURATIONS[col] * ds / 100 * weights[i]

    # === P&L contribution by maturity ===
    bm_contrib = np.zeros(len(TRADEABLE))
    st_contrib = np.zeros(len(TRADEABLE))
    for t in range(1, n_weeks):
        signal = error_pp[t - 1, TRADEABLE]
        w = np.maximum(signal, 0.0)
        if w.sum() < 0.01:
            w = np.ones(len(TRADEABLE))
        w = w / w.sum()
        for i, col in enumerate(TRADEABLE):
            ds = data_pp[t, col] - data_pp[t - 1, col]
            ret = -DURATIONS[col] * ds / 100
            bm_contrib[i] += ret / len(TRADEABLE)
            st_contrib[i] += ret * w[i]

    return dict(
        n_weeks=n_weeks,
        bm_pnl=bm_pnl,
        st_pnl=st_pnl,
        bm_contrib=bm_contrib,
        st_contrib=st_contrib,
        weight_history=weight_history,
        data_pp=data_pp,
        model_pp=model_pp,
        error_pp=error_pp,
        est_dates=est_dates,
    )


def plot_performance(res: dict) -> None:
    """Plot cumulative P&L: strategy vs benchmark."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    bm = get_metrics(res["bm_pnl"])
    st = get_metrics(res["st_pnl"])
    excess = st["total"] - bm["total"]

    weeks = np.arange(1, res["n_weeks"])
    cum_bm = np.cumsum(res["bm_pnl"][1:]) * 10000
    cum_st = np.cumsum(res["st_pnl"][1:]) * 10000
    cum_excess = cum_st - cum_bm

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    ax1 = axes[0]
    ax1.plot(weeks, cum_bm, color="#888888", linewidth=2.8,
             label=f"Benchmark: equal-weight all maturities  (+{bm['total']:.0f} bps)")
    ax1.plot(weeks, cum_st, color="#c44e00", linewidth=2.8,
             label=f"Model rotation: concentrate in cheap bonds  (+{st['total']:.0f} bps)")
    ax1.fill_between(weeks, cum_bm, cum_st, alpha=0.12, color="#c44e00")
    ax1.axhline(0, color="black", linewidth=0.4)
    ax1.set_ylabel("Cumulative Return (basis points)", fontsize=13)
    ax1.set_title(
        "Convergence Trading: Brownian Bridge Model vs. Naive Benchmark\n"
        "Italian Lira \u2013 German Mark Spread, Aug 1997 \u2013 Aug 1998",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax1.legend(loc="upper left", fontsize=11.5, framealpha=0.95,
               edgecolor="none", facecolor="white")
    ax1.set_ylim(-30, 560)
    ax1.grid(True, alpha=0.15)
    ax1.annotate(
        f"+{excess:.0f} bps\nexcess return",
        xy=(40, (cum_st[39] + cum_bm[39]) / 2),
        fontsize=12, fontweight="bold", color="#c44e00", ha="center",
    )

    stats_text = (
        f"{'':>24s}{'Benchmark':>12s}{'Model':>12s}\n"
        f"{'Total return':>24s}{bm['total']:>+10.0f} bp{st['total']:>+10.0f} bp\n"
        f"{'Sharpe ratio':>24s}{bm['sharpe']:>10.2f}  {st['sharpe']:>10.2f}  \n"
        f"{'Max drawdown':>24s}{bm['maxdd']:>+10.0f} bp{st['maxdd']:>+10.0f} bp\n"
        f"{'Win rate':>24s}{bm['win']:>9.0f}%  {st['win']:>9.0f}%  "
    )
    ax1.text(0.98, 0.35, stats_text, transform=ax1.transAxes,
             fontsize=10, fontfamily="monospace",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", alpha=0.95))

    ax2 = axes[1]
    ax2.fill_between(weeks, cum_excess, 0,
                     where=cum_excess >= 0, alpha=0.35, color="#c44e00",
                     label="Model outperforms")
    ax2.fill_between(weeks, cum_excess, 0,
                     where=cum_excess < 0, alpha=0.35, color="#888888",
                     label="Benchmark outperforms")
    ax2.plot(weeks, cum_excess, color="#c44e00", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Week (from 20 August 1997)", fontsize=13)
    ax2.set_ylabel("Excess (bps)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax2.grid(True, alpha=0.15)

    fig.savefig(OUTPUT_DIR / "backtest_final.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_mechanism(res: dict) -> None:
    """Plot 4-panel explanation of the model's trading edge."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    maturities = MATURITY_YEARS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- A: Snapshot: cheap vs rich maturities ---
    ax = axes[0, 0]
    t = 19  # 31.12.1997
    ax.plot(maturities, res["data_pp"][t], "ro-", lw=2.2, ms=8,
            label="Market (observed)", zorder=3)
    ax.plot(maturities, res["model_pp"][t], "b^--", lw=2.2, ms=8,
            label="Model (fair value)", zorder=3)
    for i in range(8):
        err = res["error_pp"][t, i]
        if err > 0.02:
            ax.annotate("CHEAP", xy=(maturities[i], res["data_pp"][t, i]),
                        xytext=(0, 12), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#006600", ha="center")
            ax.vlines(maturities[i], res["model_pp"][t, i], res["data_pp"][t, i],
                      color="#006600", linewidth=2, alpha=0.5)
        elif err < -0.02:
            ax.annotate("RICH", xy=(maturities[i], res["data_pp"][t, i]),
                        xytext=(0, -16), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#cc0000", ha="center")
            ax.vlines(maturities[i], res["data_pp"][t, i], res["model_pp"][t, i],
                      color="#cc0000", linewidth=2, alpha=0.5)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Spread (% p.a.)")
    ax.set_title("A.  What the model tells you (31.12.1997)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)

    # --- B: Average model error by maturity ---
    ax = axes[0, 1]
    avg_error = np.mean(res["error_pp"], axis=0)
    colors = ["#006600" if e > 0 else "#cc0000" for e in avg_error]
    bars = ax.bar(MAT_LABELS_ALL, avg_error, color=colors, alpha=0.7,
                  edgecolor="white", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Average model error (pp)")
    ax.set_title("B.  Systematic mispricing by maturity", fontweight="bold")
    ax.grid(alpha=0.15, axis="y")
    for i, (bar, val) in enumerate(zip(bars, avg_error)):
        if i == 0:
            continue
        label = "CHEAP\n(overweight)" if val > 0 else "RICH\n(underweight)"
        y = val + 0.02 if val > 0 else val - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, y, label,
                ha="center", fontsize=7.5, fontweight="bold",
                color="#006600" if val > 0 else "#cc0000")

    # --- C: Portfolio allocation over time ---
    ax = axes[1, 0]
    colors_mat = ["#fdae61", "#fee08b", "#ffffbf", "#d9ef8b",
                  "#a6d96a", "#66bd63", "#1a9850"]
    ax.stackplot(range(res["n_weeks"]), res["weight_history"].T,
                 labels=MAT_LABELS_SHORT, colors=colors_mat, alpha=0.8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Portfolio weight")
    ax.set_title("C.  Model-guided allocation over time", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15)

    # --- D: P&L contribution by maturity ---
    ax = axes[1, 1]
    x = np.arange(len(TRADEABLE))
    width = 0.35
    ax.bar(x - width / 2, res["bm_contrib"] * 10000, width,
           label="Benchmark", color="#888888", alpha=0.7)
    ax.bar(x + width / 2, res["st_contrib"] * 10000, width,
           label="Model", color="#c44e00", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(MAT_LABELS_SHORT)
    ax.set_ylabel("P&L contribution (bps)")
    ax.set_title("D.  Where the extra +162 bps come from", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15, axis="y")

    fig.suptitle(
        "How the Brownian Bridge Model Creates a Trading Edge",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_mechanism.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def print_summary(res: dict) -> str:
    """Print and save performance summary."""
    bm = get_metrics(res["bm_pnl"])
    st = get_metrics(res["st_pnl"])

    lines = [
        "",
        "=" * 72,
        "  Backtest Results: Model Maturity Rotation vs. Benchmark",
        "  Data: ITL-DEM spreads, 20.08.1997 -- 12.08.1998 (52 weeks)",
        "=" * 72,
        "",
        f"  {'Metric':<28s} {'Benchmark':>14s} {'Model Strategy':>14s}",
        "-" * 72,
        f"  {'Total return (bps)':<28s} {bm['total']:>+14.0f} {st['total']:>+14.0f}",
        f"  {'Annualised Sharpe':<28s} {bm['sharpe']:>14.2f} {st['sharpe']:>14.2f}",
        f"  {'Max drawdown (bps)':<28s} {bm['maxdd']:>+14.0f} {st['maxdd']:>+14.0f}",
        f"  {'Win rate':<28s} {bm['win']:>13.0f}% {st['win']:>13.0f}%",
        "",
        f"  Excess return: +{st['total'] - bm['total']:.0f} bps "
        f"({(st['total'] - bm['total']) / bm['total'] * 100:.0f}% outperformance)",
        "",
        "  P&L contribution by maturity (bps):",
        f"  {'Maturity':<10s} {'Benchmark':>12s} {'Model':>12s} {'Excess':>12s}",
        "-" * 52,
    ]
    for i, label in enumerate(MAT_LABELS_SHORT):
        bm_c = res["bm_contrib"][i] * 10000
        st_c = res["st_contrib"][i] * 10000
        lines.append(f"  {label:<10s} {bm_c:>+12.0f} {st_c:>+12.0f} {st_c - bm_c:>+12.0f}")
    lines.append("-" * 52)
    bm_t = res["bm_contrib"].sum() * 10000
    st_t = res["st_contrib"].sum() * 10000
    lines.append(f"  {'TOTAL':<10s} {bm_t:>+12.0f} {st_t:>+12.0f} {st_t - bm_t:>+12.0f}")
    lines.extend(["", "=" * 72, ""])

    text = "\n".join(lines)
    print(text)
    with open(OUTPUT_DIR / "backtest_results.txt", "w") as f:
        f.write(text)
    return text


def main():
    print("=" * 72)
    print("  Brownian Bridge Convergence -- Trading Strategy Backtest")
    print("=" * 72)
    print()

    print("[1/4] Running backtest ...")
    res = run_backtest()

    print("[2/4] Computing metrics ...")
    print_summary(res)

    print("[3/4] Generating performance chart ...")
    plot_performance(res)
    print("      [+] backtest_final.png")

    print("[4/4] Generating mechanism explanation ...")
    plot_mechanism(res)
    print("      [+] model_mechanism.png")

    print(f"\n[*] Done. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
