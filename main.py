#!/usr/bin/env python3
"""
main.py -- Replicate all results from Aevskiy & Chetverikov (2016).

Usage:
    python main.py                    # Full replication (estimation + figures)
    python main.py --figures-only     # Skip estimation, use known parameters
    python main.py --no-figures       # Estimation only, no plots
    python main.py --workers 4        # Set number of parallel workers

Output:
    output/figure1_historical_spreads.png
    output/figure2_term_structures.png
    output/figure3_time_series.png
    output/sse_surface.png
    output/estimation_results.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the src package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ESTIMATION_START_ROW, FigureConfig, OptimizerConfig, OUTPUT_DIR
from src.data_loader import load_data
from src.estimation import estimate
from src.model import BrownianBridgeModel
from src.visualization import (
    plot_error_surface,
    plot_figure2,
    plot_figure3,
    plot_historical_spreads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate Aevskiy & Chetverikov (2016) results."
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip optimisation; use the published parameter values.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Run estimation only, skip figure generation.",
    )
    parser.add_argument(
        "--data-source",
        choices=["mat", "excel"],
        default="mat",
        help="Data source format (default: mat).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for Hessian computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  Brownian Bridge Term Structure -- Replication")
    print("  Aevskiy & Chetverikov (2016), Applied Economics 48:25")
    print("=" * 62)
    print()
    print(f"[*] Loading data (source={args.data_source}) ...")
    dthat, dates = load_data(source=args.data_source)
    print(f"    Observations: {dthat.shape[0]} weeks")
    print(f"    Date range:   {dates[0]} -- {dates[-1]}")
    print(f"    Estimation window: {dates[ESTIMATION_START_ROW]} -- {dates[-1]}")
    print(f"    Estimation obs:    {dthat.shape[0] - ESTIMATION_START_ROW}")
    print()

    # ------------------------------------------------------------------
    # 2. Estimation
    # ------------------------------------------------------------------
    if args.figures_only:
        print("[*] Skipping estimation (--figures-only). Using published values.")
        lamda = 23.944243266616340
        sigma = np.exp(-15.438802868320284)
        print(f"    lambda = {lamda:.6f}")
        print(f"    sigma  = {sigma:.4e}")
    else:
        print("[*] Starting parameter estimation ...")
        result = estimate(
            dthat,
            config=OptimizerConfig(),
            parallel_hessian=True,
            n_workers=args.workers,
            verbose=True,
        )
        lamda = result.lamda
        sigma = result.sigma

        # Save results to file
        results_path = out_dir / "estimation_results.txt"
        with open(results_path, "w") as f:
            f.write(result.summary())
        print(f"    Results saved to {results_path}")

    # ------------------------------------------------------------------
    # 3. Model evaluation
    # ------------------------------------------------------------------
    print()
    print("[*] Evaluating model at optimal parameters ...")
    model = BrownianBridgeModel()
    ev = model.evaluate(lamda, sigma, dthat)

    # Report max errors at the four snapshot dates
    cfg_fig = FigureConfig()
    for idx, label in zip(cfg_fig.snapshot_indices, cfg_fig.snapshot_labels):
        max_err = np.max(np.abs(5200.0 * ev["errors"][idx, :]))
        print(f"    Max |error| at {label}: {max_err:.2f} pp")

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        print()
        print("[*] Generating figures ...")

        est_dates = dates[ESTIMATION_START_ROW:]

        fig1 = plot_historical_spreads(
            ev["dthat_est"], dates=est_dates, output_dir=out_dir
        )
        print("    [+] Figure 1: historical spreads")

        fig2 = plot_figure2(
            ev["dthat_est"], ev["Y_est"], ev["q_mat"], output_dir=out_dir
        )
        print("    [+] Figure 2: term structure snapshots")

        fig3 = plot_figure3(ev["dthat_est"], ev["Y_est"], output_dir=out_dir)
        print("    [+] Figure 3: time-series comparison")

        print()
        print("[*] Generating SSE contour surface (this may take a moment) ...")
        fig4 = plot_error_surface(dthat, output_dir=out_dir)
        print("    [+] SSE surface")

        import matplotlib
        matplotlib.pyplot.close("all")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_total
    print()
    print(f"[*] All done in {elapsed:.1f} s. Output directory: {out_dir}")
    print()


if __name__ == "__main__":
    main()
