"""
Parameter estimation via maximum-likelihood / nonlinear least squares.

Implements:
  - Optimisation of the sum-of-squared-errors objective.
  - Numerical Hessian computation (parallelised across available cores).
  - Standard-error extraction from the inverse Hessian.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .config import OptimizerConfig
from .model import BrownianBridgeModel


# ======================================================================
# Result container
# ======================================================================

@dataclass
class EstimationResult:
    """Stores optimisation output, standard errors, and timing."""

    lamda: float
    sigma: float
    log_sigma: float
    se_lamda: float
    se_sigma: float
    sse: float
    hessian: np.ndarray
    inv_hessian: np.ndarray
    converged: bool
    n_func_evals: int
    elapsed_seconds: float
    optimizer_message: str = ""

    def summary(self) -> str:
        lines = [
            "",
            "=" * 62,
            "  Estimation Results",
            "=" * 62,
            f"  {'Parameter':<16} {'Estimate':>14} {'Std. Error':>14}",
            "-" * 62,
            f"  {'lambda':<16} {self.lamda:>14.6f} {self.se_lamda:>14.6f}",
            f"  {'sigma':<16} {self.sigma:>14.4e} {self.se_sigma:>14.4e}",
            "-" * 62,
            f"  SSE:                 {self.sse:.10e}",
            f"  Converged:           {self.converged}",
            f"  Function evals:      {self.n_func_evals}",
            f"  Wall time:           {self.elapsed_seconds:.2f} s",
            "=" * 62,
            "",
        ]
        return "\n".join(lines)


# ======================================================================
# Hessian helpers (module-level for pickling by ProcessPoolExecutor)
# ======================================================================

def _obj_at_point(
    point: np.ndarray,
    T: int,
    N: int,
    k: float,
    dthat: np.ndarray,
) -> float:
    """Evaluate objective_natural at *point* (lambda, sigma)."""
    model = BrownianBridgeModel(T=T, N=N, k=k)
    return model.objective_natural(point, dthat)


def _compute_hessian_parallel(
    tstar: np.ndarray,
    fstar: float,
    dthat: np.ndarray,
    eps: float,
    T: int,
    N: int,
    k: float,
    n_workers: int,
) -> np.ndarray:
    """
    Numerical Hessian via central finite differences, parallelised.

    Evaluates the upper-triangular entries of H in parallel, then symmetrises.
    """
    d = len(tstar)
    epsmat = eps * np.eye(d)

    # Collect all (point, tag) pairs we need to evaluate
    tasks: list[tuple[str, np.ndarray]] = []

    # Diagonal / off-diagonal perturbations for the Hessian
    for i in range(d):
        tasks.append((f"e_{i}", tstar + epsmat[:, i]))
    for i in range(d):
        for j in range(i + 1):
            tasks.append((f"h_{i}_{j}", tstar + epsmat[:, i] + epsmat[:, j]))

    results: dict[str, float] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_obj_at_point, pt, T, N, k, dthat): tag
            for tag, pt in tasks
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    # Assemble the Hessian
    hessvec = np.array([results[f"e_{i}"] for i in range(d)])
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            H[i, j] = (results[f"h_{i}_{j}"] - hessvec[i] - hessvec[j] + fstar) / eps ** 2
            H[j, i] = H[i, j]
    return H


def _compute_hessian_serial(
    tstar: np.ndarray,
    fstar: float,
    dthat: np.ndarray,
    eps: float,
    model: BrownianBridgeModel,
) -> np.ndarray:
    """Numerical Hessian via central finite differences, serial fallback."""
    d = len(tstar)
    epsmat = eps * np.eye(d)

    hessvec = np.array(
        [model.objective_natural(tstar + epsmat[:, i], dthat) for i in range(d)]
    )
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            val = model.objective_natural(tstar + epsmat[:, i] + epsmat[:, j], dthat)
            H[i, j] = (val - hessvec[i] - hessvec[j] + fstar) / eps ** 2
            H[j, i] = H[i, j]
    return H


# ======================================================================
# Main estimation routine
# ======================================================================

def estimate(
    dthat: np.ndarray,
    config: OptimizerConfig | None = None,
    parallel_hessian: bool = True,
    n_workers: int | None = None,
    verbose: bool = True,
) -> EstimationResult:
    """
    Estimate lambda and sigma by minimising the SSE objective.

    Parameters
    ----------
    dthat : ndarray of shape (capt, 8)
    config : OptimizerConfig, optional
    parallel_hessian : bool
        If True, compute the numerical Hessian using multiprocessing.
    n_workers : int or None
        Number of parallel workers.  ``None`` = auto-detect.
    verbose : bool
        Print progress information.

    Returns
    -------
    EstimationResult
    """
    cfg = config or OptimizerConfig()
    model = BrownianBridgeModel()

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 6)

    # --- Warm up Numba JIT ---
    if verbose:
        print("[1/3] Warming up JIT compiler ...")
    _ = model.objective_exp(cfg.x0, dthat)

    # --- Optimise ---
    if verbose:
        print("[2/3] Running optimisation ...")
    t0 = time.perf_counter()

    res = minimize(
        model.objective_exp,
        cfg.x0,
        args=(dthat,),
        method=cfg.method,
        options={
            "ftol": cfg.tol_fun,
            "gtol": 1e-12,
            "maxiter": cfg.max_iter,
            "maxfun": cfg.max_fun_evals,
        },
    )
    t_opt = time.perf_counter() - t0

    lamda_hat = res.x[0]
    log_sigma_hat = res.x[1]
    sigma_hat = np.exp(log_sigma_hat)

    if verbose:
        print(f"    lambda = {lamda_hat:.6f}, sigma = {sigma_hat:.4e}")
        print(f"    SSE = {res.fun:.10e}  ({t_opt:.2f} s)")

    # --- Standard errors via numerical Hessian ---
    if verbose:
        print("[3/3] Computing standard errors (numerical Hessian) ...")
    t1 = time.perf_counter()

    tstar = np.array([lamda_hat, sigma_hat])
    fstar = model.objective_natural(tstar, dthat)

    if parallel_hessian and n_workers > 1:
        H = _compute_hessian_parallel(
            tstar, fstar, dthat, cfg.hessian_eps,
            model.T, model.N, model.k, n_workers,
        )
    else:
        H = _compute_hessian_serial(tstar, fstar, dthat, cfg.hessian_eps, model)

    inv_H = -np.linalg.inv(H)
    se = np.sqrt(np.maximum(np.diag(inv_H), 0.0))
    t_se = time.perf_counter() - t1

    if verbose:
        print(f"    SE(lambda) = {se[0]:.6f}, SE(sigma) = {se[1]:.4e}  ({t_se:.2f} s)")

    total_time = time.perf_counter() - t0

    result = EstimationResult(
        lamda=lamda_hat,
        sigma=sigma_hat,
        log_sigma=log_sigma_hat,
        se_lamda=se[0],
        se_sigma=se[1],
        sse=res.fun,
        hessian=H,
        inv_hessian=inv_H,
        converged=res.success or np.isclose(res.fun, fstar, atol=1e-14),
        n_func_evals=res.nfev,
        elapsed_seconds=total_time,
        optimizer_message=res.message if isinstance(res.message, str) else str(res.message),
    )
    if verbose:
        print(result.summary())

    return result
