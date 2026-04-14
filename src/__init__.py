"""
brownian-bridge-term-structure
==============================

Python replication of Aevskiy & Chetverikov (2016),
"A discrete time model of convergence for the term structure of interest
rates in the case of entering a monetary union", Applied Economics 48:25.

Submodules
----------
config          Constants, paths, and parameter defaults.
model           Core Brownian bridge affine term structure model (Numba JIT).
data_loader     Load ITL-DEM spread data from .mat or .xlsx.
estimation      MLE / NLS parameter estimation with parallel Hessian.
visualization   Replicate Figures 2 and 3 from the paper.
"""

from .config import OptimizerConfig, FigureConfig
from .model import BrownianBridgeModel
from .data_loader import load_data
from .estimation import estimate, EstimationResult

__version__ = "1.0.0"

__all__ = [
    "BrownianBridgeModel",
    "load_data",
    "estimate",
    "EstimationResult",
    "OptimizerConfig",
    "FigureConfig",
]
