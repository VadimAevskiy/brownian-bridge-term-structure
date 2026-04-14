# Changelog

## [1.0.0] -- 2026-04-13

### Initial Release

Complete Python replication of Aevskiy & Chetverikov (2016).

#### Added
- Core model implementation with Numba JIT acceleration (`src/model.py`)
- L-BFGS-B parameter estimation with parallelised numerical Hessian (`src/estimation.py`)
- Data loaders for both MATLAB `.mat` and Excel `.xlsx` formats (`src/data_loader.py`)
- Publication-quality figure generation for Figures 1, 2, 3, and SSE surface (`src/visualization.py`)
- Comprehensive test suite with 10 numerical correctness tests (`tests/test_model.py`)
- Full mathematical methodology documentation (`docs/METHODOLOGY.md`)
- Data dictionary (`docs/DATA_DICTIONARY.md`)
- MATLAB-to-Python replication notes (`docs/REPLICATION_NOTES.md`)
- Original MATLAB code for reference (`matlab_original/`)
- Published paper PDF (`docs/`)
- CLI entry point with `--figures-only`, `--no-figures`, `--workers` options

#### Verified
- Parameter estimates match paper: lambda = 23.9442 (paper: 23.9), sigma = 1.9725e-7 (paper: 2.0e-7)
- Standard errors match paper: SE(lambda) = 1.7958 (paper: 1.8), SE(sigma) = 1.4718e-8 (paper: 1.5e-8)
- Closed-form B coefficients (eqs. 16--17) verified against recursive computation
- SSE at optimum: 9.6913e-7

#### Original MATLAB Code
- `maximization.m` -- main estimation script
- `ll_k.m` -- objective function (exp-parameterised sigma)
- `ll_kse.m` -- objective function (natural-scale sigma, for Hessian)
- `ll_k_test.m` -- model evaluation and diagnostics
- `figures.m` -- plotting
