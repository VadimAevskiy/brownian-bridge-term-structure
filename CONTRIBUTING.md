# Contributing

Contributions to this replication package are welcome. Here are some areas where contributions would be particularly valuable:

## Potential Extensions

1. **Multi-factor model**: Extend the single-factor Brownian bridge to a two- or three-factor specification to improve the fit at the beginning of the sample (as discussed in the paper's Section IV).

2. **Additional country pairs**: Apply the model to other pre-EMU country pairs (e.g., ESP-DEM, FRF-DEM, NLG-DEM) or to prospective monetary union candidates.

3. **Continuous-time implementation**: Implement the continuous-time Brownian bridge version and compare with the discrete-time results.

4. **Bayesian estimation**: Replace the MLE approach with a Bayesian MCMC estimator to obtain posterior distributions over (lambda, sigma).

5. **Monte Carlo study**: Assess the finite-sample properties of the estimator via simulation.

## Development Setup

```bash
git clone https://github.com/VadimAevskiy/brownian-bridge-term-structure.git
cd brownian-bridge-term-structure
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Type hints on all public functions
- Docstrings in NumPy format
- Line length: 99 characters
- Follow PEP 8

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/multi-factor`)
3. Add tests for any new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request with a clear description of changes
