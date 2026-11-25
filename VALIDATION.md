# Validation

Release checks on 2026-11-25:

- 11 existing fusion tests passed, including dense/streaming equivalence and gradients.
- Runtime: Python 3.9.23 / PyTorch 2.7.1 / NumPy 2.0.2.
- Command: `python tools/run_core_tests.py`.
- Tests executed from an isolated copy of the release source with CPU-only visibility, not the original working directory.
- All published Python sources compiled successfully without running training.
- Source/path screening excluded dataset payloads, weights, private server addresses and credentials.

These checks do not reproduce the historical dataset benchmark or a clean installation of the complete GPU training stack. Full training, geospatial data preparation and end-to-end inference remain dependent on the documented data and environment prerequisites.
