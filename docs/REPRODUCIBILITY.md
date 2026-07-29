# Reproducibility Guide

## Required evidence bundle

Archive the following together with the submitted dissertation or in a versioned release:

- Git commit hash
- `configs/research.yaml`
- `artifacts/research/data_audit.json`
- `artifacts/research/experiment_manifest.json`
- All research CSV files
- Selected model artifact
- Feature and target scalers
- Test output
- Exact environment lock file

## Environment

Create a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

The lock file records the versions used to validate this revision. GPU builds of PyTorch may require the installation command recommended by the runtime provider.

## Data identity

Run:

```bash
otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

The SHA-256 value identifies the exact processed file. A reviewer with a different hash has not reproduced the same input data.

## Determinism

The code sets Python, NumPy and PyTorch seeds and requests deterministic PyTorch algorithms where available. Exact bitwise equality can still vary across:

- CPU and GPU implementations
- CUDA and cuDNN versions
- PyTorch versions
- Parallel tree execution
- Different hardware architectures

Therefore, report tolerance-based metric reproduction rather than promising identical floating-point values on every machine.

## Commands

```bash
pytest -q

otto-forecast research --config configs/research.yaml

otto-forecast ablate --config configs/research.yaml
```

## Validation checklist

- Processed data hash matches
- Row count and timestamp range match
- Fold boundaries match the manifest
- Seeds match the configuration
- Model parameter counts match
- Best epochs are recorded
- Metrics are calculated in original units
- No final-test value was used to tune hyperparameters
- CSV tables and paper values match exactly
