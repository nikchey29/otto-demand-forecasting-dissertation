# Reproducing the experiment

## Environment

Create a clean Python environment and install the recorded versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

A CUDA environment may require the PyTorch installation command recommended by the runtime provider.

## Data check

Run:

```bash
otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

The audit records the row count, timestamp range, event totals, file size and SHA-256 hash. Two runs should use the same processed-data hash before their results are compared.

## Commands

```bash
pytest -q
otto-forecast research --config configs/research.yaml
otto-forecast ablate --config configs/research.yaml
```

## Files to keep with the dissertation

- Git commit hash
- `configs/research.yaml`
- data-audit JSON
- experiment manifest
- metric CSV files
- prediction CSV file
- ablation outputs
- selected model and scalers in private storage
- test output
- environment lock file

## Expected variation

The code sets Python, NumPy and PyTorch seeds and requests deterministic operations where possible. Small numerical differences can still occur across CPUs, GPUs, CUDA versions, PyTorch versions and parallel tree implementations.

Reproduction should therefore compare metrics within a reasonable numerical tolerance rather than require identical binary files.
