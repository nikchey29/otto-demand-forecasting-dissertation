# Reproducing the experiment

## Environment

The package requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

On a CUDA runtime, install the PyTorch build appropriate for that environment if the pinned
wheel is not suitable.

## Data

The repository does not include the OTTO dataset. The processed file expected by the package is:

```text
data/processed/otto_hourly.csv
```

Before comparing two runs, create an audit:

```bash
otto-forecast audit-data   --input data/processed/otto_hourly.csv   --output artifacts/data_audit.json
```

The audit records the row count, timestamp range, event totals, file size and SHA-256 hash.

## Checks

```bash
pytest -q
python scripts/check_quality.py
```

A short synthetic run is available for checking the software path:

```bash
otto-forecast make-smoke-data   --output data/processed/synthetic_hourly.csv   --hours 500

otto-forecast research --config configs/smoke.yaml
```

Synthetic metrics are only a software check and should not be mixed with the dissertation
results.

## Main experiment

```bash
otto-forecast research --config configs/research.yaml
otto-forecast ablate --config configs/research.yaml
```

The main output directory is `artifacts/research/`.

For the dissertation archive I keep:

- the processed-data hash;
- the exact configuration file;
- the metric and prediction CSV files;
- the experiment manifest;
- the ablation tables;
- the selected trained model/scalers in private storage;
- the software test result.

The code fixes Python, NumPy and PyTorch seeds and asks PyTorch for deterministic behaviour
where possible. Small numerical differences can still occur across hardware, CUDA versions and
parallel tree implementations.
