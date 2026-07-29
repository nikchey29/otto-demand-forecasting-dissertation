# Google Colab Runbook

## 1. Use persistent storage

Mount Google Drive before downloading or generating large files. Changing the Colab runtime can delete files stored only on the temporary virtual machine.

## 2. Clone and install

```python
!git clone https://github.com/nikchey29/otto-demand-forecasting.git
%cd otto-demand-forecasting
!pip install -e ".[dev]"
```

## 3. Put the dataset in place

The expected raw path is:

```text
data/raw/otto-recsys-train.jsonl
```

Copy the file from persistent storage or change the paths in the YAML configuration.

## 4. Aggregate once

```python
!otto-forecast aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv
```

Aggregation uses CPU and disk I/O. Selecting a GPU does not substantially accelerate JSON parsing.

## 5. Audit

```python
!otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

Copy the processed CSV and audit JSON to Drive before changing runtimes.

## 6. Validate the code

```python
!pytest -q
!otto-forecast make-smoke-data \
  --output data/processed/synthetic_hourly.csv \
  --hours 500
!otto-forecast research --config configs/smoke.yaml
```

Synthetic results are only a software test.

## 7. Run the full experiment

Use a GPU runtime, then run:

```python
!otto-forecast research --config configs/research.yaml
```

Then run the ablations:

```python
!otto-forecast ablate --config configs/research.yaml
```

## 8. Preserve outputs

Copy the entire `artifacts/research/` directory to Drive. Keep the terminal log or notebook output showing successful completion.

## 9. Do not rerun selectively after viewing the final test

Changing configurations after examining final-holdout results converts the test interval into another validation set. Document any change and reserve a new untouched interval when possible.
