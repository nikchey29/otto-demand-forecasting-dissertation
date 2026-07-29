# Running the project in Google Colab

## 1. Select a GPU runtime

A GPU helps with GRU and Transformer training. Data download, JSON parsing and hourly aggregation remain CPU and disk operations.

## 2. Mount Google Drive

Store the processed CSV and final outputs in Drive because Colab's local machine is temporary.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Clone and install

```python
%cd /content
!rm -rf otto-demand-forecasting-dissertation
!git clone https://github.com/nikchey29/otto-demand-forecasting-dissertation.git
%cd /content/otto-demand-forecasting-dissertation
!pip install -e '.[dev]'
```

## 4. Copy the processed data

The recommended Drive location is:

```text
/content/drive/MyDrive/OTTO_Dissertation/data/processed/otto_hourly.csv
```

Copy it into the repository:

```python
!mkdir -p data/processed
!cp /content/drive/MyDrive/OTTO_Dissertation/data/processed/otto_hourly.csv \
    data/processed/otto_hourly.csv
```

When only the raw JSONL file is available, place it in `data/raw/` and run the aggregation command once.

## 5. Check the data and code

```python
!otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
!pytest -q
```

A synthetic run can be used to check the software, but its metrics must not be used in the dissertation:

```python
!otto-forecast make-smoke-data \
  --output data/processed/synthetic_hourly.csv \
  --hours 500
!otto-forecast research --config configs/smoke.yaml
```

## 6. Run the final experiments

```python
!otto-forecast research --config configs/research.yaml
!otto-forecast ablate --config configs/research.yaml
```

Copy `artifacts/research/` to Google Drive after the commands finish.

## 7. Keep the final holdout untouched

Do not change model settings after reading the final holdout results. Any later change must be documented, and ideally evaluated on a new untouched period.
