<div align="center">

# OTTO Aggregate E-commerce Demand Forecasting

### Research-grade, leakage-aware forecasting of hourly cart and order volumes

[![CI](https://github.com/nikchey29/otto-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/nikchey29/otto-demand-forecasting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformer%20%2B%20GRU-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-17%20passing-0A9EDC?logo=pytest)

A dissertation-oriented machine-learning repository that converts the OTTO clickstream dataset into an hourly operational time series and forecasts **aggregate cart and order event volumes for the next 24 hours**.

</div>

## Start here

For the complete upload-to-GitHub and real-data execution workflow, follow [`FINAL_GITHUB_AND_EXECUTION_GUIDE.md`](FINAL_GITHUB_AND_EXECUTION_GUIDE.md). A ready-to-run Colab notebook is available at [`notebooks/OTTO_Dissertation_Final_Run_Colab.ipynb`](notebooks/OTTO_Dissertation_Final_Run_Colab.ipynb).

## Scope and claim boundary

This repository forecasts **platform-level hourly event volume**. It does not forecast demand for individual products, SKUs, customers, prices, inventory positions or revenue. The academically accurate description is:

> Multi-horizon forecasting of aggregate hourly cart and order volumes from e-commerce clickstream history.

The raw OTTO data contains more than 220 million events, but the public training period becomes only 672 hourly observations after global aggregation. This is a large data-engineering task followed by a comparatively small time-series experiment. The repository makes that limitation explicit instead of treating raw event count as the forecasting sample size.

## Dissertation-grade improvements in version 2

The original project provided one chronological holdout and four models. This revision adds the experimental controls normally expected from a strong technical dissertation:

- Expanding-window rolling-origin evaluation before the final holdout
- Five repeated neural-network seeds
- An untouched final 96-hour test period
- Ridge hyperparameter selection using validation data only
- Transformer, GRU and Extra Trees comparison
- Persistence, 24-hour seasonal, 168-hour seasonal and blended seasonal baselines
- MAE, RMSE, WAPE, sMAPE, MAPE, bias, MASE and RMSSE
- Horizon-level error analysis
- Paired moving-block bootstrap confidence intervals for model differences
- Split-conformal 90% prediction intervals and coverage reporting
- Configurable lookback, feature-set and Transformer-depth ablations
- Dataset integrity checks and SHA-256 fingerprinting
- Exact experiment manifests including package versions and split boundaries
- Graceful API startup when model artifacts have not yet been generated
- Seventeen automated tests, including an end-to-end synthetic research-suite test

These changes make the repository **ready to produce stronger evidence**. They do not fabricate new OTTO results. The full research commands must be run on the processed OTTO dataset before dissertation tables are updated.

## Experimental protocol

The full experiment separates model development from final reporting:

1. Reserve the final 96 hours as an untouched holdout.
2. Run three expanding-window chronological folds before that holdout.
3. Select Ridge regularization on each fold using its validation interval.
4. Train neural models using five fixed seeds and early stopping.
5. Rank model families using mean cross-validation WAPE across both targets.
6. Evaluate all pre-specified model families on the final holdout.
7. Use block bootstrap intervals because adjacent forecast origins overlap.
8. Calibrate conformal intervals only on final-fold validation residuals.
9. Report mean, standard deviation, interval coverage and limitations.

The final test period is not used to choose hyperparameters.

## Models

| Family | Purpose |
|---|---|
| Persistence | Minimum no-change reference |
| Seasonal naive 24h | Same hour from the previous day |
| Seasonal naive 168h | Same hour from the previous week |
| Seasonal blend | Mean of daily and weekly seasonal forecasts |
| Ridge | Regularized linear multi-output benchmark |
| Extra Trees | Non-linear tree ensemble benchmark |
| GRU | Recurrent neural baseline |
| Transformer | Attention-based multi-horizon model |

The project does not assume the Transformer must win. The model selected by cross-validation is recorded in `experiment_manifest.json` and served by the API where supported.

## Legacy single-split result

The repository retains the original experiment under `artifacts/legacy_single_split/` for traceability. These values are **not** the new rolling-origin results.

| Model | Carts MAE | Orders MAE | Average MAE |
|---|---:|---:|---:|
| Ridge | **2,338.53** | 1,086.99 | **1,712.76** |
| Transformer | 3,194.64 | **1,042.02** | 2,118.33 |
| Seasonal naive 24h | 4,064.79 | 1,243.96 | 2,654.37 |
| Persistence | 14,658.26 | 4,608.62 | 9,633.44 |

The defensible conclusion from that experiment is that Ridge was best overall, while the Transformer achieved a small order-volume advantage on one four-day holdout. No claim of universal Transformer superiority is made.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   ├── legacy_single_split/       # Original reported results
│   └── research/                  # Generated by the full research suite
├── configs/
│   ├── default.yaml               # Fast single-holdout run
│   ├── research.yaml              # Full dissertation experiment
│   └── smoke.yaml                 # Synthetic pipeline validation
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── AI_USE_DISCLOSURE_TEMPLATE.md
│   ├── DISSERTATION_TECHNICAL_MAP.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS_CHECKLIST.md
│   └── RUNBOOK_COLAB.md
├── src/otto_forecasting/
│   ├── api.py
│   ├── baselines.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── research.py
│   └── training.py
├── tests/
├── DATA_CARD.md
├── MODEL_CARD.md
├── pyproject.toml
└── requirements-lock.txt
```

## Installation

```bash
git clone https://github.com/nikchey29/otto-demand-forecasting.git
cd otto-demand-forecasting
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset preparation

Download and unzip the OTTO dataset so that the training file is located at:

```text
data/raw/otto-recsys-train.jsonl
```

Aggregate it once:

```bash
otto-forecast aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv \
  --frequency 1h
```

Validate and fingerprint it:

```bash
otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

## Smoke test before the full run

The synthetic data confirms that the code executes; it is not research evidence.

```bash
otto-forecast make-smoke-data \
  --output data/processed/synthetic_hourly.csv \
  --hours 500

otto-forecast research --config configs/smoke.yaml
```

## Run the dissertation experiments

Fast diagnostic experiment:

```bash
otto-forecast train --config configs/default.yaml
```

Full repeated-seed rolling-origin evaluation:

```bash
otto-forecast research --config configs/research.yaml
```

Ablation study:

```bash
otto-forecast ablate --config configs/research.yaml
```

The full research run can take a substantial amount of time. Use a GPU runtime for Transformer and GRU training. JSON aggregation remains CPU-bound.

## Main generated evidence

After the full research run, `artifacts/research/` contains:

| File | Dissertation use |
|---|---|
| `data_audit.json` | Dataset integrity and reproducibility |
| `cv_model_ranking.csv` | Model selection without final-test tuning |
| `research_metrics_raw.csv` | Fold-, seed-, model- and target-level metrics |
| `research_metrics_summary.csv` | Mean, standard deviation and run count |
| `research_predictions.csv` | Auditable prediction-level evidence |
| `statistical_comparisons.csv` | Block-bootstrap difference intervals |
| `interval_metrics.csv` | Prediction-interval coverage and width |
| `selected_model_horizon_metrics.csv` | Error growth across forecast hours |
| `run_metadata.csv` | Runtime, parameter count and best epoch |
| `experiment_manifest.json` | Environment, splits, configuration and selected model |
| `ablation_metrics_summary.csv` | Sensitivity to lookback, features and architecture |

## Tests

```bash
pytest -q
pytest --cov=otto_forecasting --cov-report=term-missing --cov-fail-under=65
```

The seventeen-test suite checks split boundaries, training-only scaling, baselines, conformal intervals, block bootstrap logic, data auditing, API degradation, configuration compatibility and an end-to-end synthetic research run.

## API

After the research suite generates the selected model bundle:

```bash
OTTO_ARTIFACT_DIR=artifacts/research \
uvicorn otto_forecasting.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /forecast`

The API reports `degraded` rather than crashing when artifacts are absent.

## Dissertation usage warning

Do not paste repository claims into the paper until the full experiment has been run and every value is present in the generated CSV files. Do not describe the task as SKU-level or product-level demand forecasting. Do not claim a model is better when a bootstrap interval includes zero or repeated-seed results are unstable.

The university handbook states that generative-AI use may be restricted. Confirm permitted use with the supervisor, retain an audit trail, disclose assistance where required and be prepared to explain every line of code during the viva.
