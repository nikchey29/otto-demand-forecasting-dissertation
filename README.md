# OTTO Hourly Cart and Order Forecasting

[![Tests](https://github.com/nikchey29/otto-demand-forecasting-dissertation/actions/workflows/ci.yml/badge.svg)](https://github.com/nikchey29/otto-demand-forecasting-dissertation/actions/workflows/ci.yml)

**MSc dissertation project by Chaithanya Vemuri**  
MSc Data Science, AI and Digital Business  
Gisma University of Applied Sciences, Potsdam Campus  
Academic year: 2025–2026

## About the project

I built this project to study whether deep-learning models are useful for short-term forecasting when the available time series is small but the raw event data is very large.

The source data is the OTTO Recommender Systems clickstream dataset. I aggregate the session events into one global hourly time series and forecast the number of cart and order events for the next 24 hours.

This is **aggregate event-volume forecasting**. It is not a product-level or SKU-level demand forecast because product identifiers are removed during aggregation.

## Research question

> Can a compact Transformer forecast hourly cart and order volumes more accurately than seasonal, linear, tree-based and recurrent baselines when only four weeks of aggregated history are available?

## Data used

The raw training data contains more than 220 million clickstream events. After hourly aggregation, it becomes approximately 672 observations, covering 28 days.

The model uses:

- historical clicks, carts and orders;
- hour-of-day features;
- day-of-week features;
- a weekend indicator.

The forecasting targets are hourly carts and orders. The default input window is 168 hours and the forecast horizon is 24 hours.

More details are available in [DATA_CARD.md](DATA_CARD.md).

## Models compared

- Persistence
- Seasonal naive using the previous day
- Seasonal naive using the previous week
- A daily/weekly seasonal blend
- Ridge regression
- Extra Trees
- GRU
- Transformer encoder

The experiment does not assume that the Transformer must win. A simpler model is preferred when it performs better on chronological validation data.

## Evaluation design

The main experiment uses:

- chronological expanding-window folds;
- an untouched final 96-hour holdout;
- scalers fitted only on training data;
- repeated neural-network seeds;
- MAE, RMSE, WAPE, sMAPE, MASE and RMSSE;
- horizon-level error analysis;
- moving-block bootstrap comparisons;
- prediction-interval coverage;
- lookback, feature and model-depth ablations.

The full plan is described in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Current status

The code, automated tests and synthetic end-to-end checks are complete. The repository also includes the outputs from my earlier single-split experiment in `artifacts/preliminary_single_split/`.

The final rolling-origin results are generated only after running the research commands on the processed OTTO data. I have intentionally not filled the final results folder with values that were not produced by an actual run.

### Preliminary single-split results

| Model | Carts MAE | Orders MAE | Average MAE |
|---|---:|---:|---:|
| Ridge | **2,338.53** | 1,086.99 | **1,712.76** |
| Transformer | 3,194.64 | **1,042.02** | 2,118.33 |
| Seasonal naive (24h) | 4,064.79 | 1,243.96 | 2,654.37 |
| Persistence | 14,658.26 | 4,608.62 | 9,633.44 |

These numbers are preliminary. They come from one four-day holdout and should not be used as the final dissertation result.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   ├── preliminary_single_split/
│   └── research/
├── configs/
│   ├── default.yaml
│   ├── research.yaml
│   └── smoke.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── ACADEMIC_INTEGRITY.md
│   ├── COLAB_GUIDE.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── PROJECT_NOTES.md
│   └── REPRODUCIBILITY.md
├── notebooks/
├── src/otto_forecasting/
├── tests/
├── DATA_CARD.md
├── MODEL_CARD.md
├── Makefile
├── pyproject.toml
└── requirements-lock.txt
```

## Installation

```bash
git clone https://github.com/nikchey29/otto-demand-forecasting-dissertation.git
cd otto-demand-forecasting-dissertation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Preparing the dataset

Place the OTTO training file at:

```text
data/raw/otto-recsys-train.jsonl
```

Aggregate it into an hourly series:

```bash
otto-forecast aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv \
  --frequency 1h
```

Check the processed file and record its hash:

```bash
otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

## Running the project

Run the tests:

```bash
pytest -q
```

Run a quick synthetic check before using the real data:

```bash
otto-forecast make-smoke-data \
  --output data/processed/synthetic_hourly.csv \
  --hours 500

otto-forecast research --config configs/smoke.yaml
```

Run the full experiment:

```bash
otto-forecast research --config configs/research.yaml
```

Run the ablation study:

```bash
otto-forecast ablate --config configs/research.yaml
```

The Google Colab steps are in [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

## Main outputs

The full experiment writes its tables and plots to `artifacts/research/`, including:

- cross-validation model rankings;
- fold-, seed- and target-level metrics;
- final holdout predictions;
- horizon-level errors;
- bootstrap comparison intervals;
- prediction-interval coverage;
- experiment metadata and split boundaries;
- ablation results.

Large datasets and trained binary model files are excluded from GitHub.

## Limitations

The largest limitation is the short time span. Although the raw dataset is large, the forecasting experiment has only four weeks of hourly observations. This limits conclusions about monthly seasonality, holidays, promotions and long-term generalisation.

The model also has no information about price, stock, advertising, product category or promotions. Results should therefore be interpreted as a benchmark on aggregate platform activity, not as a production inventory-planning system.

## Reproducibility and academic integrity

The exact environment, data checks and output files are described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Some development and documentation work involved generative-AI assistance. The disclosure is recorded in [docs/ACADEMIC_INTEGRITY.md](docs/ACADEMIC_INTEGRITY.md). I remain responsible for checking the implementation, running the experiments, interpreting the results and following Gisma's submission rules.
