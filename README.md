# OTTO Hourly Cart and Order Forecasting
**Chaithanya Vemuri**
MSc Data Science, AI and Digital Business — Gisma University of Applied Sciences, Potsdam

This repository contains the technical work for my master's dissertation. I use the OTTO
Recommender Systems event logs as a time-series problem: the session events are aggregated
into hourly click, cart and order volumes, and the task is to forecast carts and orders for the
next 24 hours.

The project started from a simple question: **does a Transformer actually help when the raw
event log is huge, but the time series available after aggregation is short?** That distinction
became important during the work. The OTTO training data contains hundreds of millions of
events, but after global hourly aggregation there are only about four weeks of observations.

## Project at a Glance

This dissertation investigates multi-horizon demand forecasting for e-commerce cart and order activity using statistical baselines, machine-learning models, recurrent neural networks, and a Transformer encoder.

The project focuses not only on predictive performance, but also on rigorous time-series evaluation: chronological splits, rolling-origin cross-validation, fixed random seeds, statistical model comparison, prediction intervals, and a final untouched holdout period.

**Core stack:** Python, PyTorch, scikit-learn, pandas, NumPy

**Research focus:** Time-series forecasting, demand forecasting, deep learning, model evaluation, statistical comparison

### Repository Guide

- [Forecasting task](#what-i-am-forecasting)
- [Models](#models)
- [Evaluation](#evaluation)
- [Key findings](#key-findings)
- [Final evaluation](#final-evaluation)
- [Repository structure](#repository-structure)
- [Main limitation](#main-limitation)
- [Reproducibility](#reproducibility)

## What I am forecasting

One row in the processed data represents one UTC hour:

```text
timestamp | clicks | carts | orders
```

The model sees the previous **168 hours (7 days)** and predicts **24 hourly cart and order
counts**. Product identifiers are deliberately not used, so this is an aggregate activity
forecast rather than SKU-level demand forecasting.

The input features are:

- log-transformed click, cart and order history;
- sine/cosine encodings for hour of day;
- sine/cosine encodings for day of week;
- a weekend indicator.

## Forecasting Pipeline

```mermaid
flowchart LR
    A[OTTO event logs] --> B[Hourly aggregation]
    B --> C[Clicks, carts and orders]
    C --> D[Feature engineering]
    D --> E[168-hour historical window]
    E --> F[Forecasting models]
    F --> G[24-hour cart and order forecasts]
    G --> H[Rolling-origin evaluation]
    H --> I[Model comparison]
    I --> J[Final 96-hour holdout]
```

## Models

I compare the Transformer with models that are intentionally simpler:

- persistence;
- 24-hour seasonal naive;
- 168-hour seasonal naive;
- a daily/weekly seasonal blend;
- Ridge regression;
- Extra Trees;
- GRU;
- Transformer encoder with attention pooling.

The Transformer uses a small architecture because the effective time series is short:

```text
8 input features
      ↓
linear projection (d_model = 32)
      ↓
sinusoidal positional encoding
      ↓
2 Transformer encoder layers
4 attention heads
      ↓
learned attention pooling
      ↓
feed-forward head
      ↓
24 × 2 outputs (carts, orders)
```

The implementation is in [`src/otto_forecasting/model.py`](src/otto_forecasting/model.py).

## Evaluation

I use chronological splits throughout. Target windows are not allowed to cross a validation or
test boundary, and feature/target scalers are fitted only on the training part of each split.

The full evaluation code supports:

- three expanding-window development folds;
- five fixed seeds for the neural models;
- a separate 96-hour final holdout;
- MAE, RMSE, WAPE, sMAPE, bias, MASE and RMSSE;
- error by forecast horizon;
- moving-block bootstrap comparisons;
- empirically calibrated prediction intervals;
- lookback, feature-set and Transformer-depth ablations.

The reasoning behind these choices is recorded in
[`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Key Findings

- The weekly seasonal baseline achieved the strongest performance during model development, outperforming the more complex neural architectures on the available aggregated time series.
- The experiments show that greater model complexity did not automatically translate into better forecasting performance for the available aggregated time series.
- Model selection was based on the predefined cross-validation procedure rather than the final holdout results.
- Statistical comparisons, prediction intervals, and Transformer ablation experiments were used to examine performance beyond a single headline metric.

## Final Evaluation

Model selection was based on mean WAPE across three rolling-origin
development folds. The final 96 hours of the series were kept separate
from model selection and used only for the final holdout evaluation.

### Cross-validation model ranking

| Rank | Model | Mean CV WAPE |
|---:|---|---:|
| 1 | Seasonal naive 168h | 0.1393 |
| 2 | Seasonal blend 24h+168h | 0.1399 |
| 3 | Extra Trees | 0.1435 |
| 4 | GRU | 0.1542 |
| 5 | Ridge | 0.1679 |
| 6 | Transformer | 0.1767 |
| 7 | Seasonal naive 24h | 0.1917 |
| 8 | Persistence | 0.7027 |

The weekly seasonal baseline was selected before looking at the final
holdout. Greater model complexity did not translate into better
forecasting performance on this short aggregate time series.

### Final 96-hour holdout

| Target | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Carts | 2,465.15 | 3,539.27 | 0.0980 |
| Orders | 826.39 | 1,168.39 | 0.1134 |

Performance varied by target. Ridge obtained a lower carts WAPE of
0.0890, while the weekly seasonal baseline obtained a lower orders
WAPE of 0.1134 compared with Ridge's 0.1218. The final model was not
changed after observing the holdout; the model selected by the
predefined cross-validation rule was retained.

### Statistical comparison

The selected weekly baseline was compared with the cross-validation
runner-up, the 24h+168h seasonal blend, using a moving-block bootstrap.

| Target | MAE difference | 95% bootstrap interval | P(selected model better) |
|---|---:|---:|---:|
| Carts | -403.18 | [-621.09, -211.89] | 1.0000 |
| Orders | -94.83 | [-204.36, -3.63] | 0.9786 |

Negative differences indicate lower MAE for the selected 168-hour
seasonal baseline. Both bootstrap intervals remained below zero.

### Prediction intervals

Validation-calibrated prediction intervals used a nominal coverage
target of 90%.

| Target | Nominal coverage | Empirical coverage | Mean interval width |
|---|---:|---:|---:|
| Carts | 0.90 | 0.9589 | 15,530.88 |
| Orders | 0.90 | 0.9275 | 4,370.04 |

The intervals over-covered the nominal target, particularly for carts,
so they are treated as empirical uncertainty estimates rather than
guaranteed future coverage.

### Transformer ablation

The Transformer ablation varied the historical lookback, feature set
and number of encoder layers. The best tested configuration used a
24-hour lookback, historical event features only, and two Transformer
encoder layers, with a mean WAPE of 0.1584 within the ablation study.

Longer input histories did not improve Transformer performance in this
experiment. Adding explicit calendar features also generally increased
WAPE, while two encoder layers performed modestly better than one
across the tested configurations.

The ablation uses a lighter evaluation protocol than the main model
comparison, so its WAPE values are interpreted only within the
ablation experiment.

Complete result tables, predictions, uncertainty analysis, figures and
ablation outputs are available in
[`artifacts/research/`](artifacts/research/).


## Running the core Transformer in Colab

The self-contained notebook is:

[`notebooks/OTTO_Transformer_Forecasting.ipynb`](notebooks/OTTO_Transformer_Forecasting.ipynb)

It downloads/reuses the OTTO data, performs the hourly aggregation, builds the features,
trains the Transformer and baselines, and saves the resulting metrics and figures to Google
Drive. It does not need to clone this repository in order to run.

For the complete rolling-origin experiment from the package:

```bash
pip install -e ".[dev]"
otto-forecast research --config configs/research.yaml
otto-forecast ablate --config configs/research.yaml
```

## Repository structure

```text
.
├── artifacts/
│   ├── preliminary_single_split/
│   └── research/
├── configs/
│   ├── default.yaml
│   ├── research.yaml
│   └── smoke.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── AI_USE.md
│   ├── EXPERIMENT_DESIGN.md
│   ├── IMPLEMENTATION_NOTES.md
│   ├── MODEL_ARCHITECTURE.md
│   └── REPRODUCIBILITY.md
├── notebooks/
│   └── OTTO_Transformer_Forecasting.ipynb
├── scripts/
├── src/otto_forecasting/
├── tests/
├── Makefile
├── pyproject.toml
└── requirements-lock.txt
```

Raw and processed OTTO data are intentionally excluded from the repository.

## Main limitation

The central limitation is the length of the series, not the number of raw events. Four weeks
is enough to study daily/weekly patterns, but not enough to make strong claims about monthly
seasonality, holidays, promotions or long-term behaviour. The dataset also does not provide
the price, stock, advertising or promotion variables that would normally be important in a
production demand-forecasting system.

## Reproducibility

The repository records the experiment configuration, tests, dataset hash, split boundaries and
generated metric files. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

AI-assisted debugging, review and language editing used during development are disclosed in
[`docs/AI_USE.md`](docs/AI_USE.md). All reported experiments and interpretations must be
checked against the saved outputs before submission.
