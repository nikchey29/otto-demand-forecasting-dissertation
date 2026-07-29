# Experiment Protocol

## Research objective

Evaluate whether an attention-based Transformer provides reliable improvements over simpler forecasting methods when predicting aggregate hourly cart and order volumes for the next 24 hours from the preceding 168 hours.

## Unit of analysis

One observation is one UTC hour. The raw OTTO session events are aggregated globally into hourly click, cart and order counts. The forecasting sample size is therefore the number of hourly observations, not the number of raw clickstream events.

## Preprocessing

1. Stream the raw JSONL file line by line.
2. Count supported events in one-hour UTC buckets.
3. Reindex the complete hourly range and fill absent buckets with zero.
4. Validate chronological order, uniqueness, hourly spacing and non-negative counts.
5. Apply `log1p` to click, cart and order histories.
6. Add cyclical hour-of-day and day-of-week features and a weekend indicator.
7. Fit feature and target scalers only on the training interval of each fold.

## Forecast design

- Lookback: 168 hours
- Horizon: 24 hours
- Targets: cart and order event counts
- Strategy: direct multi-output forecast
- Issue time: every valid hourly forecast origin

Historical inputs may precede a validation or test boundary because those observations would be known at forecast time. Target windows must remain fully inside their assigned period.

## Development and final evaluation

### Cross-validation stage

The final 96 hours remain untouched. Three expanding-window folds are created before the final holdout. Each fold contains:

- An expanding training interval
- A 48-hour validation interval for early stopping and Ridge selection
- A 48-hour chronological assessment interval
- A 24-hour stride between fold endpoints

Model families are ranked using mean WAPE across carts and orders over all cross-validation folds and repeated neural seeds.

### Final holdout stage

The last 96 hours form the final test interval. The immediately preceding 96 hours are used for early stopping, conformal calibration and validation-only choices. No final-test information is used for hyperparameter selection.

## Models

Pre-specified model families:

- Persistence
- Seasonal naive 24 hours
- Seasonal naive 168 hours
- Daily-weekly seasonal blend
- Ridge regression
- Extra Trees regression
- GRU
- Transformer

Ridge alpha is selected from `[0.1, 1, 10, 100]` using validation error only. The neural architectures and training budget are fixed before final evaluation.

## Repetition

Transformer and GRU are trained with five seeds:

```text
42, 123, 2026, 3407, 9999
```

Report mean, standard deviation and number of runs. Do not report only the best seed.

## Metrics

Primary selection metric:

- WAPE averaged across both targets

Supporting metrics:

- MAE
- RMSE
- sMAPE
- MAPE, with caution near zero
- Bias
- MASE
- RMSSE
- MAE, RMSE and bias by forecast horizon

Metrics are calculated after inverse transformation in original event-count units.

## Statistical comparison

Forecast origins overlap, so ordinary independent-sample tests are not appropriate. The repository uses a paired moving-block bootstrap over forecast origins and reports:

- MAE difference between the cross-validation winner and runner-up
- 2.5% and 97.5% bootstrap quantiles
- Probability that model A has lower MAE than model B

A result should not be called a clear improvement when the interval contains zero.

## Prediction intervals

Split-conformal intervals use absolute validation residuals. The default nominal coverage is 90%. Report:

- Empirical coverage
- Coverage gap
- Mean interval width

These intervals quantify residual uncertainty on the selected data period; they do not guarantee coverage after distribution shift.

## Ablations

The ablation command tests:

- Lookback: 24, 72 and 168 hours
- Feature set: history only versus full temporal features
- Transformer depth: one versus two encoder layers
- Three repeated seeds

Ablations use a pre-holdout chronological fold and do not inspect the final test interval.

## Claims allowed after execution

Allowed:

- “Model X achieved the lowest mean cross-validation WAPE.”
- “On the final 96-hour holdout, model X obtained a carts MAE of …”
- “The paired block-bootstrap interval for the MAE difference was …”
- “The 90% conformal interval achieved …% empirical coverage.”

Not allowed without additional evidence:

- “The Transformer is universally superior.”
- “The model forecasts product demand.”
- “The model is production-ready.”
- “The model generalizes across seasons or promotions.”
