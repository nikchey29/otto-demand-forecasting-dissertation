# Final experiment design

## Objective

The experiment tests whether a compact Transformer improves 24-hour forecasts of aggregate cart and order volumes compared with simpler methods.

## Unit of analysis

One row represents one UTC hour. Raw session events are aggregated into hourly clicks, carts and orders. The effective sample size is therefore the number of hours rather than the number of clickstream events.

## Preprocessing

1. Aggregate events into hourly buckets.
2. Fill missing hourly buckets with zero counts.
3. Apply `log1p` to historical event counts.
4. Add cyclical hour and day-of-week features plus a weekend indicator.
5. Fit feature and target scalers only on the training part of each fold.

Historical observations before a validation or test boundary may be used as inputs because they are known at forecast time. Every target window must remain completely inside its assigned period.

## Forecast setup

- Input window: 168 hours
- Forecast horizon: 24 hours
- Targets: carts and orders
- Strategy: direct multi-output forecasting
- Forecast origin: every valid hour

## Model-development stage

The last 96 hours are reserved for the final holdout. Before that holdout, the code creates three expanding-window folds. Each fold contains a training period, a 48-hour validation period and a 48-hour assessment period.

Ridge regularisation is selected on validation data. Neural models use early stopping and five fixed random seeds. Model families are ranked using average WAPE across carts and orders.

## Final evaluation

After model development, every pre-specified model family is evaluated on the final 96-hour holdout. The holdout must not be used to change hyperparameters or select a preferred model.

## Models

- Persistence
- Seasonal naive (24 hours)
- Seasonal naive (168 hours)
- Daily/weekly seasonal blend
- Ridge regression
- Extra Trees
- GRU
- Transformer

## Metrics

Primary comparison metric:

- WAPE averaged across carts and orders

Supporting metrics:

- MAE
- RMSE
- sMAPE
- MAPE, reported cautiously near zero
- Bias
- MASE
- RMSSE
- Error by forecast horizon

All metrics are calculated after converting predictions back to event-count units.

## Repeated runs and uncertainty

GRU and Transformer models are trained with the seeds `42`, `123`, `2026`, `3407` and `9999`. The final report should include the mean, standard deviation and number of runs.

Because neighbouring forecasts overlap, model differences are compared using a moving-block bootstrap rather than treating every error as independent.

Prediction intervals are calibrated from validation residuals. Their empirical coverage and average width are reported, but they should not be described as guaranteed under distribution shift.

## Ablations

The ablation command compares:

- lookback windows of 24, 72 and 168 hours;
- historical features alone versus the complete feature set;
- one versus two Transformer encoder layers.

Ablations are run before the final holdout is inspected.

## Reporting rule

The final dissertation should report the actual model selected by the protocol. A model should not be described as clearly better when the uncertainty interval for its difference includes zero.
