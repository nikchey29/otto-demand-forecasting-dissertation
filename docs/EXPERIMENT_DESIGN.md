# Experiment design

## Research framing

I treat this as an experimental benchmarking study rather than a claim that I invented a new
forecasting architecture. The technical contribution is the reproducible pipeline, the
reformulation of the OTTO event log as an aggregate forecasting problem, and the controlled
comparison of simple and neural models under the same temporal evaluation design.

## Research question

The technical question I am testing is:

> Does a compact Transformer improve 24-hour forecasts of aggregate cart and order activity
> over simpler seasonal, linear, tree-based and recurrent models when only four weeks of
> hourly history are available?

The raw OTTO event log is very large, but the experiment is not allowed to treat the number of
events as the forecasting sample size. After aggregation, one observation is one UTC hour.

## Forecasting setup

- history available to the model: 168 hours;
- forecast horizon: 24 hours;
- targets: carts and orders;
- forecast strategy: direct multi-output;
- forecast origins: every valid hour inside each evaluation period.

Clicks are kept as an input signal but are not forecast as a target.

## Preprocessing decisions

I use `log1p` on the historical count features because the event volumes are right-skewed and
contain peaks that would otherwise dominate the scale. Hour of day and day of week are encoded
with sine/cosine pairs so that midnight remains close to 23:00 and Sunday remains close to
Monday in the representation.

The feature and target scalers are fitted only on the training observations belonging to a fold.
Historical inputs may reach backwards across a split boundary because they would be known at
forecast time, but the **target window itself must stay completely inside its assigned split**.

## Development folds and final holdout

The last 96 hours are reserved for the final holdout. Before that point, the main experiment
uses three expanding-window folds. Each development fold has a 48-hour validation period and a
48-hour assessment period.

The folds overlap because the available series is short. I therefore do not treat fold-level
results as independent samples. The fold design is mainly used to avoid choosing a model from
one arbitrary split and to expose instability across nearby periods.

The final 96-hour holdout is not used for hyperparameter selection.

## Models

The benchmark contains:

1. persistence;
2. seasonal naive using a 24-hour lag;
3. seasonal naive using a 168-hour lag;
4. an average of the daily and weekly seasonal forecasts;
5. Ridge regression;
6. Extra Trees;
7. a GRU;
8. the Transformer.

Ridge regularisation is chosen on validation data. Neural models use early stopping and the
fixed seeds `42`, `123`, `2026`, `3407` and `9999`.

## Transformer choice

I deliberately use a small Transformer (`d_model=32`, two encoder layers, four attention
heads). A larger network would be difficult to justify with only a few hundred hourly
observations after preprocessing.

The sequence is projected into the model dimension, positional information is added, and the
encoder output is reduced with learned attention pooling. One feed-forward head then predicts
all 24 future hours for both targets.

See [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) for the exact structure.

## Metrics

The primary ranking metric is WAPE averaged over carts and orders. I also save:

- MAE;
- RMSE;
- sMAPE;
- MAPE (interpreted cautiously near zero);
- bias;
- MASE;
- RMSSE;
- MAE/RMSE by forecast hour.

All metrics are calculated after predictions are transformed back into event counts.

## Repeated runs and uncertainty

Neural results are reported over five seeds rather than from the best run only.

Neighbouring forecast origins share most of their target hours, so their errors are dependent.
For the selected model versus the runner-up I use a moving-block bootstrap rather than an
independent-observation confidence interval.

Prediction intervals are calibrated from validation residuals. I treat the resulting coverage as
an empirical property of this split, not as a guarantee that will hold under distribution shift.

## Ablations

The ablation run changes one part of the Transformer setup at a time:

- lookback: 24, 72 or 168 hours;
- feature set: event history only or the full feature set;
- encoder depth: one or two layers.

Three seeds are used for each ablation configuration.

## Reporting rule

The final write-up should follow the model ranking produced by the predefined evaluation
procedure. If a simpler model wins, that is a result rather than a failure of the project. Claims
of superiority should also be tempered when the bootstrap interval for the model difference
includes zero.
