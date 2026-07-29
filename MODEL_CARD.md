# Models and intended use

## Forecasting task

The system uses a historical window of hourly activity to predict cart and order counts for each of the next 24 hours.

Default setup:

- lookback: 168 hours;
- forecast horizon: 24 hours;
- targets: carts and orders;
- forecast strategy: direct multi-output prediction.

## Models

The benchmark includes simple and complex approaches:

- persistence;
- 24-hour seasonal naive;
- 168-hour seasonal naive;
- daily/weekly seasonal blend;
- Ridge regression;
- Extra Trees;
- GRU;
- Transformer encoder with attention pooling.

## Model selection

Model families are ranked using mean rolling-origin WAPE across carts and orders. The final 96-hour period is kept separate from model development.

For neural models, repeated seeds are used to show training variation. The dissertation should report the average and standard deviation rather than only the best run.

## Intended use

This repository is intended for:

- an MSc dissertation experiment;
- comparison of forecasting methods on short aggregated clickstream data;
- studying temporal leakage, repeated runs and uncertainty;
- reproducible academic analysis.

It is not intended for automated purchasing, product-level inventory planning, financial forecasting or production decisions without further validation.

## Main risks and limitations

- The time series is short.
- Forecast origins overlap and are dependent.
- A complex model may overfit.
- Business drivers such as price and promotions are not available.
- Prediction intervals may lose coverage when the data distribution changes.
- Aggregate results cannot be interpreted as product-level demand.
