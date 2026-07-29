# Model Card

## System

A comparative forecasting system for aggregate hourly cart and order volumes. The repository evaluates naive, linear, tree-based and neural models rather than presenting one architecture as inherently superior.

## Candidate models

- Persistence
- 24-hour seasonal naive
- 168-hour seasonal naive
- Daily-weekly seasonal blend
- Ridge regression
- Extra Trees
- GRU
- Transformer encoder with attention pooling

## Inputs

A chronological lookback window containing historical click, cart and order counts plus engineered temporal features. The default lookback is 168 hours.

## Outputs

A direct multi-output forecast for the next 24 hours:

- Cart event volume
- Order event volume

Predictions are inverse-transformed and clipped to non-negative values.

## Selection

Model families are ranked using mean rolling-origin cross-validation WAPE across both targets. The final 96-hour holdout is reserved for final evaluation rather than hyperparameter selection.

## Evaluation

- MAE
- RMSE
- WAPE
- sMAPE
- MAPE
- Bias
- MASE
- RMSSE
- Horizon-level errors
- Repeated-seed variation
- Paired moving-block bootstrap intervals
- Split-conformal coverage and interval width

## Leakage controls

- Strict chronological boundaries
- Target windows fully contained in their split
- Training-only feature and target scaling in every fold
- Ridge alpha chosen on validation data
- A final holdout excluded from rolling-origin model development
- Ablations run before the final holdout

## Intended use

- Academic benchmarking
- Demonstrating time-series methodology and software engineering
- Short-horizon operational-volume research
- Portfolio evidence after the author understands and can defend every component

## Not intended for

- Product-level inventory planning
- Automated purchasing
- Financial forecasts
- Staffing decisions without independent validation
- Production deployment without monitoring, retraining and current business data

## Limitations

- The aggregated series is short
- Repeated windows share observations
- Performance may vary across seeds and folds
- The public dataset does not establish seasonal or long-term generalization
- Uncertainty intervals can fail under distribution shift
- Business drivers such as price and promotion are absent

## Responsible reporting

Report the model selected by the pre-specified protocol, not the most visually impressive model. A lower point estimate should not be described as a clear improvement when the bootstrap interval includes zero.
