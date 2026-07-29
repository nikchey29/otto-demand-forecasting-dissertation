# Interview and Viva Guide

## Thirty-second explanation

I built a leakage-aware forecasting system on the OTTO e-commerce clickstream dataset. The pipeline streams more than 220 million raw events, aggregates them into an hourly operational series, and forecasts cart and order volumes for the next 24 hours from one week of history. I compare naive, linear, tree, recurrent and attention-based models using rolling-origin evaluation, repeated seeds, original-scale metrics, block-bootstrap comparisons and conformal prediction intervals.

## Why this is not product-demand forecasting

The article identifiers are removed during global aggregation, so the model predicts platform-level event volume rather than SKU-level units. I use the narrower wording because it matches the implemented target and avoids overstating business applicability.

## Why a Transformer

A Transformer can learn interactions across the full historical window without recurrent processing. I keep the architecture compact because the aggregated series is short. The experiment includes Ridge, Extra Trees, GRU and seasonal baselines because model complexity is justified only by measured out-of-sample performance.

## How leakage is prevented

Each fold has chronological training, validation and assessment intervals. Every target window remains fully within its assigned split. Scalers are fitted only on observations before validation. Ridge alpha is selected using validation data. The final 96 hours are excluded from the rolling-origin development folds.

## Why repeated seeds

Neural-network optimization is stochastic. Reporting only the best run exaggerates performance. Five fixed seeds provide a mean and standard deviation and show whether conclusions are stable.

## Why rolling-origin evaluation

One four-day holdout may favor a particular traffic pattern. Expanding-window folds test the models at several chronological forecast origins while preserving the order of the data.

## Why block bootstrap

Hourly forecasts overlap: adjacent origins share most of their input and target observations. A moving-block bootstrap preserves local dependence better than treating all errors as independent.

## Why conformal intervals

Point forecasts do not communicate uncertainty. Validation residuals calibrate a distribution-free interval around the final prediction. I report empirical coverage and interval width rather than claiming guaranteed production coverage.

## What if Ridge wins

That is a valid and useful result. It would show that the short aggregated series does not justify the extra complexity of a neural model. The repository ranks models using a pre-specified protocol and does not force the Transformer to be the selected model.

## Main limitations to state confidently

- Only 28 days after hourly aggregation
- Highly overlapping forecast origins
- No price, promotion, inventory or holiday variables
- Global rather than product-level aggregation
- No evidence of long-term or cross-season generalization
- Public benchmark data rather than current company traffic

## Questions to expect

1. Why use 168 hours of history?
2. Why forecast 24 hours directly rather than recursively?
3. Why use `log1p` and standardization?
4. Why is WAPE the primary metric?
5. How are MASE and RMSSE calculated?
6. How is Ridge alpha selected?
7. What does the block-bootstrap confidence interval mean?
8. How are conformal intervals calibrated?
9. Why might Extra Trees or Ridge outperform deep learning?
10. What would be required for product-level forecasting?
11. How would the pipeline detect data drift?
12. Which artifacts prove reproducibility?
