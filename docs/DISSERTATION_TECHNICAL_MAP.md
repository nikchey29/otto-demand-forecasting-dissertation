# Dissertation Technical Map

This document maps repository evidence to the technical dissertation structure recommended in the module handbook.

## Abstract

Use the final outputs to state:

- The operational forecasting problem
- The 168-hour input and 24-hour horizon
- The eight pre-specified models
- The rolling-origin and repeated-seed design
- The final holdout result
- The principal limitation: only 28 days after hourly aggregation

Do not include values until `artifacts/research/` has been regenerated on the real data.

## Introduction

### Problem

Operational planning benefits from short-horizon estimates of aggregate carts and orders. The OTTO data is originally a session-recommendation dataset, so the dissertation contribution is a new aggregated forecasting formulation rather than the original recommendation task.

### Objectives

1. Build a reproducible pipeline from raw session events to hourly forecasts.
2. Prevent temporal leakage.
3. Compare neural, tree, linear and naive forecasting approaches.
4. quantify uncertainty and variation across folds and seeds.
5. identify conditions under which model complexity is justified.

### Contribution

Frame the contribution as an applied and benchmarking contribution:

- Memory-conscious event aggregation
- Leakage-aware experimental framework
- Cross-family model benchmark
- Statistical and uncertainty analysis
- Reproducible implementation and deployment interface

## Foundations and background

Explain:

- Multi-horizon forecasting
- Direct versus recursive forecasting
- Seasonal naive methods
- Regularized regression
- Tree ensembles
- GRU sequence models
- Self-attention and positional encoding
- WAPE, MASE and RMSSE
- Rolling-origin evaluation
- Conformal prediction

## Related work

The repository does not supply literature citations. The paper must independently review recent peer-reviewed work on:

- E-commerce demand forecasting
- Deep learning for short time series
- Transformer forecasting
- Strong linear and seasonal baselines
- Forecast evaluation under overlapping horizons
- Distribution-free prediction intervals

Clearly distinguish the OTTO recommendation competition from the aggregate forecasting task created here.

## Approach

Repository evidence:

| Dissertation content | Repository location |
|---|---|
| Streaming aggregation | `src/otto_forecasting/data.py` |
| Data validation and fingerprint | `audit_frame`, `audit-data` CLI |
| Temporal features | `add_time_features` |
| Split boundaries | `FoldSpec`, `make_rolling_origin_folds` |
| Training-only scaling | `prepare_fold_arrays` |
| Baselines | `baselines.py` |
| Transformer and GRU | `model.py` |
| Early stopping and seeds | `training.py` |
| Repeated evaluation | `research.py` |
| Metrics and bootstrap | `metrics.py` |
| Prediction intervals | `conformal_intervals` |
| Reproducibility manifest | `experiment_manifest.json` |
| Deployment interface | `api.py` |

Include a split diagram and state exact index or timestamp boundaries from the generated manifest.

## Evaluation and results

Minimum tables:

1. Data audit and split sizes
2. Model configuration and parameter counts
3. Cross-validation metrics by model
4. Final holdout metrics by model and target
5. Mean and standard deviation for neural seeds
6. Bootstrap comparison interval
7. Prediction-interval coverage
8. Ablation summary
9. Runtime comparison

Minimum figures:

1. Hourly target series
2. Fold and holdout diagram
3. Model comparison with repeated-run variation
4. Fold stability plot
5. Horizon-level error plot
6. Forecast example with interval
7. Residual distribution or error by hour/day

## Conclusion

State only conclusions supported by the generated artifacts. Include:

- Which model family ranked best under cross-validation
- Whether the holdout supports the same conclusion
- Whether the model difference is statistically clear
- Whether interval coverage approached the nominal level
- Why the short time span limits generalization
- Why global aggregation limits product-level business use

## Appendices

Recommended appendix materials:

- Full YAML configuration
- Data audit JSON
- Experiment manifest
- Additional seed-level tables
- Full ablation table
- API request schema
- Test summary
- AI-use disclosure, when required
