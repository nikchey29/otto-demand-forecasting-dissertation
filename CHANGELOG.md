# Changelog

## 2.0.0 — Research-grade dissertation revision

### Added

- Expanding-window rolling-origin folds
- Untouched final holdout
- Five-seed neural evaluation
- GRU and Extra Trees benchmarks
- Daily, weekly and blended seasonal baselines
- Validation-based Ridge alpha selection
- MASE, RMSSE and bias
- Moving-block bootstrap comparison
- Split-conformal prediction intervals
- Configurable ablation suite
- Data audit and SHA-256 fingerprint
- Experiment manifest and runtime metadata
- Selected-model API loading
- Graceful degraded API mode
- Synthetic smoke-data generator
- Seventeen automated tests
- Dissertation protocol, reproducibility and results documentation

### Changed

- Project scope is stated as aggregate hourly volume forecasting
- Original results moved to `artifacts/legacy_single_split/`
- Transformer uses learned attention pooling instead of simple mean pooling
- Configuration schema supports repeated experiments and multiple seasonalities

### Important

No new real-data metrics are included in this revision. The full research and ablation commands must be executed before updating dissertation results.
