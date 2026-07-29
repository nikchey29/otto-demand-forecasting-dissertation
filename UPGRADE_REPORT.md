# Upgrade Report

## Starting assessment

The original repository was technically strong in software engineering but limited by one short holdout, one neural seed, four model families and no statistical uncertainty analysis. The previous provisional technical rating was 73/100.

## Changes implemented

### Experimental rigor

- Three expanding-window folds before the final holdout
- Five neural seeds
- Final 96-hour untouched test interval
- Validation-based Ridge tuning
- GRU and Extra Trees benchmarks
- 24-hour, 168-hour and blended seasonal baselines
- MASE, RMSSE and bias
- Block-bootstrap model-difference intervals
- Conformal prediction intervals
- Lookback, feature and depth ablations

### Reproducibility

- Data validation and SHA-256 fingerprint
- Exact split boundaries in the experiment manifest
- Package and platform versions
- Runtime, parameter count and best epoch records
- Prediction-level CSV output
- Pinned validation environment file

### Engineering quality

- Selected-model API support
- Graceful degraded health state
- Synthetic smoke-data generator
- Expanded CLI and Makefile
- Coverage-enabled CI
- Seventeen passing tests
- Approximately 68% measured package coverage in the validation environment

### Academic framing

- Corrected scope to aggregate hourly volume forecasting
- Added data and model cards
- Added dissertation evidence map
- Added research protocol and results checklist
- Added AI-use disclosure template
- Preserved original results as a clearly labeled legacy experiment

## What remains before a 90+ claim

- Execute the full suite on the real processed OTTO data
- Execute the ablations
- Inspect whether results are stable across folds and seeds
- Confirm bootstrap intervals support claimed improvements
- Write and critically relate the findings to peer-reviewed literature
- Obtain supervisor approval for methodology and AI-use disclosure
- Defend every implementation and limitation in the viva

No mark is guaranteed by a repository alone. The revision is designed to remove the main technical weaknesses that prevented the original project from approaching an excellent dissertation standard.
