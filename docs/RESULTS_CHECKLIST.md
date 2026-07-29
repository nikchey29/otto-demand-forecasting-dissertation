# Results Checklist

Complete every item before writing the final results chapter.

## Execution

- [ ] The real OTTO hourly CSV exists.
- [ ] `audit-data` completed successfully.
- [ ] The full research command completed.
- [ ] The ablation command completed.
- [ ] All tests pass.
- [ ] The experiment manifest contains the expected five seeds.
- [ ] The final test interval was not used during debugging or tuning.

## Tables

- [ ] Dataset audit table
- [ ] Split and window-count table
- [ ] Model and parameter-count table
- [ ] Cross-validation ranking
- [ ] Final holdout metrics by target
- [ ] Neural mean and standard deviation
- [ ] Statistical comparison interval
- [ ] Prediction-interval coverage
- [ ] Ablation summary
- [ ] Runtime and hardware table

## Interpretation

- [ ] The task is described as aggregate volume forecasting.
- [ ] The raw event count is not presented as the forecasting sample size.
- [ ] The four-week temporal limitation is explicit.
- [ ] Overlapping forecast origins are acknowledged.
- [ ] No model is called superior solely because one point estimate is lower.
- [ ] The paper distinguishes validation, cross-validation and final holdout results.
- [ ] Business implications are proportional to the evidence.

## Consistency

- [ ] Every number in the paper exists in a generated artifact.
- [ ] Table rounding is consistent.
- [ ] Target names are consistently “carts” and “orders.”
- [ ] Model names match the CSV output.
- [ ] Figures include units and forecast horizon.
- [ ] Limitations in the paper match the model card and data card.
