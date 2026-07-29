# Data Card

## Source

OTTO Recommender Systems clickstream training data. The source dataset contains anonymized sessions and events with timestamps, article identifiers and event types.

## Use in this project

The project does not reproduce the original session-recommendation task. It aggregates all supported events into a single global hourly series:

- Clicks
- Carts
- Orders

The forecasting targets are hourly cart and order counts.

## Derived sample size

After one-hour aggregation, the public training period contains approximately 672 observations, representing 28 days. The raw event count must not be treated as the number of independent forecasting observations.

## Processing

- Stream JSONL records without loading the full source into memory
- Floor event timestamps to UTC hourly buckets
- Count events by type
- Reindex the complete hourly range
- Fill absent buckets with zero counts
- Validate uniqueness, order, spacing, finite values and non-negative counts
- Save a processed CSV or Parquet file

## Features

Historical count features:

- `log_clicks`
- `log_carts`
- `log_orders`

Calendar features:

- Hour sine and cosine
- Day-of-week sine and cosine
- Weekend indicator

## Limitations

- Only four weeks of temporal coverage
- No product-level or category-level forecasts
- No price, promotion, stock, holiday, campaign or weather features
- Article and customer heterogeneity is removed by global aggregation
- OTTO traffic may not represent another retailer or current production behavior
- Adjacent forecasting windows overlap and are statistically dependent

## Ethics and privacy

The project uses secondary anonymized behavioral event data. It does not attempt to identify users or reconstruct individual behavior. The author must still complete the university's required ethical-approval process for secondary data before submission.

## Reproducibility

`otto-forecast audit-data` records:

- Row count
- Timestamp range
- Event totals
- Zero-activity hours
- File size
- SHA-256 fingerprint

Do not publish or redistribute the raw dataset when its license or platform terms prohibit doing so.
