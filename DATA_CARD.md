# Dataset notes

## Source

This project uses the public OTTO Recommender Systems training data. Each row represents an anonymised session containing timestamped click, cart and order events.

## How I use it

The original competition is a session-recommendation task. My dissertation uses the data differently: I aggregate all supported events into one UTC hourly series.

The resulting columns are:

- `clicks`
- `carts`
- `orders`

Carts and orders are the forecasting targets. Clicks are used as an explanatory history feature.

## Processing steps

1. Read the JSONL file one session at a time so the complete raw file is not loaded into memory.
2. Convert event timestamps to UTC hourly buckets.
3. Count clicks, carts and orders in each hour.
4. Reindex the full hourly range and fill missing buckets with zero.
5. Check timestamp order, duplicate hours, missing intervals, negative values and non-finite values.
6. Save the processed hourly data locally.

## Effective sample size

The raw dataset contains more than 220 million events, but the hourly forecasting series has approximately 672 rows. The number of raw events is therefore not the statistical sample size of the forecasting experiment.

## Features

Historical values are transformed with `log1p`. The full feature set contains:

- log clicks;
- log carts;
- log orders;
- sine and cosine of hour of day;
- sine and cosine of day of week;
- weekend indicator.

## Important limitations

- Only 28 days of hourly history are available.
- Product and category information is removed during aggregation.
- Price, promotion, stock, marketing and holiday information are unavailable.
- Consecutive forecast windows overlap.
- Results may not transfer to another retailer or a different time period.

## Privacy and storage

The data is anonymised secondary data. This project does not attempt to identify individual users. The raw file and processed data are excluded from GitHub and must be obtained separately under the dataset's terms.
