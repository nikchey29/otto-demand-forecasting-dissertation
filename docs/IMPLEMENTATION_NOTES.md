# Implementation notes

These are the decisions and problems that mattered most while I was building the project.

## Aggregating the OTTO file

The training JSONL is too large for the workflow I wanted to use in Colab, so I do not load the
whole file into a DataFrame. `aggregate_jsonl` reads one session at a time and only keeps the
three hourly counters in memory.

GPU acceleration does not help this stage much because the bottleneck is JSON parsing and disk
I/O. The GPU becomes useful later for the GRU and Transformer.

## The raw-data size initially gave the wrong impression

At first, the hundreds of millions of events made the dataset look extremely large for deep
learning. After hourly aggregation there are only about 672 time points. That changed several
choices:

- the Transformer was kept small;
- strong seasonal and linear baselines became essential;
- repeated seeds were added;
- the final claims were narrowed to this short observation period.

## The first Transformer result was not the best overall

In the first chronological holdout, Ridge had the lowest average MAE, while the Transformer was
slightly better on orders. That was useful because it changed the question from “can I make a
Transformer win?” to “when does the extra model complexity help, if at all?”

The preliminary output is kept under `artifacts/preliminary_single_split/`.

## Leakage checks

The easiest way to get an unrealistically good result in this setup is to construct all windows
first and then split them, or to fit a scaler on the complete series. The implementation instead
creates target starts from explicit chronological boundaries and fits scalers on the training
segment only.

## Colab resets

I use Google Drive for the processed hourly CSV and result files. Colab's `/content` storage is
only treated as working space. The self-contained notebook was added after losing an earlier
runtime and wanting a version that could be rerun without depending on a local clone.

## What remains before the dissertation is frozen

The rolling-origin experiment and ablation study need to be run on the real processed series.
After that run I need to check the saved CSV/JSON outputs, update the results section of the
README, and make sure every number used in the dissertation can be traced to an artifact.
