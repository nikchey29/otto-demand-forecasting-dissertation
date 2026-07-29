# Project notes

These are the main practical lessons I recorded while building the project.

## Data preparation was the slowest first step

The raw OTTO file is large and stored as JSONL. Parsing it is mainly a CPU and disk task, so changing a Colab runtime to a GPU did not make the aggregation step much faster. The GPU is useful later for training the GRU and Transformer.

## Colab files are temporary

Changing or restarting a Colab runtime clears local variables and can remove files stored in the temporary machine. I therefore save the processed hourly CSV and experiment outputs in Google Drive.

## The raw event count can be misleading

The dataset contains hundreds of millions of events, but after global hourly aggregation there are only about 672 time points. This is the main reason I kept the neural architectures small and included strong simple baselines.

## A Transformer does not have to be the best model

In the preliminary experiment, Ridge regression performed best overall, while the Transformer had a small advantage only for orders. This changed the direction of the dissertation from “proving a Transformer is better” to comparing model complexity honestly.

## Leakage checks matter more than model size

The most important implementation choice was to keep target windows inside their chronological split and fit scalers only on training observations. A low validation loss is not meaningful when future information enters preprocessing or training.

## What still needs to be completed

The final rolling-origin and ablation experiments must be run on the real processed OTTO series. The dissertation results should be written only after those output files have been checked.
