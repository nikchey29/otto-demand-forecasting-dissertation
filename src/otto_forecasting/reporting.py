from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: list[dict[str, float | int]], output: str | Path) -> None:
    frame = pd.DataFrame(history)
    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.plot(frame["epoch"], frame["train_loss"], label="Training")
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Huber loss")
    axis.set_title("Training history")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_model_comparison(metrics: pd.DataFrame, output: str | Path) -> None:
    if "mae_mean" in metrics.columns:
        summary = metrics.groupby("model", as_index=False)["mae_mean"].mean()
        value_column = "mae_mean"
    else:
        summary = metrics.groupby("model", as_index=False)["mae"].mean()
        value_column = "mae"
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    axis.bar(summary["model"], summary[value_column])
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("Average forecasting MAE across targets")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_repeated_model_comparison(summary: pd.DataFrame, output: str | Path) -> None:
    phase = "holdout" if "holdout" in set(summary["phase"]) else summary["phase"].iloc[0]
    selected = summary[summary["phase"] == phase]
    grouped = selected.groupby("model", as_index=False).agg(
        mae_mean=("mae_mean", "mean"),
        mae_std=("mae_std", "mean"),
    )
    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(111)
    axis.bar(grouped["model"], grouped["mae_mean"], yerr=grouped["mae_std"].fillna(0))
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean absolute error")
    axis.set_title(f"{phase.title()} MAE with repeated-run variation")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_fold_performance(metrics: pd.DataFrame, output: str | Path) -> None:
    selected = metrics[metrics["phase"] == "cv"].copy()
    if selected.empty:
        return
    selected["average_mae"] = selected.groupby(
        ["fold", "seed", "model"], dropna=False
    )["mae"].transform("mean")
    unique = selected.drop_duplicates(["fold", "seed", "model"])
    aggregated = unique.groupby(["fold", "model"], as_index=False)["average_mae"].mean()
    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(111)
    for model, group in aggregated.groupby("model"):
        axis.plot(group["fold"], group["average_mae"], marker="o", label=model)
    axis.set_xlabel("Rolling-origin fold")
    axis.set_ylabel("Average MAE")
    axis.set_title("Model stability across chronological folds")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_horizon_error(horizon_metrics: pd.DataFrame, output: str | Path) -> None:
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    for target, group in horizon_metrics.groupby("target"):
        axis.plot(group["forecast_hour"], group["mae"], label=target)
    axis.set_xlabel("Forecast hour")
    axis.set_ylabel("MAE")
    axis.set_title("Error by forecast horizon")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_forecast_example(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_name: str,
    target_index: int,
    output: str | Path,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    hours = np.arange(1, actual.shape[1] + 1)
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    axis.plot(hours, actual[0, :, target_index], marker="o", label="Actual")
    axis.plot(hours, predicted[0, :, target_index], marker="x", label="Forecast")
    if lower is not None and upper is not None:
        axis.fill_between(
            hours,
            lower[0, :, target_index],
            upper[0, :, target_index],
            alpha=0.2,
            label="Conformal interval",
        )
    axis.set_xlabel("Forecast hour")
    axis.set_ylabel(target_name.title())
    axis.set_title(f"24-hour {target_name} forecast")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
