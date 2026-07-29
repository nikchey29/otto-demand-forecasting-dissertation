from __future__ import annotations

import numpy as np
import pandas as pd


def _scaled_error_denominators(
    training_actual: np.ndarray,
    seasonality: int,
) -> tuple[np.ndarray, np.ndarray]:
    if seasonality < 1 or len(training_actual) <= seasonality:
        raise ValueError("Training history is too short for scaled-error metrics")
    differences = training_actual[seasonality:] - training_actual[:-seasonality]
    mae_scale = np.maximum(np.mean(np.abs(differences), axis=0), 1e-8)
    mse_scale = np.maximum(np.mean(differences**2, axis=0), 1e-8)
    return mae_scale, mse_scale


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
    training_actual: np.ndarray | None = None,
    seasonality: int = 24,
) -> pd.DataFrame:
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have matching shapes")
    if actual.shape[-1] != len(target_names):
        raise ValueError("Target names do not match the final array dimension")

    mae_scale: np.ndarray | None = None
    mse_scale: np.ndarray | None = None
    if training_actual is not None:
        mae_scale, mse_scale = _scaled_error_denominators(training_actual, seasonality)

    rows: list[dict[str, float | str]] = []
    for index, target in enumerate(target_names):
        observed = actual[..., index].reshape(-1)
        forecast = predicted[..., index].reshape(-1)
        error = forecast - observed
        absolute_error = np.abs(error)
        denominator = np.maximum(np.abs(observed), 1.0)
        row: dict[str, float | str] = {
            "target": target,
            "mae": float(absolute_error.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "wape": float(absolute_error.sum() / np.maximum(np.abs(observed).sum(), 1.0)),
            "smape": float(
                np.mean(
                    2
                    * absolute_error
                    / np.maximum(np.abs(observed) + np.abs(forecast), 1.0)
                )
            ),
            "mape": float(np.mean(absolute_error / denominator)),
            "bias": float(np.mean(error)),
        }
        if mae_scale is not None and mse_scale is not None:
            row["mase"] = float(absolute_error.mean() / mae_scale[index])
            row["rmsse"] = float(np.sqrt(np.mean(error**2) / mse_scale[index]))
        rows.append(row)
    return pd.DataFrame(rows)


def horizon_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for horizon_index in range(actual.shape[1]):
        for target_index, target in enumerate(target_names):
            observed = actual[:, horizon_index, target_index]
            forecast = predicted[:, horizon_index, target_index]
            error = forecast - observed
            absolute_error = np.abs(error)
            rows.append(
                {
                    "forecast_hour": horizon_index + 1,
                    "target": target,
                    "mae": float(absolute_error.mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "bias": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)


def horizon_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    """Backward-compatible MAE-only horizon report."""
    return horizon_metrics(actual, predicted, target_names).loc[
        :, ["forecast_hour", "target", "mae"]
    ]


def conformal_intervals(
    validation_actual: np.ndarray,
    validation_prediction: np.ndarray,
    test_prediction: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one")
    if validation_actual.shape != validation_prediction.shape:
        raise ValueError("Validation actuals and predictions must match")
    if validation_actual.shape[1:] != test_prediction.shape[1:]:
        raise ValueError("Validation and test forecast shapes are incompatible")
    residuals = np.abs(validation_actual - validation_prediction)
    calibration_size = residuals.shape[0]
    quantile_level = min(
        1.0,
        np.ceil((calibration_size + 1) * (1 - alpha)) / calibration_size,
    )
    quantiles = np.quantile(residuals, quantile_level, axis=0, method="higher")
    lower = np.maximum(test_prediction - quantiles[None, ...], 0.0)
    upper = test_prediction + quantiles[None, ...]
    return lower, upper, quantiles


def interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_names: tuple[str, ...],
    nominal_coverage: float,
) -> pd.DataFrame:
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Actual, lower and upper arrays must match")
    rows: list[dict[str, float | str]] = []
    for target_index, target in enumerate(target_names):
        observed = actual[..., target_index]
        low = lower[..., target_index]
        high = upper[..., target_index]
        covered = (observed >= low) & (observed <= high)
        width = high - low
        rows.append(
            {
                "target": target,
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": float(covered.mean()),
                "coverage_gap": float(covered.mean() - nominal_coverage),
                "mean_interval_width": float(width.mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_repeated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"phase", "model", "target", "mae", "rmse", "wape"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    metric_columns = [
        column
        for column in ("mae", "rmse", "wape", "smape", "mape", "bias", "mase", "rmsse")
        if column in metrics.columns
    ]
    grouped = metrics.groupby(["phase", "model", "target"], dropna=False)[metric_columns]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def paired_block_bootstrap_mae(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target_names: tuple[str, ...],
    model_a: str,
    model_b: str,
    samples: int = 2000,
    block_size: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate MAE-difference confidence intervals while preserving local origin dependence."""
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    if actual.shape != prediction_a.shape or actual.shape != prediction_b.shape:
        raise ValueError("All arrays must have matching shapes")
    origins = actual.shape[0]
    if origins < 2:
        raise ValueError("At least two forecast origins are required")
    block_size = max(1, min(block_size, origins))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, origins - block_size + 1)
    rows: list[dict[str, float | str]] = []

    for target_index, target in enumerate(target_names):
        error_a = np.abs(
            prediction_a[..., target_index] - actual[..., target_index]
        ).mean(axis=1)
        error_b = np.abs(
            prediction_b[..., target_index] - actual[..., target_index]
        ).mean(axis=1)
        observed_difference = float(error_a.mean() - error_b.mean())
        bootstrap_differences = np.empty(samples, dtype=np.float64)
        blocks_needed = int(np.ceil(origins / block_size))
        for sample_index in range(samples):
            sampled: list[int] = []
            for _ in range(blocks_needed):
                start = int(rng.choice(starts))
                sampled.extend(range(start, start + block_size))
            indices = np.asarray(sampled[:origins], dtype=np.int64)
            bootstrap_differences[sample_index] = float(
                error_a[indices].mean() - error_b[indices].mean()
            )
        low, high = np.quantile(bootstrap_differences, [0.025, 0.975])
        rows.append(
            {
                "target": target,
                "model_a": model_a,
                "model_b": model_b,
                "mae_difference_a_minus_b": observed_difference,
                "ci_2_5": float(low),
                "ci_97_5": float(high),
                "probability_a_better": float(np.mean(bootstrap_differences < 0)),
            }
        )
    return pd.DataFrame(rows)
