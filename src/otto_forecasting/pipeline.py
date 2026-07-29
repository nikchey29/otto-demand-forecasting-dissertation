from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from otto_forecasting.baselines import (
    ExtraTreesForecaster,
    blended_seasonal_forecast,
    persistence_forecast,
    seasonal_forecast,
    select_ridge_alpha,
)
from otto_forecasting.config import ProjectConfig
from otto_forecasting.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    add_time_features,
    audit_frame,
    load_processed,
)
from otto_forecasting.dataset import build_loaders, inverse_targets, prepare_arrays
from otto_forecasting.metrics import horizon_metrics, regression_metrics
from otto_forecasting.model import DemandTransformer, count_parameters
from otto_forecasting.reporting import (
    plot_forecast_example,
    plot_horizon_error,
    plot_model_comparison,
    plot_training_history,
)
from otto_forecasting.training import predict_model, save_history, set_seed, train_model


def _raw_windows(raw_targets: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([raw_targets[start : start + horizon] for start in starts])


def run_training(config: ProjectConfig) -> dict[str, Path]:
    """Run the fast single-holdout experiment.

    Use ``otto-forecast research`` for repeated seeds, rolling-origin folds,
    confidence intervals and statistical comparison.
    """
    set_seed(config.seed)
    output_dir = Path(config.output_dir) / "single_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_processed(config.data.processed_path)
    frame = add_time_features(raw_frame)
    feature_values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    target_values = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    prepared = prepare_arrays(
        feature_values=feature_values,
        raw_target_values=target_values,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        validation_steps=config.data.validation_steps,
        test_steps=config.data.test_steps,
    )
    train_loader, validation_loader, test_loader = build_loaders(
        prepared=prepared,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandTransformer(
        num_features=len(FEATURE_COLUMNS),
        target_dim=len(TARGET_COLUMNS),
        horizon=config.data.horizon,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
    ).to(device)
    training_result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        gradient_clip=config.training.gradient_clip,
        device=device,
    )
    scaled_prediction, scaled_actual = predict_model(model, test_loader, device)
    transformer_prediction = inverse_targets(scaled_prediction, prepared.target_scaler)
    transformer_actual = inverse_targets(scaled_actual, prepared.target_scaler)

    ridge, ridge_alpha, _ = select_ridge_alpha(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        prepared.validation_starts,
        config.data.lookback,
        config.data.horizon,
        config.baselines.ridge_alphas,
    )
    ridge_prediction = inverse_targets(
        ridge.predict(prepared.features, prepared.test_starts, config.data.lookback),
        prepared.target_scaler,
    )

    extra_trees = ExtraTreesForecaster(
        n_estimators=config.baselines.extra_trees_estimators,
        min_samples_leaf=config.baselines.extra_trees_min_samples_leaf,
        max_features=config.baselines.extra_trees_max_features,
        random_state=config.seed,
    ).fit(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        config.data.lookback,
        config.data.horizon,
    )
    extra_trees_prediction = inverse_targets(
        extra_trees.predict(prepared.features, prepared.test_starts, config.data.lookback),
        prepared.target_scaler,
    )

    evaluations: dict[str, np.ndarray] = {
        "Transformer": transformer_prediction,
        "Ridge": ridge_prediction,
        "Extra Trees": extra_trees_prediction,
        "Persistence": persistence_forecast(
            prepared.raw_targets, prepared.test_starts, config.data.horizon
        ),
    }
    available_seasonalities: list[int] = []
    for seasonality in config.data.seasonalities:
        if prepared.test_starts.min() >= seasonality:
            evaluations[f"Seasonal naive {seasonality}h"] = seasonal_forecast(
                prepared.raw_targets,
                prepared.test_starts,
                config.data.horizon,
                seasonality,
            )
            available_seasonalities.append(seasonality)
    if len(available_seasonalities) >= 2:
        evaluations[
            "Seasonal blend " + "+".join(f"{value}h" for value in available_seasonalities)
        ] = blended_seasonal_forecast(
            prepared.raw_targets,
            prepared.test_starts,
            config.data.horizon,
            tuple(available_seasonalities),
        )

    training_actual = prepared.raw_targets[: prepared.fold.train_end]
    metric_frames: list[pd.DataFrame] = []
    for name, prediction in evaluations.items():
        metrics = regression_metrics(
            transformer_actual,
            prediction,
            TARGET_COLUMNS,
            training_actual=training_actual,
            seasonality=24,
        )
        metrics.insert(0, "model", name)
        metric_frames.append(metrics)
    comparison = pd.concat(metric_frames, ignore_index=True)
    horizon = horizon_metrics(transformer_actual, transformer_prediction, TARGET_COLUMNS)

    model_path = output_dir / "transformer.pt"
    torch.save(model.state_dict(), model_path)
    joblib.dump(prepared.feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(prepared.target_scaler, output_dir / "target_scaler.joblib")
    joblib.dump(ridge, output_dir / "ridge.joblib")
    joblib.dump(extra_trees, output_dir / "extra_trees.joblib")
    save_history(training_result.history, output_dir / "training_history.json")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    horizon.to_csv(output_dir / "horizon_metrics.csv", index=False)

    metadata = {
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "lookback": config.data.lookback,
        "horizon": config.data.horizon,
        "model": config.to_dict()["model"],
        "test_windows": int(len(prepared.test_starts)),
        "train_windows": int(len(prepared.train_starts)),
        "validation_windows": int(len(prepared.validation_starts)),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "best_epoch": training_result.best_epoch,
        "best_validation_loss": training_result.best_validation_loss,
        "ridge_alpha": ridge_alpha,
        "data_audit": audit_frame(raw_frame, config.data.processed_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    predictions = []
    for model_name, prediction in evaluations.items():
        for window_index, target_start in enumerate(prepared.test_starts):
            for horizon_index in range(config.data.horizon):
                timestamp_index = int(target_start) + horizon_index
                for target_index, target in enumerate(TARGET_COLUMNS):
                    predictions.append(
                        {
                            "model": model_name,
                            "window": window_index,
                            "forecast_hour": horizon_index + 1,
                            "timestamp": frame.iloc[timestamp_index]["timestamp"],
                            "target": target,
                            "actual": transformer_actual[
                                window_index, horizon_index, target_index
                            ],
                            "prediction": prediction[
                                window_index, horizon_index, target_index
                            ],
                        }
                    )
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)

    plot_training_history(training_result.history, output_dir / "training_history.png")
    plot_model_comparison(comparison, output_dir / "model_comparison.png")
    plot_horizon_error(horizon, output_dir / "horizon_metrics.png")
    for target_index, target in enumerate(TARGET_COLUMNS):
        plot_forecast_example(
            transformer_actual,
            transformer_prediction,
            target,
            target_index,
            output_dir / f"forecast_{target}.png",
        )

    return {
        "model": model_path,
        "metrics": output_dir / "model_comparison.csv",
        "predictions": output_dir / "predictions.csv",
        "metadata": output_dir / "metadata.json",
    }
