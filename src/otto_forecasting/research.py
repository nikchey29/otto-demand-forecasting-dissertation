from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
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
    select_feature_columns,
)
from otto_forecasting.dataset import (
    FoldSpec,
    PreparedData,
    build_loaders,
    inverse_targets,
    make_final_fold,
    make_rolling_origin_folds,
    prepare_fold_arrays,
)
from otto_forecasting.metrics import (
    conformal_intervals,
    horizon_metrics,
    interval_metrics,
    paired_block_bootstrap_mae,
    regression_metrics,
    summarize_repeated_metrics,
)
from otto_forecasting.model import DemandTransformer, GRUForecaster, count_parameters
from otto_forecasting.reporting import (
    plot_fold_performance,
    plot_forecast_example,
    plot_horizon_error,
    plot_repeated_model_comparison,
)
from otto_forecasting.training import predict_model, set_seed, train_model


def _raw_windows(raw_targets: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([raw_targets[start : start + horizon] for start in starts])


def _build_neural_model(
    model_name: str,
    config: ProjectConfig,
    num_features: int,
    num_layers_override: int | None = None,
) -> torch.nn.Module:
    if model_name == "Transformer":
        return DemandTransformer(
            num_features=num_features,
            target_dim=len(TARGET_COLUMNS),
            horizon=config.data.horizon,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=(
                num_layers_override
                if num_layers_override is not None
                else config.model.num_layers
            ),
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
        )
    if model_name == "GRU":
        return GRUForecaster(
            num_features=num_features,
            target_dim=len(TARGET_COLUMNS),
            horizon=config.data.horizon,
            hidden_size=config.gru.hidden_size,
            num_layers=config.gru.num_layers,
            dropout=config.gru.dropout,
        )
    raise ValueError(f"Unknown neural model: {model_name}")


def _evaluate_prediction(
    actual: np.ndarray,
    prediction: np.ndarray,
    model_name: str,
    phase: str,
    fold_id: int,
    seed: int | None,
    training_actual: np.ndarray,
) -> pd.DataFrame:
    metrics = regression_metrics(
        actual,
        prediction,
        TARGET_COLUMNS,
        training_actual=training_actual,
        seasonality=24,
    )
    metrics.insert(0, "seed", seed)
    metrics.insert(0, "fold", fold_id)
    metrics.insert(0, "phase", phase)
    metrics.insert(3, "model", model_name)
    return metrics


def _prediction_rows(
    frame: pd.DataFrame,
    starts: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    model_name: str,
    phase: str,
    fold_id: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin_index, target_start in enumerate(starts):
        origin_timestamp = frame.iloc[int(target_start) - 1]["timestamp"]
        for horizon_index in range(prediction.shape[1]):
            timestamp = frame.iloc[int(target_start) + horizon_index]["timestamp"]
            for target_index, target in enumerate(TARGET_COLUMNS):
                rows.append(
                    {
                        "phase": phase,
                        "fold": fold_id,
                        "seed": seed,
                        "model": model_name,
                        "origin": origin_index,
                        "origin_timestamp": origin_timestamp,
                        "forecast_hour": horizon_index + 1,
                        "timestamp": timestamp,
                        "target": target,
                        "actual": float(actual[origin_index, horizon_index, target_index]),
                        "prediction": float(
                            prediction[origin_index, horizon_index, target_index]
                        ),
                    }
                )
    return rows


def _run_neural(
    model_name: str,
    prepared: PreparedData,
    config: ProjectConfig,
    seed: int,
    num_features: int,
    num_layers_override: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    train_loader, validation_loader, test_loader = build_loaders(
        prepared=prepared,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_neural_model(
        model_name,
        config,
        num_features=num_features,
        num_layers_override=num_layers_override,
    ).to(device)
    started = time.perf_counter()
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
    runtime_seconds = time.perf_counter() - started

    validation_scaled, validation_actual_scaled = predict_model(
        model, validation_loader, device
    )
    test_scaled, test_actual_scaled = predict_model(model, test_loader, device)
    return {
        "model": model,
        "validation_prediction": inverse_targets(
            validation_scaled, prepared.target_scaler
        ),
        "validation_actual": inverse_targets(
            validation_actual_scaled, prepared.target_scaler
        ),
        "test_prediction": inverse_targets(test_scaled, prepared.target_scaler),
        "test_actual": inverse_targets(test_actual_scaled, prepared.target_scaler),
        "training_result": training_result,
        "runtime_seconds": runtime_seconds,
        "parameter_count": count_parameters(model),
        "device": str(device),
    }


def _run_fold_models(
    frame: pd.DataFrame,
    prepared: PreparedData,
    config: ProjectConfig,
    phase: str,
    feature_columns: tuple[str, ...],
    save_models: bool = False,
) -> dict[str, Any]:
    actual_test = _raw_windows(
        prepared.raw_targets, prepared.test_starts, config.data.horizon
    )
    actual_validation = _raw_windows(
        prepared.raw_targets, prepared.validation_starts, config.data.horizon
    )
    training_actual = prepared.raw_targets[: prepared.fold.train_end]
    metric_frames: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[np.ndarray]] = {}
    validation_predictions: dict[str, list[np.ndarray]] = {}
    fitted_models: dict[str, Any] = {}
    run_metadata: list[dict[str, Any]] = []

    ridge, alpha, validation_loss = select_ridge_alpha(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        prepared.validation_starts,
        config.data.lookback,
        config.data.horizon,
        config.baselines.ridge_alphas,
    )
    ridge_test = inverse_targets(
        ridge.predict(prepared.features, prepared.test_starts, config.data.lookback),
        prepared.target_scaler,
    )
    ridge_validation = inverse_targets(
        ridge.predict(prepared.features, prepared.validation_starts, config.data.lookback),
        prepared.target_scaler,
    )
    predictions["Ridge"] = [ridge_test]
    validation_predictions["Ridge"] = [ridge_validation]
    fitted_models["Ridge"] = ridge
    metric_frames.append(
        _evaluate_prediction(
            actual_test,
            ridge_test,
            "Ridge",
            phase,
            prepared.fold.fold_id,
            None,
            training_actual,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            frame,
            prepared.test_starts,
            actual_test,
            ridge_test,
            "Ridge",
            phase,
            prepared.fold.fold_id,
            None,
        )
    )
    run_metadata.append(
        {
            "phase": phase,
            "fold": prepared.fold.fold_id,
            "model": "Ridge",
            "selected_alpha": alpha,
            "validation_scaled_mae": validation_loss,
        }
    )

    if config.research.run_extra_trees:
        started = time.perf_counter()
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
        extra_test = inverse_targets(
            extra_trees.predict(
                prepared.features, prepared.test_starts, config.data.lookback
            ),
            prepared.target_scaler,
        )
        extra_validation = inverse_targets(
            extra_trees.predict(
                prepared.features, prepared.validation_starts, config.data.lookback
            ),
            prepared.target_scaler,
        )
        predictions["Extra Trees"] = [extra_test]
        validation_predictions["Extra Trees"] = [extra_validation]
        fitted_models["Extra Trees"] = extra_trees
        metric_frames.append(
            _evaluate_prediction(
                actual_test,
                extra_test,
                "Extra Trees",
                phase,
                prepared.fold.fold_id,
                None,
                training_actual,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                frame,
                prepared.test_starts,
                actual_test,
                extra_test,
                "Extra Trees",
                phase,
                prepared.fold.fold_id,
                None,
            )
        )
        run_metadata.append(
            {
                "phase": phase,
                "fold": prepared.fold.fold_id,
                "model": "Extra Trees",
                "runtime_seconds": time.perf_counter() - started,
            }
        )

    persistence_test = persistence_forecast(
        prepared.raw_targets, prepared.test_starts, config.data.horizon
    )
    persistence_validation = persistence_forecast(
        prepared.raw_targets, prepared.validation_starts, config.data.horizon
    )
    predictions["Persistence"] = [persistence_test]
    validation_predictions["Persistence"] = [persistence_validation]
    metric_frames.append(
        _evaluate_prediction(
            actual_test,
            persistence_test,
            "Persistence",
            phase,
            prepared.fold.fold_id,
            None,
            training_actual,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            frame,
            prepared.test_starts,
            actual_test,
            persistence_test,
            "Persistence",
            phase,
            prepared.fold.fold_id,
            None,
        )
    )

    available_seasonalities: list[int] = []
    for seasonality in config.data.seasonalities:
        if min(prepared.validation_starts.min(), prepared.test_starts.min()) < seasonality:
            continue
        model_name = f"Seasonal naive {seasonality}h"
        seasonal_test = seasonal_forecast(
            prepared.raw_targets,
            prepared.test_starts,
            config.data.horizon,
            seasonality,
        )
        seasonal_validation = seasonal_forecast(
            prepared.raw_targets,
            prepared.validation_starts,
            config.data.horizon,
            seasonality,
        )
        available_seasonalities.append(seasonality)
        predictions[model_name] = [seasonal_test]
        validation_predictions[model_name] = [seasonal_validation]
        metric_frames.append(
            _evaluate_prediction(
                actual_test,
                seasonal_test,
                model_name,
                phase,
                prepared.fold.fold_id,
                None,
                training_actual,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                frame,
                prepared.test_starts,
                actual_test,
                seasonal_test,
                model_name,
                phase,
                prepared.fold.fold_id,
                None,
            )
        )

    if len(available_seasonalities) >= 2:
        blend_name = "Seasonal blend " + "+".join(
            f"{seasonality}h" for seasonality in available_seasonalities
        )
        blend_test = blended_seasonal_forecast(
            prepared.raw_targets,
            prepared.test_starts,
            config.data.horizon,
            tuple(available_seasonalities),
        )
        blend_validation = blended_seasonal_forecast(
            prepared.raw_targets,
            prepared.validation_starts,
            config.data.horizon,
            tuple(available_seasonalities),
        )
        predictions[blend_name] = [blend_test]
        validation_predictions[blend_name] = [blend_validation]
        metric_frames.append(
            _evaluate_prediction(
                actual_test,
                blend_test,
                blend_name,
                phase,
                prepared.fold.fold_id,
                None,
                training_actual,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                frame,
                prepared.test_starts,
                actual_test,
                blend_test,
                blend_name,
                phase,
                prepared.fold.fold_id,
                None,
            )
        )

    neural_names = ["Transformer"]
    if config.research.run_gru:
        neural_names.append("GRU")
    for model_name in neural_names:
        for seed in config.research.seeds:
            result = _run_neural(
                model_name,
                prepared,
                config,
                seed,
                num_features=len(feature_columns),
            )
            prediction = result["test_prediction"]
            validation_prediction = result["validation_prediction"]
            predictions.setdefault(model_name, []).append(prediction)
            validation_predictions.setdefault(model_name, []).append(validation_prediction)
            metric_frames.append(
                _evaluate_prediction(
                    actual_test,
                    prediction,
                    model_name,
                    phase,
                    prepared.fold.fold_id,
                    seed,
                    training_actual,
                )
            )
            prediction_rows.extend(
                _prediction_rows(
                    frame,
                    prepared.test_starts,
                    actual_test,
                    prediction,
                    model_name,
                    phase,
                    prepared.fold.fold_id,
                    seed,
                )
            )
            training_result = result["training_result"]
            run_metadata.append(
                {
                    "phase": phase,
                    "fold": prepared.fold.fold_id,
                    "seed": seed,
                    "model": model_name,
                    "runtime_seconds": result["runtime_seconds"],
                    "parameter_count": result["parameter_count"],
                    "best_epoch": training_result.best_epoch,
                    "best_validation_loss": training_result.best_validation_loss,
                    "device": result["device"],
                }
            )
            if save_models:
                fitted_models.setdefault(model_name, []).append(
                    (
                        training_result.best_validation_loss,
                        seed,
                        result["model"],
                        result["training_result"],
                    )
                )

    return {
        "metrics": pd.concat(metric_frames, ignore_index=True),
        "prediction_rows": prediction_rows,
        "predictions": predictions,
        "validation_predictions": validation_predictions,
        "actual_test": actual_test,
        "actual_validation": actual_validation,
        "fitted_models": fitted_models,
        "run_metadata": run_metadata,
    }


def _average_predictions(predictions: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        model: np.mean(np.stack(model_predictions, axis=0), axis=0)
        for model, model_predictions in predictions.items()
    }


def _rank_models(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        cv_metrics.groupby("model", as_index=False)["wape"]
        .mean()
        .rename(columns={"wape": "mean_cv_wape"})
        .sort_values("mean_cv_wape")
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def run_research_suite(config: ProjectConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir) / "research"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_processed(config.data.processed_path)
    frame = add_time_features(raw_frame)
    feature_columns = FEATURE_COLUMNS
    feature_values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    target_values = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)

    audit = audit_frame(raw_frame, config.data.processed_path)
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    cv_folds = make_rolling_origin_folds(
        length=len(frame),
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        validation_steps=config.research.cv_validation_steps,
        test_steps=config.research.cv_test_steps,
        folds=config.research.cv_folds,
        stride=config.research.cv_stride,
        holdout_steps=config.research.final_test_steps,
    )
    final_fold = make_final_fold(
        length=len(frame),
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        validation_steps=config.research.final_validation_steps,
        test_steps=config.research.final_test_steps,
    )

    all_metrics: list[pd.DataFrame] = []
    all_prediction_rows: list[dict[str, Any]] = []
    all_run_metadata: list[dict[str, Any]] = []

    for fold in cv_folds:
        prepared = prepare_fold_arrays(
            feature_values,
            target_values,
            config.data.lookback,
            config.data.horizon,
            fold,
        )
        result = _run_fold_models(
            frame,
            prepared,
            config,
            phase="cv",
            feature_columns=feature_columns,
        )
        all_metrics.append(result["metrics"])
        all_prediction_rows.extend(result["prediction_rows"])
        all_run_metadata.extend(result["run_metadata"])

    cv_metrics = pd.concat(all_metrics, ignore_index=True)
    ranking = _rank_models(cv_metrics)
    ranking.to_csv(output_dir / "cv_model_ranking.csv", index=False)

    final_prepared = prepare_fold_arrays(
        feature_values,
        target_values,
        config.data.lookback,
        config.data.horizon,
        final_fold,
    )
    final_result = _run_fold_models(
        frame,
        final_prepared,
        config,
        phase="holdout",
        feature_columns=feature_columns,
        save_models=True,
    )
    all_metrics.append(final_result["metrics"])
    all_prediction_rows.extend(final_result["prediction_rows"])
    all_run_metadata.extend(final_result["run_metadata"])

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "research_metrics_raw.csv", index=False)
    summary = summarize_repeated_metrics(metrics)
    summary.to_csv(output_dir / "research_metrics_summary.csv", index=False)
    pd.DataFrame(all_prediction_rows).to_csv(
        output_dir / "research_predictions.csv", index=False
    )
    pd.DataFrame(all_run_metadata).to_csv(
        output_dir / "run_metadata.csv", index=False
    )

    holdout_predictions = _average_predictions(final_result["predictions"])
    holdout_validation_predictions = _average_predictions(
        final_result["validation_predictions"]
    )
    selected_model = str(ranking.iloc[0]["model"])
    runner_up = str(ranking.iloc[1]["model"]) if len(ranking) > 1 else selected_model
    selected_prediction = holdout_predictions[selected_model]
    runner_prediction = holdout_predictions[runner_up]

    statistical = paired_block_bootstrap_mae(
        final_result["actual_test"],
        selected_prediction,
        runner_prediction,
        TARGET_COLUMNS,
        selected_model,
        runner_up,
        samples=config.research.bootstrap_samples,
        block_size=config.research.bootstrap_block_size,
        seed=config.seed,
    )
    statistical.to_csv(output_dir / "statistical_comparisons.csv", index=False)

    lower, upper, calibration = conformal_intervals(
        final_result["actual_validation"],
        holdout_validation_predictions[selected_model],
        selected_prediction,
        alpha=config.research.conformal_alpha,
    )
    intervals = interval_metrics(
        final_result["actual_test"],
        lower,
        upper,
        TARGET_COLUMNS,
        nominal_coverage=1 - config.research.conformal_alpha,
    )
    intervals.insert(0, "model", selected_model)
    intervals.to_csv(output_dir / "interval_metrics.csv", index=False)
    np.save(output_dir / "conformal_calibration.npy", calibration)

    selected_horizon = horizon_metrics(
        final_result["actual_test"], selected_prediction, TARGET_COLUMNS
    )
    selected_horizon.insert(0, "model", selected_model)
    selected_horizon.to_csv(output_dir / "selected_model_horizon_metrics.csv", index=False)

    plot_repeated_model_comparison(summary, output_dir / "model_comparison_repeated.png")
    plot_fold_performance(metrics, output_dir / "cv_fold_stability.png")
    plot_horizon_error(selected_horizon, output_dir / "selected_model_horizon_error.png")
    for target_index, target in enumerate(TARGET_COLUMNS):
        plot_forecast_example(
            final_result["actual_test"],
            selected_prediction,
            target,
            target_index,
            output_dir / f"selected_forecast_{target}.png",
            lower=lower,
            upper=upper,
        )

    fitted = final_result["fitted_models"]
    selected_artifact: str | None = None
    selected_artifacts: list[str] = []
    selected_seeds: list[int] = []
    if selected_model in {"Ridge", "Extra Trees"}:
        selected_artifact = f"selected_{selected_model.lower().replace(' ', '_')}.joblib"
        joblib.dump(fitted[selected_model], output_dir / selected_artifact)
        selected_artifacts = [selected_artifact]
    elif selected_model in {"Transformer", "GRU"}:
        histories: list[dict[str, Any]] = []
        for validation_loss, seed, selected_torch_model, training_result in fitted[
            selected_model
        ]:
            artifact_name = f"selected_{selected_model.lower()}_seed_{seed}.pt"
            torch.save(selected_torch_model.state_dict(), output_dir / artifact_name)
            selected_artifacts.append(artifact_name)
            selected_seeds.append(int(seed))
            histories.append(
                {
                    "seed": int(seed),
                    "best_validation_loss": float(validation_loss),
                    "history": training_result.history,
                }
            )
        (output_dir / "selected_training_histories.json").write_text(
            json.dumps(histories, indent=2), encoding="utf-8"
        )

    joblib.dump(final_prepared.feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(final_prepared.target_scaler, output_dir / "target_scaler.joblib")

    manifest = {
        "selected_model_by_cv": selected_model,
        "runner_up_by_cv": runner_up,
        "selection_metric": "mean CV WAPE across carts and orders",
        "selected_artifact": selected_artifact,
        "selected_artifacts": selected_artifacts,
        "deployment_strategy": (
            "mean seed ensemble"
            if selected_model in {"Transformer", "GRU"}
            else "single fitted model"
        ),
        "feature_columns": list(feature_columns),
        "target_columns": list(TARGET_COLUMNS),
        "lookback": config.data.lookback,
        "horizon": config.data.horizon,
        "cv_folds": [fold.to_dict() for fold in cv_folds],
        "final_fold": final_fold.to_dict(),
        "seeds": list(config.research.seeds),
        "conformal_alpha": config.research.conformal_alpha,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "config": config.to_dict(),
    }
    if selected_model in {"Transformer", "GRU"}:
        manifest["selected_seeds"] = selected_seeds
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    return {
        "metrics": output_dir / "research_metrics_raw.csv",
        "summary": output_dir / "research_metrics_summary.csv",
        "ranking": output_dir / "cv_model_ranking.csv",
        "predictions": output_dir / "research_predictions.csv",
        "manifest": output_dir / "experiment_manifest.json",
        "statistics": output_dir / "statistical_comparisons.csv",
        "intervals": output_dir / "interval_metrics.csv",
    }


def run_ablation_suite(config: ProjectConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir) / "research" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.research.run_ablations:
        raise ValueError("Ablations are disabled in the configuration")

    raw_frame = load_processed(config.data.processed_path)
    frame = add_time_features(raw_frame)
    target_values = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for lookback in config.research.ablation_lookbacks:
        folds = make_rolling_origin_folds(
            length=len(frame),
            lookback=lookback,
            horizon=config.data.horizon,
            validation_steps=config.research.cv_validation_steps,
            test_steps=config.research.cv_test_steps,
            folds=1,
            stride=config.research.cv_stride,
            holdout_steps=config.research.final_test_steps,
        )
        fold = folds[0]
        for feature_set in config.research.ablation_feature_sets:
            feature_columns = select_feature_columns(feature_set)
            feature_values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
            prepared = prepare_fold_arrays(
                feature_values,
                target_values,
                lookback,
                config.data.horizon,
                fold,
            )
            local_config = replace(
                config,
                data=replace(config.data, lookback=lookback),
            )
            for layers in config.research.ablation_transformer_layers:
                for seed in config.research.ablation_seeds:
                    result = _run_neural(
                        "Transformer",
                        prepared,
                        local_config,
                        seed,
                        num_features=len(feature_columns),
                        num_layers_override=layers,
                    )
                    metrics = regression_metrics(
                        result["test_actual"],
                        result["test_prediction"],
                        TARGET_COLUMNS,
                        training_actual=prepared.raw_targets[: fold.train_end],
                        seasonality=24,
                    )
                    for record in metrics.to_dict(orient="records"):
                        rows.append(
                            {
                                "lookback": lookback,
                                "feature_set": feature_set,
                                "transformer_layers": layers,
                                "seed": seed,
                                "parameter_count": result["parameter_count"],
                                "runtime_seconds": result["runtime_seconds"],
                                **record,
                            }
                        )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "ablation_metrics_raw.csv", index=False)
    summary = (
        raw.groupby(
            ["lookback", "feature_set", "transformer_layers", "target"],
            as_index=False,
        )[["mae", "rmse", "wape", "mase"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "ablation_metrics_summary.csv", index=False)
    return {
        "raw": output_dir / "ablation_metrics_raw.csv",
        "summary": output_dir / "ablation_metrics_summary.csv",
    }
