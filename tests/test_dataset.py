import numpy as np

from otto_forecasting.dataset import (
    ForecastDataset,
    make_rolling_origin_folds,
    make_target_starts,
    prepare_arrays,
    prepare_fold_arrays,
)


def test_make_target_starts_are_separated_by_target_period():
    train, validation, test = make_target_starts(672, 168, 24, 96, 96)
    assert train.max() + 24 <= validation.min()
    assert validation.max() + 24 <= test.min()
    assert test.max() == 648


def test_rolling_origin_folds_leave_final_holdout_untouched():
    folds = make_rolling_origin_folds(
        length=672,
        lookback=168,
        horizon=24,
        validation_steps=48,
        test_steps=48,
        folds=3,
        stride=24,
        holdout_steps=96,
    )
    assert len(folds) == 3
    assert folds[-1].test_end == 576
    assert all(folds[index].test_end < folds[index + 1].test_end for index in range(2))
    assert all(fold.train_end == fold.validation_start for fold in folds)


def test_forecast_dataset_shapes():
    features = np.random.default_rng(1).normal(size=(300, 8)).astype(np.float32)
    targets = np.random.default_rng(2).normal(size=(300, 2)).astype(np.float32)
    dataset = ForecastDataset(features, targets, np.array([168]), 168, 24)
    x, y = dataset[0]
    assert x.shape == (168, 8)
    assert y.shape == (24, 2)


def test_scalers_fit_before_validation_period():
    features = np.ones((800, 8), dtype=np.float64)
    targets = np.ones((800, 2), dtype=np.float64)
    prepared = prepare_arrays(features, targets, 168, 24, 96, 96)
    fold = prepared.fold

    contaminated_features = features.copy()
    contaminated_targets = targets.copy()
    contaminated_features[fold.validation_start :] = 1_000_000
    contaminated_targets[fold.validation_start :] = 1_000_000
    contaminated = prepare_fold_arrays(
        contaminated_features,
        contaminated_targets,
        168,
        24,
        fold,
    )

    np.testing.assert_allclose(
        contaminated.feature_scaler.mean_, prepared.feature_scaler.mean_
    )
    np.testing.assert_allclose(
        contaminated.target_scaler.mean_, prepared.target_scaler.mean_
    )
