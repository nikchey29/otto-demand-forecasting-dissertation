import numpy as np

from otto_forecasting.metrics import (
    conformal_intervals,
    horizon_mae,
    interval_metrics,
    paired_block_bootstrap_mae,
    regression_metrics,
)


def test_metrics_are_zero_for_perfect_prediction():
    actual = np.ones((3, 24, 2))
    training = np.arange(240 * 2, dtype=float).reshape(240, 2) + 1
    metrics = regression_metrics(
        actual,
        actual.copy(),
        ("carts", "orders"),
        training_actual=training,
        seasonality=24,
    )
    assert metrics[["mae", "rmse", "wape", "smape", "mape"]].to_numpy().sum() == 0
    assert metrics[["mase", "rmsse"]].to_numpy().sum() == 0
    horizon = horizon_mae(actual, actual.copy(), ("carts", "orders"))
    assert horizon["mae"].sum() == 0


def test_conformal_intervals_and_coverage():
    validation_actual = np.ones((20, 3, 2)) * 10
    validation_prediction = validation_actual - 2
    test_prediction = np.ones((5, 3, 2)) * 9
    actual = np.ones((5, 3, 2)) * 10
    lower, upper, quantiles = conformal_intervals(
        validation_actual,
        validation_prediction,
        test_prediction,
        alpha=0.1,
    )
    assert np.all(quantiles == 2)
    metrics = interval_metrics(actual, lower, upper, ("carts", "orders"), 0.9)
    assert np.all(metrics["empirical_coverage"] == 1)


def test_paired_bootstrap_detects_better_model():
    actual = np.ones((40, 4, 2)) * 10
    better = actual.copy()
    worse = actual + 5
    result = paired_block_bootstrap_mae(
        actual,
        better,
        worse,
        ("carts", "orders"),
        "better",
        "worse",
        samples=200,
        block_size=4,
    )
    assert np.all(result["mae_difference_a_minus_b"] < 0)
    assert np.all(result["probability_a_better"] == 1)
