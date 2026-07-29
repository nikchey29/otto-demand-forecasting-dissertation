import numpy as np

from otto_forecasting.baselines import (
    blended_seasonal_forecast,
    persistence_forecast,
    seasonal_forecast,
)


def test_persistence_and_seasonal_forecasts_use_only_history():
    values = np.arange(300 * 2, dtype=np.float64).reshape(300, 2)
    starts = np.array([200, 201])
    persistence = persistence_forecast(values, starts, horizon=24)
    seasonal = seasonal_forecast(values, starts, horizon=24, seasonality=168)
    assert persistence.shape == (2, 24, 2)
    np.testing.assert_array_equal(persistence[0, 0], values[199])
    np.testing.assert_array_equal(seasonal[0], values[32:56])


def test_blended_seasonal_is_mean_of_components():
    values = np.arange(400 * 2, dtype=np.float64).reshape(400, 2)
    starts = np.array([250])
    daily = seasonal_forecast(values, starts, 24, 24)
    weekly = seasonal_forecast(values, starts, 24, 168)
    blended = blended_seasonal_forecast(values, starts, 24, (24, 168))
    np.testing.assert_allclose(blended, (daily + weekly) / 2)
