from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


def _window_matrix(
    features: np.ndarray,
    target_starts: np.ndarray,
    lookback: int,
) -> np.ndarray:
    return np.stack(
        [features[start - lookback : start].reshape(-1) for start in target_starts]
    )


def _target_matrix(
    targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.stack(
        [targets[start : start + horizon].reshape(-1) for start in target_starts]
    )


class RidgeForecaster:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha)
        self.horizon = 0
        self.target_dim = 0
        self.lookback = 0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> "RidgeForecaster":
        x = _window_matrix(features, target_starts, lookback)
        y = _target_matrix(targets, target_starts, horizon)
        self.horizon = horizon
        self.target_dim = targets.shape[1]
        self.lookback = lookback
        self.model.fit(x, y)
        return self

    def predict(
        self,
        features: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        x = _window_matrix(features, target_starts, lookback)
        prediction = self.model.predict(x)
        return prediction.reshape(-1, self.horizon, self.target_dim)

    def predict_history(self, history: np.ndarray) -> np.ndarray:
        if history.shape[0] != self.lookback:
            raise ValueError(f"Expected {self.lookback} history rows")
        prediction = self.model.predict(history.reshape(1, -1))
        return prediction.reshape(self.horizon, self.target_dim)


class ExtraTreesForecaster:
    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 3,
        max_features: float = 0.7,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.horizon = 0
        self.target_dim = 0
        self.lookback = 0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> "ExtraTreesForecaster":
        x = _window_matrix(features, target_starts, lookback)
        y = _target_matrix(targets, target_starts, horizon)
        self.horizon = horizon
        self.target_dim = targets.shape[1]
        self.lookback = lookback
        self.model.fit(x, y)
        return self

    def predict(
        self,
        features: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        prediction = self.model.predict(_window_matrix(features, target_starts, lookback))
        return prediction.reshape(-1, self.horizon, self.target_dim)

    def predict_history(self, history: np.ndarray) -> np.ndarray:
        if history.shape[0] != self.lookback:
            raise ValueError(f"Expected {self.lookback} history rows")
        prediction = self.model.predict(history.reshape(1, -1))
        return prediction.reshape(self.horizon, self.target_dim)


def select_ridge_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    train_starts: np.ndarray,
    validation_starts: np.ndarray,
    lookback: int,
    horizon: int,
    candidates: tuple[float, ...],
) -> tuple[RidgeForecaster, float, float]:
    if not candidates:
        raise ValueError("At least one Ridge alpha is required")
    best_model: RidgeForecaster | None = None
    best_alpha = float(candidates[0])
    best_loss = float("inf")
    validation_actual = _target_matrix(targets, validation_starts, horizon).reshape(
        -1, horizon, targets.shape[1]
    )
    for alpha in candidates:
        candidate = RidgeForecaster(alpha=alpha).fit(
            features, targets, train_starts, lookback, horizon
        )
        prediction = candidate.predict(features, validation_starts, lookback)
        loss = float(np.mean(np.abs(prediction - validation_actual)))
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
            best_model = candidate
    assert best_model is not None
    return best_model, best_alpha, best_loss


def persistence_forecast(
    raw_targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.stack(
        [np.repeat(raw_targets[start - 1][None, :], horizon, axis=0) for start in target_starts]
    )


def seasonal_forecast(
    raw_targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
    seasonality: int,
) -> np.ndarray:
    if seasonality < horizon:
        raise ValueError("Seasonality must be at least as large as the forecast horizon")
    if int(target_starts.min()) < seasonality:
        raise ValueError("Not enough history for the requested seasonality")
    return np.stack(
        [
            raw_targets[start - seasonality : start - seasonality + horizon]
            for start in target_starts
        ]
    )


def blended_seasonal_forecast(
    raw_targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
    seasonalities: tuple[int, ...],
) -> np.ndarray:
    if len(seasonalities) < 2:
        raise ValueError("At least two seasonalities are needed for a blended forecast")
    forecasts = [
        seasonal_forecast(raw_targets, target_starts, horizon, seasonality)
        for seasonality in seasonalities
    ]
    return np.mean(np.stack(forecasts, axis=0), axis=0)
