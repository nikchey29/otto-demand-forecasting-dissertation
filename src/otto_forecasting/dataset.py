from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class ForecastDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.target_starts = target_starts.astype(np.int64, copy=False)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_start = int(self.target_starts[index])
        x = self.features[target_start - self.lookback : target_start]
        y = self.targets[target_start : target_start + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_end: int
    validation_start: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return {
            "fold_id": self.fold_id,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


@dataclass(frozen=True)
class PreparedData:
    features: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    train_starts: np.ndarray
    validation_starts: np.ndarray
    test_starts: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    fold: FoldSpec


def _target_starts_for_fold(
    length: int,
    lookback: int,
    horizon: int,
    fold: FoldSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries_are_valid = (
        lookback < fold.train_end <= fold.validation_start < fold.test_start < fold.test_end
    )
    if not boundaries_are_valid:
        raise ValueError("Fold boundaries are not strictly chronological")
    if fold.test_end > length:
        raise ValueError("Fold extends beyond the available observations")
    if fold.test_start - fold.validation_start < horizon:
        raise ValueError("Validation period must contain at least one full forecast horizon")
    if fold.test_end - fold.test_start < horizon:
        raise ValueError("Test period must contain at least one full forecast horizon")

    all_starts = np.arange(lookback, length - horizon + 1, dtype=np.int64)
    train = all_starts[all_starts + horizon <= fold.train_end]
    validation = all_starts[
        (all_starts >= fold.validation_start)
        & (all_starts + horizon <= fold.test_start)
    ]
    test = all_starts[
        (all_starts >= fold.test_start)
        & (all_starts + horizon <= fold.test_end)
    ]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("One or more dataset splits are empty")
    return train, validation, test


def make_target_starts(
    length: int,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold = make_final_fold(
        length=length,
        lookback=lookback,
        horizon=horizon,
        validation_steps=validation_steps,
        test_steps=test_steps,
    )
    return _target_starts_for_fold(length, lookback, horizon, fold)


def make_final_fold(
    length: int,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
) -> FoldSpec:
    if validation_steps < horizon or test_steps < horizon:
        raise ValueError("Validation and test periods must each be at least one horizon")
    test_start = length - test_steps
    validation_start = test_start - validation_steps
    if validation_start <= lookback:
        raise ValueError("Not enough observations for the requested lookback and splits")
    return FoldSpec(
        fold_id=0,
        train_end=validation_start,
        validation_start=validation_start,
        test_start=test_start,
        test_end=length,
    )


def make_rolling_origin_folds(
    length: int,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
    folds: int,
    stride: int,
    holdout_steps: int = 0,
) -> list[FoldSpec]:
    """Build expanding-window folds that finish before an optional untouched holdout."""
    if folds < 1:
        raise ValueError("At least one fold is required")
    if stride < 1:
        raise ValueError("Stride must be positive")
    usable_end = length - holdout_steps
    results: list[FoldSpec] = []
    for fold_index in range(folds):
        distance_from_latest = (folds - 1 - fold_index) * stride
        test_end = usable_end - distance_from_latest
        test_start = test_end - test_steps
        validation_start = test_start - validation_steps
        if validation_start <= lookback:
            raise ValueError(
                "Not enough history for the requested number of rolling-origin folds"
            )
        fold = FoldSpec(
            fold_id=fold_index + 1,
            train_end=validation_start,
            validation_start=validation_start,
            test_start=test_start,
            test_end=test_end,
        )
        _target_starts_for_fold(length, lookback, horizon, fold)
        results.append(fold)
    return results


def prepare_arrays(
    feature_values: np.ndarray,
    raw_target_values: np.ndarray,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
) -> PreparedData:
    fold = make_final_fold(
        length=len(feature_values),
        lookback=lookback,
        horizon=horizon,
        validation_steps=validation_steps,
        test_steps=test_steps,
    )
    return prepare_fold_arrays(
        feature_values,
        raw_target_values,
        lookback,
        horizon,
        fold,
    )


def prepare_fold_arrays(
    feature_values: np.ndarray,
    raw_target_values: np.ndarray,
    lookback: int,
    horizon: int,
    fold: FoldSpec,
) -> PreparedData:
    if len(feature_values) != len(raw_target_values):
        raise ValueError("Feature and target arrays must contain the same number of rows")
    if not np.isfinite(feature_values).all() or not np.isfinite(raw_target_values).all():
        raise ValueError("Feature and target arrays must be finite")
    if (raw_target_values < 0).any():
        raise ValueError("Target counts cannot be negative")

    train_starts, validation_starts, test_starts = _target_starts_for_fold(
        length=len(feature_values),
        lookback=lookback,
        horizon=horizon,
        fold=fold,
    )
    feature_scaler = StandardScaler().fit(feature_values[: fold.train_end])
    log_targets = np.log1p(raw_target_values.astype(np.float64))
    target_scaler = StandardScaler().fit(log_targets[: fold.train_end])
    features = feature_scaler.transform(feature_values)
    targets = target_scaler.transform(log_targets)
    return PreparedData(
        features=features,
        targets=targets,
        raw_targets=raw_target_values.astype(np.float64),
        train_starts=train_starts,
        validation_starts=validation_starts,
        test_starts=test_starts,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        fold=fold,
    )


def build_loaders(
    prepared: PreparedData,
    lookback: int,
    horizon: int,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        lookback,
        horizon,
    )
    validation_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.validation_starts,
        lookback,
        horizon,
    )
    test_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.test_starts,
        lookback,
        horizon,
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    return train_loader, validation_loader, test_loader


def inverse_targets(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = values.shape
    restored = scaler.inverse_transform(values.reshape(-1, shape[-1]))
    return np.maximum(np.expm1(restored), 0.0).reshape(shape)
