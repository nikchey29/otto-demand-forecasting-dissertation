from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_TYPES = ("clicks", "carts", "orders")
HISTORY_FEATURE_COLUMNS = ("log_clicks", "log_carts", "log_orders")
TEMPORAL_FEATURE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
)
FEATURE_COLUMNS = HISTORY_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS
TARGET_COLUMNS = ("carts", "orders")


def parse_fixed_frequency(frequency: str) -> pd.Timedelta:
    """Return a Timedelta for a fixed pandas offset such as ``1h``."""
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
        delta = pd.Timedelta(offset.nanos, unit="ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frequency must be a fixed duration such as '1h' or '30min'"
        ) from exc

    if delta <= pd.Timedelta(0):
        raise ValueError("Frequency must be positive")
    return delta


def aggregate_jsonl(input_path: str | Path, frequency: str = "1h") -> pd.DataFrame:
    """Read OTTO sessions one line at a time and count events in fixed UTC buckets."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    interval_ms = int(parse_fixed_frequency(frequency).total_seconds() * 1000)
    counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {event_type: 0 for event_type in EVENT_TYPES}
    )

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                session = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            for event in session.get("events", []):
                event_type = event.get("type")
                timestamp = event.get("ts")
                if event_type not in EVENT_TYPES or timestamp is None:
                    continue

                bucket = int(timestamp) // interval_ms * interval_ms
                counts[bucket][event_type] += 1

    if not counts:
        raise ValueError("No supported OTTO events were found")

    rows = [
        {"timestamp": timestamp, **event_counts}
        for timestamp, event_counts in counts.items()
    ]
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)

    full_range = pd.date_range(
        start=frame["timestamp"].min(),
        end=frame["timestamp"].max(),
        freq=frequency,
        tz="UTC",
    )
    frame = (
        frame.set_index("timestamp")
        .reindex(full_range, fill_value=0)
        .rename_axis("timestamp")
        .reset_index()
    )

    validate_hourly_frame(frame, frequency)
    return frame


def validate_hourly_frame(frame: pd.DataFrame, frequency: str = "1h") -> None:
    required = {"timestamp", *EVENT_TYPES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The processed dataset is empty")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps were found")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be in chronological order")

    if len(timestamps) > 1:
        expected = parse_fixed_frequency(frequency)
        if not timestamps.diff().dropna().eq(expected).all():
            raise ValueError(f"Timestamps must be consecutive at frequency {frequency}")

    event_values = frame.loc[:, EVENT_TYPES].to_numpy(dtype=np.float64)
    if not np.isfinite(event_values).all():
        raise ValueError("Event columns contain non-finite values")
    if (event_values < 0).any():
        raise ValueError("Event counts cannot be negative")


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    validate_hourly_frame(frame)
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    hour = result["timestamp"].dt.hour.to_numpy()
    day = result["timestamp"].dt.dayofweek.to_numpy()

    result["log_clicks"] = np.log1p(result["clicks"].to_numpy(dtype=np.float64))
    result["log_carts"] = np.log1p(result["carts"].to_numpy(dtype=np.float64))
    result["log_orders"] = np.log1p(result["orders"].to_numpy(dtype=np.float64))
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)
    result["is_weekend"] = (day >= 5).astype(np.float32)
    return result


def select_feature_columns(feature_set: str) -> tuple[str, ...]:
    normalized = feature_set.strip().lower()
    if normalized == "full":
        return FEATURE_COLUMNS
    if normalized == "history_only":
        return HISTORY_FEATURE_COLUMNS
    if normalized == "temporal_only":
        return TEMPORAL_FEATURE_COLUMNS
    raise ValueError(f"Unknown feature set: {feature_set}")


def save_processed(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination


def load_processed(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Processed dataset not found: {source}")

    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    validate_hourly_frame(frame)
    return frame


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_frame(frame: pd.DataFrame, source_path: str | Path | None = None) -> dict[str, Any]:
    validate_hourly_frame(frame)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    event_values = frame.loc[:, EVENT_TYPES]

    audit: dict[str, Any] = {
        "rows": int(len(frame)),
        "start_timestamp": timestamps.iloc[0].isoformat(),
        "end_timestamp": timestamps.iloc[-1].isoformat(),
        "duration_hours": int(len(frame)),
        "frequency": "1h",
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "zero_activity_hours": int((event_values.sum(axis=1) == 0).sum()),
        "event_totals": {column: int(event_values[column].sum()) for column in EVENT_TYPES},
        "event_means": {column: float(event_values[column].mean()) for column in EVENT_TYPES},
        "event_maxima": {column: int(event_values[column].max()) for column in EVENT_TYPES},
    }

    if source_path is not None and Path(source_path).exists():
        audit["source_path"] = str(source_path)
        audit["sha256"] = sha256_file(source_path)
        audit["size_bytes"] = int(Path(source_path).stat().st_size)

    return audit


def generate_synthetic_hourly(
    hours: int = 960,
    seed: int = 42,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Generate a small seasonal series for testing the pipeline."""
    if hours < 300:
        raise ValueError("Synthetic series must contain at least 300 hours")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=hours, freq="1h", tz="UTC")
    index = np.arange(hours, dtype=np.float64)
    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofweek.to_numpy()

    daily = 1.0 + 0.35 * np.sin(2 * np.pi * (hour - 8) / 24)
    weekly = np.where(day >= 5, 0.82, 1.0)
    trend = 1.0 + 0.0007 * index
    campaign = 1.0 + 0.15 * ((index > hours * 0.65) & (index < hours * 0.75))

    clicks_mean = np.maximum(25000 * daily * weekly * trend * campaign, 100.0)
    clicks = rng.poisson(clicks_mean)
    cart_rate = np.clip(0.075 + 0.008 * np.sin(2 * np.pi * hour / 24), 0.03, 0.15)
    carts = rng.binomial(clicks, cart_rate)
    order_rate = np.clip(0.31 + 0.025 * np.cos(2 * np.pi * hour / 24), 0.15, 0.50)
    orders = rng.binomial(carts, order_rate)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "clicks": clicks.astype(np.int64),
            "carts": carts.astype(np.int64),
            "orders": orders.astype(np.int64),
        }
    )
