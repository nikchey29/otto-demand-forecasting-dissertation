import json

import pandas as pd

from otto_forecasting.data import (
    add_time_features,
    aggregate_jsonl,
    audit_frame,
    generate_synthetic_hourly,
    validate_hourly_frame,
)


def test_aggregate_jsonl_fills_missing_intervals(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": 1659301200000, "type": "clicks"},
                {"aid": 11, "ts": 1659308400000, "type": "orders"},
            ],
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    frame = aggregate_jsonl(path, "1h")
    assert len(frame) == 3
    assert frame.iloc[1][["clicks", "carts", "orders"]].sum() == 0


def test_add_time_features_returns_expected_columns():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "clicks": [10],
            "carts": [2],
            "orders": [1],
        }
    )
    result = add_time_features(frame)
    assert result.loc[0, "log_clicks"] > 0
    assert result.loc[0, "hour_sin"] == 0


def test_synthetic_data_and_audit_are_valid():
    frame = generate_synthetic_hourly(hours=320, seed=7)
    validate_hourly_frame(frame)
    report = audit_frame(frame)
    assert report["rows"] == 320
    assert report["event_totals"]["clicks"] > report["event_totals"]["carts"]
    assert report["event_totals"]["carts"] > report["event_totals"]["orders"]
