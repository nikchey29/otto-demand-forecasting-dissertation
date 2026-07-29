from pathlib import Path

from otto_forecasting.config import load_config


def test_load_config_accepts_legacy_seasonality(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
output_dir: artifacts
data:
  raw_path: raw.jsonl
  processed_path: processed.csv
  frequency: 1h
  lookback: 24
  horizon: 6
  validation_steps: 12
  test_steps: 12
  seasonality: 24
model:
  d_model: 16
  nhead: 4
  num_layers: 1
  dim_feedforward: 32
  dropout: 0.1
training:
  batch_size: 8
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  patience: 1
  gradient_clip: 1.0
  num_workers: 0
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.data.seasonalities == (24,)
    assert config.research.seeds[0] == 42
