from pathlib import Path

from otto_forecasting.config import load_config
from otto_forecasting.data import generate_synthetic_hourly, save_processed
from otto_forecasting.research import run_research_suite


def test_research_suite_runs_end_to_end_on_synthetic_data(tmp_path: Path):
    data_path = tmp_path / "synthetic.csv"
    save_processed(generate_synthetic_hourly(hours=360, seed=9), data_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
seed: 42
output_dir: {tmp_path.as_posix()}/artifacts
data:
  raw_path: not-used.jsonl
  processed_path: {data_path.as_posix()}
  frequency: 1h
  lookback: 24
  horizon: 6
  validation_steps: 24
  test_steps: 24
  seasonalities: [24]
model:
  d_model: 8
  nhead: 2
  num_layers: 1
  dim_feedforward: 16
  dropout: 0.1
gru:
  hidden_size: 8
  num_layers: 1
  dropout: 0.1
baselines:
  ridge_alphas: [0.1, 1.0]
  extra_trees_estimators: 10
  extra_trees_min_samples_leaf: 2
  extra_trees_max_features: 0.8
training:
  batch_size: 32
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  patience: 1
  gradient_clip: 1.0
  num_workers: 0
research:
  seeds: [42]
  cv_folds: 1
  cv_stride: 6
  cv_validation_steps: 12
  cv_test_steps: 12
  final_validation_steps: 24
  final_test_steps: 24
  conformal_alpha: 0.1
  bootstrap_samples: 100
  bootstrap_block_size: 3
  run_gru: false
  run_extra_trees: false
  run_ablations: false
  ablation_seeds: [42]
  ablation_lookbacks: [24]
  ablation_feature_sets: [full]
  ablation_transformer_layers: [1]
""",
        encoding="utf-8",
    )
    outputs = run_research_suite(load_config(config_path))
    assert outputs["metrics"].exists()
    assert outputs["manifest"].exists()
    assert outputs["statistics"].exists()
