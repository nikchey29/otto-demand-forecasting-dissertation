from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    processed_path: str
    frequency: str
    lookback: int
    horizon: int
    validation_steps: int
    test_steps: int
    seasonalities: tuple[int, ...] = (24, 168)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float


@dataclass(frozen=True)
class GRUConfig:
    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.15


@dataclass(frozen=True)
class BaselineConfig:
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    extra_trees_estimators: int = 300
    extra_trees_min_samples_leaf: int = 3
    extra_trees_max_features: float = 0.7


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    gradient_clip: float
    num_workers: int


@dataclass(frozen=True)
class ResearchConfig:
    seeds: tuple[int, ...] = (42, 123, 2026, 3407, 9999)
    cv_folds: int = 3
    cv_stride: int = 24
    cv_validation_steps: int = 48
    cv_test_steps: int = 48
    final_validation_steps: int = 96
    final_test_steps: int = 96
    conformal_alpha: float = 0.10
    bootstrap_samples: int = 2000
    bootstrap_block_size: int = 24
    run_gru: bool = True
    run_extra_trees: bool = True
    run_ablations: bool = True
    ablation_seeds: tuple[int, ...] = (42, 123, 2026)
    ablation_lookbacks: tuple[int, ...] = (24, 72, 168)
    ablation_feature_sets: tuple[str, ...] = ("history_only", "full")
    ablation_transformer_layers: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    gru: GRUConfig = field(default_factory=GRUConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_tuple(values: Any, cast: type) -> tuple:
    if values is None:
        return tuple()
    if isinstance(values, (str, bytes)):
        raise TypeError("Expected a sequence, not a string")
    return tuple(cast(value) for value in values)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data_raw = dict(raw["data"])
    legacy_seasonality = data_raw.pop("seasonality", None)
    seasonalities = data_raw.pop("seasonalities", None)
    if seasonalities is None:
        seasonalities = [legacy_seasonality] if legacy_seasonality is not None else [24, 168]
    data = DataConfig(
        **data_raw,
        seasonalities=_as_tuple(seasonalities, int),
    )

    baseline_raw = dict(raw.get("baselines", {}))
    if "ridge_alphas" in baseline_raw:
        baseline_raw["ridge_alphas"] = _as_tuple(baseline_raw["ridge_alphas"], float)
    baselines = BaselineConfig(**baseline_raw)

    research_raw = dict(raw.get("research", {}))
    tuple_fields = {
        "seeds": int,
        "ablation_seeds": int,
        "ablation_lookbacks": int,
        "ablation_feature_sets": str,
        "ablation_transformer_layers": int,
    }
    for field_name, cast in tuple_fields.items():
        if field_name in research_raw:
            research_raw[field_name] = _as_tuple(research_raw[field_name], cast)
    research = ResearchConfig(**research_raw)

    return ProjectConfig(
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        data=data,
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        gru=GRUConfig(**raw.get("gru", {})),
        baselines=baselines,
        research=research,
    )
