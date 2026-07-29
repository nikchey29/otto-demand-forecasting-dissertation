from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from otto_forecasting.data import TARGET_COLUMNS, add_time_features
from otto_forecasting.dataset import inverse_targets
from otto_forecasting.model import DemandTransformer, GRUForecaster


class Observation(BaseModel):
    timestamp: str
    clicks: float = Field(ge=0)
    carts: float = Field(ge=0)
    orders: float = Field(ge=0)


class ForecastRequest(BaseModel):
    history: list[Observation]


class ForecastPoint(BaseModel):
    forecast_hour: int
    carts: float
    orders: float


class ForecastResponse(BaseModel):
    model: str
    forecasts: list[ForecastPoint]


class ModelService:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.root = Path(artifact_dir)
        manifest_path = self.root / "experiment_manifest.json"
        metadata_path = self.root / "metadata.json"
        if manifest_path.exists():
            self.metadata: dict[str, Any] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.model_name = str(self.metadata["selected_model_by_cv"])
            artifacts = self.metadata.get("selected_artifacts") or []
            if not artifacts and self.metadata.get("selected_artifact"):
                artifacts = [self.metadata["selected_artifact"]]
            self.artifact_names = [str(value) for value in artifacts]
        elif metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_name = "Transformer"
            self.artifact_names = ["transformer.pt"]
        else:
            raise FileNotFoundError(f"Missing model metadata in {self.root}")

        self.feature_columns = tuple(self.metadata["feature_columns"])
        self.target_columns = tuple(self.metadata.get("target_columns", TARGET_COLUMNS))
        self.lookback = int(self.metadata["lookback"])
        self.horizon = int(self.metadata["horizon"])
        self.feature_scaler = joblib.load(self.root / "feature_scaler.joblib")
        self.target_scaler = joblib.load(self.root / "target_scaler.joblib")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = self._load_model()

    def _load_model(self) -> Any:
        if self.model_name in {"Transformer", "GRU"}:
            if not self.artifact_names:
                raise FileNotFoundError("No neural model artifacts are listed")
            models: list[torch.nn.Module] = []
            for artifact_name in self.artifact_names:
                if self.model_name == "Transformer":
                    model_config = self.metadata.get("model") or self.metadata["config"][
                        "model"
                    ]
                    model: torch.nn.Module = DemandTransformer(
                        num_features=len(self.feature_columns),
                        target_dim=len(self.target_columns),
                        horizon=self.horizon,
                        **model_config,
                    ).to(self.device)
                else:
                    gru_config = self.metadata["config"]["gru"]
                    model = GRUForecaster(
                        num_features=len(self.feature_columns),
                        target_dim=len(self.target_columns),
                        horizon=self.horizon,
                        **gru_config,
                    ).to(self.device)
                state = torch.load(
                    self.root / artifact_name,
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()
                models.append(model)
            return models
        if self.model_name in {"Ridge", "Extra Trees"}:
            if not self.artifact_names:
                raise FileNotFoundError("No fitted model artifact is listed")
            return joblib.load(self.root / self.artifact_names[0])
        if self.model_name.startswith("Seasonal") or self.model_name == "Persistence":
            return None
        raise ValueError(f"Unsupported selected model: {self.model_name}")

    def _validate_history(self, history: list[Observation]) -> pd.DataFrame:
        if len(history) != self.lookback:
            raise ValueError(f"Expected exactly {self.lookback} observations")
        frame = pd.DataFrame([item.model_dump() for item in history])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("History timestamps must be unique and chronological")
        expected_frequency = timestamps.diff().dropna()
        if not expected_frequency.eq(pd.Timedelta(hours=1)).all():
            raise ValueError("History must contain consecutive hourly observations")
        return add_time_features(frame)

    def _baseline_forecast(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, self.target_columns].to_numpy(dtype=np.float64)
        if self.model_name == "Persistence":
            return np.repeat(raw[-1][None, :], self.horizon, axis=0)
        seasonalities = [
            int(token[:-1])
            for token in self.model_name.split()
            if token.endswith("h") and token[:-1].isdigit()
        ]
        if not seasonalities and "blend" in self.model_name.lower():
            suffix = self.model_name.split("blend", maxsplit=1)[1].strip()
            seasonalities = [int(value.rstrip("h")) for value in suffix.split("+")]
        forecasts = []
        for seasonality in seasonalities:
            if len(raw) < seasonality:
                raise ValueError(f"History is shorter than {seasonality} hours")
            forecasts.append(raw[-seasonality : -seasonality + self.horizon])
        if not forecasts:
            raise ValueError(f"Cannot interpret baseline model {self.model_name}")
        return np.mean(np.stack(forecasts, axis=0), axis=0)

    def forecast(self, history: list[Observation]) -> np.ndarray:
        frame = self._validate_history(history)
        if self.model is None:
            return self._baseline_forecast(frame)

        values = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        scaled = self.feature_scaler.transform(values).astype(np.float32)
        if self.model_name in {"Ridge", "Extra Trees"}:
            scaled_prediction = self.model.predict_history(scaled)
            return inverse_targets(scaled_prediction[None, ...], self.target_scaler)[0]

        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            predictions = [model(tensor).cpu().numpy() for model in self.model]
        prediction = np.mean(np.stack(predictions, axis=0), axis=0)
        return inverse_targets(prediction, self.target_scaler)[0]


_service: ModelService | None = None
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _service, _startup_error
    try:
        _service = ModelService(os.getenv("OTTO_ARTIFACT_DIR", "artifacts/research"))
        _startup_error = None
    except (FileNotFoundError, ValueError, KeyError) as exc:
        _service = None
        _startup_error = str(exc)
    yield
    _service = None


app = FastAPI(
    title="OTTO Aggregate Demand Forecasting API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    if _service is None:
        return {"status": "degraded", "detail": _startup_error or "Model is not loaded"}
    return {"status": "ok", "model": _service.model_name}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if _service is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Model is not loaded")
    return {
        "model": _service.model_name,
        "lookback": _service.lookback,
        "horizon": _service.horizon,
        "features": list(_service.feature_columns),
        "targets": list(_service.target_columns),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Model is not loaded")
    try:
        prediction = _service.forecast(request.history)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    points = [
        ForecastPoint(
            forecast_hour=index + 1,
            carts=float(values[0]),
            orders=float(values[1]),
        )
        for index, values in enumerate(prediction)
    ]
    return ForecastResponse(model=_service.model_name, forecasts=points)
