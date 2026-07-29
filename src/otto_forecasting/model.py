from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.size(1)]


class AttentionPooling(nn.Module):
    """Learn a normalized importance weight for every historical time step."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(values), dim=1)
        return torch.sum(values * weights, dim=1)


class DemandTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        target_dim: int,
        horizon: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.horizon = horizon
        self.target_dim = target_dim
        self.input_projection = nn.Linear(num_features, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.normalization = nn.LayerNorm(d_model)
        self.pooling = AttentionPooling(d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon * target_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values)
        encoded = self.position(encoded)
        encoded = self.encoder(encoded)
        pooled = self.pooling(self.normalization(encoded))
        forecast = self.output(pooled)
        return forecast.view(values.size(0), self.horizon, self.target_dim)


class GRUForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,
        target_dim: int,
        horizon: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, horizon * target_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(values)
        forecast = self.output(hidden[-1])
        return forecast.view(values.size(0), self.horizon, self.target_dim)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
