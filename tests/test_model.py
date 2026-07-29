import torch

from otto_forecasting.model import DemandTransformer


def test_model_output_shape():
    model = DemandTransformer(
        num_features=8,
        target_dim=2,
        horizon=24,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    output = model(torch.randn(4, 168, 8))
    assert output.shape == (4, 24, 2)
