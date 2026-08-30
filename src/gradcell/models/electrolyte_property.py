from __future__ import annotations

import math

import torch
from torch import nn


class ElectrolytePropertyNetwork(nn.Module):
    """Predict log ionic conductivity in mS/cm from formulation features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        minimum_conductivity_ms_cm: float = 0.05,
        maximum_conductivity_ms_cm: float = 50.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or depth <= 0:
            raise ValueError("input_dim and depth must be positive")
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.log_min = math.log(minimum_conductivity_ms_cm)
        self.log_max = math.log(maximum_conductivity_ms_cm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features).squeeze(-1)
        return self.log_min + (self.log_max - self.log_min) * torch.sigmoid(raw)
