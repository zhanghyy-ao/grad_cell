from __future__ import annotations

import torch
from torch import nn


class ElectrolytePropertyNetwork(nn.Module):
    """Predict log ionic conductivity in mS/cm from formulation features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict an unbounded, standardized log-conductivity target."""
        return self.network(features).squeeze(-1)
