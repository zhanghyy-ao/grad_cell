from __future__ import annotations

import torch
from torch import nn


class ElectrolytePropertyNetwork(nn.Module):
    """Predict a bounded log multiplier for the DFN conductivity function.

    The output is a dimensionless ``log_conductivity_scale`` used as
    ``kappa = kappa_Chen2020 * exp(output)``.  It is intentionally not a
    standardized CALiSol conductivity prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        min_log_scale: float = -0.6931471805599453,
        max_log_scale: float = 0.6931471805599453,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or depth <= 0:
            raise ValueError("input_dim and depth must be positive")
        if min_log_scale >= max_log_scale:
            raise ValueError("min_log_scale must be smaller than max_log_scale")
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.register_buffer("min_log_scale", torch.tensor(float(min_log_scale)))
        self.register_buffer("max_log_scale", torch.tensor(float(max_log_scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return bounded ``log_conductivity_scale`` values."""
        raw = self.network(features).squeeze(-1)
        unit = 0.5 * (torch.tanh(raw) + 1.0)
        return self.min_log_scale + unit * (self.max_log_scale - self.min_log_scale)
