from __future__ import annotations

import torch
from torch import nn


class Chen2020Surrogate(nn.Module):
    """MLP mapping seven feasible-design latents to standardized targets."""

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 4,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)
