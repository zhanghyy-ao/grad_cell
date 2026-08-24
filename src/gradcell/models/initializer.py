from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class DesignInitializer(nn.Module):
    def __init__(
        self,
        task_dim: int = 128,
        latent_dim: int = 7,
        width: int = 256,
        blocks: int = 4,
        initial_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.initial_scale = initial_scale
        self.input = nn.Sequential(nn.Linear(task_dim, width), nn.SiLU(), nn.LayerNorm(width))
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.output = nn.Linear(width, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        delta = self.output(self.blocks(self.input(task_embedding)))
        return self.initial_scale * torch.tanh(delta)

