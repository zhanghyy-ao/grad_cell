from __future__ import annotations

import math

import torch
from torch import nn


class FourierPreferenceEncoder(nn.Module):
    def __init__(self, frequencies: int = 8, embedding_dim: int = 128) -> None:
        super().__init__()
        self.frequencies = frequencies
        input_dim = 1 + 2 * frequencies
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, preference: torch.Tensor) -> torch.Tensor:
        if preference.ndim == 1:
            preference = preference[:, None]
        features = [preference]
        for frequency in range(1, self.frequencies + 1):
            phase = 2.0 * math.pi * frequency * preference
            features.extend([torch.sin(phase), torch.cos(phase)])
        return self.network(torch.cat(features, dim=-1))

