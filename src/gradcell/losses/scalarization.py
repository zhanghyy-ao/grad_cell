from __future__ import annotations

import torch
from torch import nn


class SmoothTchebycheff(nn.Module):
    def __init__(
        self,
        energy_ideal: float = 260.0,
        energy_nadir: float = 100.0,
        retention_ideal: float = 1.0,
        retention_nadir: float = 0.0,
        temperature: float = 0.05,
        augmented_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.energy_ideal = energy_ideal
        self.energy_nadir = energy_nadir
        self.retention_ideal = retention_ideal
        self.retention_nadir = retention_nadir
        self.temperature = temperature
        self.augmented_weight = augmented_weight

    def forward(
        self,
        energy: torch.Tensor,
        retention: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        preference = preference.reshape(-1)
        distances = torch.stack(
            [
                (self.energy_ideal - energy) / (self.energy_ideal - self.energy_nadir),
                (self.retention_ideal - retention) / (self.retention_ideal - self.retention_nadir),
            ],
            dim=-1,
        )
        weights = torch.stack([preference, 1.0 - preference], dim=-1)
        weighted = weights * distances
        smooth_max = self.temperature * torch.logsumexp(weighted / self.temperature, dim=-1)
        return smooth_max + self.augmented_weight * weighted.sum(dim=-1)
