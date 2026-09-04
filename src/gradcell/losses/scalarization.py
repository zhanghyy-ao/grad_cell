from __future__ import annotations

import torch
from torch import nn


class SmoothTchebycheff(nn.Module):
    def __init__(
        self,
        energy_ideal: float = 260.0,
        energy_nadir: float = 100.0,
        high_rate_ideal: float = 1.0,
        high_rate_nadir: float = 0.0,
        retention_5c_min: float = 0.55,
        retention_6c_min: float = 0.45,
        constraint_weight: float = 2.0,
        temperature: float = 0.05,
        augmented_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.energy_ideal = energy_ideal
        self.energy_nadir = energy_nadir
        self.high_rate_ideal = high_rate_ideal
        self.high_rate_nadir = high_rate_nadir
        self.retention_5c_min = retention_5c_min
        self.retention_6c_min = retention_6c_min
        self.constraint_weight = constraint_weight
        self.temperature = temperature
        self.augmented_weight = augmented_weight

    def forward(
        self,
        energy: torch.Tensor,
        retention_5c: torch.Tensor,
        retention_6c: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        preference = preference.reshape(-1)
        high_rate = torch.minimum(retention_5c, retention_6c)
        distances = torch.stack(
            [
                (self.energy_ideal - energy) / (self.energy_ideal - self.energy_nadir),
                (self.high_rate_ideal - high_rate)
                / (self.high_rate_ideal - self.high_rate_nadir),
            ],
            dim=-1,
        )
        weights = torch.stack([preference, 1.0 - preference], dim=-1)
        weighted = weights * distances
        smooth_max = self.temperature * torch.logsumexp(weighted / self.temperature, dim=-1)
        constraint = (
            torch.relu(self.retention_5c_min - retention_5c).square()
            + torch.relu(self.retention_6c_min - retention_6c).square()
        )
        return (
            smooth_max
            + self.augmented_weight * weighted.sum(dim=-1)
            + self.constraint_weight * constraint
        )
