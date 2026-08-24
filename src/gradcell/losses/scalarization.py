from __future__ import annotations

import torch
from torch import nn


class SmoothTchebycheff(nn.Module):
    def __init__(
        self,
        energy_ideal: float = 260.0,
        energy_nadir: float = 100.0,
        power_ideal: float = 900.0,
        power_nadir: float = 200.0,
        temperature: float = 0.05,
        augmented_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.energy_ideal = energy_ideal
        self.energy_nadir = energy_nadir
        self.power_ideal = power_ideal
        self.power_nadir = power_nadir
        self.temperature = temperature
        self.augmented_weight = augmented_weight

    def forward(
        self,
        energy: torch.Tensor,
        power: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        preference = preference.reshape(-1)
        distances = torch.stack(
            [
                (self.energy_ideal - energy) / (self.energy_ideal - self.energy_nadir),
                (self.power_ideal - power) / (self.power_ideal - self.power_nadir),
            ],
            dim=-1,
        )
        weights = torch.stack([preference, 1.0 - preference], dim=-1)
        weighted = weights * distances
        smooth_max = self.temperature * torch.logsumexp(weighted / self.temperature, dim=-1)
        return smooth_max + self.augmented_weight * weighted.sum(dim=-1)

