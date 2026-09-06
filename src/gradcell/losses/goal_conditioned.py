from __future__ import annotations

import torch
from torch import nn


class GoalConditionedObjective(nn.Module):
    """Per-task target violations plus preference-controlled feasible improvement."""

    def __init__(
        self,
        energy_scale: float = 160.0,
        constraint_weight: float = 5.0,
        improvement_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.energy_scale = energy_scale
        self.constraint_weight = constraint_weight
        self.improvement_weight = improvement_weight

    def forward(
        self,
        energy: torch.Tensor,
        retention_5c: torch.Tensor,
        retention_6c: torch.Tensor,
        preference: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.shape[-1] != 3:
            raise ValueError("targets must contain [energy, min_r5, min_r6]")
        target_energy, min_r5, min_r6 = targets.unbind(dim=-1)
        energy_violation = torch.relu(target_energy - energy) / self.energy_scale
        r5_violation = torch.relu(min_r5 - retention_5c)
        r6_violation = torch.relu(min_r6 - retention_6c)
        constraint_loss = energy_violation + self.constraint_weight * (
            r5_violation + r6_violation
        )
        preference = preference.reshape(-1)
        high_rate_margin = torch.minimum(retention_5c - min_r5, retention_6c - min_r6)
        improvement = -(
            preference * energy / self.energy_scale
            + (1.0 - preference) * high_rate_margin
        )
        return constraint_loss + self.improvement_weight * improvement
