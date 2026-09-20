"""Differentiable components for language-conditioned battery inverse design."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_PERFORMANCE_FIELDS = (
    "capacity_1c_ah",
    "energy_1c_wh",
    "capacity_5c_ah",
    "energy_5c_wh",
    "energy_retention_5c",
    "capacity_6c_ah",
    "energy_6c_wh",
    "energy_retention_6c",
)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class DFNPerformanceSurrogate(nn.Module):
    """Map normalized log design multipliers to normalized log DFN metrics."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            *(ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, normalized_log_design: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_log_design)


class SingleDesignPhysicsMLP(nn.Module):
    """Predict one bounded design from a frozen language embedding."""

    def __init__(
        self,
        input_dim: int,
        design_lower: Sequence[float] | torch.Tensor,
        design_upper: Sequence[float] | torch.Tensor,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lower = torch.as_tensor(design_lower, dtype=torch.float32)
        upper = torch.as_tensor(design_upper, dtype=torch.float32)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("design bounds must be one-dimensional and have equal shape")
        if not torch.all(upper > lower):
            raise ValueError("every upper design bound must exceed its lower bound")
        self.register_buffer("design_center", 0.5 * (lower + upper))
        self.register_buffer("design_half_range", 0.5 * (upper - lower))
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            *(ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)),
            nn.LayerNorm(hidden_dim),
        )
        self.design_head = nn.Linear(hidden_dim, len(lower))

    @property
    def design_dim(self) -> int:
        return int(self.design_center.numel())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        raw = self.design_head(self.trunk(embeddings))
        return self.design_center + self.design_half_range * torch.tanh(raw)


def freeze_surrogate(model: DFNPerformanceSurrogate) -> DFNPerformanceSurrogate:
    """Freeze surrogate weights while preserving gradients with respect to its input."""

    model.eval()
    model.requires_grad_(False)
    return model
