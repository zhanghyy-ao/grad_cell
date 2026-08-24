from __future__ import annotations

import torch
from torch import nn


class DiagonalPhysicsRefiner(nn.Module):
    def __init__(
        self,
        task_dim: int = 128,
        latent_dim: int = 7,
        feature_dim: int = 5,
        hidden_dim: int = 128,
        max_step_size: float = 0.20,
        diagonal_min: float = 0.25,
        diagonal_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.max_step_size = max_step_size
        self.diagonal_min = diagonal_min
        self.diagonal_max = diagonal_max
        input_dim = task_dim + latent_dim + feature_dim + latent_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.step_head = nn.Linear(hidden_dim, 1)
        self.diagonal_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        latent: torch.Tensor,
        physics_features: torch.Tensor,
        gradient: torch.Tensor,
        state: torch.Tensor | None,
    ):
        normalized_gradient = gradient / gradient.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        refiner_input = torch.cat(
            [task_embedding, latent, physics_features, normalized_gradient], dim=-1
        )
        if state is None:
            state = torch.zeros(
                latent.shape[0], self.gru.hidden_size, dtype=latent.dtype, device=latent.device
            )
        state = self.gru(refiner_input, state)
        step = self.max_step_size * torch.sigmoid(self.step_head(state))
        diagonal = self.diagonal_min + (self.diagonal_max - self.diagonal_min) * torch.sigmoid(
            self.diagonal_head(state)
        )
        return step, diagonal, state

