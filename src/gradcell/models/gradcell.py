from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gradcell.design.feasible_decoder import CellDesign, DesignSpace
from gradcell.losses.scalarization import SmoothTchebycheff
from gradcell.physics.soft_metrics import discharge_metrics

from .initializer import DesignInitializer
from .refiner import DiagonalPhysicsRefiner
from .task_encoder import FourierPreferenceEncoder


@dataclass
class GradCellStep:
    latent: torch.Tensor
    design: CellDesign
    loss: torch.Tensor
    energy: torch.Tensor
    retention_5c: torch.Tensor
    retention_6c: torch.Tensor
    status: torch.Tensor


@dataclass
class GradCellOutput:
    steps: list[GradCellStep]

    @property
    def final(self) -> GradCellStep:
        return self.steps[-1]


class GradCell(nn.Module):
    def __init__(
        self,
        physics_1c: nn.Module,
        physics_5c: nn.Module,
        physics_6c: nn.Module,
        design_space: DesignSpace | None = None,
        objective: nn.Module | None = None,
        task_dim: int = 128,
        max_refinement_update_norm: float = 0.25,
    ) -> None:
        super().__init__()
        self.design_space = design_space or DesignSpace()
        self.task_encoder = FourierPreferenceEncoder(embedding_dim=task_dim)
        self.initializer = DesignInitializer(
            task_dim=task_dim, latent_dim=self.design_space.latent_dim
        )
        self.refiner = DiagonalPhysicsRefiner(
            task_dim=task_dim, latent_dim=self.design_space.latent_dim
        )
        self.physics_1c = physics_1c
        self.physics_5c = physics_5c
        self.physics_6c = physics_6c
        self.objective = objective or SmoothTchebycheff()
        if max_refinement_update_norm <= 0.0:
            raise ValueError("max_refinement_update_norm must be positive")
        self.max_refinement_update_norm = max_refinement_update_norm

    def evaluate(self, latent: torch.Tensor, preference: torch.Tensor) -> GradCellStep:
        design = self.design_space(latent)
        y1, status1, _ = self.physics_1c(design.physics_tensor(1.0))
        y5, status5, _ = self.physics_5c(design.physics_tensor(5.0))
        y6, status6, _ = self.physics_6c(design.physics_tensor(6.0))
        finite1 = torch.isfinite(y1).flatten(start_dim=1).all(dim=1)
        finite5 = torch.isfinite(y5).flatten(start_dim=1).all(dim=1)
        finite6 = torch.isfinite(y6).flatten(start_dim=1).all(dim=1)
        valid = status1.bool() & status5.bool() & status6.bool() & finite1 & finite5 & finite6
        valid_indices = valid.nonzero(as_tuple=True)[0]

        batch_size = latent.shape[0]
        energy = latent.new_zeros(batch_size)
        retention_5c = latent.new_zeros(batch_size)
        retention_6c = latent.new_zeros(batch_size)
        loss = 100.0 + 0.1 * latent.square().mean(dim=-1)

        if valid_indices.numel() > 0:
            metrics1 = discharge_metrics(
                y1[valid],
                design.nominal_capacity_ah[valid],
                design.stack_mass_kg[valid],
                3600.0,
            )
            metrics5 = discharge_metrics(
                y5[valid], 5.0 * design.nominal_capacity_ah[valid],
                design.stack_mass_kg[valid],
                720.0,
            )
            metrics6 = discharge_metrics(
                y6[valid], 6.0 * design.nominal_capacity_ah[valid],
                design.stack_mass_kg[valid], 600.0,
            )
            valid_energy = metrics1.specific_energy_wh_kg
            energy_1c = metrics1.specific_energy_wh_kg.clamp_min(1e-8)
            valid_retention_5c = metrics5.specific_energy_wh_kg / energy_1c
            valid_retention_6c = metrics6.specific_energy_wh_kg / energy_1c
            valid_loss = self.objective(
                valid_energy, valid_retention_5c, valid_retention_6c, preference[valid]
            )
            energy = energy.index_copy(0, valid_indices, valid_energy)
            retention_5c = retention_5c.index_copy(0, valid_indices, valid_retention_5c)
            retention_6c = retention_6c.index_copy(0, valid_indices, valid_retention_6c)
            loss = loss.index_copy(0, valid_indices, valid_loss)

        status = valid.to(torch.int64)
        return GradCellStep(
            latent=latent,
            design=design,
            loss=loss,
            energy=energy,
            retention_5c=retention_5c,
            retention_6c=retention_6c,
            status=status,
        )

    def forward(self, preference: torch.Tensor, num_steps: int = 0) -> GradCellOutput:
        task_embedding = self.task_encoder(preference)
        latent = self.initializer(task_embedding)
        # During frozen-refiner training the initializer has no trainable parameters,
        # but the refiner still needs dL/du from the differentiable physics chain.
        if not latent.requires_grad:
            latent = latent.detach().requires_grad_(True)
        state = None
        steps: list[GradCellStep] = []
        for index in range(num_steps + 1):
            step = self.evaluate(latent, preference)
            steps.append(step)
            if index == num_steps:
                break
            (gradient,) = torch.autograd.grad(
                step.loss.sum(), latent, create_graph=False, retain_graph=True
            )
            gradient = gradient.detach()
            physics_features = torch.stack(
                [
                    step.energy / 250.0,
                    torch.minimum(step.retention_5c, step.retention_6c),
                    step.loss,
                    step.status.to(step.loss.dtype),
                    latent.square().mean(dim=-1),
                ],
                dim=-1,
            )
            alpha, diagonal, state = self.refiner(
                task_embedding, latent, physics_features, gradient, state
            )
            update = alpha * diagonal * gradient
            update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            update_scale = torch.clamp(
                self.max_refinement_update_norm / update_norm, max=1.0
            )
            latent = latent - update_scale * update
        return GradCellOutput(steps)
