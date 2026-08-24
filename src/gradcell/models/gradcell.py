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
    power: torch.Tensor
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
        physics_3c: nn.Module,
        design_space: DesignSpace | None = None,
        objective: nn.Module | None = None,
        task_dim: int = 128,
    ) -> None:
        super().__init__()
        self.design_space = design_space or DesignSpace()
        self.task_encoder = FourierPreferenceEncoder(embedding_dim=task_dim)
        self.initializer = DesignInitializer(task_dim=task_dim)
        self.refiner = DiagonalPhysicsRefiner(task_dim=task_dim)
        self.physics_1c = physics_1c
        self.physics_3c = physics_3c
        self.objective = objective or SmoothTchebycheff()

    def evaluate(self, latent: torch.Tensor, preference: torch.Tensor) -> GradCellStep:
        design = self.design_space(latent)
        y1, status1, _ = self.physics_1c(design.physics_tensor(1.0))
        y3, status3, _ = self.physics_3c(design.physics_tensor(3.0))
        metrics1 = discharge_metrics(
            y1, design.nominal_capacity_ah, design.stack_mass_kg, 3600.0
        )
        metrics3 = discharge_metrics(
            y3, 3.0 * design.nominal_capacity_ah, design.stack_mass_kg, 1200.0
        )
        loss = self.objective(
            metrics1.specific_energy_wh_kg,
            metrics3.specific_power_w_kg,
            preference,
        )
        voltage_penalty = torch.nn.functional.softplus(
            (2.4 - torch.minimum(metrics1.minimum_voltage_v, metrics3.minimum_voltage_v)) / 0.02
        ).square()
        status = status1 * status3
        failure_penalty = (1 - status).to(loss.dtype) * (
            100.0 + 0.1 * latent.square().sum(dim=-1)
        )
        loss = loss + 10.0 * voltage_penalty + failure_penalty
        return GradCellStep(
            latent=latent,
            design=design,
            loss=loss,
            energy=metrics1.specific_energy_wh_kg,
            power=metrics3.specific_power_w_kg,
            status=status,
        )

    def forward(self, preference: torch.Tensor, num_steps: int = 0) -> GradCellOutput:
        task_embedding = self.task_encoder(preference)
        latent = self.initializer(task_embedding)
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
                    step.power / 800.0,
                    step.loss,
                    step.status.to(step.loss.dtype),
                    latent.square().mean(dim=-1),
                ],
                dim=-1,
            )
            alpha, diagonal, state = self.refiner(
                task_embedding, latent, physics_features, gradient, state
            )
            latent = latent - alpha * diagonal * gradient
        return GradCellOutput(steps)
