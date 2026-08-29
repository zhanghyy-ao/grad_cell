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
        finite1 = torch.isfinite(y1).flatten(start_dim=1).all(dim=1)
        finite3 = torch.isfinite(y3).flatten(start_dim=1).all(dim=1)
        valid = status1.bool() & status3.bool() & finite1 & finite3
        valid_indices = valid.nonzero(as_tuple=True)[0]

        batch_size = latent.shape[0]
        energy = latent.new_zeros(batch_size)
        power = latent.new_zeros(batch_size)
        loss = 100.0 + 0.1 * latent.square().mean(dim=-1)

        if valid_indices.numel() > 0:
            metrics1 = discharge_metrics(
                y1[valid],
                design.nominal_capacity_ah[valid],
                design.stack_mass_kg[valid],
                3600.0,
            )
            metrics3 = discharge_metrics(
                y3[valid],
                3.0 * design.nominal_capacity_ah[valid],
                design.stack_mass_kg[valid],
                1200.0,
            )
            valid_energy = metrics1.specific_energy_wh_kg
            valid_power = metrics3.specific_power_w_kg
            valid_loss = self.objective(valid_energy, valid_power, preference[valid])
            minimum_voltage = torch.minimum(
                metrics1.minimum_voltage_v, metrics3.minimum_voltage_v
            )
            voltage_penalty = torch.nn.functional.softplus(
                (2.4 - minimum_voltage) / 0.02
            ).square()
            valid_loss = valid_loss + 10.0 * voltage_penalty
            energy = energy.index_copy(0, valid_indices, valid_energy)
            power = power.index_copy(0, valid_indices, valid_power)
            loss = loss.index_copy(0, valid_indices, valid_loss)

        status = valid.to(torch.int64)
        return GradCellStep(
            latent=latent,
            design=design,
            loss=loss,
            energy=energy,
            power=power,
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
