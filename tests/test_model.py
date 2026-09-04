import numpy as np
import torch

from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer
from gradcell.physics.backend import PhysicsBatch


class ControlledFailureBackend:
    def __init__(self, fail_all: bool = False):
        self.fail_all = fail_all

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        batch_size, parameter_count = inputs.shape
        trajectories = np.full((batch_size, 1, 11), 3.7, dtype=np.float64)
        jacobian = np.full((batch_size, 1, 11, parameter_count), 1e-3, dtype=np.float64)
        status = np.ones(batch_size, dtype=np.int64)
        failed = np.ones(batch_size, dtype=bool) if self.fail_all else np.arange(batch_size) == 1
        trajectories[failed] = np.nan
        jacobian[failed] = 0.0
        status[failed] = 0
        return PhysicsBatch(
            trajectories=trajectories,
            jacobian=jacobian,
            status=status,
            runtime_s=np.zeros(batch_size, dtype=np.float64),
        )


def test_gradcell_k0_backward():
    model = GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    ).double()
    output = model(torch.tensor([0.2, 0.8], dtype=torch.float64), num_steps=0)
    loss = output.final.loss.mean()
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.norm())
        for parameter in model.initializer.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert gradient_norm > 0.0


def test_gradcell_refinement_runs():
    model = GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    ).double()
    output = model(torch.tensor([0.5], dtype=torch.float64), num_steps=1)
    assert len(output.steps) == 2
    assert torch.isfinite(output.final.loss).all()


def test_failed_sample_does_not_contaminate_valid_sample():
    model = GradCell(
        DifferentiablePhysicsLayer(ControlledFailureBackend()),
        DifferentiablePhysicsLayer(ControlledFailureBackend()),
        DifferentiablePhysicsLayer(ControlledFailureBackend()),
    ).double()
    latent = torch.full(
        (2, model.design_space.latent_dim), 0.2, dtype=torch.float64, requires_grad=True
    )
    step = model.evaluate(latent, torch.tensor([0.3, 0.7], dtype=torch.float64))

    assert step.status.tolist() == [1, 0]
    assert torch.isfinite(step.loss).all()
    assert torch.isfinite(step.energy).all()
    assert torch.isfinite(step.retention_5c).all()
    assert torch.isfinite(step.retention_6c).all()
    assert step.energy[1] == 0.0
    assert step.retention_5c[1] == 0.0
    assert step.retention_6c[1] == 0.0

    step.loss.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_all_failed_batch_has_finite_loss_and_gradient():
    model = GradCell(
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
    ).double()
    latent = torch.full(
        (2, model.design_space.latent_dim), 0.2, dtype=torch.float64, requires_grad=True
    )
    step = model.evaluate(latent, torch.tensor([0.3, 0.7], dtype=torch.float64))

    assert step.status.tolist() == [0, 0]
    assert torch.isfinite(step.loss).all()
    step.loss.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
