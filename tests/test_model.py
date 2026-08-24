import torch

from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer


def test_gradcell_k0_backward():
    model = GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=1200.0)),
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
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=1200.0)),
    ).double()
    output = model(torch.tensor([0.5], dtype=torch.float64), num_steps=1)
    assert len(output.steps) == 2
    assert torch.isfinite(output.final.loss).all()

