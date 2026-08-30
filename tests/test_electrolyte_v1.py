import math

import torch

from gradcell.models import ElectrolytePropertyNetwork
from gradcell.physics import AnalyticElectrolyteBackend, DifferentiablePhysicsLayer
from gradcell.physics.gradient_validation import directional_derivative_check


def test_electrolyte_property_network_is_positive_and_bounded():
    model = ElectrolytePropertyNetwork(input_dim=6, hidden_dim=16, depth=2).double()
    prediction = model(torch.randn(8, 6, dtype=torch.float64))
    assert prediction.shape == (8,)
    assert bool((prediction >= math.log(0.05)).all())
    assert bool((prediction <= math.log(50.0)).all())


def test_analytic_electrolyte_backend_gradient_matches_finite_difference():
    layer = DifferentiablePhysicsLayer(
        AnalyticElectrolyteBackend(time_points=31, probe_horizon_s=300.0)
    )
    point = torch.tensor([[0.1, 5.0]], dtype=torch.float64)

    def objective(value):
        voltage, status, _ = layer(value)
        assert status.item() == 1
        return voltage.square().mean()

    result = directional_derivative_check(
        objective,
        point,
        direction=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        eps=1e-6,
    )
    assert result["relative_directional_error"] < 1e-5


def test_hybrid_voltage_loss_reaches_property_network():
    model = ElectrolytePropertyNetwork(input_dim=4, hidden_dim=16, depth=2).double()
    backend = AnalyticElectrolyteBackend(time_points=21, probe_horizon_s=200.0)
    layer = DifferentiablePhysicsLayer(backend)
    features = torch.randn(3, 4, dtype=torch.float64)
    log_k = model(features)
    current = torch.full_like(log_k, 5.0)
    physics_inputs = torch.stack([log_k - math.log(10.0), current], dim=-1)
    voltage, status, _ = layer(physics_inputs)
    assert bool((status == 1).all())
    loss = voltage.square().mean()
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0
