import torch

from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer
from gradcell.physics.gradient_validation import directional_derivative_check


def test_custom_backward_matches_directional_finite_difference():
    layer = DifferentiablePhysicsLayer(AnalyticToyBackend(time_points=41))
    point = torch.tensor(
        [[0.30, 0.30, 0.45, 0.55, 0.58, 1.0, 1.0, 5.0]],
        dtype=torch.float64,
    )

    def objective(value):
        y, _, _ = layer(value)
        return y.square().mean()

    result = directional_derivative_check(objective, point, eps=1e-6)
    assert result["relative_directional_error"] < 1e-5


def test_toy_backend_records_termination_metadata():
    backend = AnalyticToyBackend(horizon_s=1200.0)
    inputs = torch.tensor(
        [[0.3, 0.3, 0.45, 0.55, 0.58, 1.0, 1.0, 2.0]], dtype=torch.float64
    )
    DifferentiablePhysicsLayer(backend)(inputs)
    diagnostics = backend.last_solve_diagnostics[0]
    assert diagnostics["requested_end_time_s"] == 1200.0
    assert diagnostics["actual_end_time_s"] == 1200.0
    assert diagnostics["completed_requested_horizon"] is True
