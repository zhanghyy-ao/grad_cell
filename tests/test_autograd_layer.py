import numpy as np
import torch

from gradcell.physics import (
    AnalyticToyBackend,
    DifferentiablePhysicsLayer,
    summarize_discharge,
)
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


def test_physical_discharge_summary_uses_actual_termination_time():
    summary = summarize_discharge(
        time_s=np.array([0.0, 900.0, 1800.0]),
        voltage_v=np.array([4.0, 3.5, 3.0]),
        current_a=np.array([2.0, 2.0, 2.0]),
    )

    assert summary["delivered_capacity_ah"] == 1.0
    assert summary["delivered_energy_wh"] == 3.5
    assert summary["average_voltage_v"] == 3.5
    assert summary["minimum_voltage_v"] == 3.0
    assert summary["discharge_time_s"] == 1800.0


def test_physical_discharge_summary_rejects_non_finite_values():
    with np.testing.assert_raises(ValueError):
        summarize_discharge(
            time_s=np.array([0.0, 1.0]),
            voltage_v=np.array([4.0, np.nan]),
            current_a=np.array([1.0, 1.0]),
        )
