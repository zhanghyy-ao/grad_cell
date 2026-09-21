import torch
import numpy as np

from gradcell.language.direct_dfn import _discharge_summary
from gradcell.physics import DifferentiablePhysicsLayer
from gradcell.physics.backend import PhysicsBatch


def test_discharge_summary_propagates_voltage_gradient() -> None:
    voltage = torch.full((2, 1, 11), 3.5, dtype=torch.float64, requires_grad=True)
    current = torch.tensor([2.0, 4.0], dtype=torch.float64)
    capacity, energy = _discharge_summary(
        voltage,
        current,
        horizon_s=3600.0,
        cutoff_v=2.5,
        gate_temperature_v=0.02,
    )
    assert capacity.shape == (2,)
    assert energy.shape == (2,)
    assert torch.all(capacity > 0.0)
    assert torch.all(energy > 0.0)
    energy.sum().backward()
    assert voltage.grad is not None
    assert torch.isfinite(voltage.grad).all()
    assert float(voltage.grad.abs().sum()) > 0.0


class _SensitivityRecordingBackend:
    def __init__(self) -> None:
        self.calculate_sensitivities = True
        self.settings_seen: list[bool] = []

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        self.settings_seen.append(self.calculate_sensitivities)
        batch, parameters = inputs.shape
        jacobian = np.ones((batch, 1, 2, parameters), dtype=np.float64)
        return PhysicsBatch(
            trajectories=np.ones((batch, 1, 2), dtype=np.float64),
            jacobian=jacobian,
            status=np.ones(batch, dtype=np.int64),
            runtime_s=np.zeros(batch, dtype=np.float64),
        )


def test_physics_layer_skips_sensitivities_for_evaluation_inputs() -> None:
    backend = _SensitivityRecordingBackend()
    layer = DifferentiablePhysicsLayer(backend)
    layer(torch.ones(1, 2))
    assert backend.settings_seen == [False]
    assert backend.calculate_sensitivities is True


def test_physics_layer_keeps_sensitivities_for_training_inputs() -> None:
    backend = _SensitivityRecordingBackend()
    layer = DifferentiablePhysicsLayer(backend)
    inputs = torch.ones(1, 2, requires_grad=True)
    output, _, _ = layer(inputs)
    output.sum().backward()
    assert backend.settings_seen == [True]
    assert inputs.grad is not None
