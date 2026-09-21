import torch

from gradcell.language.direct_dfn import _discharge_summary


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
