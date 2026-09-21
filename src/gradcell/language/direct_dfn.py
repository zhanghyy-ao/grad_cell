"""Direct differentiable PyBaMM DFN performance layer for language design."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend

from .physics_guided import DEFAULT_PERFORMANCE_FIELDS


@dataclass(frozen=True)
class DirectDFNOutput:
    performance: torch.Tensor
    status: torch.Tensor
    runtime_s: torch.Tensor


def _discharge_summary(
    trajectory: torch.Tensor,
    current_a: torch.Tensor,
    horizon_s: float,
    cutoff_v: float,
    gate_temperature_v: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    voltage = trajectory[:, 0]
    gate = torch.sigmoid((voltage - cutoff_v) / gate_temperature_v)
    dt = horizon_s / (voltage.shape[-1] - 1)
    effective_h = torch.trapezoid(gate, dx=dt, dim=-1) / 3600.0
    capacity_ah = current_a * effective_h
    energy_wh = torch.trapezoid(current_a[:, None] * voltage * gate, dx=dt, dim=-1) / 3600.0
    return capacity_ah, energy_wh


class DirectDFNPerformanceLayer(nn.Module):
    """Run differentiable 1C/5C/6C DFN solves and return eight performance metrics."""

    performance_fields = DEFAULT_PERFORMANCE_FIELDS

    def __init__(
        self,
        parameter_set: str = "Chen2020",
        time_points: int = 151,
        maximum_duration_factor: float = 1.5,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        cutoff_v: float = 2.5,
        gate_temperature_v: float = 0.02,
        current_ramp_time_s: float = 1.0,
        calculate_sensitivities: bool = True,
    ) -> None:
        super().__init__()
        if time_points < 2:
            raise ValueError("time_points must be at least two")
        if maximum_duration_factor <= 0.0 or gate_temperature_v <= 0.0:
            raise ValueError("duration factor and gate temperature must be positive")
        self.rates = (1.0, 5.0, 6.0)
        self.cutoff_v = float(cutoff_v)
        self.gate_temperature_v = float(gate_temperature_v)
        self.horizons = {
            rate: float(maximum_duration_factor) * 3600.0 / rate for rate in self.rates
        }
        backends = {}
        for rate in self.rates:
            backend = PyBaMMBackend(
                model_name="DFN",
                parameter_set=parameter_set,
                output_variables=("Voltage [V]",),
                time_points=time_points,
                horizon_s=self.horizons[rate],
                rtol=rtol,
                atol=atol,
                calculate_sensitivities=calculate_sensitivities,
                current_ramp_time_s=current_ramp_time_s,
                physical_voltage_cutoffs=False,
            )
            backends[f"rate_{rate:g}c"] = DifferentiablePhysicsLayer(backend)
        self.layers = nn.ModuleDict(backends)
        nominal = backends["rate_1c"].backend.nominal_input_values
        self.register_buffer(
            "nominal_parameter_values", torch.as_tensor(nominal, dtype=torch.float32)
        )

    def forward(
        self, parameter_values: torch.Tensor, reference_capacity_ah: torch.Tensor
    ) -> DirectDFNOutput:
        if parameter_values.ndim != 2 or parameter_values.shape[1] != 7:
            raise ValueError("parameter_values must have shape [batch, 7]")
        if reference_capacity_ah.shape != (parameter_values.shape[0],):
            raise ValueError("reference_capacity_ah must have shape [batch]")
        summaries = {}
        statuses = []
        runtimes = []
        for rate in self.rates:
            current = rate * reference_capacity_ah
            inputs = torch.cat([parameter_values, current[:, None]], dim=1)
            trajectory, status, runtime = self.layers[f"rate_{rate:g}c"](inputs)
            capacity, energy = _discharge_summary(
                trajectory,
                current,
                self.horizons[rate],
                self.cutoff_v,
                self.gate_temperature_v,
            )
            summaries[rate] = (capacity, energy)
            statuses.append(status.bool())
            runtimes.append(runtime)
        capacity_1c, energy_1c = summaries[1.0]
        capacity_5c, energy_5c = summaries[5.0]
        capacity_6c, energy_6c = summaries[6.0]
        performance = torch.stack(
            [
                capacity_1c,
                energy_1c,
                capacity_5c,
                energy_5c,
                energy_5c / energy_1c.clamp_min(1e-8),
                capacity_6c,
                energy_6c,
                energy_6c / energy_1c.clamp_min(1e-8),
            ],
            dim=1,
        )
        status = torch.stack(statuses, dim=1).all(dim=1)
        runtime_s = torch.stack(runtimes, dim=1).sum(dim=1)
        finite = torch.isfinite(performance).all(dim=1)
        positive = (performance > 0.0).all(dim=1)
        return DirectDFNOutput(performance, status & finite & positive, runtime_s)
