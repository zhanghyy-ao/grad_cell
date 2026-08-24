from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerformanceMetrics:
    specific_energy_wh_kg: torch.Tensor
    specific_power_w_kg: torch.Tensor
    minimum_voltage_v: torch.Tensor


def voltage_gate(voltage: torch.Tensor, cutoff_v: float = 2.5, temperature_v: float = 0.02):
    return torch.sigmoid((voltage - cutoff_v) / temperature_v)


def discharge_metrics(
    voltage: torch.Tensor,
    current_a: torch.Tensor,
    mass_kg: torch.Tensor,
    horizon_s: float,
    cutoff_v: float = 2.5,
    gate_temperature_v: float = 0.02,
) -> PerformanceMetrics:
    if voltage.ndim == 3:
        voltage = voltage[:, 0]
    gate = voltage_gate(voltage, cutoff_v, gate_temperature_v)
    dt = horizon_s / (voltage.shape[-1] - 1)
    usable_wh = torch.trapezoid(
        current_a[:, None] * voltage * gate,
        dx=dt,
        dim=-1,
    ) / 3600.0
    effective_h = torch.trapezoid(gate, dx=dt, dim=-1) / 3600.0
    specific_energy = usable_wh / mass_kg
    specific_power = specific_energy / effective_h.clamp_min(1e-6)
    soft_min_voltage = -0.02 * torch.logsumexp(-voltage / 0.02, dim=-1)
    return PerformanceMetrics(specific_energy, specific_power, soft_min_voltage)

