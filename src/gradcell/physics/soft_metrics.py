from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerformanceMetrics:
    """由放电电压轨迹计算得到的性能指标。"""

    # 单位 Wh/kg，按软截止门积分得到的可用比能量。
    specific_energy_wh_kg: torch.Tensor
    # 单位 W/kg，用比能量除以有效放电时间得到的比功率。
    specific_power_w_kg: torch.Tensor
    # 单位 V，使用 log-sum-exp 近似轨迹最小电压。
    minimum_voltage_v: torch.Tensor


def voltage_gate(voltage: torch.Tensor, cutoff_v: float = 2.5, temperature_v: float = 0.02):
    """将硬截止电压平滑为可微 sigmoid 门函数。"""
    return torch.sigmoid((voltage - cutoff_v) / temperature_v)


def discharge_metrics(
    voltage: torch.Tensor,
    current_a: torch.Tensor,
    mass_kg: torch.Tensor,
    horizon_s: float,
    cutoff_v: float = 2.5,
    gate_temperature_v: float = 0.02,
) -> PerformanceMetrics:
    """从电压轨迹计算软截止下的比能量、比功率和最低电压。"""
    if voltage.ndim == 3:
        # 后端输出通常为 [B,1,T]；指标计算只需要唯一输出通道 [B,T]。
        voltage = voltage[:, 0]
    # gate 接近 1 的时间段计入放电，低于 cutoff 的部分平滑衰减。
    gate = voltage_gate(voltage, cutoff_v, gate_temperature_v)
    # 固定仿真时域下相邻采样点的时间间隔，单位 s。
    dt = horizon_s / (voltage.shape[-1] - 1)
    # 对 I*V*gate 积分并从焦耳/瓦秒换算为 Wh。
    usable_wh = torch.trapezoid(
        current_a[:, None] * voltage * gate,
        dx=dt,
        dim=-1,
    ) / 3600.0
    # gate 的积分表示有效放电时长，单位 h。
    effective_h = torch.trapezoid(gate, dx=dt, dim=-1) / 3600.0
    # 用 stack mass 归一化能量，并用有效时间计算平均比功率。
    specific_energy = usable_wh / mass_kg
    specific_power = specific_energy / effective_h.clamp_min(1e-6)
    # -tau*logsumexp(-V/tau) 是 min(V) 的平滑近似，保持可微。
    soft_min_voltage = -0.02 * torch.logsumexp(-voltage / 0.02, dim=-1)
    return PerformanceMetrics(specific_energy, specific_power, soft_min_voltage)
