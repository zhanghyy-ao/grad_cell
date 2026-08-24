from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MassConstants:
    area_m2: float = 0.1027
    positive_thickness_m: float = 7.56e-5
    negative_thickness_m: float = 8.52e-5
    separator_thickness_m: float = 1.2e-5
    positive_active_density_kg_m3: float = 3262.0
    negative_active_density_kg_m3: float = 2266.0
    inactive_density_kg_m3: float = 1800.0
    electrolyte_density_kg_m3: float = 1270.0
    fixed_collector_mass_kg: float = 0.010


def stack_mass_kg(
    eps_p: torch.Tensor,
    eps_n: torch.Tensor,
    eps_s: torch.Tensor,
    phi_p: torch.Tensor,
    phi_n: torch.Tensor,
    constants: MassConstants,
) -> torch.Tensor:
    inactive_p = 1.0 - eps_p - phi_p
    inactive_n = 1.0 - eps_n - phi_n
    positive = constants.area_m2 * constants.positive_thickness_m * (
        constants.positive_active_density_kg_m3 * phi_p
        + constants.inactive_density_kg_m3 * inactive_p
    )
    negative = constants.area_m2 * constants.negative_thickness_m * (
        constants.negative_active_density_kg_m3 * phi_n
        + constants.inactive_density_kg_m3 * inactive_n
    )
    electrolyte = constants.area_m2 * constants.electrolyte_density_kg_m3 * (
        constants.positive_thickness_m * eps_p
        + constants.negative_thickness_m * eps_n
        + constants.separator_thickness_m * eps_s
    )
    return positive + negative + electrolyte + constants.fixed_collector_mass_kg

