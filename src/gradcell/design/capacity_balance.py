from __future__ import annotations

from dataclasses import dataclass

import torch

FARADAY_C_PER_MOL = 96485.33212


@dataclass(frozen=True)
class CapacityConstants:
    positive_thickness_m: float = 7.56e-5
    negative_thickness_m: float = 8.52e-5
    positive_cmax_mol_m3: float = 63104.0
    negative_cmax_mol_m3: float = 33133.0
    positive_stoich_window: float = 0.75
    negative_stoich_window: float = 0.75
    electrode_area_m2: float = 0.1027


def negative_active_fraction(
    positive_active_fraction: torch.Tensor,
    np_ratio: torch.Tensor,
    constants: CapacityConstants,
) -> torch.Tensor:
    """Derive negative active fraction so the requested N/P ratio is exact."""
    numerator = (
        constants.positive_thickness_m
        * constants.positive_cmax_mol_m3
        * constants.positive_stoich_window
    )
    denominator = (
        constants.negative_thickness_m
        * constants.negative_cmax_mol_m3
        * constants.negative_stoich_window
    )
    return np_ratio * (numerator / denominator) * positive_active_fraction


def nominal_capacity_ah(
    positive_active_fraction: torch.Tensor,
    constants: CapacityConstants,
) -> torch.Tensor:
    areal_capacity = (
        FARADAY_C_PER_MOL
        * constants.positive_thickness_m
        * positive_active_fraction
        * constants.positive_cmax_mol_m3
        * constants.positive_stoich_window
        / 3600.0
    )
    return areal_capacity * constants.electrode_area_m2

