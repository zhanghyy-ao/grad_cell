from __future__ import annotations

from dataclasses import dataclass

import torch

FARADAY_C_PER_MOL = 96485.33212


@dataclass(frozen=True)
class CapacityConstants:
    """用于正负极容量平衡和额定容量计算的固定物理常数。"""

    # 正极涂层厚度，单位 m。
    positive_thickness_m: float = 7.56e-5

    # 负极涂层厚度，单位 m。
    negative_thickness_m: float = 8.52e-5

    # 正极活性材料的最大锂浓度，单位 mol/m^3。
    positive_cmax_mol_m3: float = 63104.0

    # 负极活性材料的最大锂浓度，单位 mol/m^3。
    negative_cmax_mol_m3: float = 33133.0

    # 正极在工作 SOC 范围内可利用的化学计量窗口 Delta theta_p，无量纲。
    positive_stoich_window: float = 0.75

    # 负极在工作 SOC 范围内可利用的化学计量窗口 Delta theta_n，无量纲。
    negative_stoich_window: float = 0.75

    # 电极有效面积，单位 m^2，用于将单位面积容量换算为整片电极容量。
    electrode_area_m2: float = 0.1027


def negative_active_fraction(
    positive_active_fraction: torch.Tensor,
    np_ratio: torch.Tensor,
    constants: CapacityConstants,
) -> torch.Tensor:
    """根据正极活性材料体积分数推导满足目标 N/P 比的负极活性材料体积分数。

    Args:
        positive_active_fraction: 正极活性材料体积分数 ``phi_p``。
        np_ratio: 目标负极/正极容量比 ``Q_n / Q_p``。
        constants: 正负极厚度、最大锂浓度和化学计量窗口等容量常数。

    Returns:
        负极活性材料体积分数 ``phi_n``，使 ``Q_n / Q_p`` 精确等于
        ``np_ratio``。
    """
    # 正极单位面积容量的比例因子：L_p * c_max,p * Delta theta_p。
    numerator = (
        constants.positive_thickness_m
        * constants.positive_cmax_mol_m3
        * constants.positive_stoich_window
    )

    # 负极单位面积容量的比例因子：L_n * c_max,n * Delta theta_n。
    denominator = (
        constants.negative_thickness_m
        * constants.negative_cmax_mol_m3
        * constants.negative_stoich_window
    )

    # 由 Q_n / Q_p = np_ratio 解出：
    # phi_n = np_ratio * (L_p c_max,p Delta theta_p)
    #                    / (L_n c_max,n Delta theta_n) * phi_p。
    negative_active_fraction_value = (
        np_ratio * (numerator / denominator) * positive_active_fraction
    )
    return negative_active_fraction_value
# 已知目标容量比（NP），反推负极需要填多少料

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


def chen2020_scaled_capacity_ah(
    positive_active_fraction: torch.Tensor,
    nominal_capacity_ah: float = 5.0,
    nominal_positive_active_fraction: float = 0.665,
) -> torch.Tensor:
    """Scale Chen2020's nominal 5 Ah capacity by positive active fraction.

    This deliberately conservative alternative keeps the published Chen2020
    nominal cell as its calibration point instead of assuming a fixed 0.75
    stoichiometric utilization window for every redesigned cell.
    """
    return (
        positive_active_fraction
        / nominal_positive_active_fraction
        * nominal_capacity_ah
    )

#正极标称容量
