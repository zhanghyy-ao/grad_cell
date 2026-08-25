from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MassConstants:
    """用于商业电芯 BOM 级质量估算的几何尺寸、密度和硬件质量常数。

    该模型显式区分涂层固相、孔隙中的电解液、隔膜聚合物、集流体箔材
    和固定硬件。粘结剂/导电剂仍使用 ``inactive_density_kg_m3`` 的等效
    密度；壳体、极耳、绝缘件和制造余量由 ``fixed_collector_mass_kg``
    汇总表示，因此仍不是厂家级逐零件 BOM。
    """

    # 电极有效面积，单位 m^2。
    area_m2: float = 0.1027

    # 正极涂层厚度，单位 m。
    positive_thickness_m: float = 7.56e-5

    # 负极涂层厚度，单位 m。
    negative_thickness_m: float = 8.52e-5

    # 隔膜厚度，单位 m。
    separator_thickness_m: float = 1.2e-5

    # 正极集流体（通常为铝箔）厚度和密度，单位分别为 m 和 kg/m^3。
    positive_collector_thickness_m: float = 1.5e-5
    positive_collector_density_kg_m3: float = 2700.0

    # 负极集流体（通常为铜箔）厚度和密度，单位分别为 m 和 kg/m^3。
    negative_collector_thickness_m: float = 1.0e-5
    negative_collector_density_kg_m3: float = 8960.0

    # 隔膜聚合物基材的等效密度，单位 kg/m^3。
    separator_density_kg_m3: float = 930.0

    # 正极活性材料密度，单位 kg/m^3。
    positive_active_density_kg_m3: float = 3262.0

    # 负极活性材料密度，单位 kg/m^3。
    negative_active_density_kg_m3: float = 2266.0

    # 正负极导电剂、黏结剂等非活性固相的等效密度，单位 kg/m^3。
    inactive_density_kg_m3: float = 1800.0

    # 电解液密度，单位 kg/m^3。
    electrolyte_density_kg_m3: float = 1270.0

    # 壳体、极耳、绝缘件、封装和制造余量等固定硬件质量，单位 kg。
    # 为保持与旧版接口兼容，字段名沿用 fixed_collector_mass_kg。
    fixed_collector_mass_kg: float = 0.010


def stack_mass_kg(
    eps_p: torch.Tensor,
    eps_n: torch.Tensor,
    eps_s: torch.Tensor,
    phi_p: torch.Tensor,
    phi_n: torch.Tensor,
    constants: MassConstants,
) -> torch.Tensor:
    """按商业电芯 BOM 的主要质量项估算 stack-level mass，返回单位 kg。

    质量项包括：正负极涂层固相、正负极集流体、隔膜聚合物、电解液，
    以及壳体/极耳等固定硬件。该模型仍是工程级 proxy，不包含卷绕压实、
    涂布损失、注液余量、焊点和制造良率等厂家专有细节。
    """

    # 正极中非活性固相的体积分数：总固相空间减去活性材料部分。
    inactive_p = 1.0 - eps_p - phi_p

    # 负极中非活性固相的体积分数：总固相空间减去活性材料部分。
    inactive_n = 1.0 - eps_n - phi_n

    # 正极涂层固相质量 = 面积 * 厚度 *（活性材料质量密度贡献
    #                         + 非活性固相质量密度贡献）。
    positive = constants.area_m2 * constants.positive_thickness_m * (
        constants.positive_active_density_kg_m3 * phi_p
        + constants.inactive_density_kg_m3 * inactive_p
    )

    # 负极涂层固相质量 = 面积 * 厚度 *（活性材料质量密度贡献
    #                         + 非活性固相质量密度贡献）。
    negative = constants.area_m2 * constants.negative_thickness_m * (
        constants.negative_active_density_kg_m3 * phi_n
        + constants.inactive_density_kg_m3 * inactive_n
    )

    # 正极和负极孔隙中的电解液质量。
    electrolyte = constants.area_m2 * constants.electrolyte_density_kg_m3 * (
        constants.positive_thickness_m * eps_p
        + constants.negative_thickness_m * eps_n
    )

    # 隔膜聚合物基材质量：隔膜总体积乘以非孔隙体积分数。
    separator = constants.area_m2 * constants.separator_thickness_m * (
        constants.separator_density_kg_m3 * (1.0 - eps_s)
    )

    # 隔膜孔隙中的电解液质量。
    separator_electrolyte = (
        constants.area_m2
        * constants.separator_thickness_m
        * constants.electrolyte_density_kg_m3
        * eps_s
    )

    # 正、负极集流体箔材质量。
    collectors = constants.area_m2 * (
        constants.positive_collector_thickness_m
        * constants.positive_collector_density_kg_m3
        + constants.negative_collector_thickness_m
        * constants.negative_collector_density_kg_m3
    )

    # 总质量 = 涂层固相 + 电解液 + 隔膜 + 集流体 + 固定硬件。
    return (
        positive
        + negative
        + electrolyte
        + separator
        + separator_electrolyte
        + collectors
        + constants.fixed_collector_mass_kg
    )
