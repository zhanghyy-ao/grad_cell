from __future__ import annotations

from dataclasses import dataclass

import torch


TASK_TAGS = (
    "<TASK>",
    "</TASK>",
    "<MATERIAL_CONTEXT>",
    "</MATERIAL_CONTEXT>",
    "<MATERIAL_PARAMETER_SET>",
    "</MATERIAL_PARAMETER_SET>",
    "<MATERIAL_PROPERTIES_MODE>",
    "</MATERIAL_PROPERTIES_MODE>",
    "<OPERATING_CONDITION>",
    "</OPERATING_CONDITION>",
    "<TEMPERATURE_K>",
    "</TEMPERATURE_K>",
    "<DISCHARGE_PROTOCOL>",
    "</DISCHARGE_PROTOCOL>",
    "<PERFORMANCE_REQUIREMENTS>",
    "</PERFORMANCE_REQUIREMENTS>",
    "<OBJECTIVE_A>",
    "</OBJECTIVE_A>",
    "<OBJECTIVE_B>",
    "</OBJECTIVE_B>",
    "<PREFERENCE>",
    "</PREFERENCE>",
    "<TARGET_ENERGY>",
    "</TARGET_ENERGY>",
    "<MIN_R5>",
    "</MIN_R5>",
    "<MIN_R6>",
    "</MIN_R6>",
    "<PREFERENCE_PROFILE>",
    "</PREFERENCE_PROFILE>",
    "<ENERGY_PRIORITY>",
    "</ENERGY_PRIORITY>",
    "<RATE_PRIORITY>",
    "</RATE_PRIORITY>",
    "<DESIGN_CONSTRAINTS>",
    "</DESIGN_CONSTRAINTS>",
    "<POSITIVE_POROSITY_RANGE>",
    "</POSITIVE_POROSITY_RANGE>",
    "<NEGATIVE_POROSITY_RANGE>",
    "</NEGATIVE_POROSITY_RANGE>",
    "<SEPARATOR_POROSITY_RANGE>",
    "</SEPARATOR_POROSITY_RANGE>",
    "<NP_RATIO_RANGE>",
    "</NP_RATIO_RANGE>",
    "<CAPACITY_BALANCE>",
    "</CAPACITY_BALANCE>",
    "<FEASIBILITY_POLICY>",
    "</FEASIBILITY_POLICY>",
    "<OUTPUT_CONTRACT>",
    "</OUTPUT_CONTRACT>",
    "<OUTPUT_SCHEMA>",
    "</OUTPUT_SCHEMA>",
    "<REQUIRED_DESIGN_FIELDS>",
    "</REQUIRED_DESIGN_FIELDS>",
    "<FORBIDDEN_OUTPUT_FIELDS>",
    "</FORBIDDEN_OUTPUT_FIELDS>",
    "<OUTPUT_JSON_TEMPLATE>",
    "</OUTPUT_JSON_TEMPLATE>",
    "<SELECTION_POLICY>",
    "</SELECTION_POLICY>",
    "<DESIGN>",
    "</DESIGN>",
    "<U0>",
    "</U0>",
    "<U1>",
    "</U1>",
    "<U2>",
    "</U2>",
    "<U3>",
    "</U3>",
    "<U4>",
    "</U4>",
)


@dataclass(frozen=True)
class StructuredPreferenceTask:
    preference: float
    target_energy: float = 150.0
    min_retention_5c: float = 0.50
    min_retention_6c: float = 0.44
    temperature_k: float = 298.15
    material_parameter_set: str = "Chen2020"

    def __post_init__(self) -> None:
        if not 0.0 <= self.preference <= 1.0:
            raise ValueError("preference must be in [0, 1]")
        if self.target_energy <= 0.0:
            raise ValueError("target_energy must be positive")
        if not 0.0 <= self.min_retention_5c <= 1.0:
            raise ValueError("min_retention_5c must be in [0, 1]")
        if not 0.0 <= self.min_retention_6c <= 1.0:
            raise ValueError("min_retention_6c must be in [0, 1]")
        if self.temperature_k != 298.15:
            raise ValueError(
                "the current GradCell physics backend only supports the fixed 298.15 K task context"
            )
        if self.material_parameter_set != "Chen2020":
            raise ValueError("the current design space only supports the Chen2020 parameter set")


class GradCellLanguageCodec:
    """Deterministic grammar for preference tasks and quantized GradCell latents."""

    def __init__(self, bins: int = 256, latent_limit: float = 4.0) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if latent_limit <= 0.0:
            raise ValueError("latent_limit must be positive")
        self.bins = bins
        self.latent_limit = latent_limit

    @property
    def special_tokens(self) -> list[str]:
        return [*TASK_TAGS, *[f"<LEVEL_{index:03d}>" for index in range(self.bins)]]

    def preference_level(self, preference: float) -> int:
        StructuredPreferenceTask(preference)
        return round(preference * (self.bins - 1))

    def serialize_task(self, task: StructuredPreferenceTask) -> str:
        level = self.preference_level(task.preference)
        energy_priority = self._priority_label(task.preference)
        rate_priority = self._priority_label(1.0 - task.preference)
        return (
            "<TASK>\n"
            "<MATERIAL_CONTEXT>\n"
            f"<MATERIAL_PARAMETER_SET>{task.material_parameter_set}</MATERIAL_PARAMETER_SET>\n"
            "<MATERIAL_PROPERTIES_MODE>FIXED</MATERIAL_PROPERTIES_MODE>\n"
            "</MATERIAL_CONTEXT>\n"
            "<OPERATING_CONDITION>\n"
            f"<TEMPERATURE_K>{task.temperature_k:.2f}</TEMPERATURE_K>\n"
            "<DISCHARGE_PROTOCOL>1C,5C,6C_CONSTANT_CURRENT</DISCHARGE_PROTOCOL>\n"
            "</OPERATING_CONDITION>\n"
            "<PERFORMANCE_REQUIREMENTS>\n"
            "<OBJECTIVE_A>SPECIFIC_ENERGY_1C_WH_KG</OBJECTIVE_A>\n"
            "<OBJECTIVE_B>ENERGY_RETENTION_5C_6C</OBJECTIVE_B>\n"
            f"<TARGET_ENERGY>{task.target_energy:.4f}</TARGET_ENERGY>\n"
            f"<MIN_R5>{task.min_retention_5c:.6f}</MIN_R5>\n"
            f"<MIN_R6>{task.min_retention_6c:.6f}</MIN_R6>\n"
            "</PERFORMANCE_REQUIREMENTS>\n"
            "<PREFERENCE_PROFILE>\n"
            f"<PREFERENCE><LEVEL_{level:03d}></PREFERENCE>\n"
            f"<ENERGY_PRIORITY>{energy_priority}</ENERGY_PRIORITY>\n"
            f"<RATE_PRIORITY>{rate_priority}</RATE_PRIORITY>\n"
            "</PREFERENCE_PROFILE>\n"
            "<DESIGN_CONSTRAINTS>\n"
            "<POSITIVE_POROSITY_RANGE>0.20,0.42</POSITIVE_POROSITY_RANGE>\n"
            "<NEGATIVE_POROSITY_RANGE>0.20,0.42</NEGATIVE_POROSITY_RANGE>\n"
            "<SEPARATOR_POROSITY_RANGE>0.35,0.60</SEPARATOR_POROSITY_RANGE>\n"
            "<NP_RATIO_RANGE>1.02,1.25</NP_RATIO_RANGE>\n"
            "<CAPACITY_BALANCE>ANALYTIC_NEGATIVE_ACTIVE_FRACTION</CAPACITY_BALANCE>\n"
            "<FEASIBILITY_POLICY>HARD_FEASIBLE_DECODER</FEASIBILITY_POLICY>\n"
            "</DESIGN_CONSTRAINTS>\n"
            "<OUTPUT_CONTRACT>\n"
            "<OUTPUT_SCHEMA>gradcell.material_design.v1</OUTPUT_SCHEMA>\n"
            "<REQUIRED_DESIGN_FIELDS>positive_electrode_porosity,negative_electrode_porosity,"
            "separator_porosity,positive_active_material_fraction,"
            "negative_to_positive_capacity_ratio</REQUIRED_DESIGN_FIELDS>\n"
            "<FORBIDDEN_OUTPUT_FIELDS>np_ratio,positive_active_fraction,negative_active_fraction,"
            "positive_electrode_active_fraction,negative_electrode_active_fraction,physics_loss,"
            "validation_loss</FORBIDDEN_OUTPUT_FIELDS>\n"
            "<OUTPUT_JSON_TEMPLATE>{\"schema\":\"gradcell.material_design.v1\","
            "\"material_parameter_set\":\"Chen2020\",\"fixed_material_properties\":true,"
            "\"design\":{\"positive_electrode_porosity\":FLOAT,"
            "\"negative_electrode_porosity\":FLOAT,\"separator_porosity\":FLOAT,"
            "\"positive_active_material_fraction\":FLOAT,"
            "\"negative_to_positive_capacity_ratio\":FLOAT},"
            "\"derived\":{\"negative_active_material_fraction\":FLOAT,"
            "\"nominal_capacity_ah\":FLOAT,\"stack_mass_kg\":FLOAT}}"
            "</OUTPUT_JSON_TEMPLATE>\n"
            "<SELECTION_POLICY>MINIMUM_PHYSICS_LOSS</SELECTION_POLICY>\n"
            "</OUTPUT_CONTRACT>\n"
            "</TASK>\n<DESIGN>"
        )

    @staticmethod
    def _priority_label(value: float) -> str:
        if value >= 2.0 / 3.0:
            return "HIGH"
        if value <= 1.0 / 3.0:
            return "LOW"
        return "MEDIUM"

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = (latent.clamp(-self.latent_limit, self.latent_limit) + self.latent_limit)
        normalized = normalized / (2.0 * self.latent_limit)
        return torch.round(normalized * (self.bins - 1)).to(torch.long)

    def dequantize(self, levels: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if torch.any(levels < 0) or torch.any(levels >= self.bins):
            raise ValueError("levels contain a value outside the codec vocabulary")
        result = levels.to(dtype=dtype or torch.get_default_dtype()) / (self.bins - 1)
        return result * (2.0 * self.latent_limit) - self.latent_limit

    def serialize_design(self, latent: torch.Tensor) -> str:
        if latent.ndim != 1 or latent.numel() != 5:
            raise ValueError("a serialized design must be a one-dimensional five-value latent")
        levels = self.quantize(latent).tolist()
        fields = [f"<U{i}><LEVEL_{level:03d}></U{i}>" for i, level in enumerate(levels)]
        return "\n".join(fields) + "\n</DESIGN>"
