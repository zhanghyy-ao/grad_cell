from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from gradcell.design import CellDesign, DesignSpace


DESIGN_KEYS = (
    "positive_electrode_porosity",
    "negative_electrode_porosity",
    "separator_porosity",
    "positive_active_material_fraction",
    "negative_to_positive_capacity_ratio",
)


@dataclass(frozen=True)
class ParsedMaterialDesign:
    eps_p: float
    eps_n: float
    eps_s: float
    phi_p: float
    np_ratio: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(DESIGN_KEYS, (self.eps_p, self.eps_n, self.eps_s, self.phi_p, self.np_ratio)))


class MaterialDesignJSONCodec:
    """Canonical JSON representation and strict validation for GradCell designs."""

    def __init__(self, design_space: DesignSpace | None = None, precision: int = 8) -> None:
        self.design_space = design_space or DesignSpace()
        self.precision = precision

    def design_dict(self, design: CellDesign, index: int = 0) -> dict[str, object]:
        def value(tensor: torch.Tensor) -> float:
            flat = tensor.detach().cpu().reshape(-1)
            return round(float(flat[index]), self.precision)

        return {
            "schema": "gradcell.material_design.v1",
            "material_parameter_set": "Chen2020",
            "fixed_material_properties": True,
            "design": {
                "positive_electrode_porosity": value(design.eps_p),
                "negative_electrode_porosity": value(design.eps_n),
                "separator_porosity": value(design.eps_s),
                "positive_active_material_fraction": value(design.phi_p),
                "negative_to_positive_capacity_ratio": value(design.np_ratio),
            },
            "derived": {
                "negative_active_material_fraction": value(design.phi_n),
                "nominal_capacity_ah": value(design.nominal_capacity_ah),
                "stack_mass_kg": value(design.stack_mass_kg),
            },
        }

    def dumps_design(self, design: CellDesign, index: int = 0) -> str:
        return json.dumps(
            self.design_dict(design, index), ensure_ascii=False, separators=(",", ":")
        )

    def dumps_latent(self, latent: torch.Tensor, index: int = 0) -> str:
        design = self.design_space(latent)
        return self.dumps_design(design, index=index)

    @staticmethod
    def extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model output does not contain a JSON object")
        return text[start : end + 1]

    def loads(self, text: str) -> ParsedMaterialDesign:
        payload = json.loads(self.extract_json(text))
        if payload.get("schema") != "gradcell.material_design.v1":
            raise ValueError("unsupported or missing material-design schema")
        values = payload.get("design")
        if not isinstance(values, dict) or tuple(values.keys()) != DESIGN_KEYS:
            raise ValueError("design fields are missing, reordered, or unexpected")
        parsed = ParsedMaterialDesign(*(float(values[key]) for key in DESIGN_KEYS))
        self.validate(parsed)
        return parsed

    def validate(self, design: ParsedMaterialDesign) -> None:
        space = self.design_space
        bounded = (
            (design.eps_p, space.eps_p_bounds, "positive electrode porosity"),
            (design.eps_n, space.eps_n_bounds, "negative electrode porosity"),
            (design.eps_s, space.eps_s_bounds, "separator porosity"),
            (design.np_ratio, space.np_bounds, "N/P ratio"),
        )
        for value, (lower, upper), name in bounded:
            if not lower <= value <= upper:
                raise ValueError(f"{name}={value} lies outside [{lower}, {upper}]")
        constants = space.capacity_constants
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
        kappa = design.np_ratio * numerator / denominator
        phi_p_max = min(
            1.0 - design.eps_p - space.inactive_p_min,
            (1.0 - design.eps_n - space.inactive_n_min) / kappa,
        )
        if not space.phi_p_min <= design.phi_p <= phi_p_max:
            raise ValueError(
                f"positive active fraction={design.phi_p} lies outside the coupled feasible "
                f"interval [{space.phi_p_min}, {phi_p_max}]"
            )


def differentiable_design_penalty(design: CellDesign, space: DesignSpace) -> torch.Tensor:
    """Soft diagnostic penalty; zero for a hard-feasible decoded design."""

    penalties = [
        torch.relu(space.eps_p_bounds[0] - design.eps_p),
        torch.relu(design.eps_p - space.eps_p_bounds[1]),
        torch.relu(space.eps_n_bounds[0] - design.eps_n),
        torch.relu(design.eps_n - space.eps_n_bounds[1]),
        torch.relu(space.eps_s_bounds[0] - design.eps_s),
        torch.relu(design.eps_s - space.eps_s_bounds[1]),
        torch.relu(space.phi_p_min - design.phi_p),
        torch.relu(design.phi_p + design.eps_p + space.inactive_p_min - 1.0),
        torch.relu(design.phi_n + design.eps_n + space.inactive_n_min - 1.0),
        torch.relu(space.np_bounds[0] - design.np_ratio),
        torch.relu(design.np_ratio - space.np_bounds[1]),
    ]
    return torch.stack([value.square() for value in penalties], dim=-1).mean(dim=-1)
