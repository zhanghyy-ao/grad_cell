from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from gradcell.design.mass_model import MassConstants, stack_mass_kg


@dataclass(frozen=True)
class PhysicalNominalDesign:
    """A parameter-set design that bypasses GradCell's latent decoder."""

    parameter_set: str
    physics_inputs: np.ndarray
    initial_capacity_ah: np.ndarray
    stack_mass_kg: np.ndarray
    parameters: dict[str, float]


REQUIRED_PARAMETERS = (
    "Positive electrode porosity",
    "Negative electrode porosity",
    "Separator porosity",
    "Positive electrode active material volume fraction",
    "Negative electrode active material volume fraction",
    "Nominal cell capacity [A.h]",
)


def nominal_design_from_parameter_values(
    parameter_values: Mapping[str, object],
    *,
    parameter_set: str = "Chen2020",
    mass_constants: MassConstants | None = None,
) -> PhysicalNominalDesign:
    """Build the physical nominal baseline from unmodified parameter values."""
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameter_values]
    if missing:
        raise KeyError(f"Parameter set lacks required nominal fields: {missing}")
    values = {name: float(parameter_values[name]) for name in REQUIRED_PARAMETERS}
    eps_p = values["Positive electrode porosity"]
    eps_n = values["Negative electrode porosity"]
    eps_s = values["Separator porosity"]
    phi_p = values["Positive electrode active material volume fraction"]
    phi_n = values["Negative electrode active material volume fraction"]
    capacity = values["Nominal cell capacity [A.h]"]
    fractions = (
        ("eps_p", eps_p),
        ("eps_n", eps_n),
        ("eps_s", eps_s),
        ("phi_p", phi_p),
        ("phi_n", phi_n),
    )
    for name, value in fractions:
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0,1), got {value}")
    if capacity <= 0.0:
        raise ValueError(f"Nominal capacity must be positive, got {capacity}")

    tensor = lambda value: torch.tensor([value], dtype=torch.float64)
    mass = stack_mass_kg(
        tensor(eps_p),
        tensor(eps_n),
        tensor(eps_s),
        tensor(phi_p),
        tensor(phi_n),
        mass_constants or MassConstants(),
    ).detach().numpy()
    physics_inputs = np.asarray(
        [[eps_p, eps_n, eps_s, phi_p, phi_n, 1.0, 1.0, capacity]],
        dtype=np.float64,
    )
    return PhysicalNominalDesign(
        parameter_set=parameter_set,
        physics_inputs=physics_inputs,
        initial_capacity_ah=np.asarray([capacity], dtype=np.float64),
        stack_mass_kg=mass,
        parameters=values,
    )


def load_pybamm_nominal_design(
    parameter_set: str = "Chen2020",
) -> PhysicalNominalDesign:
    """Read a physical baseline directly from a PyBaMM parameter set."""
    try:
        import pybamm
    except ImportError as exc:
        raise ImportError("Install GradCell with `pip install -e .[physics]`") from exc
    values = pybamm.ParameterValues(parameter_set)
    return nominal_design_from_parameter_values(values, parameter_set=parameter_set)
