from __future__ import annotations

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.losses import SmoothTchebycheff
from gradcell.physics import PyBaMMBackend


def hard_cutoff_metrics(
    latent: torch.Tensor,
    model_name: str,
    capacity_formula: str,
    time_points: int = 151,
    calibration_rate: float = 0.1,
    calibration_iterations: int = 2,
    capacity_multiplier: float = 1.0,
) -> dict[str, np.ndarray]:
    """Evaluate decoded designs with physical voltage cutoffs enabled."""
    decoder = DesignSpace(
        capacity_formula=capacity_formula,
        capacity_multiplier=capacity_multiplier,
    )
    design = decoder(latent.detach().cpu())
    base = design.physics_tensor(1.0).detach().numpy()
    mass = design.stack_mass_kg.detach().numpy()
    calibration_backend = PyBaMMBackend(
        model_name=model_name,
        horizon_s=1.5 * 3600.0 / calibration_rate,
        time_points=time_points,
        calculate_sensitivities=False,
        current_ramp_time_s=0.0,
        physical_voltage_cutoffs=True,
    )
    backends = {
        "1c": PyBaMMBackend(
            model_name=model_name,
            horizon_s=1.5 * 3600.0,
            time_points=time_points,
            calculate_sensitivities=False,
            current_ramp_time_s=0.0,
            physical_voltage_cutoffs=True,
        ),
        "3c": PyBaMMBackend(
            model_name=model_name,
            horizon_s=1.5 * 1200.0,
            time_points=time_points,
            calculate_sensitivities=False,
            current_ramp_time_s=0.0,
            physical_voltage_cutoffs=True,
        ),
    }
    reference_capacity = design.nominal_capacity_ah.detach().numpy().copy()
    calibration = None
    for _ in range(calibration_iterations):
        calibration_inputs = base.copy()
        calibration_inputs[:, -1] = calibration_rate * reference_capacity
        calibration = calibration_backend.solve_discharge_batch(calibration_inputs)
        valid = calibration.status == 1
        reference_capacity[valid] = calibration.delivered_capacity_ah[valid]
    assert calibration is not None
    calibration_cutoff = np.asarray(
        [row["reached_voltage_cutoff"] for row in calibration_backend.last_solve_diagnostics]
    )
    results = {}
    cutoffs = {}
    for rate_name, rate in (("1c", 1.0), ("3c", 3.0)):
        inputs = base.copy()
        inputs[:, -1] = rate * reference_capacity
        results[rate_name] = backends[rate_name].solve_discharge_batch(inputs)
        cutoffs[rate_name] = np.asarray(
            [row["reached_voltage_cutoff"] for row in backends[rate_name].last_solve_diagnostics]
        )
    result_1c, result_3c = results["1c"], results["3c"]
    status = (
        (calibration.status == 1)
        & calibration_cutoff
        & (result_1c.status == 1)
        & cutoffs["1c"]
        & (result_3c.status == 1)
        & cutoffs["3c"]
    )
    duration_3c_h = np.maximum(result_3c.discharge_time_s / 3600.0, 1e-12)
    return {
        "status": status.astype(np.int64),
        "reference_capacity_ah": reference_capacity,
        "energy_wh_kg": result_1c.delivered_energy_wh / mass,
        "power_w_kg": result_3c.delivered_energy_wh / duration_3c_h / mass,
        "capacity_retention_3c": result_3c.delivered_capacity_ah
        / np.maximum(result_1c.delivered_capacity_ah, 1e-12),
        "capacity_1c_ah": result_1c.delivered_capacity_ah,
        "capacity_3c_ah": result_3c.delivered_capacity_ah,
        "time_1c_s": result_1c.discharge_time_s,
        "time_3c_s": result_3c.discharge_time_s,
    }


def scalarized_loss(
    energy: np.ndarray,
    retention: np.ndarray,
    preferences: np.ndarray,
    bounds: dict[str, float],
) -> np.ndarray:
    objective = SmoothTchebycheff(**bounds).double()
    with torch.no_grad():
        values = objective(
            torch.from_numpy(np.asarray(energy)).double(),
            torch.from_numpy(np.asarray(retention)).double(),
            torch.from_numpy(np.asarray(preferences)).double(),
        )
    return values.numpy()
