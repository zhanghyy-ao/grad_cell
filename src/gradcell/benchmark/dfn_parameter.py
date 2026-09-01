from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAMETER_FIELDS = (
    "Positive electrode porosity",
    "Negative electrode porosity",
    "Separator porosity",
    "Positive electrode active material volume fraction",
    "Negative electrode active material volume fraction",
    "Positive particle diffusivity multiplier",
    "Negative particle diffusivity multiplier",
)


@dataclass(frozen=True)
class BenchmarkFilter:
    min_capacity_change_fraction: float = 0.01
    min_voltage_rmse_v: float = 0.005

    def accepts(
        self,
        status: int,
        capacity_ah: float,
        nominal_capacity_ah: float,
        voltage_v: np.ndarray,
        nominal_voltage_v: np.ndarray,
    ) -> tuple[bool, str, float, float]:
        """Reject failed or practically indistinguishable simulations."""
        if status != 1:
            return False, "solver_or_cutoff_failure", float("nan"), float("nan")
        capacity_change = abs(capacity_ah - nominal_capacity_ah) / max(
            abs(nominal_capacity_ah), 1e-12
        )
        voltage_rmse = float(
            np.sqrt(np.mean((np.asarray(voltage_v) - np.asarray(nominal_voltage_v)) ** 2))
        )
        informative = (
            capacity_change >= self.min_capacity_change_fraction
            or voltage_rmse >= self.min_voltage_rmse_v
        )
        reason = "accepted" if informative else "indistinguishable_from_nominal"
        return informative, reason, float(capacity_change), voltage_rmse


def sample_log_multipliers(
    family: str,
    count: int,
    parameter_count: int,
    low: float,
    high: float,
    seed: int,
    multi_min_parameters: int = 2,
    multi_max_parameters: int = 4,
) -> np.ndarray:
    """Generate reproducible single- or multi-parameter perturbations."""
    if family not in {"single", "multi"}:
        raise ValueError("family must be 'single' or 'multi'")
    if count < 1 or parameter_count < 1:
        raise ValueError("count and parameter_count must be positive")
    if not (0.0 < low < 1.0 < high):
        raise ValueError("bounds must satisfy 0 < low < 1 < high")
    rng = np.random.default_rng(seed)
    result = np.ones((count, parameter_count), dtype=np.float64)
    log_low, log_high = np.log(low), np.log(high)
    for row in range(count):
        if family == "single":
            selected = np.asarray([row % parameter_count])
        else:
            maximum = min(multi_max_parameters, parameter_count)
            minimum = min(max(multi_min_parameters, 1), maximum)
            selected = rng.choice(
                parameter_count, size=int(rng.integers(minimum, maximum + 1)), replace=False
            )
        result[row, selected] = np.exp(
            rng.uniform(log_low, log_high, size=selected.size)
        )
    return result


def apply_multipliers(nominal: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    nominal = np.asarray(nominal, dtype=np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64)
    if multipliers.shape[-1] != nominal.size:
        raise ValueError("multiplier width must match nominal parameter count")
    values = multipliers * nominal
    # Volume fractions must remain physically meaningful even with custom bounds.
    values[..., :5] = np.clip(values[..., :5], 1e-4, 0.9999)
    return values


def structural_feasibility(values: np.ndarray) -> np.ndarray:
    """Check that porosity and active fraction do not exceed electrode volume."""
    values = np.asarray(values, dtype=np.float64)
    return (
        np.isfinite(values).all(axis=-1)
        & (values[..., :5] > 0.0).all(axis=-1)
        & (values[..., 5:] > 0.0).all(axis=-1)
        & (values[..., 0] + values[..., 3] <= 1.0 + 1e-10)
        & (values[..., 1] + values[..., 4] <= 1.0 + 1e-10)
    )
