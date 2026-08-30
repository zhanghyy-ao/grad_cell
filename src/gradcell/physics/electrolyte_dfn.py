from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .backend import PhysicsBatch

LOG_CONDUCTIVITY_SCALE = "Log electrolyte conductivity scale"
CURRENT_A = "Current function [A]"


@dataclass(frozen=True)
class HardCutoffResult:
    discharge_time_s: np.ndarray
    delivered_capacity_ah: np.ndarray
    delivered_energy_wh: np.ndarray
    termination: tuple[str, ...]
    status: np.ndarray


class AnalyticElectrolyteBackend:
    """Two-input analytic backend for fast tests of the electrolyte pipeline."""

    input_names = (LOG_CONDUCTIVITY_SCALE, CURRENT_A)

    def __init__(self, time_points: int = 41, probe_horizon_s: float = 600.0) -> None:
        self.t_eval = np.linspace(0.0, probe_horizon_s, time_points, dtype=np.float64)
        self.probe_horizon_s = probe_horizon_s
        self.last_solve_diagnostics: list[dict] = []

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        started = time.perf_counter()
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("Electrolyte backend expects [B, 2] = [log_scale, current_a]")
        log_scale = x[:, 0:1]
        current = x[:, 1:2]
        tau = self.t_eval[None, :] / max(self.probe_horizon_s, 1.0)
        resistance = 0.035 + 0.025 * np.exp(-log_scale)
        voltage = 4.15 - 0.55 * tau - current * resistance - 0.04 * tau**2
        jacobian = np.zeros((len(x), 1, len(self.t_eval), 2), dtype=np.float64)
        jacobian[:, 0, :, 0] = (current * 0.025 * np.exp(-log_scale)).repeat(
            len(self.t_eval), axis=1
        )
        jacobian[:, 0, :, 1] = (-resistance).repeat(len(self.t_eval), axis=1)
        elapsed = (time.perf_counter() - started) / max(len(x), 1)
        self.last_solve_diagnostics = [
            {
                "model": "analytic-electrolyte",
                "physical_voltage_cutoffs": True,
                "probe_horizon_s": self.probe_horizon_s,
                "termination": "final time",
                "error": None,
            }
            for _ in x
        ]
        return PhysicsBatch(
            trajectories=voltage[:, None, :],
            jacobian=jacobian,
            status=np.ones(len(x), dtype=np.int64),
            runtime_s=np.full(len(x), elapsed, dtype=np.float64),
        )


class PyBaMMElectrolyteDFNBackend:
    """Chen2020 DFN with a differentiable electrolyte-conductivity correction.

    The differentiable probe retains the physical voltage events, but evaluates a
    short common observation window that is expected to end before cutoff. This
    avoids pretending that IDAKLU provides the derivative of event time. Full
    discharge-to-cutoff evaluation is exposed separately by ``solve_hard_cutoff``.
    """

    input_names = (LOG_CONDUCTIVITY_SCALE, CURRENT_A)

    def __init__(
        self,
        time_points: int = 41,
        probe_horizon_s: float = 600.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        calculate_sensitivities: bool = True,
        current_ramp_time_s: float = 0.0,
    ) -> None:
        try:
            import pybamm
        except ImportError as exc:
            raise ImportError("Install GradCell with `pip install -e .[physics]`") from exc
        if probe_horizon_s <= 0.0:
            raise ValueError("probe_horizon_s must be positive")
        if current_ramp_time_s < 0.0:
            raise ValueError("current_ramp_time_s must be non-negative")
        self.pybamm = pybamm
        self.model = pybamm.lithium_ion.DFN()
        self.parameters = pybamm.ParameterValues("Chen2020")
        base_conductivity = self.parameters["Electrolyte conductivity [S.m-1]"]

        def scaled_conductivity(concentration, temperature):
            base = (
                base_conductivity(concentration, temperature)
                if callable(base_conductivity)
                else base_conductivity
            )
            return base * pybamm.exp(pybamm.InputParameter(LOG_CONDUCTIVITY_SCALE))

        if current_ramp_time_s == 0.0:
            current_function = pybamm.InputParameter(CURRENT_A)
        else:

            def current_function(time_value):
                ramp = 1.0 - pybamm.exp(-time_value / current_ramp_time_s)
                return pybamm.InputParameter(CURRENT_A) * ramp

        self.parameters.update(
            {
                "Electrolyte conductivity [S.m-1]": scaled_conductivity,
                CURRENT_A: current_function,
            }
        )
        self.solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        self.simulation = pybamm.Simulation(
            self.model,
            parameter_values=self.parameters,
            solver=self.solver,
        )
        self.simulation.build()
        self.t_eval = np.linspace(0.0, probe_horizon_s, time_points, dtype=np.float64)
        self.probe_horizon_s = probe_horizon_s
        self.calculate_sensitivities = calculate_sensitivities
        self.current_ramp_time_s = current_ramp_time_s
        self.last_solve_diagnostics: list[dict] = []

    def _sensitivity(self, variable, name: str, solution_time: np.ndarray) -> np.ndarray:
        value = variable.sensitivities[name]
        if hasattr(value, "entries"):
            value = value.entries
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        solution_time = np.asarray(solution_time, dtype=np.float64).reshape(-1)
        if value.size == self.t_eval.size:
            return value
        if value.size != solution_time.size:
            raise ValueError(
                f"Sensitivity {name!r} has {value.size} values for "
                f"{solution_time.size} solution times"
            )
        return np.interp(self.t_eval, solution_time, value)

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("Electrolyte DFN expects [B, 2] inputs")
        trajectories, jacobians, statuses, runtimes, diagnostics = [], [], [], [], []
        for row in x:
            started = time.perf_counter()
            try:
                input_dict = dict(zip(self.input_names, row.tolist()))
                solution = self.simulation.solve(
                    self.t_eval,
                    inputs=input_dict,
                    calculate_sensitivities=(
                        list(self.input_names) if self.calculate_sensitivities else False
                    ),
                )
                actual_end = float(np.asarray(solution.t).reshape(-1)[-1])
                termination = str(getattr(solution, "termination", "unknown"))
                completed_probe = actual_end >= self.probe_horizon_s - 1e-8
                variable = solution["Voltage [V]"]
                voltage = np.asarray(variable(self.t_eval), dtype=np.float64).reshape(-1)
                if self.calculate_sensitivities:
                    jacobian = np.stack(
                        [self._sensitivity(variable, name, solution.t) for name in self.input_names],
                        axis=-1,
                    )
                else:
                    jacobian = np.zeros((len(self.t_eval), 2), dtype=np.float64)
                finite = bool(np.isfinite(voltage).all() and np.isfinite(jacobian).all())
                success = completed_probe and finite
                trajectories.append(voltage[None, :] if success else np.zeros((1, len(voltage))))
                jacobians.append(
                    jacobian[None, :, :] if success else np.zeros((1, len(voltage), 2))
                )
                statuses.append(int(success))
                diagnostics.append(
                    {
                        "model": "DFN",
                        "parameter_set": "Chen2020",
                        "physical_voltage_cutoffs": True,
                        "probe_horizon_s": self.probe_horizon_s,
                        "actual_end_time_s": actual_end,
                        "completed_probe": completed_probe,
                        "termination": termination,
                        "current_ramp_time_s": self.current_ramp_time_s,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                trajectories.append(np.zeros((1, len(self.t_eval)), dtype=np.float64))
                jacobians.append(np.zeros((1, len(self.t_eval), 2), dtype=np.float64))
                statuses.append(0)
                diagnostics.append(
                    {
                        "model": "DFN",
                        "parameter_set": "Chen2020",
                        "physical_voltage_cutoffs": True,
                        "probe_horizon_s": self.probe_horizon_s,
                        "actual_end_time_s": None,
                        "completed_probe": False,
                        "termination": "exception",
                        "current_ramp_time_s": self.current_ramp_time_s,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            runtimes.append(time.perf_counter() - started)
        self.last_solve_diagnostics = diagnostics
        return PhysicsBatch(
            trajectories=np.stack(trajectories),
            jacobian=np.stack(jacobians),
            status=np.asarray(statuses, dtype=np.int64),
            runtime_s=np.asarray(runtimes, dtype=np.float64),
        )

    def solve_hard_cutoff(
        self, inputs: np.ndarray, maximum_time_s: float = 7200.0
    ) -> HardCutoffResult:
        x = np.asarray(inputs, dtype=np.float64)
        durations, capacities, energies, terminations, statuses = [], [], [], [], []
        for row in x:
            try:
                solution = self.simulation.solve(
                    np.array([0.0, maximum_time_s]),
                    inputs=dict(zip(self.input_names, row.tolist())),
                    calculate_sensitivities=False,
                )
                time_s = np.asarray(solution.t, dtype=np.float64).reshape(-1)
                voltage = np.asarray(solution["Voltage [V]"].entries, dtype=np.float64).reshape(-1)
                current = np.asarray(solution["Current [A]"].entries, dtype=np.float64).reshape(-1)
                integrate = getattr(np, "trapezoid", np.trapz)
                capacity = float(integrate(np.clip(current, 0.0, None), time_s) / 3600.0)
                energy = float(integrate(np.clip(current, 0.0, None) * voltage, time_s) / 3600.0)
                termination = str(getattr(solution, "termination", "unknown"))
                valid = capacity > 0.0 and np.isfinite(energy)
                durations.append(float(time_s[-1] - time_s[0]))
                capacities.append(capacity)
                energies.append(energy)
                terminations.append(termination)
                statuses.append(int(valid))
            except Exception as exc:  # noqa: BLE001
                durations.append(0.0)
                capacities.append(0.0)
                energies.append(0.0)
                terminations.append(f"{type(exc).__name__}: {exc}")
                statuses.append(0)
        return HardCutoffResult(
            discharge_time_s=np.asarray(durations),
            delivered_capacity_ah=np.asarray(capacities),
            delivered_energy_wh=np.asarray(energies),
            termination=tuple(terminations),
            status=np.asarray(statuses, dtype=np.int64),
        )
