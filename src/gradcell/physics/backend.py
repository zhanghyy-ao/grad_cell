from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PhysicsBatch:
    trajectories: np.ndarray
    jacobian: np.ndarray
    status: np.ndarray
    runtime_s: np.ndarray


class PhysicsBackend(Protocol):
    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch: ...


class AnalyticToyBackend:
    """Fast differentiable surrogate used to test the complete training loop.

    Input order matches the PyBaMM backend. Output is a voltage trajectory with
    a closed-form Jacobian, so autograd tests do not require PyBaMM.
    """

    def __init__(self, time_points: int = 101, horizon_s: float = 3600.0) -> None:
        self.t = np.linspace(0.0, 1.0, time_points, dtype=np.float64)
        self.horizon_s = horizon_s

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        import time

        started = time.perf_counter()
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError("Toy backend expects [B, 8] physics inputs")
        b = x.shape[0]
        t = self.t[None, :]
        eps_p, eps_n, eps_s, phi_p, phi_n, dp, dn, current = [x[:, i : i + 1] for i in range(8)]
        resistance = (
            0.10
            + 0.10 * (0.32 - eps_p) ** 2
            + 0.10 * (0.32 - eps_n) ** 2
            + 0.02 / eps_s
            + 0.018 / dp
            + 0.015 / dn
        )
        capacity_factor = 0.7 * phi_p + 0.3 * phi_n
        voltage = (
            4.2
            - 1.15 * t
            - current * resistance
            - 0.14 * t**2 / capacity_factor
            + 0.02 * np.sin(np.pi * t)
        )
        jac = np.zeros((b, 1, self.t.size, 8), dtype=np.float64)
        dres_deps_p = 0.20 * (eps_p - 0.32)
        dres_deps_n = 0.20 * (eps_n - 0.32)
        dres_deps_s = -0.02 / eps_s**2
        jac[:, 0, :, 0] = (-current * dres_deps_p).repeat(self.t.size, axis=1)
        jac[:, 0, :, 1] = (-current * dres_deps_n).repeat(self.t.size, axis=1)
        jac[:, 0, :, 2] = (-current * dres_deps_s).repeat(self.t.size, axis=1)
        jac[:, 0, :, 3] = 0.14 * t**2 * 0.7 / capacity_factor**2
        jac[:, 0, :, 4] = 0.14 * t**2 * 0.3 / capacity_factor**2
        jac[:, 0, :, 5] = (current * 0.018 / dp**2).repeat(self.t.size, axis=1)
        jac[:, 0, :, 6] = (current * 0.015 / dn**2).repeat(self.t.size, axis=1)
        jac[:, 0, :, 7] = (-resistance).repeat(self.t.size, axis=1)
        elapsed = time.perf_counter() - started
        return PhysicsBatch(
            trajectories=voltage[:, None, :],
            jacobian=jac,
            status=np.ones(b, dtype=np.int64),
            runtime_s=np.full(b, elapsed / max(b, 1), dtype=np.float64),
        )


class PyBaMMBackend:
    """SPMe trajectory and forward-sensitivity backend.

    This wrapper intentionally uses only non-geometric parameters supported as
    InputParameter by standard PyBaMM meshes.
    """

    input_names = (
        "Positive electrode porosity",
        "Negative electrode porosity",
        "Separator porosity",
        "Positive electrode active material volume fraction",
        "Negative electrode active material volume fraction",
        "Positive particle diffusivity multiplier",
        "Negative particle diffusivity multiplier",
        "Current function [A]",
    )

    def __init__(
        self,
        model_name: str = "SPMe",
        parameter_set: str = "Chen2020",
        output_variables: tuple[str, ...] = ("Voltage [V]",),
        time_points: int = 151,
        horizon_s: float = 3600.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> None:
        try:
            import pybamm
        except ImportError as exc:
            raise ImportError("Install GradCell with `pip install -e .[physics]`") from exc
        self.pybamm = pybamm
        model_cls = getattr(pybamm.lithium_ion, model_name)
        self.model = model_cls()
        self.parameters = pybamm.ParameterValues(parameter_set)
        # Training uses a fixed horizon and a smooth voltage gate in PyTorch.
        # Move hard voltage events away from the operating region so their event
        # time does not create missing trajectory tails or discontinuous gradients.
        self.parameters.update(
            {
                "Lower voltage cut-off [V]": 0.0,
                "Upper voltage cut-off [V]": 10.0,
            }
        )
        direct_inputs = self.input_names[:5] + self.input_names[7:]
        for name in direct_inputs:
            self.parameters.update({name: pybamm.InputParameter(name)})
        positive_diffusivity = self.parameters["Positive particle diffusivity [m2.s-1]"]
        negative_diffusivity = self.parameters["Negative particle diffusivity [m2.s-1]"]

        def scaled_positive_diffusivity(stoichiometry, temperature):
            base = (
                positive_diffusivity(stoichiometry, temperature)
                if callable(positive_diffusivity)
                else positive_diffusivity
            )
            return base * pybamm.InputParameter(self.input_names[5])

        def scaled_negative_diffusivity(stoichiometry, temperature):
            base = (
                negative_diffusivity(stoichiometry, temperature)
                if callable(negative_diffusivity)
                else negative_diffusivity
            )
            return base * pybamm.InputParameter(self.input_names[6])

        self.parameters.update(
            {
                "Positive particle diffusivity [m2.s-1]": scaled_positive_diffusivity,
                "Negative particle diffusivity [m2.s-1]": scaled_negative_diffusivity,
            }
        )
        self.solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        self.t_eval = np.linspace(0.0, horizon_s, time_points)
        self.output_variables = output_variables
        self.simulation = pybamm.Simulation(
            self.model,
            parameter_values=self.parameters,
            solver=self.solver,
        )
        self.simulation.build()

    def _extract_sensitivity(self, variable, name: str, solution_time: np.ndarray) -> np.ndarray:
        sensitivities = variable.sensitivities
        if name not in sensitivities:
            raise KeyError(f"No sensitivity for {name!r}; available: {list(sensitivities)}")
        value = sensitivities[name]
        if callable(value):
            value = value(self.t_eval)
        elif hasattr(value, "entries"):
            value = value.entries
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        solution_time = np.asarray(solution_time, dtype=np.float64).reshape(-1)
        if value.size != self.t_eval.size:
            if value.size != solution_time.size:
                raise ValueError(
                    f"Sensitivity has {value.size} points but solution has {solution_time.size}"
                )
            value = np.interp(self.t_eval, solution_time, value)
        return value

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        import time

        trajectories, jacobians, statuses, runtimes = [], [], [], []
        for row in np.asarray(inputs, dtype=np.float64):
            started = time.perf_counter()
            try:
                input_dict = dict(zip(self.input_names, row.tolist()))
                solution = self.simulation.solve(
                    self.t_eval,
                    inputs=input_dict,
                    calculate_sensitivities=list(self.input_names),
                )
                output_rows, sensitivity_rows = [], []
                for output_name in self.output_variables:
                    variable = solution[output_name]
                    values = np.asarray(variable(self.t_eval), dtype=np.float64).reshape(-1)
                    output_rows.append(values)
                    sensitivity_rows.append(
                        np.stack(
                            [
                                self._extract_sensitivity(variable, name, solution.t)
                                for name in self.input_names
                            ],
                            axis=-1,
                        )
                    )
                trajectories.append(np.stack(output_rows, axis=0))
                jacobians.append(np.stack(sensitivity_rows, axis=0))
                statuses.append(1)
            # Numerical solver failures are data in this experiment: convert them
            # into status=0 and let the PyTorch loss apply a recovery barrier.
            except Exception:  # noqa: BLE001
                shape = (len(self.output_variables), self.t_eval.size)
                trajectories.append(np.full(shape, np.nan, dtype=np.float64))
                jacobians.append(np.zeros((*shape, len(self.input_names)), dtype=np.float64))
                statuses.append(0)
            runtimes.append(time.perf_counter() - started)
        return PhysicsBatch(
            trajectories=np.stack(trajectories),
            jacobian=np.stack(jacobians),
            status=np.asarray(statuses, dtype=np.int64),
            runtime_s=np.asarray(runtimes, dtype=np.float64),
        )
