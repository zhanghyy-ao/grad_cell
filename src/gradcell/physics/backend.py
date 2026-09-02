from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PhysicsBatch:
    """一次 batch 物理求解的统一返回结构。"""

    trajectories: np.ndarray  # 物理输出轨迹
    jacobian: np.ndarray  # 轨迹对输入参数的 雅可比矩阵（真的可以求得吗？）
    status: np.ndarray  # 求解是否成功
    runtime_s: np.ndarray  # 每个样本的求解时间


@dataclass(frozen=True)
class DischargeBatch:
    """Non-differentiable physical-discharge summaries for supervised labels."""

    delivered_capacity_ah: np.ndarray
    delivered_energy_wh: np.ndarray
    average_voltage_v: np.ndarray
    minimum_voltage_v: np.ndarray
    discharge_time_s: np.ndarray
    status: np.ndarray
    runtime_s: np.ndarray


@dataclass(frozen=True)
class NormalizedDischargeBatch:
    """Physical-cutoff discharge curves resampled on normalized cycle time."""

    normalized_time: np.ndarray
    voltage_v: np.ndarray
    discharge_time_s: np.ndarray
    delivered_capacity_ah: np.ndarray
    status: np.ndarray
    runtime_s: np.ndarray


def summarize_discharge(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
) -> dict[str, float]:
    """Integrate a discharge trajectory using its actual termination time."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    voltage_v = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
    current_a = np.asarray(current_a, dtype=np.float64).reshape(-1)
    if not (time_s.size == voltage_v.size == current_a.size) or time_s.size < 2:
        raise ValueError("Discharge trajectory arrays must have the same length >= 2")
    if not (
        np.isfinite(time_s).all() and np.isfinite(voltage_v).all() and np.isfinite(current_a).all()
    ):
        raise ValueError("Discharge trajectory contains non-finite values")
    discharge_current = np.clip(current_a, 0.0, None)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    capacity_ah = float(integrate(discharge_current, time_s) / 3600.0)
    energy_wh = float(integrate(discharge_current * voltage_v, time_s) / 3600.0)
    return {
        "delivered_capacity_ah": capacity_ah,
        "delivered_energy_wh": energy_wh,
        "average_voltage_v": energy_wh / max(capacity_ah, 1e-12),
        "minimum_voltage_v": float(voltage_v.min()),
        "discharge_time_s": float(time_s[-1] - time_s[0]),
    }


class PhysicsBackend(Protocol):
    """所有物理后端必须实现的最小求解接口。"""

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch: ...


class AnalyticToyBackend:
    """Fast differentiable surrogate used to test the complete training loop.

    Input order matches the PyBaMM backend. Output is a voltage trajectory with
    a closed-form Jacobian, so autograd tests do not require PyBaMM.
    """

    def __init__(self, time_points: int = 101, horizon_s: float = 3600.0) -> None:
        # 归一化时间网格；horizon_s 由上层性能指标换算为实际时间步长。
        self.t = np.linspace(0.0, 1.0, time_points, dtype=np.float64)
        self.horizon_s = horizon_s
        self.last_solve_diagnostics: list[dict] = []

    def solve_batch(self, inputs: np.ndarray) -> PhysicsBatch:
        import time

        started = time.perf_counter()
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError("Toy backend expects [B, 8] physics inputs")
        b = x.shape[0]
        t = self.t[None, :]
        # 输入顺序：正/负极孔隙率、隔膜孔隙率、正/负极活性比例、
        # 正/负极扩散率乘子和电流。
        eps_p, eps_n, eps_s, phi_p, phi_n, dp, dn, current = [x[:, i : i + 1] for i in range(8)]
        # 解析 surrogate 的等效内阻。
        resistance = (
            0.10
            + 0.10 * (0.32 - eps_p) ** 2
            + 0.10 * (0.32 - eps_n) ** 2
            + 0.02 / eps_s
            + 0.018 / dp
            + 0.015 / dn
        )
        # 用正负极活性材料比例构造简化容量因子。
        capacity_factor = 0.7 * phi_p + 0.3 * phi_n
        # 生成人工电压轨迹，仅用于软件/梯度测试，不代表真实电化学结果。
        voltage = (
            4.2
            - 1.15 * t
            - current * resistance
            - 0.14 * t**2 / capacity_factor
            + 0.02 * np.sin(np.pi * t)
        )
        # 对人工公式逐输入求导，提供自定义 autograd 所需 Jacobian。
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
        self.last_solve_diagnostics = [
            {
                "requested_end_time_s": self.horizon_s,
                "actual_end_time_s": self.horizon_s,
                "completed_requested_horizon": True,
                "termination": "analytic toy trajectory completed",
                "error": None,
            }
            for _ in range(b)
        ]
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
        calculate_sensitivities: bool = True,
        current_ramp_time_s: float = 1.0,
        physical_voltage_cutoffs: bool = False,
        training_voltage_floor_v: float = -10.0,
    ) -> None:
        # 延迟导入 PyBaMM，使 toy 后端和基础测试不依赖该可选依赖。
        try:
            import pybamm
        except ImportError as exc:
            raise ImportError("Install GradCell with `pip install -e .[physics]`") from exc
        self.pybamm = pybamm
        if current_ramp_time_s < 0.0:
            raise ValueError("current_ramp_time_s must be non-negative")
        self.current_ramp_time_s = current_ramp_time_s
        # 创建指定锂离子模型和参数集。
        model_cls = getattr(pybamm.lithium_ion, model_name)
        self.model = model_cls()
        self.parameters = pybamm.ParameterValues(parameter_set)
        self.physical_voltage_cutoffs = physical_voltage_cutoffs
        if not physical_voltage_cutoffs:
            # Differentiable training uses a fixed horizon and a smooth voltage gate.
            # Physical supervised runs retain the parameter-set voltage events.
            self.parameters.update(
                {
                    "Lower voltage cut-off [V]": training_voltage_floor_v,
                    "Upper voltage cut-off [V]": 10.0,
                }
            )
        direct_inputs = self.input_names[:5]
        self.nominal_input_values = np.asarray(
            [float(self.parameters[name]) for name in direct_inputs] + [1.0, 1.0],
            dtype=np.float64,
        )
        for name in direct_inputs:
            # 将孔隙率、活性比例和电流暴露为运行时输入。
            self.parameters.update({name: pybamm.InputParameter(name)})
        if current_ramp_time_s == 0.0:
            self.parameters.update(
                {self.input_names[7]: pybamm.InputParameter(self.input_names[7])}
            )
        else:

            def smooth_current(time):
                target = pybamm.InputParameter(self.input_names[7])
                ramp = 1.0 - pybamm.exp(-time / current_ramp_time_s)
                return target * ramp

            self.parameters.update({self.input_names[7]: smooth_current})
        positive_diffusivity = self.parameters["Positive particle diffusivity [m2.s-1]"]
        negative_diffusivity = self.parameters["Negative particle diffusivity [m2.s-1]"]

        def scaled_positive_diffusivity(stoichiometry, temperature):
            # 保留 Chen2020 的浓度/温度依赖，再乘以可优化的扩散率因子。
            base = (
                positive_diffusivity(stoichiometry, temperature)
                if callable(positive_diffusivity)
                else positive_diffusivity
            )
            return base * pybamm.InputParameter(self.input_names[5])

        def scaled_negative_diffusivity(stoichiometry, temperature):
            # 对负极扩散率执行与正极相同的基准值缩放。
            base = (
                negative_diffusivity(
                    stoichiometry, temperature
                )  # stoichiometry 当前材料颗粒内部已经填充了锂离子的“空位”占所有可用空位的百分比。
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
        self.horizon_s = horizon_s
        self.output_variables = output_variables
        self.calculate_sensitivities = calculate_sensitivities
        self.last_solve_diagnostics: list[dict] = []
        self.simulation = pybamm.Simulation(
            self.model,
            parameter_values=self.parameters,
            solver=self.solver,
        )
        self.simulation.build()

    def _extract_sensitivity(self, variable, name: str, solution_time: np.ndarray) -> np.ndarray:
        """提取单个输出对输入的 sensitivity，并插值到固定时间网格。"""
        sensitivities = variable.sensitivities
        if name not in sensitivities:
            raise KeyError(f"No sensitivity for {name!r}; available: {list(sensitivities)}")
        value = sensitivities[name]
        if callable(value):
            value = value(solution_time)
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
        """逐样本调用 PyBaMM，返回轨迹和对 8 个输入的 forward sensitivity。"""
        import time

        trajectories, jacobians, statuses, runtimes, diagnostics = [], [], [], [], []
        for row in np.asarray(inputs, dtype=np.float64):
            started = time.perf_counter()
            try:
                # 将一行输入转换为 PyBaMM 参数名到数值的字典。
                input_dict = dict(zip(self.input_names, row.tolist()))
                solution = self.simulation.solve(
                    self.t_eval,
                    inputs=input_dict,
                    calculate_sensitivities=(
                        list(self.input_names) if self.calculate_sensitivities else False
                    ),
                )
                actual_end_time_s = float(np.asarray(solution.t).reshape(-1)[-1])
                output_rows, sensitivity_rows = [], []
                for output_name in self.output_variables:
                    # 逐输出提取轨迹，并堆叠其对所有输入的导数。
                    variable = solution[output_name]
                    raw_values = np.asarray(variable(solution.t), dtype=np.float64).reshape(-1)
                    values = np.interp(self.t_eval, solution.t, raw_values)
                    output_rows.append(values)
                    if self.calculate_sensitivities:
                        sensitivity_rows.append(
                            np.stack(
                                [
                                    self._extract_sensitivity(variable, name, solution.t)
                                    for name in self.input_names
                                ],
                                axis=-1,
                            )
                        )
                    else:
                        sensitivity_rows.append(
                            np.zeros((self.t_eval.size, len(self.input_names)), dtype=np.float64)
                        )
                trajectory = np.stack(output_rows, axis=0)
                jacobian = np.stack(sensitivity_rows, axis=0)
                termination = str(getattr(solution, "termination", "unknown"))
                safely_depleted = termination.startswith("event: Minimum voltage")
                completed_horizon = actual_end_time_s >= self.horizon_s - 1e-8
                trajectory_finite = bool(np.isfinite(trajectory).all())
                jacobian_finite = bool(np.isfinite(jacobian).all())
                success = (
                    (completed_horizon or safely_depleted)
                    and trajectory_finite
                    and (not self.calculate_sensitivities or jacobian_finite)
                )
                if success:
                    trajectories.append(trajectory)
                    jacobians.append(jacobian)
                else:
                    # Finite placeholders cross into PyTorch, but status=0 prevents
                    # callers from treating them as physical labels.
                    trajectories.append(np.zeros_like(trajectory))
                    jacobians.append(np.zeros_like(jacobian))
                statuses.append(int(success))
                diagnostics.append(
                    {
                        "requested_end_time_s": self.horizon_s,
                        "actual_end_time_s": actual_end_time_s,
                        "completed_requested_horizon": bool(completed_horizon),
                        "padded_after_safe_depletion": bool(
                            safely_depleted and not completed_horizon
                        ),
                        "trajectory_finite": trajectory_finite,
                        "jacobian_finite": jacobian_finite,
                        "termination": termination,
                        "current_ramp_time_s": self.current_ramp_time_s,
                        "error": None,
                    }
                )
            # Numerical solver failures are data in this experiment: convert them
            # into status=0 and let the PyTorch loss apply a recovery barrier.
            except Exception as exc:  # noqa: BLE001
                shape = (len(self.output_variables), self.t_eval.size)
                trajectories.append(np.zeros(shape, dtype=np.float64))
                jacobians.append(np.zeros((*shape, len(self.input_names)), dtype=np.float64))
                statuses.append(0)
                diagnostics.append(
                    {
                        "requested_end_time_s": self.horizon_s,
                        "actual_end_time_s": None,
                        "completed_requested_horizon": False,
                        "trajectory_finite": False,
                        "jacobian_finite": False,
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

    def solve_discharge_batch(self, inputs: np.ndarray) -> DischargeBatch:
        """Run to a physical cutoff and summarize delivered capacity and energy.

        A minimum-voltage event is a successful physical outcome here. This path
        is non-differentiable and intended for capacity calibration and labels.
        """
        if not self.physical_voltage_cutoffs:
            raise RuntimeError("solve_discharge_batch requires physical_voltage_cutoffs=True")
        import time

        capacities, energies, average_voltages = [], [], []
        minimum_voltages, durations, statuses, runtimes, diagnostics = [], [], [], [], []
        for row in np.asarray(inputs, dtype=np.float64):
            started = time.perf_counter()
            try:
                input_dict = dict(zip(self.input_names, row.tolist()))
                solution = self.simulation.solve(
                    self.t_eval,
                    inputs=input_dict,
                    calculate_sensitivities=False,
                )
                time_values = np.asarray(solution.t, dtype=np.float64).reshape(-1)
                voltage_values = np.asarray(
                    solution["Voltage [V]"].entries, dtype=np.float64
                ).reshape(-1)
                current_values = np.asarray(
                    solution["Current [A]"].entries, dtype=np.float64
                ).reshape(-1)
                summary = summarize_discharge(time_values, voltage_values, current_values)
                termination = str(getattr(solution, "termination", "unknown"))
                reached_voltage_cutoff = "minimum voltage" in termination.lower()
                valid = summary["delivered_capacity_ah"] > 0.0 and summary["discharge_time_s"] > 0.0
                capacities.append(summary["delivered_capacity_ah"])
                energies.append(summary["delivered_energy_wh"])
                average_voltages.append(summary["average_voltage_v"])
                minimum_voltages.append(summary["minimum_voltage_v"])
                durations.append(summary["discharge_time_s"])
                statuses.append(int(valid))
                diagnostics.append(
                    {
                        **summary,
                        "requested_end_time_s": self.horizon_s,
                        "actual_end_time_s": float(time_values[-1]),
                        "reached_voltage_cutoff": reached_voltage_cutoff,
                        "termination": termination,
                        "current_ramp_time_s": self.current_ramp_time_s,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                capacities.append(0.0)
                energies.append(0.0)
                average_voltages.append(0.0)
                minimum_voltages.append(0.0)
                durations.append(0.0)
                statuses.append(0)
                diagnostics.append(
                    {
                        "requested_end_time_s": self.horizon_s,
                        "actual_end_time_s": None,
                        "reached_voltage_cutoff": False,
                        "termination": "exception",
                        "current_ramp_time_s": self.current_ramp_time_s,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            runtimes.append(time.perf_counter() - started)
        self.last_solve_diagnostics = diagnostics
        return DischargeBatch(
            delivered_capacity_ah=np.asarray(capacities, dtype=np.float64),
            delivered_energy_wh=np.asarray(energies, dtype=np.float64),
            average_voltage_v=np.asarray(average_voltages, dtype=np.float64),
            minimum_voltage_v=np.asarray(minimum_voltages, dtype=np.float64),
            discharge_time_s=np.asarray(durations, dtype=np.float64),
            status=np.asarray(statuses, dtype=np.int64),
            runtime_s=np.asarray(runtimes, dtype=np.float64),
        )

    def solve_normalized_discharge_batch(self, inputs: np.ndarray) -> NormalizedDischargeBatch:
        """Run to voltage cutoff and retain curve shape despite variable duration.

        Every successful curve is interpolated onto ``[0, 1]``. Its physical
        duration and delivered capacity are returned separately, so downstream
        benchmarks do not silently pretend that all cells discharge for the
        same amount of time.
        """
        if not self.physical_voltage_cutoffs:
            raise RuntimeError(
                "solve_normalized_discharge_batch requires physical_voltage_cutoffs=True"
            )
        import time

        normalized_time = np.linspace(0.0, 1.0, self.t_eval.size, dtype=np.float64)
        voltages, durations, capacities, statuses, runtimes, diagnostics = [], [], [], [], [], []
        for row in np.asarray(inputs, dtype=np.float64):
            started = time.perf_counter()
            try:
                input_dict = dict(zip(self.input_names, row.tolist()))
                solution = self.simulation.solve(
                    self.t_eval, inputs=input_dict, calculate_sensitivities=False
                )
                time_values = np.asarray(solution.t, dtype=np.float64).reshape(-1)
                voltage_values = np.asarray(
                    solution["Voltage [V]"].entries, dtype=np.float64
                ).reshape(-1)
                current_values = np.asarray(
                    solution["Current [A]"].entries, dtype=np.float64
                ).reshape(-1)
                summary = summarize_discharge(time_values, voltage_values, current_values)
                termination = str(getattr(solution, "termination", "unknown"))
                reached_cutoff = "minimum voltage" in termination.lower()
                finite = bool(np.isfinite(voltage_values).all())
                valid = finite and reached_cutoff and summary["delivered_capacity_ah"] > 0.0
                if valid:
                    relative_time = (time_values - time_values[0]) / max(
                        summary["discharge_time_s"], 1e-12
                    )
                    voltage = np.interp(normalized_time, relative_time, voltage_values)
                else:
                    voltage = np.zeros_like(normalized_time)
                voltages.append(voltage)
                durations.append(summary["discharge_time_s"] if valid else 0.0)
                capacities.append(summary["delivered_capacity_ah"] if valid else 0.0)
                statuses.append(int(valid))
                diagnostics.append(
                    {
                        **summary,
                        "reached_voltage_cutoff": reached_cutoff,
                        "trajectory_finite": finite,
                        "termination": termination,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                voltages.append(np.zeros_like(normalized_time))
                durations.append(0.0)
                capacities.append(0.0)
                statuses.append(0)
                diagnostics.append(
                    {
                        "reached_voltage_cutoff": False,
                        "trajectory_finite": False,
                        "termination": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            runtimes.append(time.perf_counter() - started)
        self.last_solve_diagnostics = diagnostics
        return NormalizedDischargeBatch(
            normalized_time=normalized_time,
            voltage_v=np.stack(voltages),
            discharge_time_s=np.asarray(durations, dtype=np.float64),
            delivered_capacity_ah=np.asarray(capacities, dtype=np.float64),
            status=np.asarray(statuses, dtype=np.int64),
            runtime_s=np.asarray(runtimes, dtype=np.float64),
        )
