from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PhysicsBatch:
    """一次 batch 物理求解的统一返回结构。"""

    trajectories: np.ndarray # 物理输出轨迹
    jacobian: np.ndarray   # 轨迹对输入参数的 雅可比矩阵（真的可以求得吗？）
    status: np.ndarray # 求解是否成功
    runtime_s: np.ndarray # 每个样本的求解时间


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
    ) -> None:
        # 延迟导入 PyBaMM，使 toy 后端和基础测试不依赖该可选依赖。
        try:
            import pybamm
        except ImportError as exc:
            raise ImportError("Install GradCell with `pip install -e .[physics]`") from exc
        self.pybamm = pybamm
        # 创建指定锂离子模型和参数集。
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
            # 将孔隙率、活性比例和电流暴露为运行时输入。
            self.parameters.update({name: pybamm.InputParameter(name)})
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
                negative_diffusivity(stoichiometry, temperature) #stoichiometry 当前材料颗粒内部已经填充了锂离子的“空位”占所有可用空位的百分比。
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
        self.calculate_sensitivities = calculate_sensitivities
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
        """逐样本调用 PyBaMM，返回轨迹和对 8 个输入的 forward sensitivity。"""
        import time

        trajectories, jacobians, statuses, runtimes = [], [], [], []
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
                output_rows, sensitivity_rows = [], []
                for output_name in self.output_variables:
                    # 逐输出提取轨迹，并堆叠其对所有输入的导数。
                    variable = solution[output_name]
                    values = np.asarray(variable(self.t_eval), dtype=np.float64).reshape(-1)
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
