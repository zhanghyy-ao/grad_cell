from __future__ import annotations

import torch
from torch import nn


class _PhysicsFunction(torch.autograd.Function):
    """将 NumPy/PyBaMM 后端包装成可参与 PyTorch 反向传播的算子。"""

    @staticmethod
    def forward(ctx, physics_inputs: torch.Tensor, backend):
        # PyBaMM 后端在 CPU 上接收 NumPy 数组；detach 只切断前向转换，
        # 反向梯度由后端返回的显式 Jacobian 提供。
        batch = backend.solve_batch(physics_inputs.detach().cpu().double().numpy())
        # 将后端轨迹、Jacobian、状态和运行时间恢复为与输入相容的 Tensor。
        y = torch.as_tensor(
            batch.trajectories,
            dtype=physics_inputs.dtype,
            device=physics_inputs.device,
        )
        jac = torch.as_tensor(
            batch.jacobian,
            dtype=physics_inputs.dtype,
            device=physics_inputs.device,
        )
        status = torch.as_tensor(batch.status, device=physics_inputs.device)
        runtime = torch.as_tensor(
            batch.runtime_s, dtype=physics_inputs.dtype, device=physics_inputs.device
        )
        # 保存 dy/dx，供 backward 计算向量-雅可比积。
        ctx.save_for_backward(jac)
        return y, status, runtime

    @staticmethod
    def backward(ctx, grad_y, grad_status, grad_runtime):
        (jacobian,) = ctx.saved_tensors
        # grad_y 形状为 [B,O,T]，Jacobian 形状为 [B,O,T,P]；
        # 对输出维和时间维求和，得到 loss 对 P 个物理输入的梯度。
        grad_inputs = torch.einsum("bot,botp->bp", grad_y, jacobian)
        # backend 不是 Tensor，因此没有梯度；status/runtime 也只作为记录量。
        return grad_inputs, None


class DifferentiablePhysicsLayer(nn.Module):
    """把任意 PhysicsBackend 封装为标准 PyTorch 模块。"""

    def __init__(self, backend) -> None:
        super().__init__()
        self.backend = backend

    def forward(self, physics_inputs: torch.Tensor):
        """返回物理轨迹、求解状态和每个样本的运行时间。"""
        return _PhysicsFunction.apply(physics_inputs, self.backend)
