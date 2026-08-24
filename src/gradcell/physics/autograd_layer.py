from __future__ import annotations

import torch
from torch import nn


class _PhysicsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, physics_inputs: torch.Tensor, backend):
        batch = backend.solve_batch(physics_inputs.detach().cpu().double().numpy())
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
        ctx.save_for_backward(jac)
        return y, status, runtime

    @staticmethod
    def backward(ctx, grad_y, grad_status, grad_runtime):
        (jacobian,) = ctx.saved_tensors
        grad_inputs = torch.einsum("bot,botp->bp", grad_y, jacobian)
        return grad_inputs, None


class DifferentiablePhysicsLayer(nn.Module):
    def __init__(self, backend) -> None:
        super().__init__()
        self.backend = backend

    def forward(self, physics_inputs: torch.Tensor):
        return _PhysicsFunction.apply(physics_inputs, self.backend)

