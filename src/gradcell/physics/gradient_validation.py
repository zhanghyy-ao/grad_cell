from __future__ import annotations

import torch


def directional_derivative_check(
    function,
    point: torch.Tensor,
    direction: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> dict[str, float]:
    point = point.detach().clone().requires_grad_(True)
    if direction is None:
        direction = torch.randn_like(point)
    direction = direction / direction.norm().clamp_min(1e-12)
    value = function(point)
    (gradient,) = torch.autograd.grad(value, point)
    autodiff = torch.sum(gradient * direction)
    with torch.no_grad():
        finite_difference = (function(point + eps * direction) - function(point - eps * direction)) / (
            2.0 * eps
        )
    error = torch.abs(autodiff - finite_difference) / (1.0 + torch.abs(autodiff))
    return {
        "autodiff": float(autodiff),
        "finite_difference": float(finite_difference),
        "relative_directional_error": float(error),
    }

