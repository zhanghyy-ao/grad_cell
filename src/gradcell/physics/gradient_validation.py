from __future__ import annotations

import torch


def directional_derivative_check(
    function,
    point: torch.Tensor,
    direction: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> dict[str, float]:
    """用中心有限差分检查自动微分的方向导数。

    ``function`` 应返回标量；该检查比较解析/自动梯度与沿 ``direction``
    的有限差分结果，适合验证自定义物理层或损失函数。
    """
    # 克隆输入，避免修改调用方张量，并打开对输入的梯度跟踪。
    point = point.detach().clone().requires_grad_(True)
    if direction is None:
        # 未指定方向时使用随机方向，随后归一化以控制扰动尺度。
        direction = torch.randn_like(point)
    direction = direction / direction.norm().clamp_min(1e-12)
    # 通过 autograd 计算方向导数 grad(f) · direction。
    value = function(point)
    (gradient,) = torch.autograd.grad(value, point)
    autodiff = torch.sum(gradient * direction)
    with torch.no_grad():
        # 中心差分具有二阶截断精度：
        # [f(x+eps*d)-f(x-eps*d)]/(2 eps)。
        finite_difference = (function(point + eps * direction) - function(point - eps * direction)) / (
            2.0 * eps
        )
    # 用 1+|autodiff| 做尺度归一化，避免小导数导致相对误差爆炸。
    error = torch.abs(autodiff - finite_difference) / (1.0 + torch.abs(autodiff))
    return {
        "autodiff": float(autodiff),
        "finite_difference": float(finite_difference),
        "relative_directional_error": float(error),
    }
