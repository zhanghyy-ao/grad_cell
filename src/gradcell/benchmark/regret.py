from __future__ import annotations

import torch


def normalized_regret(
    achieved_loss: torch.Tensor,
    optimal_loss: torch.Tensor,
    nominal_loss: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (achieved_loss - optimal_loss) / (nominal_loss - optimal_loss + eps)

