from __future__ import annotations

import torch


def sample_preferences(batch_size: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    component = torch.rand(batch_size, device=device)
    uniform = torch.rand(batch_size, device=device)
    power_end = torch.distributions.Beta(0.5, 2.0).sample((batch_size,)).to(device=device)
    energy_end = torch.distributions.Beta(2.0, 0.5).sample((batch_size,)).to(device=device)
    preference = torch.where(component < 0.5, uniform, torch.where(component < 0.75, power_end, energy_end))
    return preference.to(dtype=dtype)

