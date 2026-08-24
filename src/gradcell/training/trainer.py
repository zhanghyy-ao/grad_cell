from __future__ import annotations

from dataclasses import dataclass

import torch

from .task_sampler import sample_preferences


@dataclass
class TrainResult:
    losses: list[float]


def train(
    model,
    steps: int = 300,
    batch_size: int = 8,
    refinement_steps: int = 0,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    log_every: int = 25,
) -> TrainResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []
    dtype = next(model.parameters()).dtype
    for iteration in range(1, steps + 1):
        preference = sample_preferences(batch_size, dtype=dtype)
        optimizer.zero_grad(set_to_none=True)
        output = model(preference, num_steps=refinement_steps)
        intermediate = torch.stack([step.loss for step in output.steps], dim=0)
        loss = intermediate.mean()
        if len(output.steps) > 1:
            monotonic = torch.nn.functional.softplus(
                intermediate[1:] - intermediate[:-1] + 1e-3
            ).mean()
            step_penalty = torch.stack(
                [
                    (output.steps[i + 1].latent - output.steps[i].latent).square().mean()
                    for i in range(len(output.steps) - 1)
                ]
            ).mean()
            loss = loss + 0.5 * monotonic + 1e-3 * step_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        if log_every and iteration % log_every == 0:
            print(f"step={iteration:05d} loss={losses[-1]:.6f}")
    return TrainResult(losses=losses)

