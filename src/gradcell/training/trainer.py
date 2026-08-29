from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .task_sampler import sample_preferences


@dataclass
class TrainResult:
    losses: list[float]
    validation_losses: list[float]
    best_validation_loss: float
    best_step: int
    stopped_early: bool


def train(
    model,
    steps: int = 300,
    batch_size: int = 8,
    refinement_steps: int = 0,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    log_every: int = 25,
    validation_preferences: torch.Tensor | None = None,
    validation_interval: int = 25,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    log_path: str | Path | None = None,
) -> TrainResult:
    """训练模型，并支持验证、早停、断点恢复和 JSONL 结构化日志。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []
    validation_losses: list[float] = []
    dtype = next(model.parameters()).dtype
    if validation_preferences is None:
        validation_preferences = torch.linspace(0.0, 1.0, 11, dtype=dtype)
    else:
        validation_preferences = validation_preferences.to(dtype=dtype)
    start_step = 0
    best_validation_loss = float("inf")
    best_step = 0
    stale_steps = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = list(checkpoint.get("losses", []))
        validation_losses = list(checkpoint.get("validation_losses", []))
        start_step = int(checkpoint.get("step", len(losses)))
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        stale_steps = int(checkpoint.get("stale_steps", 0))
    log_file = Path(log_path) if log_path is not None else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(record: dict) -> None:
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stopped_early = False
    for iteration in range(start_step + 1, steps + 1):
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
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss detected at step {iteration}: {loss.detach()}"
            )
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradient detected in {name!r} at step {iteration}"
                )
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        record = {"step": iteration, "train_loss": losses[-1]}
        if validation_interval and iteration % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(validation_preferences, num_steps=refinement_steps)
                val_loss = float(torch.stack([step.loss for step in val_output.steps]).mean())
            model.train()
            validation_losses.append(val_loss)
            record["validation_loss"] = val_loss
            if val_loss < best_validation_loss - min_delta:
                best_validation_loss, best_step, stale_steps = val_loss, iteration, 0
            else:
                stale_steps += validation_interval
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": iteration, "losses": losses,
                            "validation_losses": validation_losses,
                            "best_validation_loss": best_validation_loss,
                            "best_step": best_step, "stale_steps": stale_steps}, path)
            if early_stopping_patience is not None and stale_steps >= early_stopping_patience:
                stopped_early = True
        emit(record)
        if log_every and iteration % log_every == 0:
            print(f"step={iteration:05d} loss={losses[-1]:.6f}")
        if stopped_early:
            break
    return TrainResult(losses, validation_losses, best_validation_loss, best_step, stopped_early)
