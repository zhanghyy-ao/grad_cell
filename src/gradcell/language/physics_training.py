from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.losses import SmoothTchebycheff
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.training.task_sampler import sample_preferences

from .codec import GradCellLanguageCodec, StructuredPreferenceTask
from .model import LanguageGradCell, QwenBackbone


@dataclass(frozen=True)
class PhysicsTrainingConfig:
    stage: int
    steps: int
    batch_size: int
    refinement_steps: int
    learning_rate: float
    validation_interval: int = 25
    auxiliary_weight: float = 0.1
    monotonic_weight: float = 0.1
    step_weight: float = 1e-3


def build_gradcell(
    backend: str,
    model_name: str,
    reference_front: Path | None,
) -> GradCell:
    objective = None
    capacity_multiplier = 1.0
    if reference_front is not None:
        with np.load(reference_front, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
        objective = SmoothTchebycheff(**metadata["bounds"])
        capacity_multiplier = float(metadata.get("capacity_multiplier", 1.0))
    if backend == "toy":
        backends = (
            AnalyticToyBackend(horizon_s=3600.0),
            AnalyticToyBackend(horizon_s=720.0),
            AnalyticToyBackend(horizon_s=600.0),
        )
    elif backend == "pybamm":
        backends = (
            PyBaMMBackend(model_name=model_name, horizon_s=3600.0),
            PyBaMMBackend(model_name=model_name, horizon_s=720.0),
            PyBaMMBackend(model_name=model_name, horizon_s=600.0),
        )
    else:
        raise ValueError(f"unknown physics backend: {backend}")
    return GradCell(
        *(DifferentiablePhysicsLayer(item) for item in backends),
        design_space=DesignSpace(
            capacity_formula="chen2020_scaled",
            capacity_multiplier=capacity_multiplier,
        ),
        objective=objective,
    ).double()


def load_language_model(
    *,
    stage1_dir: Path,
    gradcell: GradCell,
    load_in_4bit: bool,
    train_lora: bool = False,
) -> tuple[LanguageGradCell, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('Install `pip install -e ".[language-gpu]"`') from exc
    heads = torch.load(stage1_dir / "language_heads.pt", map_location="cpu", weights_only=False)
    model_name = str(heads["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = QwenBackbone(
        model_name,
        load_in_4bit=load_in_4bit,
        use_lora=train_lora,
        adapter_path=str(stage1_dir / "qwen_adapter"),
    )
    language_model = LanguageGradCell(
        backbone,
        gradcell,
        codec=GradCellLanguageCodec(bins=int(heads["bins"])),
        freeze_backbone=not train_lora,
    )
    language_model.projector.load_state_dict(heads["projector"])
    language_model.continuous_head.load_state_dict(heads["continuous_head"])
    language_model.level_head.load_state_dict(heads["level_head"])
    language_device = backbone.model.get_input_embeddings().weight.device
    language_model.projector.to(language_device)
    language_model.continuous_head.to(language_device)
    language_model.level_head.to(language_device)
    gradcell.to(language_device)
    return language_model, tokenizer


def load_physics_checkpoint(model: LanguageGradCell, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(checkpoint["projector"])
    model.continuous_head.load_state_dict(checkpoint["continuous_head"])
    model.level_head.load_state_dict(checkpoint["level_head"])
    model.gradcell.load_state_dict(checkpoint["gradcell"])
    return checkpoint


def save_physics_checkpoint(
    model: LanguageGradCell,
    output: Path,
    config: PhysicsTrainingConfig,
    history: list[dict],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": config.stage,
            "config": config.__dict__,
            "projector": model.projector.state_dict(),
            "continuous_head": model.continuous_head.state_dict(),
            "level_head": model.level_head.state_dict(),
            "gradcell": model.gradcell.state_dict(),
            "history": history,
        },
        output,
    )


def sample_targets(preferences: torch.Tensor) -> torch.Tensor:
    """Sample a conservative first task domain; formal bounds should come from the reference set."""

    count = preferences.shape[0]
    device, dtype = preferences.device, preferences.dtype
    energy = 145.0 + 15.0 * torch.rand(count, device=device, dtype=dtype)
    min_r5 = 0.48 + 0.07 * torch.rand(count, device=device, dtype=dtype)
    min_r6 = 0.42 + 0.06 * torch.rand(count, device=device, dtype=dtype)
    return torch.stack([energy, min_r5, min_r6], dim=-1)


def tokenize_preferences(
    tokenizer,
    codec,
    preferences: torch.Tensor,
    device,
    targets: torch.Tensor | None = None,
) -> dict:
    if targets is None:
        targets = preferences.new_tensor([150.0, 0.50, 0.44]).expand(
            preferences.shape[0], -1
        )
    texts = [
        codec.serialize_task(
            StructuredPreferenceTask(
                float(preference),
                target_energy=float(target[0]),
                min_retention_5c=float(target[1]),
                min_retention_6c=float(target[2]),
            )
        )
        for preference, target in zip(preferences.detach().cpu(), targets.detach().cpu())
    ]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    return {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
    }


def train_physics_stage(
    model: LanguageGradCell,
    tokenizer,
    config: PhysicsTrainingConfig,
    output: Path,
) -> list[dict]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("physics stage has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=1e-5)
    physics_parameter = next(model.gradcell.parameters())
    physics_device, physics_dtype = physics_parameter.device, physics_parameter.dtype
    language_device = model.backbone.model.get_input_embeddings().weight.device
    validation_preferences = torch.linspace(0.0, 1.0, 11, dtype=physics_dtype, device=physics_device)
    history = []

    def objective(
        preferences: torch.Tensor,
        targets: torch.Tensor,
        refinement_steps: int,
    ) -> tuple[torch.Tensor, dict]:
        tokens = tokenize_preferences(
            tokenizer, model.codec, preferences, language_device, targets=targets
        )
        output_value = model(
            tokens["input_ids"],
            tokens["attention_mask"],
            preferences,
            num_steps=refinement_steps,
            targets=targets,
        )
        losses = torch.stack([step.loss for step in output_value.gradcell.steps], dim=0)
        best_loss = losses.min(dim=0).values.mean()
        final_loss = losses[-1].mean()
        auxiliary = final_loss.new_zeros(())
        monotonic = final_loss.new_zeros(())
        step_penalty = final_loss.new_zeros(())
        total = final_loss
        if refinement_steps:
            auxiliary = losses[:-1].mean()
            monotonic = torch.relu(losses[1:] - losses[:-1]).mean()
            step_penalty = torch.stack(
                [
                    (current.latent - previous.latent).square().mean()
                    for previous, current in zip(
                        output_value.gradcell.steps, output_value.gradcell.steps[1:]
                    )
                ]
            ).mean()
            total = (
                best_loss
                + config.auxiliary_weight * auxiliary
                + config.monotonic_weight * monotonic
                + config.step_weight * step_penalty
            )
        metrics = {
            "total_loss": float(total.detach()),
            "initial_loss": float(losses[0].mean().detach()),
            "final_loss": float(final_loss.detach()),
            "best_loss": float(best_loss.detach()),
            "success_rate": float(output_value.gradcell.final.status.double().mean().detach()),
            "auxiliary_loss": float(auxiliary.detach()),
            "monotonic_penalty": float(monotonic.detach()),
            "step_penalty": float(step_penalty.detach()),
        }
        return total, metrics

    for step in range(1, config.steps + 1):
        model.train()
        preferences = sample_preferences(
            config.batch_size, dtype=physics_dtype, device=physics_device
        )
        targets = sample_targets(preferences)
        optimizer.zero_grad(set_to_none=True)
        loss, record = objective(preferences, targets, config.refinement_steps)
        if record["success_rate"] == 0.0:
            raise RuntimeError("all PyBaMM simulations failed")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        record.update({"stage": config.stage, "step": step})
        if config.validation_interval and step % config.validation_interval == 0:
            model.eval()
            with torch.enable_grad():
                validation_targets = validation_preferences.new_tensor(
                    [150.0, 0.50, 0.44]
                ).expand(validation_preferences.shape[0], -1)
                _, validation = objective(
                    validation_preferences, validation_targets, config.refinement_steps
                )
            record.update({f"validation_{key}": value for key, value in validation.items()})
            print(json.dumps(record))
        history.append(record)
        save_physics_checkpoint(model, output, config, history)
    return history
