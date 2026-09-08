from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from gradcell.models.gradcell import GradCell, GradCellOutput

from .codec import GradCellLanguageCodec
from .json_design import MaterialDesignJSONCodec


class HiddenStateBackbone(Protocol):
    hidden_size: int

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...


class QwenBackbone(nn.Module):
    """Lazy Hugging Face wrapper for Qwen3-8B hidden states.

    Transformers is imported only when this class is instantiated, so the core
    GradCell package and its tests do not require language-model dependencies.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        torch_dtype: str = "bfloat16",
        device_map: str | None = "auto",
        load_in_4bit: bool = False,
        use_lora: bool = False,
        adapter_path: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError('Install GradCell with `pip install -e ".[language]"`') from exc
        dtype = getattr(torch, torch_dtype)
        quantization_config = None
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit Qwen loading requires a CUDA-enabled PyTorch build")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("loading a LoRA adapter requires the peft package") from exc
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=use_lora
            )
        elif use_lora:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except ImportError as exc:
                raise ImportError("LoRA training requires the peft package") from exc
            if load_in_4bit:
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=True
                )
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=(
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ),
            )
            self.model = get_peft_model(self.model, config)
        if use_lora:
            # Causal-LM activations dominate memory for an 8B model.  This is
            # needed for BF16 LoRA as well as QLoRA; the k-bit helper only
            # enables it automatically in the quantized branch.
            self.model.config.use_cache = False
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.model.gradient_checkpointing_enable()
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        causal_lm = (
            self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        )
        outputs = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def causal_forward(self, **kwargs):
        """Run the native causal-LM objective for Stage 1 JSON training."""
        return self.model(**kwargs, return_dict=True)

    def generate(self, **kwargs) -> torch.Tensor:
        return self.model.generate(**kwargs)

    def save_adapter(self, output_dir: str) -> None:
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)


@dataclass
class LanguageGradCellOutput:
    gradcell: GradCellOutput
    continuous_latent: torch.Tensor
    token_latent: torch.Tensor
    level_logits: torch.Tensor
    best_step_index: torch.Tensor
    best_latent: torch.Tensor


class LanguageGradCell(nn.Module):
    """Qwen-conditioned GradCell with differentiable continuous and token heads."""

    def __init__(
        self,
        backbone: HiddenStateBackbone,
        gradcell: GradCell,
        *,
        codec: GradCellLanguageCodec | None = None,
        task_dim: int = 128,
        freeze_backbone: bool = True,
        token_blend: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= token_blend <= 1.0:
            raise ValueError("token_blend must be in [0, 1]")
        self.backbone = backbone
        self.gradcell = gradcell
        self.codec = codec or GradCellLanguageCodec()
        self.token_blend = token_blend
        latent_dim = gradcell.design_space.latent_dim
        self.projector = nn.Sequential(
            nn.Linear(backbone.hidden_size, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, task_dim),
            nn.LayerNorm(task_dim),
        )
        self.continuous_head = nn.Linear(task_dim, latent_dim)
        self.level_head = nn.Linear(task_dim, latent_dim * self.codec.bins)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _pool_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.to(torch.long).sum(dim=-1).sub(1).clamp_min(0)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, positions]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        pooled = self._pool_last(hidden, attention_mask)
        return self.project_hidden(pooled)

    def project_hidden(self, pooled: torch.Tensor) -> torch.Tensor:
        """Project one selected Qwen hidden state into the GradCell task space."""
        target_dtype = self.projector[0].weight.dtype
        return self.projector(pooled.to(dtype=target_dtype))

    def propose_from_hidden_positions(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a proposal from selected sequence positions without another Qwen pass."""
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        task_embedding = self.project_hidden(hidden[batch, positions.to(hidden.device)])
        return self.propose_from_embedding(task_embedding)

    def propose_from_embedding(
        self, task_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a task embedding into continuous and discretized design latents."""
        batch_size = task_embedding.shape[0]
        continuous = 4.0 * torch.tanh(self.continuous_head(task_embedding))
        logits = self.level_head(task_embedding).reshape(
            batch_size, self.gradcell.design_space.latent_dim, self.codec.bins
        )
        levels = torch.arange(self.codec.bins, dtype=logits.dtype, device=logits.device)
        expected_levels = (torch.softmax(logits, dim=-1) * levels).sum(dim=-1)
        token_latent = self.codec.dequantize(expected_levels, dtype=continuous.dtype)
        latent = (1.0 - self.token_blend) * continuous + self.token_blend * token_latent
        return task_embedding, latent, logits

    def propose(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_embedding = self.encode(input_ids, attention_mask)
        return self.propose_from_embedding(task_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        preference: torch.Tensor,
        *,
        num_steps: int = 0,
        targets: torch.Tensor | None = None,
    ) -> LanguageGradCellOutput:
        task_embedding, latent, logits = self.propose(input_ids, attention_mask)
        token_levels = logits.argmax(dim=-1)
        token_latent = self.codec.dequantize(token_levels, dtype=latent.dtype)
        physics_parameter = next(self.gradcell.parameters())
        physics_targets = None
        if targets is not None:
            physics_targets = targets.to(
                dtype=physics_parameter.dtype, device=physics_parameter.device
            )
        steps = self.gradcell.run_from_embedding(
            task_embedding.to(dtype=physics_parameter.dtype, device=physics_parameter.device),
            latent.to(dtype=physics_parameter.dtype, device=physics_parameter.device),
            preference.to(dtype=physics_parameter.dtype, device=physics_parameter.device),
            num_steps=num_steps,
            targets=physics_targets,
        )
        losses = torch.stack([step.loss for step in steps.steps], dim=0)
        best_step_index = losses.argmin(dim=0)
        latents = torch.stack([step.latent for step in steps.steps], dim=0)
        gather_index = best_step_index.reshape(1, -1, 1).expand(1, -1, latents.shape[-1])
        best_latent = latents.gather(0, gather_index).squeeze(0)
        return LanguageGradCellOutput(
            steps, latent, token_latent, logits, best_step_index, best_latent
        )

    def render_best_json(
        self,
        output: LanguageGradCellOutput,
        targets: torch.Tensor | None = None,
    ) -> list[str]:
        json_codec = MaterialDesignJSONCodec(self.gradcell.design_space)
        design = self.gradcell.design_space(output.best_latent)
        results = []
        for index in range(output.best_latent.shape[0]):
            step_index = int(output.best_step_index[index])
            step = output.gradcell.steps[step_index]
            payload = json_codec.design_dict(design, index)
            performance = {
                "specific_energy_1c_wh_kg": float(step.energy[index].detach().cpu()),
                "energy_retention_5c": float(step.retention_5c[index].detach().cpu()),
                "energy_retention_6c": float(step.retention_6c[index].detach().cpu()),
                "physics_loss": float(step.loss[index].detach().cpu()),
                "solver_success": bool(step.status[index]),
            }
            payload.update(
                {
                    "selected_refinement_step": step_index,
                    "predicted_performance": performance,
                }
            )
            if targets is not None:
                target = targets[index].detach().cpu()
                requirements = {
                    "target_energy_1c_wh_kg": float(target[0]),
                    "minimum_energy_retention_5c": float(target[1]),
                    "minimum_energy_retention_6c": float(target[2]),
                }
                payload["requirements"] = requirements
                payload["requirements_satisfied"] = bool(
                    performance["solver_success"]
                    and performance["specific_energy_1c_wh_kg"] >= requirements["target_energy_1c_wh_kg"]
                    and performance["energy_retention_5c"] >= requirements["minimum_energy_retention_5c"]
                    and performance["energy_retention_6c"] >= requirements["minimum_energy_retention_6c"]
                )
            results.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return results
