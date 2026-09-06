from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from gradcell.models.gradcell import GradCell, GradCellOutput

from .codec import GradCellLanguageCodec


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
        if use_lora:
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
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
            self.model = get_peft_model(self.model, config)
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


@dataclass
class LanguageGradCellOutput:
    gradcell: GradCellOutput
    continuous_latent: torch.Tensor
    token_latent: torch.Tensor
    level_logits: torch.Tensor


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
        target_dtype = self.projector[0].weight.dtype
        return self.projector(pooled.to(dtype=target_dtype))

    def propose(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_embedding = self.encode(input_ids, attention_mask)
        continuous = 4.0 * torch.tanh(self.continuous_head(task_embedding))
        logits = self.level_head(task_embedding).reshape(
            input_ids.shape[0], self.gradcell.design_space.latent_dim, self.codec.bins
        )
        levels = torch.arange(self.codec.bins, dtype=logits.dtype, device=logits.device)
        expected_levels = (torch.softmax(logits, dim=-1) * levels).sum(dim=-1)
        token_latent = self.codec.dequantize(expected_levels, dtype=continuous.dtype)
        latent = (1.0 - self.token_blend) * continuous + self.token_blend * token_latent
        return task_embedding, latent, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        preference: torch.Tensor,
        *,
        num_steps: int = 0,
    ) -> LanguageGradCellOutput:
        task_embedding, latent, logits = self.propose(input_ids, attention_mask)
        token_levels = logits.argmax(dim=-1)
        token_latent = self.codec.dequantize(token_levels, dtype=latent.dtype)
        steps = self.gradcell.run_from_embedding(
            task_embedding, latent, preference, num_steps=num_steps
        )
        return LanguageGradCellOutput(steps, latent, token_latent, logits)
