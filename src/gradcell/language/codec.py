from __future__ import annotations

from dataclasses import dataclass

import torch


TASK_TAGS = (
    "<TASK>",
    "</TASK>",
    "<OBJECTIVE_A>",
    "</OBJECTIVE_A>",
    "<OBJECTIVE_B>",
    "</OBJECTIVE_B>",
    "<PREFERENCE>",
    "</PREFERENCE>",
    "<DESIGN>",
    "</DESIGN>",
    "<U0>",
    "</U0>",
    "<U1>",
    "</U1>",
    "<U2>",
    "</U2>",
    "<U3>",
    "</U3>",
    "<U4>",
    "</U4>",
)


@dataclass(frozen=True)
class StructuredPreferenceTask:
    preference: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.preference <= 1.0:
            raise ValueError("preference must be in [0, 1]")


class GradCellLanguageCodec:
    """Deterministic grammar for preference tasks and quantized GradCell latents."""

    def __init__(self, bins: int = 256, latent_limit: float = 4.0) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if latent_limit <= 0.0:
            raise ValueError("latent_limit must be positive")
        self.bins = bins
        self.latent_limit = latent_limit

    @property
    def special_tokens(self) -> list[str]:
        return [*TASK_TAGS, *[f"<LEVEL_{index:03d}>" for index in range(self.bins)]]

    def preference_level(self, preference: float) -> int:
        StructuredPreferenceTask(preference)
        return round(preference * (self.bins - 1))

    def serialize_task(self, task: StructuredPreferenceTask) -> str:
        level = self.preference_level(task.preference)
        return (
            "<TASK>\n"
            "<OBJECTIVE_A>ENERGY_1C</OBJECTIVE_A>\n"
            "<OBJECTIVE_B>MIN_RETENTION_5C_6C</OBJECTIVE_B>\n"
            f"<PREFERENCE><LEVEL_{level:03d}></PREFERENCE>\n"
            "</TASK>\n<DESIGN>"
        )

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = (latent.clamp(-self.latent_limit, self.latent_limit) + self.latent_limit)
        normalized = normalized / (2.0 * self.latent_limit)
        return torch.round(normalized * (self.bins - 1)).to(torch.long)

    def dequantize(self, levels: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if torch.any(levels < 0) or torch.any(levels >= self.bins):
            raise ValueError("levels contain a value outside the codec vocabulary")
        result = levels.to(dtype=dtype or torch.get_default_dtype()) / (self.bins - 1)
        return result * (2.0 * self.latent_limit) - self.latent_limit

    def serialize_design(self, latent: torch.Tensor) -> str:
        if latent.ndim != 1 or latent.numel() != 5:
            raise ValueError("a serialized design must be a one-dimensional five-value latent")
        levels = self.quantize(latent).tolist()
        fields = [f"<U{i}><LEVEL_{level:03d}></U{i}>" for i, level in enumerate(levels)]
        return "\n".join(fields) + "\n</DESIGN>"
