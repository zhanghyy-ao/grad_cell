from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradcell.language import (
    DFNPerformanceSurrogate,
    SingleDesignPhysicsMLP,
    freeze_surrogate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine Qwen input device")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


def load_models(
    design_checkpoint_path: Path,
    surrogate_checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    SingleDesignPhysicsMLP,
    DFNPerformanceSurrogate,
    dict[str, Any],
    dict[str, Any],
]:
    design_checkpoint = torch.load(design_checkpoint_path, map_location="cpu", weights_only=False)
    surrogate_checkpoint = torch.load(
        surrogate_checkpoint_path, map_location="cpu", weights_only=False
    )
    if design_checkpoint.get("schema") != ("gradcell.language_single_design_physics_guided.v1"):
        raise ValueError("Unsupported single-design checkpoint schema")
    if surrogate_checkpoint.get("schema") != "gradcell.dfn_performance_surrogate.v1":
        raise ValueError("Unsupported surrogate checkpoint schema")
    expected_hash = design_checkpoint.get("surrogate_checkpoint_sha256")
    if expected_hash and expected_hash != sha256(surrogate_checkpoint_path):
        raise ValueError("Surrogate checkpoint does not match the design checkpoint")
    design_model = SingleDesignPhysicsMLP(**design_checkpoint["model_config"])
    design_model.load_state_dict(design_checkpoint["model_state"])
    design_model.to(device).eval()
    surrogate = DFNPerformanceSurrogate(**surrogate_checkpoint["model_config"])
    surrogate.load_state_dict(surrogate_checkpoint["model_state"])
    freeze_surrogate(surrogate.to(device))
    return design_model, surrogate, design_checkpoint, surrogate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one battery design and predicted performance from natural language."
    )
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--description")
    text.add_argument("--description-file", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    description = (
        args.description
        if args.description is not None
        else args.description_file.read_text(encoding="utf-8").strip()
    )
    if not description.strip():
        raise ValueError("Battery description cannot be empty")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    design_model, surrogate, design_checkpoint, surrogate_checkpoint = load_models(
        args.design_checkpoint, args.surrogate_checkpoint, device
    )

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        args.model_name,
        dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    qwen.eval()
    qwen.requires_grad_(False)
    qwen_device = input_device(qwen)
    with torch.inference_mode():
        tokens = tokenizer(
            [description],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = {name: value.to(qwen_device) for name, value in tokens.items()}
        embedding = mean_pool(
            qwen(**tokens, return_dict=True).last_hidden_state,
            tokens["attention_mask"],
        ).float()
        if embedding.shape[1] != design_checkpoint["model_config"]["input_dim"]:
            raise ValueError("Qwen embedding dimension does not match the design checkpoint")
        normalized_design = design_model(embedding.to(device))
        normalized_performance = surrogate(normalized_design)
    design_log = (
        normalized_design.cpu() * design_checkpoint["design_log_std"]
        + design_checkpoint["design_log_mean"]
    )[0]
    performance_log = (
        normalized_performance.cpu() * design_checkpoint["performance_log_std"]
        + design_checkpoint["performance_log_mean"]
    )[0]
    multipliers = torch.exp(design_log).numpy()
    performance = torch.exp(performance_log).numpy()
    nominal = surrogate_checkpoint["nominal_parameter_values"].numpy()
    values = multipliers * nominal
    structurally_feasible = bool(
        np.isfinite(values).all()
        and np.all(values > 0.0)
        and values[0] + values[3] <= 1.0 + 1e-10
        and values[1] + values[4] <= 1.0 + 1e-10
    )
    result = {
        "schema": "gradcell.language_single_design_inference.v1",
        "battery_description": description,
        "base_parameter_set": design_checkpoint["parameter_sets"][0],
        "generation_mode": design_checkpoint["generation_modes"][0],
        "parameter_multipliers": dict(
            zip(design_checkpoint["parameter_names"], multipliers.tolist(), strict=True)
        ),
        "parameter_values": dict(
            zip(design_checkpoint["parameter_names"], values.tolist(), strict=True)
        ),
        "surrogate_predicted_performance": dict(
            zip(
                design_checkpoint["performance_fields"],
                performance.tolist(),
                strict=True,
            )
        ),
        "structurally_feasible": structurally_feasible,
        "performance_provenance": (
            "Differentiable surrogate prediction; strict DFN replay is required for "
            "physical verification."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
