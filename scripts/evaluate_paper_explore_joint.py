from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extract_paper_explore_embeddings import dtype_from_name, masked_pool, model_input_device
from train_paper_explore_mlp import PaperExploreMLP

from gradcell.design import DesignSpace
from gradcell.evaluation import hard_cutoff_metrics


DESIGN_FIELDS = (
    "eps_p",
    "eps_n",
    "eps_s",
    "phi_p",
    "phi_n",
    "np_ratio",
    "nominal_capacity_ah",
    "stack_mass_kg",
)
PERFORMANCE_FIELDS = ("energy_1c_wh_kg", "retention_5c", "retention_6c")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_inputs(args: argparse.Namespace) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if args.data:
        rows = [row for row in read_jsonl(args.data) if args.split == "all" or row["split"] == args.split]
        if not rows:
            raise ValueError(f"No rows found for split {args.split!r}")
        if args.max_samples:
            rows = rows[: args.max_samples]
        return (
            [str(row["task_id"]) for row in rows],
            [str(row["requirement_text"]) for row in rows],
            rows,
        )
    texts = list(args.prompt)
    if args.prompt_file:
        texts.extend(line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not texts:
        raise ValueError("Provide --data, --prompt, or --prompt-file")
    return [f"prompt-{index:04d}" for index in range(len(texts))], texts, rows


def load_cached_embeddings(path: Path, task_ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        all_ids = arrays["task_ids"].astype(str)
        all_embeddings = arrays["embeddings"].astype(np.float32)
    lookup = {task_id: index for index, task_id in enumerate(all_ids)}
    missing = [task_id for task_id in task_ids if task_id not in lookup]
    if missing:
        raise ValueError(f"Cached embeddings are missing {len(missing)} task IDs")
    return all_embeddings[[lookup[task_id] for task_id in task_ids]]


def embed_texts(
    texts: list[str],
    *,
    model_name: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    dtype: str,
    device_name: str,
    load_in_4bit: bool,
    local_files_only: bool,
    trust_remote_code: bool,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype_from_name(dtype)
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype_from_name(dtype)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    if not load_in_4bit:
        model.to(torch.device(device_name))
    model.eval().requires_grad_(False)
    device = model_input_device(model)
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                pad_to_multiple_of=8 if device.type == "cuda" else None,
                return_tensors="pt",
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            output = model(**tokens, return_dict=True)
            pooled = masked_pool(output.last_hidden_state, tokens["attention_mask"], pooling)
            parts.append(pooled.float().cpu().numpy())
    return np.concatenate(parts)


def decode_design(latent: np.ndarray, capacity_formula: str, capacity_multiplier: float) -> list[dict[str, float]]:
    decoder = DesignSpace(
        capacity_formula=capacity_formula, capacity_multiplier=capacity_multiplier
    ).double()
    with torch.inference_mode():
        design = decoder(torch.from_numpy(latent).double())
    return [
        {field: float(getattr(design, field)[index]) for field in DESIGN_FIELDS}
        for index in range(len(latent))
    ]


def label_metrics(records: list[dict[str, Any]], latent: np.ndarray, feasibility: np.ndarray) -> dict[str, float]:
    distances = []
    labels = []
    for row, predicted in zip(records, latent, strict=True):
        candidates = row.get("teacher_candidates") or [{"latent": row["teacher_latent"]}]
        candidate_latents = np.asarray([candidate["latent"] for candidate in candidates])
        if bool(row["requirement_feasible"]):
            distances.append(float(np.abs(candidate_latents - predicted).mean(axis=1).min()))
        labels.append(float(bool(row["requirement_feasible"])))
    labels_array = np.asarray(labels)
    predicted_labels = feasibility >= 0.5
    return {
        "latent_topk_mae_feasible": float(np.mean(distances)) if distances else float("nan"),
        "feasibility_accuracy": float(np.mean(predicted_labels == labels_array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint Qwen + MLP + decoder/SPMe evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--embeddings", type=Path, help="Dataset-only fast path using aligned cached embeddings")
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--pooling", choices=("mean", "last"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--physics-model", choices=("none", "SPMe", "DFN"), default="none")
    parser.add_argument("--physics-batch-size", type=int, default=8)
    parser.add_argument("--time-points", type=int, default=301)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.embeddings and not args.data:
        parser.error("--embeddings can only be used with --data")
    if args.data and (args.prompt or args.prompt_file):
        parser.error("Use either dataset mode or prompt mode, not both")
    if args.batch_size < 1 or args.physics_batch_size < 1:
        parser.error("Batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu if intended")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    embedding_config = checkpoint.get("embedding_metadata", {})
    model_name = args.model_name or embedding_config.get("model_name")
    pooling = args.pooling or embedding_config.get("pooling", "mean")
    max_length = args.max_length or int(embedding_config.get("max_length", 512))
    dtype = args.dtype or embedding_config.get("dtype", "bfloat16")
    task_ids, texts, records = load_inputs(args)

    if args.embeddings:
        embeddings = load_cached_embeddings(args.embeddings, task_ids)
    else:
        if not model_name:
            raise ValueError("Checkpoint has no model_name metadata; pass --model-name")
        embeddings = embed_texts(
            texts,
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            batch_size=args.batch_size,
            dtype=dtype,
            device_name=args.device,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        torch.cuda.empty_cache()
    if embeddings.shape[1] != config["input_dim"]:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match checkpoint {config['input_dim']}"
        )

    device = torch.device(args.device)
    mlp = PaperExploreMLP(**config).to(device)
    mlp.load_state_dict(checkpoint["model_state"])
    mlp.eval().requires_grad_(False)
    with torch.inference_mode():
        output = mlp(torch.from_numpy(embeddings).to(device))
    latent = output["latent"].float().cpu().numpy()
    feasibility = output["feasibility_logit"].sigmoid().float().cpu().numpy()
    unsupported = output["unsupported_logits"].sigmoid().float().cpu().numpy()
    performance = (
        output["performance_standardized"].float().cpu()
        * checkpoint["performance_std"].float()
        + checkpoint["performance_mean"].float()
    ).numpy()
    designs = decode_design(latent, args.capacity_formula, args.capacity_multiplier)

    physics: dict[str, np.ndarray] | None = None
    if args.physics_model != "none":
        parts: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(latent), args.physics_batch_size):
            result = hard_cutoff_metrics(
                torch.from_numpy(latent[start : start + args.physics_batch_size]).double(),
                args.physics_model,
                args.capacity_formula,
                args.time_points,
                args.calibration_rate,
                args.calibration_iterations,
                args.capacity_multiplier,
            )
            for name, values in result.items():
                parts.setdefault(name, []).append(values)
            print(json.dumps({"physics_processed": min(start + args.physics_batch_size, len(latent)), "total": len(latent)}))
        physics = {name: np.concatenate(values) for name, values in parts.items()}

    labels = checkpoint.get("unsupported_labels", ())
    output_rows = []
    for index, (task_id, text) in enumerate(zip(task_ids, texts, strict=True)):
        row: dict[str, Any] = {
            "task_id": task_id,
            "requirement_text": text,
            "predicted_latent": latent[index].tolist(),
            "predicted_design": designs[index],
            "predicted_feasibility_probability": float(feasibility[index]),
            "predicted_performance_proxy": dict(zip(PERFORMANCE_FIELDS, performance[index].tolist(), strict=True)),
            "unsupported_requirement_probability": dict(zip(labels, unsupported[index].tolist(), strict=True)),
        }
        if physics is not None:
            row["physics"] = {name: value[index].item() for name, value in physics.items()}
        output_rows.append(row)

    summary: dict[str, Any] = {
        "samples": len(output_rows),
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "pooling": pooling,
        "max_length": max_length,
        "physics_model": args.physics_model,
        "mean_feasibility_probability": float(feasibility.mean()),
    }
    if records:
        summary.update(label_metrics(records, latent, feasibility))
    if physics is not None:
        valid = physics["status"] == 1
        summary["physics_valid_rate"] = float(valid.mean())
        for key in ("energy_wh_kg", "energy_retention_5c", "energy_retention_6c"):
            summary[f"physics_mean_{key}"] = float(np.mean(physics[key][valid])) if valid.any() else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
