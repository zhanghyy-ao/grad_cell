from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Dataset is empty")
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")
    if any(not str(row.get("requirement_text", "")).strip() for row in rows):
        raise ValueError("Every row must contain a non-empty requirement_text")
    return rows


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        expanded = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)
    if method == "last":
        token_positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        positions = (token_positions * mask.long()).max(dim=1).values
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, positions]
    raise ValueError(f"Unknown pooling method: {method}")


def atomic_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return values[name]


def model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine a real model input device")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Qwen embeddings")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if (
        args.dtype == "bfloat16"
        and args.device.startswith("cuda")
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("The selected CUDA device does not support bfloat16")

    rows = load_rows(args.data)
    task_ids = np.asarray([str(row["task_id"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    texts = [str(row["requirement_text"]) for row in rows]
    dataset_hash = file_sha256(args.data)
    partial = args.output.with_suffix(args.output.suffix + ".partial")

    existing_embeddings: list[np.ndarray] = []
    start = 0
    if partial.exists() and not args.overwrite:
        with np.load(partial, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            expected = {
                "dataset_sha256": dataset_hash,
                "model_name": args.model_name,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "dtype": args.dtype,
                "load_in_4bit": args.load_in_4bit,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Partial embedding cache does not match the current settings")
            saved_ids = saved["task_ids"].astype(str)
            if not np.array_equal(saved_ids, task_ids[: len(saved_ids)]):
                raise RuntimeError("Partial embedding task IDs do not match the dataset prefix")
            existing_embeddings.append(saved["embeddings"].astype(np.float32))
            start = len(saved_ids)
            print(f"Resuming embedding extraction at row {start}/{len(rows)}")
    elif args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}; use --overwrite to replace it")
        return

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(args.dtype),
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype_from_name(args.dtype)

    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    if not args.load_in_4bit:
        model.to(torch.device(args.device))
    model.eval()
    model.requires_grad_(False)
    device = model_input_device(model)
    metadata = {
        "dataset": str(args.data),
        "dataset_sha256": dataset_hash,
        "model_name": args.model_name,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "device": str(device),
    }

    embedding_parts = existing_embeddings
    with torch.inference_mode():
        for begin in range(start, len(texts), args.batch_size):
            end = min(begin + args.batch_size, len(texts))
            tokens = tokenizer(
                texts[begin:end],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                pad_to_multiple_of=8 if device.type == "cuda" else None,
                return_tensors="pt",
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            outputs = model(**tokens, return_dict=True)
            pooled = masked_pool(outputs.last_hidden_state, tokens["attention_mask"], args.pooling)
            embedding_parts.append(pooled.float().cpu().numpy())
            combined = np.concatenate(embedding_parts, axis=0)
            atomic_save(
                partial,
                embeddings=combined,
                task_ids=task_ids[:end],
                splits=splits[:end],
                metadata=np.asarray(json.dumps(metadata)),
            )
            print(f"Embedded {end}/{len(texts)}")

    embeddings = np.concatenate(embedding_parts, axis=0)
    if embeddings.shape[0] != len(rows) or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding output is incomplete or contains non-finite values")
    metadata["rows"] = len(rows)
    metadata["embedding_dim"] = int(embeddings.shape[1])
    atomic_save(
        args.output,
        embeddings=embeddings,
        task_ids=task_ids,
        splits=splits,
        metadata=np.asarray(json.dumps(metadata)),
    )
    partial.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
