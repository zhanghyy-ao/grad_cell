from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class BatteryDescriptionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        parameter_count: int,
        parameter_set_count: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            *(ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)),
            nn.LayerNorm(hidden_dim),
        )
        self.parameter_head = nn.Linear(hidden_dim, parameter_count)
        self.parameter_set_head = nn.Linear(hidden_dim, parameter_set_count)
        self.mode_head = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(inputs)
        return {
            "parameters_standardized": self.parameter_head(hidden),
            "parameter_set_logits": self.parameter_set_head(hidden),
            "mode_logits": self.mode_head(hidden),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(
    data_path: Path, embedding_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = read_jsonl(data_path)
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Dataset task IDs must be unique")
    with np.load(embedding_path, allow_pickle=False) as arrays:
        embeddings = arrays["embeddings"].astype(np.float32)
        task_ids = arrays["task_ids"].astype(str)
        splits = arrays["splits"].astype(str)
        embedding_metadata = json.loads(str(arrays["metadata"]))
    if embedding_metadata.get("dataset_sha256") != sha256(data_path):
        raise ValueError("Embedding cache was generated from different dataset content")
    ordered = [by_id[task_id] for task_id in task_ids]
    if not np.array_equal(splits, np.asarray([row["split"] for row in ordered])):
        raise ValueError("Embedding split labels do not match dataset")

    parameter_names = list(ordered[0]["teacher_design"]["parameter_updates"])
    parameter_sets = sorted(
        {row["teacher_design"]["base_parameter_set"] for row in ordered}
    )
    modes = ("regular", "extreme")
    set_to_index = {name: index for index, name in enumerate(parameter_sets)}
    mode_to_index = {name: index for index, name in enumerate(modes)}
    log_parameters = np.asarray(
        [
            [
                np.log(row["teacher_design"]["parameter_updates"][name]["multiplier"])
                for name in parameter_names
            ]
            for row in ordered
        ],
        dtype=np.float32,
    )
    parameter_set_targets = np.asarray(
        [set_to_index[row["teacher_design"]["base_parameter_set"]] for row in ordered],
        dtype=np.int64,
    )
    mode_targets = np.asarray(
        [mode_to_index[row["teacher_design"]["generation_mode"]] for row in ordered],
        dtype=np.int64,
    )
    train = splits == "train"
    mean = log_parameters[train].mean(axis=0)
    std = np.maximum(log_parameters[train].std(axis=0), 1e-6)
    standardized = (log_parameters - mean) / std
    tensors = {
        "embeddings": torch.from_numpy(embeddings),
        "parameters": torch.from_numpy(standardized),
        "parameter_sets": torch.from_numpy(parameter_set_targets),
        "modes": torch.from_numpy(mode_targets),
    }
    metadata = {
        "task_ids": task_ids,
        "splits": splits,
        "parameter_names": parameter_names,
        "parameter_sets": parameter_sets,
        "modes": list(modes),
        "parameter_log_mean": torch.from_numpy(mean),
        "parameter_log_std": torch.from_numpy(std),
        "embedding_metadata": embedding_metadata,
    }
    return tensors, metadata


def make_loader(
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    index = torch.from_numpy(indices.astype(np.int64))
    dataset = TensorDataset(
        tensors["embeddings"][index],
        tensors["parameters"][index],
        tensors["parameter_sets"][index],
        tensors["modes"][index],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        pin_memory=pin_memory,
    )


def batch_loss(
    outputs: dict[str, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    parameters, parameter_sets, modes = targets
    parameter_loss = torch.nn.functional.smooth_l1_loss(
        outputs["parameters_standardized"], parameters
    )
    parameter_set_loss = torch.nn.functional.cross_entropy(
        outputs["parameter_set_logits"], parameter_sets
    )
    mode_loss = torch.nn.functional.cross_entropy(outputs["mode_logits"], modes)
    total = (
        weights["parameter"] * parameter_loss
        + weights["parameter_set"] * parameter_set_loss
        + weights["mode"] * mode_loss
    )
    return total, {
        "loss": float(total.detach()),
        "parameter_loss": float(parameter_loss.detach()),
        "parameter_set_loss": float(parameter_set_loss.detach()),
        "mode_loss": float(mode_loss.detach()),
    }


def evaluate(
    model: BatteryDescriptionMLP,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    log_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    absolute_error = 0.0
    parameter_count = 0
    set_correct = 0
    mode_correct = 0
    rows = 0
    with torch.inference_mode():
        for embedding, parameters, parameter_sets, modes in loader:
            embedding = embedding.to(device)
            parameters = parameters.to(device)
            parameter_sets = parameter_sets.to(device)
            modes = modes.to(device)
            outputs = model(embedding)
            _, parts = batch_loss(
                outputs, (parameters, parameter_sets, modes), weights
            )
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
            predicted_log = outputs["parameters_standardized"] * log_std.to(device)
            target_log = parameters * log_std.to(device)
            absolute_error += float((predicted_log - target_log).abs().sum())
            parameter_count += target_log.numel()
            set_correct += int(
                (outputs["parameter_set_logits"].argmax(dim=-1) == parameter_sets).sum()
            )
            mode_correct += int((outputs["mode_logits"].argmax(dim=-1) == modes).sum())
            rows += len(embedding)
            batches += 1
    result = {name: value / max(batches, 1) for name, value in totals.items()}
    result.update(
        {
            "log_multiplier_mae": absolute_error / max(parameter_count, 1),
            "parameter_set_accuracy": set_correct / max(rows, 1),
            "mode_accuracy": mode_correct / max(rows, 1),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-Qwen embedding MLP for battery parameter reconstruction."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--parameter-weight", type=float, default=1.0)
    parser.add_argument("--parameter-set-weight", type=float, default=0.25)
    parser.add_argument("--mode-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    tensors, metadata = prepare_data(args.data, args.embeddings)
    splits = metadata["splits"]
    indices = {
        name: np.flatnonzero(splits == name)
        for name in ("train", "validation", "test")
    }
    if any(len(index) == 0 for index in indices.values()):
        raise ValueError("train, validation, and test must all be non-empty")
    loaders = {
        name: make_loader(
            tensors,
            index,
            args.batch_size,
            name == "train",
            args.seed,
            device.type == "cuda",
        )
        for name, index in indices.items()
    }
    model_config = {
        "input_dim": tensors["embeddings"].shape[1],
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "parameter_count": len(metadata["parameter_names"]),
        "parameter_set_count": len(metadata["parameter_sets"]),
    }
    model = BatteryDescriptionMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    weights = {
        "parameter": args.parameter_weight,
        "parameter_set": args.parameter_set_weight,
        "mode": args.mode_weight,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for embedding, parameters, parameter_sets, modes in loaders["train"]:
            embedding = embedding.to(device, non_blocking=True)
            targets = (
                parameters.to(device, non_blocking=True),
                parameter_sets.to(device, non_blocking=True),
                modes.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model(embedding), targets, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            weights,
            metadata["parameter_log_std"],
        )
        record = {
            "epoch": epoch,
            "train_loss": total / max(batches, 1),
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "parameter_names": metadata["parameter_names"],
                    "parameter_sets": metadata["parameter_sets"],
                    "modes": metadata["modes"],
                    "parameter_log_mean": metadata["parameter_log_mean"],
                    "parameter_log_std": metadata["parameter_log_std"],
                    "embedding_metadata": metadata["embedding_metadata"],
                    "training_args": vars(args),
                },
                args.output_dir / "best_model.pt",
            )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break
    checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        weights,
        metadata["parameter_log_std"],
    )
    metrics = {
        "best_validation_loss": best_validation,
        "test": test_metrics,
        "split_sizes": {name: len(index) for name, index in indices.items()},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
