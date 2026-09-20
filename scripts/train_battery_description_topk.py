from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
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


class BatteryDescriptionTopKMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        parameter_count: int,
        candidate_count: int,
    ) -> None:
        super().__init__()
        self.parameter_count = parameter_count
        self.candidate_count = candidate_count
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            *(ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)),
            nn.LayerNorm(hidden_dim),
        )
        self.parameter_head = nn.Linear(
            hidden_dim, candidate_count * parameter_count
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.parameter_head(self.trunk(inputs))
        return values.view(-1, self.candidate_count, self.parameter_count)


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


def design_log_vector(design: dict[str, Any], names: list[str]) -> np.ndarray:
    updates = design["parameter_updates"]
    if set(updates) != set(names):
        raise ValueError("Every teacher design must contain the same parameter fields")
    values = np.asarray(
        [float(updates[name]["multiplier"]) for name in names], dtype=np.float32
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Teacher multipliers must be finite and positive")
    return np.log(values)


def row_targets(row: dict[str, Any], names: list[str]) -> list[np.ndarray]:
    designs = [row["teacher_design"]]
    designs.extend(
        alternative["teacher_design"]
        for alternative in row.get("alternative_teacher_designs", [])
    )
    unique = []
    seen = set()
    base_set = row["teacher_design"]["base_parameter_set"]
    for design in designs:
        if design["base_parameter_set"] != base_set:
            raise ValueError("Top-K target alternatives cannot cross parameter sets")
        vector = design_log_vector(design, names)
        key = tuple(float(value) for value in vector)
        if key not in seen:
            seen.add(key)
            unique.append(vector)
    return unique


def prepare_data(
    data_path: Path, embedding_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError("Dataset is empty")
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
    if len(task_ids) != len(rows) or set(task_ids) != set(by_id):
        raise ValueError("Embedding task IDs do not match the dataset")
    ordered = [by_id[task_id] for task_id in task_ids]
    expected_splits = np.asarray([row["split"] for row in ordered])
    if not np.array_equal(splits, expected_splits):
        raise ValueError("Embedding split labels do not match the strict dataset")
    parameter_names = list(ordered[0]["teacher_design"]["parameter_updates"])
    parameter_sets = sorted(
        {row["teacher_design"]["base_parameter_set"] for row in ordered}
    )
    modes = sorted({row["teacher_design"]["generation_mode"] for row in ordered})
    if len(parameter_sets) != 1 or len(modes) != 1:
        raise ValueError(
            "This checkpoint trainer expects one parameter set and one generation mode; "
            f"found parameter_sets={parameter_sets}, modes={modes}"
        )

    target_lists = [row_targets(row, parameter_names) for row in ordered]
    maximum_targets = max(map(len, target_lists))
    primary = np.stack([targets[0] for targets in target_lists])
    train = splits == "train"
    if not np.any(train):
        raise ValueError("Strict dataset has no training records")
    mean = primary[train].mean(axis=0)
    std = np.maximum(primary[train].std(axis=0), 1e-6)
    padded_targets = np.zeros(
        (len(rows), maximum_targets, len(parameter_names)), dtype=np.float32
    )
    target_mask = np.zeros((len(rows), maximum_targets), dtype=bool)
    for index, targets in enumerate(target_lists):
        count = len(targets)
        padded_targets[index, :count] = (np.stack(targets) - mean) / std
        target_mask[index, :count] = True
    tensors = {
        "embeddings": torch.from_numpy(embeddings),
        "targets": torch.from_numpy(padded_targets),
        "target_mask": torch.from_numpy(target_mask),
    }
    metadata = {
        "task_ids": task_ids,
        "splits": splits,
        "parameter_names": parameter_names,
        "parameter_sets": parameter_sets,
        "modes": modes,
        "parameter_log_mean": torch.from_numpy(mean),
        "parameter_log_std": torch.from_numpy(std),
        "maximum_targets": maximum_targets,
        "target_count_distribution": dict(Counter(map(len, target_lists))),
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
        tensors["targets"][index],
        tensors["target_mask"][index],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        pin_memory=pin_memory,
    )


def pairwise_huber(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    difference = predictions[:, :, None, :] - targets[:, None, :, :]
    absolute = difference.abs()
    elementwise = torch.where(absolute < 1.0, 0.5 * difference.square(), absolute - 0.5)
    return elementwise.mean(dim=-1)


def set_matching_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    precision_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distances = pairwise_huber(predictions, targets)
    coverage = distances.min(dim=1).values
    coverage_loss = (coverage * target_mask).sum() / target_mask.sum().clamp_min(1)
    masked = distances.masked_fill(~target_mask[:, None, :], torch.inf)
    precision_loss = masked.min(dim=2).values.mean()
    total = coverage_loss + precision_weight * precision_loss
    return total, {
        "loss": total,
        "coverage_loss": coverage_loss,
        "precision_loss": precision_loss,
    }


def candidate_spread(predictions_log: torch.Tensor) -> tuple[float, int]:
    candidate_count = predictions_log.shape[1]
    if candidate_count < 2:
        return 0.0, 0
    differences = predictions_log[:, :, None, :] - predictions_log[:, None, :, :]
    rms = differences.square().mean(dim=-1).sqrt()
    triangle = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            dtype=torch.bool,
            device=predictions_log.device,
        ),
        diagonal=1,
    )
    values = rms[:, triangle]
    return float(values.sum()), values.numel()


def evaluate(
    model: BatteryDescriptionTopKMLP,
    loader: DataLoader,
    device: torch.device,
    precision_weight: float,
    log_mean: torch.Tensor,
    log_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = Counter()
    rows = 0
    target_values = 0
    primary_best_sum = 0.0
    coverage_sum = 0.0
    precision_sum = 0.0
    ambiguous_coverage_sum = 0.0
    ambiguous_targets = 0
    spread_sum = 0.0
    spread_values = 0
    mean = log_mean.to(device)
    std = log_std.to(device)
    with torch.inference_mode():
        for embeddings, targets, target_mask in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)
            predictions = model(embeddings)
            _, parts = set_matching_loss(
                predictions, targets, target_mask, precision_weight
            )
            batch_rows = len(embeddings)
            for name, value in parts.items():
                totals[name] += float(value) * batch_rows
            predictions_log = predictions * std + mean
            targets_log = targets * std + mean
            pair_mae = (
                predictions_log[:, :, None, :] - targets_log[:, None, :, :]
            ).abs().mean(dim=-1)
            target_best = pair_mae.min(dim=1).values
            prediction_best = pair_mae.masked_fill(
                ~target_mask[:, None, :], torch.inf
            ).min(dim=2).values
            primary_best_sum += float(target_best[:, 0].sum())
            coverage_sum += float((target_best * target_mask).sum())
            precision_sum += float(prediction_best.sum())
            target_values += int(target_mask.sum())
            ambiguous = target_mask.sum(dim=1) > 1
            if ambiguous.any():
                ambiguous_mask = target_mask & ambiguous[:, None]
                ambiguous_coverage_sum += float(
                    (target_best * ambiguous_mask).sum()
                )
                ambiguous_targets += int(ambiguous_mask.sum())
            batch_spread, batch_spread_values = candidate_spread(predictions_log)
            spread_sum += batch_spread
            spread_values += batch_spread_values
            rows += batch_rows
    result = {name: value / max(rows, 1) for name, value in totals.items()}
    result.update(
        {
            "primary_best_of_k_log_mae": primary_best_sum / max(rows, 1),
            "target_coverage_log_mae": coverage_sum / max(target_values, 1),
            "candidate_precision_log_mae": precision_sum
            / max(rows * model.candidate_count, 1),
            "ambiguous_target_coverage_log_mae": ambiguous_coverage_sum
            / max(ambiguous_targets, 1),
            "candidate_pairwise_log_rms": spread_sum / max(spread_values, 1),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-Qwen embedding MLP with Top-K set matching."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision-weight", type=float, default=0.25)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.candidate_count < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("candidate count, batch size, and epochs must be positive")
    if args.precision_weight < 0:
        parser.error("--precision-weight must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    tensors, metadata = prepare_data(args.data, args.embeddings)
    if args.candidate_count < metadata["maximum_targets"]:
        raise ValueError(
            f"candidate_count={args.candidate_count} cannot cover the maximum "
            f"{metadata['maximum_targets']} valid labels per row"
        )
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
        "candidate_count": args.candidate_count,
    }
    model = BatteryDescriptionTopKMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        rows_seen = 0
        for embeddings, targets, target_mask in loaders["train"]:
            embeddings = embeddings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(embeddings)
            loss, _ = set_matching_loss(
                predictions, targets, target_mask, args.precision_weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(embeddings)
            rows_seen += len(embeddings)
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            args.precision_weight,
            metadata["parameter_log_mean"],
            metadata["parameter_log_std"],
        )
        record = {
            "epoch": epoch,
            "train_loss": total / max(rows_seen, 1),
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
                    "maximum_targets": metadata["maximum_targets"],
                    "target_count_distribution": metadata[
                        "target_count_distribution"
                    ],
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
        args.precision_weight,
        metadata["parameter_log_mean"],
        metadata["parameter_log_std"],
    )
    metrics = {
        "best_validation_loss": best_validation,
        "test": test_metrics,
        "split_sizes": {name: len(index) for name, index in indices.items()},
        "candidate_count": args.candidate_count,
        "maximum_targets": metadata["maximum_targets"],
        "target_count_distribution": metadata["target_count_distribution"],
        "parameter_sets": metadata["parameter_sets"],
        "modes": metadata["modes"],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
