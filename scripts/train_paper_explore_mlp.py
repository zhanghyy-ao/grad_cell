from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_explore_reporting import (
    plot_supervised_benchmark,
    plot_training_history,
    write_metrics_csv,
)

UNSUPPORTED_LABELS = (
    "cycle_life",
    "safety",
    "cost",
    "low_temperature",
    "finished_cell_geometry",
)


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


class PaperExploreMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float, latent_limit: float
    ) -> None:
        super().__init__()
        self.latent_limit = latent_limit
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            *(ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)),
            nn.LayerNorm(hidden_dim),
        )
        self.latent_head = nn.Linear(hidden_dim, 5)
        self.feasibility_head = nn.Linear(hidden_dim, 1)
        self.performance_head = nn.Linear(hidden_dim, 3)
        self.unsupported_head = nn.Linear(hidden_dim, len(UNSUPPORTED_LABELS))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(inputs)
        return {
            "latent": self.latent_limit * torch.tanh(self.latent_head(hidden)),
            "feasibility_logit": self.feasibility_head(hidden).squeeze(-1),
            "performance_standardized": self.performance_head(hidden),
            "unsupported_logits": self.unsupported_head(hidden),
        }


@dataclass(frozen=True)
class PreparedData:
    task_ids: np.ndarray
    splits: np.ndarray
    embeddings: torch.Tensor
    candidates: torch.Tensor
    candidate_mask: torch.Tensor
    performance: torch.Tensor
    feasibility: torch.Tensor
    unsupported: torch.Tensor
    performance_mean: torch.Tensor
    performance_std: torch.Tensor
    embedding_metadata: dict[str, Any]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(data_path: Path, embedding_path: Path) -> PreparedData:
    rows = load_rows(data_path)
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Dataset task IDs must be unique")
    with np.load(embedding_path, allow_pickle=False) as arrays:
        embeddings = arrays["embeddings"].astype(np.float32)
        task_ids = arrays["task_ids"].astype(str)
        splits = arrays["splits"].astype(str)
        metadata = json.loads(str(arrays["metadata"]))
    expected_hash = metadata.get("dataset_sha256")
    if expected_hash is not None and expected_hash != file_sha256(data_path):
        raise ValueError("Embedding cache was generated from a different dataset content hash")
    if len(task_ids) != len(rows) or any(task_id not in by_id for task_id in task_ids):
        raise ValueError("Embedding task IDs do not match the dataset")
    ordered = [by_id[task_id] for task_id in task_ids]
    row_splits = np.asarray([str(row["split"]) for row in ordered])
    if not np.array_equal(row_splits, splits):
        raise ValueError("Embedding split labels do not match the dataset")

    max_candidates = max(len(row.get("teacher_candidates", [])) for row in ordered)
    if max_candidates < 1:
        raise ValueError("No teacher candidates were found")
    candidates = np.zeros((len(ordered), max_candidates, 5), dtype=np.float32)
    candidate_mask = np.zeros((len(ordered), max_candidates), dtype=bool)
    performance = np.zeros((len(ordered), 3), dtype=np.float32)
    feasibility = np.zeros(len(ordered), dtype=np.float32)
    unsupported = np.zeros((len(ordered), len(UNSUPPORTED_LABELS)), dtype=np.float32)
    for row_index, row in enumerate(ordered):
        candidate_rows = row.get("teacher_candidates") or [
            {
                "latent": row["teacher_latent"],
                "energy_1c_wh_kg": row["teacher_performance"]["energy_1c_wh_kg"],
                "retention_5c": row["teacher_performance"]["retention_5c"],
                "retention_6c": row["teacher_performance"]["retention_6c"],
            }
        ]
        for candidate_index, candidate in enumerate(candidate_rows):
            latent = np.asarray(candidate["latent"], dtype=np.float32)
            if latent.shape != (5,) or not np.isfinite(latent).all():
                raise ValueError(f"Invalid teacher latent for {row['task_id']}")
            candidates[row_index, candidate_index] = latent
            candidate_mask[row_index, candidate_index] = True
        primary = candidate_rows[0]
        performance[row_index] = [
            primary["energy_1c_wh_kg"],
            primary["retention_5c"],
            primary["retention_6c"],
        ]
        feasibility[row_index] = float(bool(row["requirement_feasible"]))
        flags = set(row["requirements_canonical"].get("unverified_requirements", []))
        unsupported[row_index] = [float(name in flags) for name in UNSUPPORTED_LABELS]

    train_feasible = (splits == "train") & (feasibility > 0.5)
    if not train_feasible.any():
        raise ValueError("Training set contains no feasible samples")
    performance_mean = performance[train_feasible].mean(axis=0)
    performance_std = performance[train_feasible].std(axis=0)
    performance_std = np.maximum(performance_std, 1e-6)
    performance = (performance - performance_mean) / performance_std
    return PreparedData(
        task_ids=task_ids,
        splits=splits,
        embeddings=torch.from_numpy(embeddings),
        candidates=torch.from_numpy(candidates),
        candidate_mask=torch.from_numpy(candidate_mask),
        performance=torch.from_numpy(performance),
        feasibility=torch.from_numpy(feasibility),
        unsupported=torch.from_numpy(unsupported),
        performance_mean=torch.from_numpy(performance_mean),
        performance_std=torch.from_numpy(performance_std),
        embedding_metadata=metadata,
    )


def subset(data: PreparedData, indices: np.ndarray) -> TensorDataset:
    index = torch.from_numpy(indices.astype(np.int64))
    return TensorDataset(
        data.embeddings[index],
        data.candidates[index],
        data.candidate_mask[index],
        data.performance[index],
        data.feasibility[index],
        data.unsupported[index],
        index,
    )


def batch_loss(
    outputs: dict[str, torch.Tensor],
    batch: tuple[torch.Tensor, ...],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    _, candidates, candidate_mask, performance, feasibility, unsupported, _ = batch
    feasible = feasibility > 0.5
    distance = torch.nn.functional.smooth_l1_loss(
        outputs["latent"].unsqueeze(1).expand_as(candidates), candidates, reduction="none"
    ).mean(dim=-1)
    distance = distance.masked_fill(~candidate_mask, torch.inf)
    latent_per_row = distance.min(dim=1).values
    latent_loss = latent_per_row[feasible].mean() if feasible.any() else latent_per_row.sum() * 0.0
    performance_loss = (
        torch.nn.functional.smooth_l1_loss(
            outputs["performance_standardized"][feasible], performance[feasible]
        )
        if feasible.any()
        else outputs["performance_standardized"].sum() * 0.0
    )
    feasibility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["feasibility_logit"], feasibility
    )
    unsupported_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["unsupported_logits"], unsupported
    )
    total = (
        weights["latent"] * latent_loss
        + weights["performance"] * performance_loss
        + weights["feasibility"] * feasibility_loss
        + weights["unsupported"] * unsupported_loss
    )
    parts = {
        "loss": float(total.detach()),
        "latent_loss": float(latent_loss.detach()),
        "performance_loss": float(performance_loss.detach()),
        "feasibility_loss": float(feasibility_loss.detach()),
        "unsupported_loss": float(unsupported_loss.detach()),
    }
    return total, parts


def evaluate(
    model: PaperExploreMLP,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    stored: dict[str, list[np.ndarray]] = {
        "indices": [], "latent": [], "feasibility": [], "performance_standardized": []
    }
    with torch.inference_mode():
        for raw_batch in loader:
            batch = tuple(value.to(device) for value in raw_batch)
            outputs = model(batch[0])
            _, parts = batch_loss(outputs, batch, weights)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
            stored["indices"].append(batch[-1].cpu().numpy())
            stored["latent"].append(outputs["latent"].cpu().numpy())
            stored["feasibility"].append(torch.sigmoid(outputs["feasibility_logit"]).cpu().numpy())
            stored["performance_standardized"].append(
                outputs["performance_standardized"].cpu().numpy()
            )
    return (
        {name: value / max(batches, 1) for name, value in totals.items()},
        {name: np.concatenate(values) for name, values in stored.items()},
    )


def classification_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    truth = target >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper_explore embedding MLP")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-limit", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--performance-weight", type=float, default=0.3)
    parser.add_argument("--feasibility-weight", type=float, default=0.3)
    parser.add_argument("--unsupported-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.plot_every < 1:
        parser.error("--plot-every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu if intended")
    device = torch.device(args.device)
    data = prepare_data(args.data, args.embeddings)
    split_indices = {
        name: np.flatnonzero(data.splits == name) for name in ("train", "validation", "test")
    }
    if args.max_train_samples:
        split_indices["train"] = split_indices["train"][: args.max_train_samples]
    if any(len(values) == 0 for values in split_indices.values()):
        raise ValueError("train, validation, and test splits must all be non-empty")

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            subset(data, indices),
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, indices in split_indices.items()
    }
    model = PaperExploreMLP(
        input_dim=data.embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        latent_limit=args.latent_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    weights = {
        "latent": args.latent_weight,
        "performance": args.performance_weight,
        "feasibility": args.feasibility_weight,
        "unsupported": args.unsupported_weight,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_batches = 0
        for raw_batch in loaders["train"]:
            batch = tuple(value.to(device, non_blocking=True) for value in raw_batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[0])
            loss, parts = batch_loss(outputs, batch, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            for name, value in parts.items():
                train_totals[name] = train_totals.get(name, 0.0) + value
            train_batches += 1
        validation, _ = evaluate(model, loaders["validation"], device, weights)
        record = {
            "epoch": float(epoch),
            "train_loss": train_totals["loss"] / max(train_batches, 1),
            "validation_loss": validation["loss"],
        }
        record.update(
            {
                f"train_{name}": value / max(train_batches, 1)
                for name, value in train_totals.items()
                if name != "loss"
            }
        )
        record.update(
            {f"validation_{name}": value for name, value in validation.items() if name != "loss"}
        )
        history.append(record)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if not args.no_plots and (epoch == 1 or epoch % args.plot_every == 0):
            plot_training_history(history, args.output_dir)
        print(json.dumps(record))
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "input_dim": data.embeddings.shape[1],
                        "hidden_dim": args.hidden_dim,
                        "num_blocks": args.num_blocks,
                        "dropout": args.dropout,
                        "latent_limit": args.latent_limit,
                    },
                    "performance_mean": data.performance_mean,
                    "performance_std": data.performance_std,
                    "unsupported_labels": UNSUPPORTED_LABELS,
                    "embedding_metadata": data.embedding_metadata,
                    "training_args": vars(args),
                    "best_validation_loss": best_validation,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_losses, prediction = evaluate(model, loaders["test"], device, weights)
    test_indices = prediction["indices"].astype(np.int64)
    feasible = data.feasibility[test_indices].numpy() > 0.5
    candidates = data.candidates[test_indices].numpy()
    candidate_mask = data.candidate_mask[test_indices].numpy()
    predicted_latent = prediction["latent"]
    distances = np.abs(predicted_latent[:, None, :] - candidates).mean(axis=-1)
    distances[~candidate_mask] = np.inf
    latent_mae = float(distances.min(axis=1)[feasible].mean()) if feasible.any() else float("nan")
    predicted_performance = (
        prediction["performance_standardized"] * data.performance_std.numpy()
        + data.performance_mean.numpy()
    )
    target_performance = (
        data.performance[test_indices].numpy() * data.performance_std.numpy()
        + data.performance_mean.numpy()
    )
    metrics = {
        "best_validation_loss": best_validation,
        "epochs_trained": len(history),
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "test_losses": test_losses,
        "test_latent_topk_mae": latent_mae,
        "test_feasibility": classification_metrics(
            prediction["feasibility"], data.feasibility[test_indices].numpy()
        ),
        "test_performance_mae_feasible": {
            name: float(np.abs(predicted_performance[feasible, column] - target_performance[feasible, column]).mean())
            for column, name in enumerate(("energy_1c_wh_kg", "retention_5c", "retention_6c"))
        },
        "embedding_metadata": data.embedding_metadata,
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_metrics_csv(metrics, args.output_dir / "benchmark_metrics.csv")
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        task_ids=data.task_ids[test_indices],
        latent=predicted_latent,
        feasibility_probability=prediction["feasibility"],
        performance=predicted_performance,
    )
    if not args.no_plots:
        plot_training_history(history, args.output_dir)
        plot_supervised_benchmark(
            predicted_latent=predicted_latent,
            candidates=candidates,
            candidate_mask=candidate_mask,
            feasible=feasible,
            feasibility_probability=prediction["feasibility"],
            predicted_performance=predicted_performance,
            target_performance=target_performance,
            output_dir=args.output_dir,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
