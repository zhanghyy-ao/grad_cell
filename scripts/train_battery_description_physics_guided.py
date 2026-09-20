from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gradcell.language import (
    DFNPerformanceSurrogate,
    SingleDesignPhysicsMLP,
    freeze_surrogate,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_log_vector(row: dict[str, Any], names: list[str]) -> np.ndarray:
    updates = row["teacher_design"]["parameter_updates"]
    if set(updates) != set(names):
        raise ValueError("Teacher design fields do not match the surrogate")
    values = np.asarray([float(updates[name]["multiplier"]) for name in names], dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Teacher multipliers must be finite and positive")
    return np.log(values)


def performance_log_vector(row: dict[str, Any], fields: list[str]) -> np.ndarray:
    performance = row["verified_performance"]
    missing = [name for name in fields if name not in performance]
    if missing:
        raise ValueError(f"Verified performance is missing fields: {missing}")
    values = np.asarray([float(performance[name]) for name in fields], dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Selected performance values must be finite and positive")
    return np.log(values)


def load_surrogate(
    checkpoint_path: Path, device: torch.device
) -> tuple[DFNPerformanceSurrogate, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "gradcell.dfn_performance_surrogate.v1":
        raise ValueError("Unsupported performance surrogate checkpoint schema")
    surrogate = DFNPerformanceSurrogate(**checkpoint["model_config"])
    surrogate.load_state_dict(checkpoint["model_state"])
    return freeze_surrogate(surrogate.to(device)), checkpoint


def prepare_data(
    data_path: Path,
    embedding_path: Path,
    surrogate_checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(data_path)
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Dataset task IDs must be unique")
    if surrogate_checkpoint["dataset_sha256"] != sha256(data_path):
        raise ValueError("Surrogate checkpoint was trained from different dataset content")
    with np.load(embedding_path, allow_pickle=False) as arrays:
        embeddings = arrays["embeddings"].astype(np.float32)
        task_ids = arrays["task_ids"].astype(str)
        embedding_splits = arrays["splits"].astype(str)
        embedding_metadata = json.loads(str(arrays["metadata"]))
    if embedding_metadata.get("dataset_sha256") != sha256(data_path):
        raise ValueError("Embedding cache was generated from different dataset content")
    if len(task_ids) != len(rows) or set(task_ids) != set(by_id):
        raise ValueError("Embedding task IDs do not match the strict dataset")
    ordered = [by_id[task_id] for task_id in task_ids]
    splits = np.asarray([str(row["split"]) for row in ordered])
    if not np.array_equal(splits, embedding_splits):
        raise ValueError("Embedding split labels do not match the strict dataset")
    parameter_sets = sorted({row["teacher_design"]["base_parameter_set"] for row in ordered})
    modes = sorted({row["teacher_design"]["generation_mode"] for row in ordered})
    if len(parameter_sets) != 1 or len(modes) != 1:
        raise ValueError(
            "This trainer requires one parameter set and one generation mode; "
            f"found parameter_sets={parameter_sets}, modes={modes}"
        )

    names = list(surrogate_checkpoint["parameter_names"])
    fields = list(surrogate_checkpoint["performance_fields"])
    design_mean = surrogate_checkpoint["design_log_mean"].numpy()
    design_std = surrogate_checkpoint["design_log_std"].numpy()
    performance_mean = surrogate_checkpoint["performance_log_mean"].numpy()
    performance_std = surrogate_checkpoint["performance_log_std"].numpy()
    design_log = np.stack([design_log_vector(row, names) for row in ordered])
    performance_log = np.stack([performance_log_vector(row, fields) for row in ordered])
    normalized_design = (design_log - design_mean) / design_std
    normalized_performance = (performance_log - performance_mean) / performance_std
    tensors = {
        "embeddings": torch.from_numpy(embeddings),
        "design": torch.from_numpy(normalized_design.astype(np.float32)),
        "performance": torch.from_numpy(normalized_performance.astype(np.float32)),
    }
    metadata = {
        "task_ids": task_ids,
        "splits": splits,
        "parameter_names": names,
        "performance_fields": fields,
        "parameter_sets": parameter_sets,
        "generation_modes": modes,
        "nominal_parameter_values": surrogate_checkpoint["nominal_parameter_values"].float(),
        "design_log_mean": torch.from_numpy(design_mean.astype(np.float32)),
        "design_log_std": torch.from_numpy(design_std.astype(np.float32)),
        "performance_log_mean": torch.from_numpy(performance_mean.astype(np.float32)),
        "performance_log_std": torch.from_numpy(performance_std.astype(np.float32)),
        "embedding_metadata": embedding_metadata,
    }
    return tensors, metadata, ordered


def make_loader(
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    index = torch.from_numpy(indices.astype(np.int64))
    return DataLoader(
        TensorDataset(
            tensors["embeddings"][index],
            tensors["design"][index],
            tensors["performance"][index],
            index,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        pin_memory=pin_memory,
    )


def batch_loss(
    predicted_design: torch.Tensor,
    target_design: torch.Tensor,
    predicted_performance: torch.Tensor,
    target_performance: torch.Tensor,
    design_weight: float,
    performance_weight: float,
    feasibility_weight: float,
    support_weight: float,
    design_log_mean: torch.Tensor,
    design_log_std: torch.Tensor,
    nominal_parameter_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    design_loss = torch.nn.functional.smooth_l1_loss(predicted_design, target_design)
    performance_loss = torch.nn.functional.smooth_l1_loss(predicted_performance, target_performance)
    predicted_log_multipliers = predicted_design * design_log_std + design_log_mean
    predicted_multipliers = torch.exp(predicted_log_multipliers)
    physical_values = predicted_multipliers * nominal_parameter_values
    positive_overfill = torch.relu(physical_values[:, 0] + physical_values[:, 3] - 1.0)
    negative_overfill = torch.relu(physical_values[:, 1] + physical_values[:, 4] - 1.0)
    feasibility_loss = (positive_overfill.square() + negative_overfill.square()).mean()
    squared_changes = predicted_log_multipliers.square()
    support_loss = (squared_changes.sum(dim=1) - squared_changes.max(dim=1).values).mean()
    total = (
        design_weight * design_loss
        + performance_weight * performance_loss
        + feasibility_weight * feasibility_loss
        + support_weight * support_loss
    )
    return total, design_loss, performance_loss, feasibility_loss, support_loss


def evaluate(
    model: SingleDesignPhysicsMLP,
    surrogate: DFNPerformanceSurrogate,
    loader: DataLoader,
    device: torch.device,
    metadata: dict[str, Any],
    design_weight: float,
    performance_weight: float,
    feasibility_weight: float,
    support_weight: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    prediction_designs = []
    target_designs = []
    prediction_performance = []
    target_performance = []
    stored_indices = []
    total_loss = 0.0
    design_loss_sum = 0.0
    performance_loss_sum = 0.0
    feasibility_loss_sum = 0.0
    support_loss_sum = 0.0
    seen = 0
    with torch.inference_mode():
        for embeddings, design, performance, indices in loader:
            embeddings = embeddings.to(device)
            design = design.to(device)
            performance = performance.to(device)
            predicted_design = model(embeddings)
            predicted_physics = surrogate(predicted_design)
            loss, design_loss, physics_loss, feasibility_loss, support_loss = batch_loss(
                predicted_design,
                design,
                predicted_physics,
                performance,
                design_weight,
                performance_weight,
                feasibility_weight,
                support_weight,
                metadata["design_log_mean"].to(device),
                metadata["design_log_std"].to(device),
                metadata["nominal_parameter_values"].to(device),
            )
            count = len(embeddings)
            total_loss += float(loss) * count
            design_loss_sum += float(design_loss) * count
            performance_loss_sum += float(physics_loss) * count
            feasibility_loss_sum += float(feasibility_loss) * count
            support_loss_sum += float(support_loss) * count
            seen += count
            prediction_designs.append(predicted_design.cpu())
            target_designs.append(design.cpu())
            prediction_performance.append(predicted_physics.cpu())
            target_performance.append(performance.cpu())
            stored_indices.append(indices.numpy())
    predicted_design = torch.cat(prediction_designs)
    target_design = torch.cat(target_designs)
    predicted_performance = torch.cat(prediction_performance)
    target_performance = torch.cat(target_performance)
    design_mean = metadata["design_log_mean"]
    design_std = metadata["design_log_std"]
    performance_mean = metadata["performance_log_mean"]
    performance_std = metadata["performance_log_std"]
    predicted_design_log = predicted_design * design_std + design_mean
    target_design_log = target_design * design_std + design_mean
    predicted_performance_log = predicted_performance * performance_std + performance_mean
    target_performance_log = target_performance * performance_std + performance_mean
    design_difference = predicted_design_log - target_design_log
    performance_difference = predicted_performance_log - target_performance_log
    design_fraction = (design_difference.exp() - 1.0).abs()
    performance_fraction = (performance_difference.exp() - 1.0).abs()
    metrics = {
        "loss": total_loss / max(seen, 1),
        "design_loss": design_loss_sum / max(seen, 1),
        "surrogate_performance_loss": performance_loss_sum / max(seen, 1),
        "structural_feasibility_loss": feasibility_loss_sum / max(seen, 1),
        "regular_mode_support_loss": support_loss_sum / max(seen, 1),
        "design_log_mae": float(design_difference.abs().mean()),
        "design_mean_absolute_percentage_error": float(design_fraction.mean()),
        "surrogate_performance_log_mae": float(performance_difference.abs().mean()),
        "surrogate_performance_mean_absolute_percentage_error": float(performance_fraction.mean()),
        "surrogate_all_metrics_within_5pct_rate": float(
            (performance_fraction <= 0.05).all(dim=1).float().mean()
        ),
        "per_parameter": {
            name: {
                "log_mae": float(design_difference[:, column].abs().mean()),
                "mean_absolute_percentage_error": float(design_fraction[:, column].mean()),
            }
            for column, name in enumerate(metadata["parameter_names"])
        },
        "per_performance_field": {
            name: {
                "log_mae": float(performance_difference[:, column].abs().mean()),
                "mean_absolute_percentage_error": float(performance_fraction[:, column].mean()),
            }
            for column, name in enumerate(metadata["performance_fields"])
        },
    }
    arrays = {
        "indices": np.concatenate(stored_indices),
        "predicted_design_log": predicted_design_log.numpy(),
        "target_design_log": target_design_log.numpy(),
        "predicted_performance_log": predicted_performance_log.numpy(),
        "target_performance_log": target_performance_log.numpy(),
    }
    return metrics, arrays


def physics_gradient_qa(
    model: SingleDesignPhysicsMLP,
    surrogate: DFNPerformanceSurrogate,
    embeddings: torch.Tensor,
    target_performance: torch.Tensor,
) -> dict[str, Any]:
    model.train()
    predicted_design = model(embeddings)
    predicted_performance = surrogate(predicted_design)
    physics_loss = torch.nn.functional.smooth_l1_loss(predicted_performance, target_performance)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(physics_loss, parameters, allow_unused=True)
    finite = all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    total_norm = (
        sum(
            float(gradient.detach().square().sum())
            for gradient in gradients
            if gradient is not None
        )
        ** 0.5
    )
    surrogate_has_gradient = any(parameter.grad is not None for parameter in surrogate.parameters())
    result = {
        "finite": bool(finite),
        "design_model_gradient_norm": total_norm,
        "surrogate_parameter_gradients_present": surrogate_has_gradient,
    }
    if not finite or total_norm <= 0.0 or surrogate_has_gradient:
        raise RuntimeError(f"Physics-gradient QA failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train natural language -> one design with gradients from a frozen "
            "differentiable DFN performance surrogate."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--surrogate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--design-weight", type=float, default=0.25)
    parser.add_argument("--performance-weight", type=float, default=1.0)
    parser.add_argument("--feasibility-weight", type=float, default=10.0)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--performance-warmup-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.hidden_dim < 1:
        parser.error("batch size, epochs, and hidden dimension must be positive")
    if (
        args.design_weight < 0.0
        or args.performance_weight <= 0.0
        or args.feasibility_weight < 0.0
        or args.support_weight < 0.0
    ):
        parser.error("loss weights must be non-negative and performance weight positive")
    if args.performance_warmup_epochs < 0:
        parser.error("performance warmup epochs cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    surrogate, surrogate_checkpoint = load_surrogate(args.surrogate_checkpoint, device)
    tensors, metadata, ordered_rows = prepare_data(args.data, args.embeddings, surrogate_checkpoint)
    splits = metadata["splits"]
    split_indices = {
        name: np.flatnonzero(splits == name) for name in ("train", "validation", "test")
    }
    if any(len(value) == 0 for value in split_indices.values()):
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
        for name, index in split_indices.items()
    }
    lower = surrogate_checkpoint["design_lower_standardized"].float()
    upper = surrogate_checkpoint["design_upper_standardized"].float()
    model_config = {
        "input_dim": tensors["embeddings"].shape[1],
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "design_lower": lower.tolist(),
        "design_upper": upper.tolist(),
    }
    model = SingleDesignPhysicsMLP(**model_config).to(device)
    probe_count = min(args.batch_size, len(split_indices["train"]))
    probe_indices = torch.from_numpy(split_indices["train"][:probe_count])
    gradient_qa = physics_gradient_qa(
        model,
        surrogate,
        tensors["embeddings"][probe_indices].to(device),
        tensors["performance"][probe_indices].to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ramp = (
            1.0
            if args.performance_warmup_epochs == 0
            else min(epoch / args.performance_warmup_epochs, 1.0)
        )
        epoch_performance_weight = args.performance_weight * ramp
        total = 0.0
        total_design = 0.0
        total_performance = 0.0
        total_feasibility = 0.0
        total_support = 0.0
        seen = 0
        for embeddings, design, performance, _ in loaders["train"]:
            embeddings = embeddings.to(device, non_blocking=True)
            design = design.to(device, non_blocking=True)
            performance = performance.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_design = model(embeddings)
            predicted_performance = surrogate(predicted_design)
            (
                loss,
                design_loss,
                performance_loss,
                feasibility_loss,
                support_loss,
            ) = batch_loss(
                predicted_design,
                design,
                predicted_performance,
                performance,
                args.design_weight,
                epoch_performance_weight,
                args.feasibility_weight,
                args.support_weight,
                metadata["design_log_mean"].to(device),
                metadata["design_log_std"].to(device),
                metadata["nominal_parameter_values"].to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            count = len(embeddings)
            total += float(loss.detach()) * count
            total_design += float(design_loss.detach()) * count
            total_performance += float(performance_loss.detach()) * count
            total_feasibility += float(feasibility_loss.detach()) * count
            total_support += float(support_loss.detach()) * count
            seen += count
        validation, _ = evaluate(
            model,
            surrogate,
            loaders["validation"],
            device,
            metadata,
            args.design_weight,
            args.performance_weight,
            args.feasibility_weight,
            args.support_weight,
        )
        record = {
            "epoch": epoch,
            "performance_weight_ramp": epoch_performance_weight,
            "train_loss": total / max(seen, 1),
            "train_design_loss": total_design / max(seen, 1),
            "train_surrogate_performance_loss": total_performance / max(seen, 1),
            "train_structural_feasibility_loss": total_feasibility / max(seen, 1),
            "train_regular_mode_support_loss": total_support / max(seen, 1),
            **{
                f"validation_{name}": value
                for name, value in validation.items()
                if not isinstance(value, dict)
            },
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
                    "schema": "gradcell.language_single_design_physics_guided.v1",
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "parameter_names": metadata["parameter_names"],
                    "performance_fields": metadata["performance_fields"],
                    "parameter_sets": metadata["parameter_sets"],
                    "generation_modes": metadata["generation_modes"],
                    "design_log_mean": metadata["design_log_mean"],
                    "design_log_std": metadata["design_log_std"],
                    "performance_log_mean": metadata["performance_log_mean"],
                    "performance_log_std": metadata["performance_log_std"],
                    "embedding_metadata": metadata["embedding_metadata"],
                    "surrogate_checkpoint": str(args.surrogate_checkpoint),
                    "surrogate_checkpoint_sha256": sha256(args.surrogate_checkpoint),
                    "dataset": str(args.data),
                    "dataset_sha256": sha256(args.data),
                    "gradient_qa": gradient_qa,
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
    validation_metrics, _ = evaluate(
        model,
        surrogate,
        loaders["validation"],
        device,
        metadata,
        args.design_weight,
        args.performance_weight,
        args.feasibility_weight,
        args.support_weight,
    )
    test_metrics, test_arrays = evaluate(
        model,
        surrogate,
        loaders["test"],
        device,
        metadata,
        args.design_weight,
        args.performance_weight,
        args.feasibility_weight,
        args.support_weight,
    )
    prediction_rows = []
    predicted_multipliers = np.exp(test_arrays["predicted_design_log"])
    predicted_performance = np.exp(test_arrays["predicted_performance_log"])
    target_performance = np.exp(test_arrays["target_performance_log"])
    for position, row_index in enumerate(test_arrays["indices"]):
        source = ordered_rows[int(row_index)]
        prediction_rows.append(
            {
                "schema": "gradcell.language_single_design_prediction.v1",
                "task_id": source["task_id"],
                "physical_design_id": source["physical_design_id"],
                "split": source["split"],
                "battery_description": source["battery_description"],
                "base_parameter_set": source["teacher_design"]["base_parameter_set"],
                "generation_mode": source["teacher_design"]["generation_mode"],
                "predicted_parameter_multipliers": dict(
                    zip(
                        metadata["parameter_names"],
                        predicted_multipliers[position].tolist(),
                        strict=True,
                    )
                ),
                "surrogate_predicted_performance": dict(
                    zip(
                        metadata["performance_fields"],
                        predicted_performance[position].tolist(),
                        strict=True,
                    )
                ),
                "target_verified_performance": dict(
                    zip(
                        metadata["performance_fields"],
                        target_performance[position].tolist(),
                        strict=True,
                    )
                ),
            }
        )
    write_jsonl(args.output_dir / "test_predictions.jsonl", prediction_rows)
    report = {
        "schema": "gradcell.language_single_design_physics_metrics.v1",
        "best_validation_loss": best_validation,
        "validation": validation_metrics,
        "test": test_metrics,
        "split_sizes": {name: len(index) for name, index in split_indices.items()},
        "single_design_output": True,
        "gradient_source": "frozen_differentiable_DFN_performance_surrogate",
        "gradient_qa": gradient_qa,
        "parameter_names": metadata["parameter_names"],
        "performance_fields": metadata["performance_fields"],
        "warning": (
            "Performance metrics in this file come from the frozen surrogate; run "
            "strict DFN replay before making physical-performance claims."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
