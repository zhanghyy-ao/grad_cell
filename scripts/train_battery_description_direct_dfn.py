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

from gradcell.benchmark.dfn_parameter import PARAMETER_FIELDS
from gradcell.language import (
    DEFAULT_PERFORMANCE_FIELDS,
    DirectDFNPerformanceLayer,
    SingleDesignPhysicsMLP,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def design_vector(row: dict[str, Any]) -> np.ndarray:
    updates = row["teacher_design"]["parameter_updates"]
    if set(updates) != set(PARAMETER_FIELDS):
        raise ValueError("Teacher design fields do not match the direct DFN inputs")
    values = np.asarray(
        [float(updates[name]["multiplier"]) for name in PARAMETER_FIELDS], dtype=np.float32
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Teacher parameter multipliers must be finite and positive")
    return np.log(values)


def performance_vector(row: dict[str, Any]) -> np.ndarray:
    performance = row["verified_performance"]
    values = np.asarray(
        [float(performance[name]) for name in DEFAULT_PERFORMANCE_FIELDS], dtype=np.float32
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("DFN performance targets must be finite and positive")
    return np.log(values)


def prepare_data(
    data_path: Path, embedding_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(data_path)
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Dataset task IDs must be unique")
    with np.load(embedding_path, allow_pickle=False) as arrays:
        embeddings = arrays["embeddings"].astype(np.float32)
        task_ids = arrays["task_ids"].astype(str)
        embedding_splits = arrays["splits"].astype(str)
        embedding_metadata = json.loads(str(arrays["metadata"]))
    data_digest = sha256(data_path)
    if embedding_metadata.get("dataset_sha256") != data_digest:
        raise ValueError("Embedding cache was generated from different dataset content")
    if len(task_ids) != len(rows) or set(task_ids) != set(by_id):
        raise ValueError("Embedding task IDs do not match the dataset")
    ordered = [by_id[task_id] for task_id in task_ids]
    splits = np.asarray([str(row["split"]) for row in ordered])
    if not np.array_equal(splits, embedding_splits):
        raise ValueError("Embedding split labels do not match the dataset")
    parameter_sets = sorted({row["teacher_design"]["base_parameter_set"] for row in ordered})
    modes = sorted({row["teacher_design"]["generation_mode"] for row in ordered})
    if len(parameter_sets) != 1 or len(modes) != 1:
        raise ValueError(
            "Direct DFN training requires one parameter set and one generation mode; "
            f"found parameter_sets={parameter_sets}, modes={modes}"
        )
    design_log = np.stack([design_vector(row) for row in ordered])
    performance_log = np.stack([performance_vector(row) for row in ordered])
    reference_capacity = np.asarray(
        [float(row["verified_performance"]["reference_capacity_ah"]) for row in ordered],
        dtype=np.float32,
    )
    train = splits == "train"
    if not train.any():
        raise ValueError("Training split is empty")
    design_mean = design_log[train].mean(axis=0)
    design_std = design_log[train].std(axis=0).clip(min=1e-4)
    performance_mean = performance_log[train].mean(axis=0)
    performance_std = performance_log[train].std(axis=0).clip(min=1e-4)
    normalized_design = (design_log - design_mean) / design_std
    normalized_performance = (performance_log - performance_mean) / performance_std
    lower = normalized_design[train].min(axis=0) - 0.05
    upper = normalized_design[train].max(axis=0) + 0.05
    tensors = {
        "embeddings": torch.from_numpy(embeddings),
        "design": torch.from_numpy(normalized_design.astype(np.float32)),
        "performance": torch.from_numpy(normalized_performance.astype(np.float32)),
        "reference_capacity": torch.from_numpy(reference_capacity),
    }
    metadata = {
        "task_ids": task_ids,
        "splits": splits,
        "parameter_sets": parameter_sets,
        "generation_modes": modes,
        "parameter_names": list(PARAMETER_FIELDS),
        "performance_fields": list(DEFAULT_PERFORMANCE_FIELDS),
        "design_log_mean": torch.from_numpy(design_mean),
        "design_log_std": torch.from_numpy(design_std),
        "performance_log_mean": torch.from_numpy(performance_mean),
        "performance_log_std": torch.from_numpy(performance_std),
        "design_lower_standardized": torch.from_numpy(lower.astype(np.float32)),
        "design_upper_standardized": torch.from_numpy(upper.astype(np.float32)),
        "embedding_metadata": embedding_metadata,
        "dataset_sha256": data_digest,
    }
    return tensors, metadata, ordered


def make_loader(
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    index = torch.from_numpy(indices.astype(np.int64))
    return DataLoader(
        TensorDataset(
            tensors["embeddings"][index],
            tensors["design"][index],
            tensors["performance"][index],
            tensors["reference_capacity"][index],
            index,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def decode_design(
    normalized_design: torch.Tensor,
    design_mean: torch.Tensor,
    design_std: torch.Tensor,
    nominal_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    design_log = normalized_design * design_std + design_mean
    multipliers = torch.exp(design_log)
    return design_log, multipliers, multipliers * nominal_values


def structural_losses(
    design_log: torch.Tensor, physical_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_overfill = torch.relu(physical_values[:, 0] + physical_values[:, 3] - 1.0)
    negative_overfill = torch.relu(physical_values[:, 1] + physical_values[:, 4] - 1.0)
    feasibility = (positive_overfill.square() + negative_overfill.square()).mean()
    squared_changes = design_log.square()
    support = (squared_changes.sum(dim=1) - squared_changes.max(dim=1).values).mean()
    return feasibility, support


def direct_loss(
    predicted_design: torch.Tensor,
    target_design: torch.Tensor,
    target_performance: torch.Tensor,
    reference_capacity: torch.Tensor,
    physics: DirectDFNPerformanceLayer,
    metadata: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    design_mean = metadata["design_log_mean"].to(predicted_design.device)
    design_std = metadata["design_log_std"].to(predicted_design.device)
    performance_mean = metadata["performance_log_mean"].to(predicted_design.device)
    performance_std = metadata["performance_log_std"].to(predicted_design.device)
    nominal = physics.nominal_parameter_values.to(predicted_design)
    design_log, _, physical_values = decode_design(
        predicted_design, design_mean, design_std, nominal
    )
    dfn = physics(physical_values, reference_capacity)
    normalized_performance = (
        torch.log(dfn.performance.clamp_min(1e-8)) - performance_mean
    ) / performance_std
    valid = dfn.status
    design_loss = torch.nn.functional.smooth_l1_loss(predicted_design, target_design)
    if valid.any():
        performance_loss = torch.nn.functional.smooth_l1_loss(
            normalized_performance[valid], target_performance[valid]
        )
    else:
        performance_loss = predicted_design.sum() * 0.0
    feasibility_loss, support_loss = structural_losses(design_log, physical_values)
    failure_loss = (~valid).to(predicted_design.dtype).mean()
    total = (
        weights["design"] * design_loss
        + weights["performance"] * performance_loss
        + weights["feasibility"] * feasibility_loss
        + weights["support"] * support_loss
    )
    components = {
        "design": design_loss,
        "performance": performance_loss,
        "feasibility": feasibility_loss,
        "support": support_loss,
        "failure_rate": failure_loss,
        "runtime_s": dfn.runtime_s.sum(),
    }
    return total, components, dfn.performance, valid


def evaluate(
    model: SingleDesignPhysicsMLP,
    physics: DirectDFNPerformanceLayer,
    loader: DataLoader,
    device: torch.device,
    metadata: dict[str, Any],
    weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    sums = {name: 0.0 for name in ("loss", "design", "performance", "feasibility", "support")}
    seen = 0
    successes = 0
    runtime_s = 0.0
    predicted_designs = []
    predicted_performances = []
    targets = []
    indices = []
    with torch.no_grad():
        for embedding, design, performance, capacity, index in loader:
            embedding = embedding.to(device)
            design = design.to(device)
            performance = performance.to(device)
            capacity = capacity.to(device)
            predicted_design = model(embedding)
            loss, components, predicted_performance, valid = direct_loss(
                predicted_design,
                design,
                performance,
                capacity,
                physics,
                metadata,
                weights,
            )
            count = len(embedding)
            sums["loss"] += float(loss) * count
            for name in ("design", "performance", "feasibility", "support"):
                sums[name] += float(components[name]) * count
            successes += int(valid.sum())
            runtime_s += float(components["runtime_s"])
            seen += count
            predicted_designs.append(predicted_design.cpu())
            predicted_performances.append(predicted_performance.cpu())
            targets.append(performance.cpu())
            indices.append(index.numpy())
    metrics = {name: value / max(seen, 1) for name, value in sums.items()}
    metrics["dfn_success_rate"] = successes / max(seen, 1)
    metrics["dfn_runtime_s"] = runtime_s
    arrays = {
        "indices": np.concatenate(indices),
        "predicted_design": torch.cat(predicted_designs).numpy(),
        "predicted_performance": torch.cat(predicted_performances).numpy(),
        "target_performance": torch.cat(targets).numpy(),
    }
    return metrics, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train language-to-design MLP with direct PyBaMM DFN sensitivities."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--design-weight", type=float, default=0.25)
    parser.add_argument("--performance-weight", type=float, default=1.0)
    parser.add_argument("--feasibility-weight", type=float, default=10.0)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--maximum-duration-factor", type=float, default=1.5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--cutoff-v", type=float, default=2.5)
    parser.add_argument("--gate-temperature-v", type=float, default=0.02)
    parser.add_argument("--current-ramp-time-s", type=float, default=1.0)
    parser.add_argument("--training-voltage-floor-v", type=float, default=2.0)
    parser.add_argument("--minimum-dfn-success-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.hidden_dim, args.time_points) < 1:
        parser.error("batch size, epochs, hidden dimension, and time points must be positive")
    if not 0.0 <= args.minimum_dfn_success_rate <= 1.0:
        parser.error("minimum DFN success rate must be between zero and one")
    if args.training_voltage_floor_v >= args.cutoff_v:
        parser.error("training voltage floor must be below the soft cutoff voltage")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    print(json.dumps({"direct_dfn_stage": "load_data"}), flush=True)
    tensors, metadata, ordered_rows = prepare_data(args.data, args.embeddings)
    split_indices = {
        name: np.flatnonzero(metadata["splits"] == name) for name in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise ValueError("train, validation, and test must all be non-empty")
    loaders = {
        name: make_loader(tensors, indices, args.batch_size, name == "train", args.seed)
        for name, indices in split_indices.items()
    }
    print(
        json.dumps(
            {
                "direct_dfn_stage": "initialize_physics",
                "pybamm_model": "DFN",
                "parameter_set": metadata["parameter_sets"][0],
            }
        ),
        flush=True,
    )
    physics = DirectDFNPerformanceLayer(
        parameter_set=metadata["parameter_sets"][0],
        time_points=args.time_points,
        maximum_duration_factor=args.maximum_duration_factor,
        rtol=args.rtol,
        atol=args.atol,
        cutoff_v=args.cutoff_v,
        gate_temperature_v=args.gate_temperature_v,
        current_ramp_time_s=args.current_ramp_time_s,
        training_voltage_floor_v=args.training_voltage_floor_v,
    ).to(device)
    print(json.dumps({"direct_dfn_stage": "initialize_mlp"}), flush=True)
    model_config = {
        "input_dim": tensors["embeddings"].shape[1],
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "design_lower": metadata["design_lower_standardized"].tolist(),
        "design_upper": metadata["design_upper_standardized"].tolist(),
    }
    model = SingleDesignPhysicsMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    weights = {
        "design": args.design_weight,
        "performance": args.performance_weight,
        "feasibility": args.feasibility_weight,
        "support": args.support_weight,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    patience = 0
    history = []
    gradient_qa = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "design", "performance", "feasibility", "support")}
        successes = 0
        seen = 0
        runtime_s = 0.0
        for embedding, design, performance, capacity, _ in loaders["train"]:
            embedding = embedding.to(device)
            design = design.to(device)
            performance = performance.to(device)
            capacity = capacity.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_design = model(embedding)
            loss, components, _, valid = direct_loss(
                predicted_design,
                design,
                performance,
                capacity,
                physics,
                metadata,
                weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            )
            if gradient_qa is None:
                gradient_qa = {
                    "source": "PyBaMM_DFN_forward_sensitivities",
                    "finite": bool(np.isfinite(gradient_norm)),
                    "design_model_gradient_norm": gradient_norm,
                }
                if not gradient_qa["finite"] or gradient_norm <= 0.0:
                    raise RuntimeError(f"Direct DFN gradient QA failed: {gradient_qa}")
            optimizer.step()
            count = len(embedding)
            totals["loss"] += float(loss.detach()) * count
            for name in ("design", "performance", "feasibility", "support"):
                totals[name] += float(components[name].detach()) * count
            successes += int(valid.sum())
            runtime_s += float(components["runtime_s"])
            seen += count
        validation, _ = evaluate(model, physics, loaders["validation"], device, metadata, weights)
        train_success_rate = successes / max(seen, 1)
        if train_success_rate < args.minimum_dfn_success_rate:
            raise RuntimeError(
                "Direct DFN success rate is too low for trustworthy physics-gradient training: "
                f"{train_success_rate:.1%} < {args.minimum_dfn_success_rate:.1%}"
            )
        record = {
            "epoch": epoch,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            "train_dfn_success_rate": train_success_rate,
            "train_dfn_runtime_s": runtime_s,
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
                    "schema": "gradcell.language_single_design_direct_dfn.v1",
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "parameter_names": metadata["parameter_names"],
                    "performance_fields": metadata["performance_fields"],
                    "parameter_sets": metadata["parameter_sets"],
                    "generation_modes": metadata["generation_modes"],
                    "nominal_parameter_values": physics.nominal_parameter_values.cpu(),
                    "design_log_mean": metadata["design_log_mean"],
                    "design_log_std": metadata["design_log_std"],
                    "performance_log_mean": metadata["performance_log_mean"],
                    "performance_log_std": metadata["performance_log_std"],
                    "embedding_metadata": metadata["embedding_metadata"],
                    "dataset": str(args.data),
                    "dataset_sha256": metadata["dataset_sha256"],
                    "gradient_qa": gradient_qa,
                    "physics_config": {
                        "model": "DFN",
                        "time_points": args.time_points,
                        "maximum_duration_factor": args.maximum_duration_factor,
                        "rtol": args.rtol,
                        "atol": args.atol,
                        "cutoff_v": args.cutoff_v,
                        "gate_temperature_v": args.gate_temperature_v,
                        "current_ramp_time_s": args.current_ramp_time_s,
                        "training_voltage_floor_v": args.training_voltage_floor_v,
                    },
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
        model, physics, loaders["validation"], device, metadata, weights
    )
    test_metrics, arrays = evaluate(model, physics, loaders["test"], device, metadata, weights)
    design_mean = metadata["design_log_mean"].numpy()
    design_std = metadata["design_log_std"].numpy()
    predicted_multipliers = np.exp(arrays["predicted_design"] * design_std + design_mean)
    target_log = (
        arrays["target_performance"] * metadata["performance_log_std"].numpy()
        + metadata["performance_log_mean"].numpy()
    )
    target_performance = np.exp(target_log)
    prediction_rows = []
    for position, row_index in enumerate(arrays["indices"]):
        source = ordered_rows[int(row_index)]
        prediction_rows.append(
            {
                "schema": "gradcell.language_single_design_direct_dfn_prediction.v1",
                "task_id": source["task_id"],
                "physical_design_id": source["physical_design_id"],
                "split": source["split"],
                "battery_description": source["battery_description"],
                "base_parameter_set": source["teacher_design"]["base_parameter_set"],
                "generation_mode": source["teacher_design"]["generation_mode"],
                "predicted_parameter_multipliers": dict(
                    zip(PARAMETER_FIELDS, predicted_multipliers[position].tolist(), strict=True)
                ),
                "direct_dfn_predicted_performance": dict(
                    zip(
                        DEFAULT_PERFORMANCE_FIELDS,
                        arrays["predicted_performance"][position].tolist(),
                        strict=True,
                    )
                ),
                "target_verified_performance": dict(
                    zip(
                        DEFAULT_PERFORMANCE_FIELDS,
                        target_performance[position].tolist(),
                        strict=True,
                    )
                ),
            }
        )
    write_jsonl(args.output_dir / "test_predictions.jsonl", prediction_rows)
    report = {
        "schema": "gradcell.language_single_design_direct_dfn_metrics.v1",
        "best_validation_loss": best_validation,
        "validation": validation_metrics,
        "test": test_metrics,
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "single_design_output": True,
        "gradient_source": "PyBaMM_DFN_forward_sensitivities",
        "uses_performance_surrogate": False,
        "gradient_qa": gradient_qa,
        "parameter_names": list(PARAMETER_FIELDS),
        "performance_fields": list(DEFAULT_PERFORMANCE_FIELDS),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
