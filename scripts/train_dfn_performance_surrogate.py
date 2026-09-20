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

from gradcell.language import DEFAULT_PERFORMANCE_FIELDS, DFNPerformanceSurrogate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
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
        raise ValueError("Every teacher design must use the same parameter fields")
    values = np.asarray([float(updates[name]["multiplier"]) for name in names], dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Design multipliers must be finite and positive")
    return np.log(values)


def performance_log_vector(performance: dict[str, Any], fields: list[str]) -> np.ndarray:
    missing = [name for name in fields if name not in performance]
    if missing:
        raise ValueError(f"Verified performance is missing fields: {missing}")
    values = np.asarray([float(performance[name]) for name in fields], dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Selected performance metrics must be finite and positive")
    return np.log(values)


def nominal_parameter_values(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    nominal_rows = []
    for row in rows:
        updates = row["teacher_design"]["parameter_updates"]
        nominal_rows.append(
            [float(updates[name]["value"]) / float(updates[name]["multiplier"]) for name in names]
        )
    nominal = np.asarray(nominal_rows, dtype=np.float64)
    if np.any(~np.isfinite(nominal)) or np.any(nominal <= 0.0):
        raise ValueError("Nominal parameter values must be finite and positive")
    reference = nominal[0]
    if not np.allclose(nominal, reference[None, :], rtol=1e-6, atol=1e-10):
        raise ValueError("Dataset does not use one consistent nominal parameter vector")
    return reference.astype(np.float32)


def unique_physical_designs(
    rows: list[dict[str, Any]], performance_fields: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if not rows:
        raise ValueError("Dataset is empty")
    parameter_names = list(rows[0]["teacher_design"]["parameter_updates"])
    by_design: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for row in rows:
        design = design_log_vector(row["teacher_design"], parameter_names)
        performance = performance_log_vector(row["verified_performance"], performance_fields)
        value = (design, performance, str(row["split"]))
        design_id = str(row["physical_design_id"])
        if design_id in by_design:
            previous = by_design[design_id]
            if (
                previous[2] != value[2]
                or not np.allclose(previous[0], design, rtol=0.0, atol=1e-7)
                or not np.allclose(previous[1], performance, rtol=0.0, atol=1e-7)
            ):
                raise ValueError(f"Physical design {design_id!r} has inconsistent labels or splits")
        else:
            by_design[design_id] = value
    design_ids = list(by_design)
    designs = np.stack([by_design[key][0] for key in design_ids])
    performance = np.stack([by_design[key][1] for key in design_ids])
    splits = np.asarray([by_design[key][2] for key in design_ids])
    return designs, performance, splits, design_ids, parameter_names


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    index = torch.from_numpy(indices.astype(np.int64))
    return DataLoader(
        TensorDataset(x[index], y[index]),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def regression_metrics(
    prediction_standardized: torch.Tensor,
    target_standardized: torch.Tensor,
    performance_mean: torch.Tensor,
    performance_std: torch.Tensor,
    fields: list[str],
) -> dict[str, Any]:
    prediction_log = prediction_standardized * performance_std + performance_mean
    target_log = target_standardized * performance_std + performance_mean
    difference = prediction_log - target_log
    absolute_fraction = (difference.exp() - 1.0).abs()
    centered = target_log - target_log.mean(dim=0)
    r2 = 1.0 - difference.square().sum(dim=0) / centered.square().sum(dim=0).clamp_min(1e-12)
    return {
        "standardized_mae": float((prediction_standardized - target_standardized).abs().mean()),
        "mean_log_mae": float(difference.abs().mean()),
        "mean_absolute_percentage_error": float(absolute_fraction.mean()),
        "per_field": {
            name: {
                "log_mae": float(difference[:, index].abs().mean()),
                "mean_absolute_percentage_error": float(absolute_fraction[:, index].mean()),
                "r2_log_space": float(r2[index]),
            }
            for index, name in enumerate(fields)
        },
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    performance_mean: torch.Tensor,
    performance_std: torch.Tensor,
    fields: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for features, target in loader:
            predictions.append(model(features.to(device)).cpu())
            targets.append(target)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    loss = float(torch.nn.functional.mse_loss(prediction, target))
    return loss, regression_metrics(prediction, target, performance_mean, performance_std, fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a differentiable surrogate for strict DFN performance labels."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--performance-fields", nargs="+", default=list(DEFAULT_PERFORMANCE_FIELDS))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-validation-mape", type=float, default=0.10)
    parser.add_argument("--max-validation-field-mape", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.hidden_dim < 1:
        parser.error("batch size, epochs, and hidden dimension must be positive")
    if len(set(args.performance_fields)) != len(args.performance_fields):
        parser.error("--performance-fields cannot contain duplicates")
    if args.max_validation_mape <= 0.0 or args.max_validation_field_mape <= 0.0:
        parser.error("surrogate validation thresholds must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    rows = read_jsonl(args.data)
    designs, performance, splits, design_ids, parameter_names = unique_physical_designs(
        rows, args.performance_fields
    )
    nominal_values = nominal_parameter_values(rows, parameter_names)
    indices = {name: np.flatnonzero(splits == name) for name in ("train", "validation", "test")}
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError("train, validation, and test physical designs must be non-empty")
    train = indices["train"]
    design_mean = designs[train].mean(axis=0)
    design_std = np.maximum(designs[train].std(axis=0), 1e-6)
    performance_mean = performance[train].mean(axis=0)
    performance_std = np.maximum(performance[train].std(axis=0), 1e-6)
    normalized_designs = (designs - design_mean) / design_std
    normalized_performance = (performance - performance_mean) / performance_std
    design_lower = normalized_designs[train].min(axis=0)
    design_upper = normalized_designs[train].max(axis=0)
    if np.any(design_upper <= design_lower):
        raise ValueError("Every design parameter must vary in the training split")

    x = torch.from_numpy(normalized_designs.astype(np.float32))
    y = torch.from_numpy(normalized_performance.astype(np.float32))
    loaders = {
        name: make_loader(
            x,
            y,
            index,
            args.batch_size,
            name == "train",
            args.seed,
        )
        for name, index in indices.items()
    }
    model_config = {
        "input_dim": x.shape[1],
        "output_dim": y.shape[1],
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
    }
    model = DFNPerformanceSurrogate(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    mean_tensor = torch.from_numpy(performance_mean.astype(np.float32))
    std_tensor = torch.from_numpy(performance_std.astype(np.float32))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, target in loaders["train"]:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite surrogate loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        validation_loss, validation_metrics = evaluate(
            model,
            loaders["validation"],
            device,
            mean_tensor,
            std_tensor,
            args.performance_fields,
        )
        record = {
            "epoch": epoch,
            "train_standardized_mse": total / max(seen, 1),
            "validation_standardized_mse": validation_loss,
            "validation_mean_absolute_percentage_error": validation_metrics[
                "mean_absolute_percentage_error"
            ],
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            patience = 0
            torch.save(
                {
                    "schema": "gradcell.dfn_performance_surrogate.v1",
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "parameter_names": parameter_names,
                    "nominal_parameter_values": torch.from_numpy(nominal_values),
                    "performance_fields": args.performance_fields,
                    "design_log_mean": torch.from_numpy(design_mean),
                    "design_log_std": torch.from_numpy(design_std),
                    "performance_log_mean": mean_tensor,
                    "performance_log_std": std_tensor,
                    "design_lower_standardized": torch.from_numpy(design_lower),
                    "design_upper_standardized": torch.from_numpy(design_upper),
                    "dataset": str(args.data),
                    "dataset_sha256": sha256(args.data),
                    "split_sizes": {name: len(index) for name, index in indices.items()},
                    "physical_design_ids": design_ids,
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
    validation_loss, validation_metrics = evaluate(
        model,
        loaders["validation"],
        device,
        mean_tensor,
        std_tensor,
        args.performance_fields,
    )
    test_loss, test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        mean_tensor,
        std_tensor,
        args.performance_fields,
    )
    probe = torch.zeros(1, x.shape[1], device=device, requires_grad=True)
    model(probe).sum().backward()
    gradient = probe.grad
    gradient_qa = {
        "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
        "nonzero": bool(gradient is not None and gradient.abs().max() > 0),
        "maximum_absolute_gradient": float(gradient.abs().max()) if gradient is not None else 0.0,
    }
    if not gradient_qa["finite"] or not gradient_qa["nonzero"]:
        raise RuntimeError(f"Surrogate input-gradient QA failed: {gradient_qa}")
    maximum_field_mape = max(
        value["mean_absolute_percentage_error"]
        for value in validation_metrics["per_field"].values()
    )
    quality_gate = {
        "maximum_mean_absolute_percentage_error": args.max_validation_mape,
        "maximum_per_field_mean_absolute_percentage_error": (args.max_validation_field_mape),
        "observed_mean_absolute_percentage_error": validation_metrics[
            "mean_absolute_percentage_error"
        ],
        "observed_maximum_per_field_mean_absolute_percentage_error": (maximum_field_mape),
        "passed": bool(
            validation_metrics["mean_absolute_percentage_error"] <= args.max_validation_mape
            and maximum_field_mape <= args.max_validation_field_mape
        ),
    }
    report = {
        "schema": "gradcell.dfn_performance_surrogate_metrics.v1",
        "dataset": str(args.data),
        "physical_designs": len(designs),
        "split_sizes": {name: len(index) for name, index in indices.items()},
        "parameter_names": parameter_names,
        "nominal_parameter_values": nominal_values.tolist(),
        "performance_fields": args.performance_fields,
        "best_validation_standardized_mse": best_validation,
        "validation": validation_metrics,
        "test_standardized_mse": test_loss,
        "test": test_metrics,
        "gradient_qa": gradient_qa,
        "quality_gate": quality_gate,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not quality_gate["passed"]:
        raise RuntimeError(
            "DFN performance surrogate failed its validation quality gate; "
            "physics-guided language training must not use this checkpoint"
        )


if __name__ == "__main__":
    main()
