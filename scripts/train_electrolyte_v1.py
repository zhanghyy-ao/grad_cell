from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gradcell.data import group_split
from gradcell.models import ElectrolytePropertyNetwork
from gradcell.physics import (
    AnalyticElectrolyteBackend,
    DifferentiablePhysicsLayer,
    PyBaMMElectrolyteDFNBackend,
)


def conductivity_diagnostics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Diagnostics only; conductivity labels do not train or select the model."""
    error = prediction - target
    relative = (prediction.exp() - target.exp()).abs() / target.exp()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return {
        "mae_log_ms_cm": float(error.abs().mean()),
        "rmse_log_ms_cm": float(error.square().mean().sqrt()),
        "r2_log_ms_cm": float(1.0 - error.square().sum() / total),
        "median_relative_error": float(relative.median()),
    }


def select_rows(partition, code, mask, labels, rank, limit=None):
    selected = partition[mask[partition] & (labels[partition] == code)]
    selected = selected[np.argsort(rank[selected])]
    return selected if limit is None else selected[:limit]


def evaluate_physics(
    model, normalized, indices, target_voltage, current_a, layer, scale, batch_size
):
    loader = DataLoader(
        TensorDataset(normalized[indices], target_voltage[indices], current_a[indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    voltage_sse = scaled_sse = runtime_sum = 0.0
    points = successes = rows = 0
    model.eval()
    for features, voltage_y, current in loader:
        with torch.no_grad():
            log_scale = model(features)
        voltage, status, runtime = layer(torch.stack([log_scale, current], dim=-1))
        valid = status == 1
        rows += len(features)
        successes += int(valid.sum())
        runtime_sum += float(runtime.sum())
        if bool(valid.any()):
            error = voltage[valid, 0] - voltage_y[valid]
            voltage_sse += float(error.square().sum())
            scaled_sse += float((error / scale).square().sum())
            points += error.numel()
    return {
        "rows": rows,
        "success_rate": successes / max(rows, 1),
        "voltage_rmse_v": math.sqrt(voltage_sse / points) if points else None,
        "voltage_rmse_scaled": math.sqrt(scaled_sse / points) if points else None,
        "mean_runtime_s": runtime_sum / max(rows, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train formulation -> bounded log scale -> PyBaMM DFN from voltage only."
    )
    parser.add_argument("--data", type=Path, default=Path("data/electrolyte_v1.npz"))
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--physics-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--physics-samples", type=int)
    parser.add_argument("--min-conductivity-scale", type=float, default=0.5)
    parser.add_argument("--max-conductivity-scale", type=float, default=2.0)
    parser.add_argument("--voltage-scale-v", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1"))
    args = parser.parse_args()
    if args.physics_samples is not None and args.physics_samples <= 0:
        parser.error("--physics-samples must be positive")
    if not 0.0 < args.min_conductivity_scale < args.max_conductivity_scale:
        parser.error("conductivity scales must satisfy 0 < min < max")
    if args.voltage_scale_v <= 0.0:
        parser.error("--voltage-scale-v must be positive")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    with np.load(args.data, allow_pickle=False) as arrays:
        features = torch.from_numpy(arrays["features"].copy()).double()
        target_log_k = torch.from_numpy(arrays["target_log_conductivity"].copy()).double()
        groups = arrays["group_id"].copy()
        physics_mask = arrays["physics_mask"].copy()
        physics_partition = arrays["physics_partition"].copy()
        physics_rank = arrays["physics_rank"].copy()
        target_voltage = torch.from_numpy(arrays["target_voltage"].copy()).double()
        current_a = torch.from_numpy(arrays["current_a"].copy()).double()
        feature_names = arrays["feature_names"].astype(str).tolist()
        metadata = json.loads(str(arrays["metadata"]))

    prepared_seed = int(metadata.get("split_seed", metadata.get("seed", args.seed)))
    split_seed = prepared_seed if args.split_seed is None else args.split_seed
    if split_seed != prepared_seed:
        raise RuntimeError("Split seed differs from prepared physics partitions")
    train_np, validation_np, test_np = group_split(groups, split_seed)
    train_np_phys = select_rows(
        train_np, 0, physics_mask, physics_partition, physics_rank, args.physics_samples
    )
    validation_np_phys = select_rows(
        validation_np, 1, physics_mask, physics_partition, physics_rank
    )
    test_np_phys = select_rows(test_np, 2, physics_mask, physics_partition, physics_rank)
    if not len(train_np_phys):
        raise RuntimeError("No training physics rows are available")
    if not len(validation_np_phys):
        raise RuntimeError("No validation physics rows; prepare with --physics-validation-samples")

    train_indices = torch.from_numpy(train_np)
    x_mean = features[train_indices].mean(0)
    x_std = features[train_indices].std(0).clamp_min(1e-8)
    normalized = (features - x_mean) / x_std
    train_phys = torch.from_numpy(train_np_phys)
    validation_phys = torch.from_numpy(validation_np_phys)
    test_phys = torch.from_numpy(test_np_phys)
    train_loader = DataLoader(
        TensorDataset(normalized[train_phys], target_voltage[train_phys], current_a[train_phys]),
        batch_size=args.physics_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    backend_cls = (
        PyBaMMElectrolyteDFNBackend if args.physics_backend == "dfn" else AnalyticElectrolyteBackend
    )
    backend = backend_cls(
        time_points=metadata["time_points"], probe_horizon_s=metadata["probe_horizon_s"]
    )
    layer = DifferentiablePhysicsLayer(backend)
    min_log_scale, max_log_scale = map(
        math.log, (args.min_conductivity_scale, args.max_conductivity_scale)
    )
    model = ElectrolytePropertyNetwork(
        features.shape[1], args.hidden_dim, args.depth, min_log_scale, max_log_scale
    ).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_validation, best_state, stale = float("inf"), None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = updates = 0
        gradient_norm = torch.tensor(float("nan"))
        for physics_x, voltage_y, current in train_loader:
            inputs = torch.stack([model(physics_x), current], dim=-1)
            voltage, status, _ = layer(inputs)
            if not bool((status == 1).all()):
                raise RuntimeError(f"Online DFN batch failed: {backend.last_solve_diagnostics}")
            loss = torch.nn.functional.huber_loss(
                voltage[:, 0] / args.voltage_scale_v, voltage_y / args.voltage_scale_v
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            updates += 1
        validation = evaluate_physics(
            model,
            normalized,
            validation_phys,
            target_voltage,
            current_a,
            layer,
            args.voltage_scale_v,
            args.physics_batch_size,
        )
        score = validation["voltage_rmse_scaled"]
        if score is None or validation["success_rate"] < 1.0:
            score = float("inf")
        record = {
            "epoch": epoch,
            "physics_voltage_loss_scaled": loss_sum / updates,
            "validation_voltage_rmse_v": validation["voltage_rmse_v"],
            "validation_voltage_rmse_scaled": validation["voltage_rmse_scaled"],
            "validation_success_rate": validation["success_rate"],
            "last_gradient_norm_before_clip": float(gradient_norm),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score < best_validation:
            best_validation = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no valid physics checkpoint")
    model.load_state_dict(best_state)

    evaluation = {
        "validation": evaluate_physics(
            model,
            normalized,
            validation_phys,
            target_voltage,
            current_a,
            layer,
            args.voltage_scale_v,
            args.physics_batch_size,
        )
    }
    if len(test_phys):
        evaluation["test"] = evaluate_physics(
            model,
            normalized,
            test_phys,
            target_voltage,
            current_a,
            layer,
            args.voltage_scale_v,
            args.physics_batch_size,
        )
    test_indices = torch.from_numpy(test_np)
    with torch.no_grad():
        predicted_scale = model(normalized[test_indices])
    implied_log_k = predicted_scale + math.log(metadata["reference_conductivity_ms_cm"])
    report = {
        "data": str(args.data),
        "training_objective": "physics_voltage_only",
        "property_loss_enabled": False,
        "physics_backend": args.physics_backend,
        "physics_model": "PyBaMM DFN" if args.physics_backend == "dfn" else "analytic QA",
        "model_output": "bounded_log_conductivity_scale",
        "conductivity_scale_bounds": [args.min_conductivity_scale, args.max_conductivity_scale],
        "split_policy": "grouped by source DOI",
        "split_sizes": {
            "train": len(train_np),
            "validation": len(validation_np),
            "test": len(test_np),
        },
        "physics_rows": {
            "train": len(train_np_phys),
            "validation": len(validation_np_phys),
            "test": len(test_np_phys),
        },
        "best_validation_voltage_rmse_scaled": best_validation,
        "physics_evaluation": evaluation,
        "conductivity_diagnostics_not_used_for_training": conductivity_diagnostics(
            implied_log_k, target_log_k[test_indices]
        ),
        "model_seed": args.seed,
        "split_seed": split_seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {
                "input_dim": features.shape[1],
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "min_log_scale": min_log_scale,
                "max_log_scale": max_log_scale,
            },
            "normalization": {"x_mean": x_mean, "x_std": x_std},
            "feature_names": feature_names,
            "output_semantics": "log_conductivity_scale",
            "reference_conductivity_ms_cm": metadata["reference_conductivity_ms_cm"],
        },
        args.output_dir / "best_model.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        indices=test_np,
        predicted_log_conductivity_scale=predicted_scale.numpy(),
        implied_log_conductivity_ms_cm=implied_log_k.numpy(),
        observed_log_conductivity_ms_cm=target_log_k[test_indices].numpy(),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
