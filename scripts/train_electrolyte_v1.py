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


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    relative_error = (prediction.exp() - target.exp()).abs() / target.exp()
    ss_total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return {
        "mae_log_ms_cm": float(error.abs().mean()),
        "rmse_log_ms_cm": float(error.square().mean().sqrt()),
        "r2_log_ms_cm": float(1.0 - error.square().sum() / ss_total),
        "median_relative_error": float(relative_error.median()),
        "relative_error_p75": float(torch.quantile(relative_error, 0.75)),
        "relative_error_p90": float(torch.quantile(relative_error, 0.90)),
        "relative_error_p95": float(torch.quantile(relative_error, 0.95)),
        "relative_error_max": float(relative_error.max()),
    }


def error_analysis(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: np.ndarray,
    group_names: list[str],
) -> dict:
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    relative = np.abs(np.exp(prediction_np) - np.exp(target_np)) / np.exp(target_np)
    doi_rows = []
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        doi_rows.append(
            {
                "group_id": int(group_id),
                "doi": group_names[int(group_id)] if int(group_id) < len(group_names) else None,
                "rows": int(mask.sum()),
                "median_relative_error": float(np.median(relative[mask])),
                "mean_relative_error": float(relative[mask].mean()),
            }
        )
    doi_rows.sort(key=lambda row: row["mean_relative_error"], reverse=True)
    edges = np.quantile(target_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    conductivity_bins = []
    for index in range(4):
        mask = (target_np >= edges[index]) & (
            target_np <= edges[index + 1]
            if index == 3
            else target_np < edges[index + 1]
        )
        conductivity_bins.append(
            {
                "bin": index,
                "log_k_min": float(edges[index]),
                "log_k_max": float(edges[index + 1]),
                "rows": int(mask.sum()),
                "median_relative_error": float(np.median(relative[mask])),
                "mean_relative_error": float(relative[mask].mean()),
            }
        )
    return {"worst_doi_groups": doi_rows[:20], "conductivity_quartiles": conductivity_bins}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CALiSol conductivity prediction with an online differentiable DFN loss."
    )
    parser.add_argument("--data", type=Path, default=Path("data/electrolyte_v1.npz"))
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--physics-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument(
        "--physics-samples",
        type=int,
        help="Use the first N ranked training physics rows from a nested prepared dataset.",
    )
    parser.add_argument("--voltage-scale-v", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="DOI split seed; defaults to the split seed stored in the dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1"))
    args = parser.parse_args()
    if args.physics_weight < 0.0:
        parser.error("--physics-weight must be non-negative")
    if args.voltage_scale_v <= 0.0:
        parser.error("--voltage-scale-v must be positive")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    with np.load(args.data, allow_pickle=False) as arrays:
        features = torch.from_numpy(arrays["features"].copy()).double()
        target_log_k = torch.from_numpy(arrays["target_log_conductivity"].copy()).double()
        groups = arrays["group_id"].copy()
        group_names = (
            arrays["group_names"].astype(str).tolist()
            if "group_names" in arrays
            else [str(value) for value in np.unique(groups)]
        )
        physics_mask = arrays["physics_mask"].copy()
        physics_partition = (
            arrays["physics_partition"].copy()
            if "physics_partition" in arrays
            else np.where(physics_mask, 0, -1)
        )
        physics_rank = (
            arrays["physics_rank"].copy()
            if "physics_rank" in arrays
            else np.arange(len(physics_mask), dtype=np.int64)
        )
        target_voltage = torch.from_numpy(arrays["target_voltage"].copy()).double()
        current_a = torch.from_numpy(arrays["current_a"].copy()).double()
        feature_names = arrays["feature_names"].astype(str).tolist()
        metadata = json.loads(str(arrays["metadata"]))
    dataset_split_seed = int(metadata.get("split_seed", metadata.get("seed", args.seed)))
    split_seed = dataset_split_seed if args.split_seed is None else args.split_seed
    if split_seed != dataset_split_seed:
        raise RuntimeError(
            f"Requested split seed {split_seed} differs from prepared dataset split seed "
            f"{dataset_split_seed}; regenerate the dataset to prevent physics-partition leakage"
        )
    train_idx_np, validation_idx_np, test_idx_np = group_split(groups, split_seed)
    if min(len(train_idx_np), len(validation_idx_np), len(test_idx_np)) == 0:
        raise RuntimeError("DOI-group split produced an empty partition")
    train_idx = torch.from_numpy(train_idx_np)
    validation_idx = torch.from_numpy(validation_idx_np)
    test_idx = torch.from_numpy(test_idx_np)
    x_mean = features[train_idx].mean(0)
    x_std = features[train_idx].std(0).clamp_min(1e-8)
    normalized = (features - x_mean) / x_std
    target_mean = target_log_k[train_idx].mean()
    target_std = target_log_k[train_idx].std().clamp_min(1e-8)
    target_standardized = (target_log_k - target_mean) / target_std
    property_loader = DataLoader(
        TensorDataset(normalized[train_idx], target_standardized[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    use_physics = args.physics_weight > 0.0
    train_physics_np = train_idx_np[
        physics_mask[train_idx_np] & (physics_partition[train_idx_np] == 0)
    ]
    train_physics_np = train_physics_np[np.argsort(physics_rank[train_physics_np])]
    if args.physics_samples is not None:
        if args.physics_samples < 0:
            parser.error("--physics-samples must be non-negative")
        train_physics_np = train_physics_np[: args.physics_samples]
    if use_physics and len(train_physics_np) == 0:
        raise RuntimeError("No physics-supervised rows fell in the training split")
    train_physics = torch.from_numpy(train_physics_np)
    physics_loader = None
    backend = None
    physics_layer = None
    if use_physics:
        physics_loader = DataLoader(
            TensorDataset(
                normalized[train_physics],
                target_voltage[train_physics],
                current_a[train_physics],
            ),
            batch_size=args.physics_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 1),
        )
        backend_cls = (
            PyBaMMElectrolyteDFNBackend
            if args.physics_backend == "dfn"
            else AnalyticElectrolyteBackend
        )
        backend = backend_cls(
            time_points=metadata["time_points"],
            probe_horizon_s=metadata["probe_horizon_s"],
        )
        physics_layer = DifferentiablePhysicsLayer(backend)
    model = ElectrolytePropertyNetwork(
        input_dim=features.shape[1], hidden_dim=args.hidden_dim, depth=args.depth
    ).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    reference_log_k = math.log(metadata["reference_conductivity_ms_cm"])
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        physics_iterator = iter(physics_loader) if physics_loader is not None else None
        property_sum = physics_sum = total_sum = 0.0
        updates = 0
        for property_x, property_y in property_loader:
            predicted_standardized_log_k = model(property_x)
            property_loss = torch.nn.functional.huber_loss(
                predicted_standardized_log_k, property_y
            )
            voltage_loss = property_loss.new_zeros(())
            if use_physics:
                assert physics_iterator is not None
                assert physics_loader is not None
                assert physics_layer is not None
                assert backend is not None
                try:
                    physics_x, voltage_y, physics_current = next(physics_iterator)
                except StopIteration:
                    physics_iterator = iter(physics_loader)
                    physics_x, voltage_y, physics_current = next(physics_iterator)
                predicted_physics_log_k = model(physics_x) * target_std + target_mean
                physics_inputs = torch.stack(
                    [predicted_physics_log_k - reference_log_k, physics_current], dim=-1
                )
                predicted_voltage, status, _ = physics_layer(physics_inputs)
                if not bool((status == 1).all()):
                    raise RuntimeError(
                        f"Online physics batch failed: {backend.last_solve_diagnostics}"
                    )
                voltage_loss = torch.nn.functional.huber_loss(
                    predicted_voltage[:, 0] / args.voltage_scale_v,
                    voltage_y / args.voltage_scale_v,
                )
            loss = property_loss + args.physics_weight * voltage_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            property_sum += float(property_loss.detach())
            physics_sum += float(voltage_loss.detach())
            total_sum += float(loss.detach())
            updates += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.nn.functional.huber_loss(
                    model(normalized[validation_idx]),
                    target_standardized[validation_idx],
                )
            )
        mean_property_loss = property_sum / updates
        mean_physics_loss = physics_sum / updates
        mean_total_loss = total_sum / updates
        weighted_physics_loss = args.physics_weight * mean_physics_loss
        record = {
            "epoch": epoch,
            "property_loss": mean_property_loss,
            "physics_voltage_loss_scaled": mean_physics_loss,
            "physics_voltage_loss_weighted": weighted_physics_loss,
            "physics_loss_fraction": weighted_physics_loss / max(mean_total_loss, 1e-12),
            "total_loss": mean_total_loss,
            "validation_property_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prediction = model(normalized[test_idx]) * target_std + target_mean
    physics_evaluation = {}
    if use_physics:
        assert physics_layer is not None
        for partition_name, partition_code, partition_indices in (
            ("validation", 1, validation_idx_np),
            ("test", 2, test_idx_np),
        ):
            selected_np = partition_indices[
                physics_mask[partition_indices]
                & (physics_partition[partition_indices] == partition_code)
            ]
            if len(selected_np) == 0:
                continue
            selected = torch.from_numpy(selected_np)
            with torch.no_grad():
                predicted_log_k = model(normalized[selected]) * target_std + target_mean
            physics_inputs = torch.stack(
                [predicted_log_k - reference_log_k, current_a[selected]], dim=-1
            )
            predicted_voltage, status, runtime_s = physics_layer(physics_inputs)
            valid = status == 1
            voltage_error = predicted_voltage[:, 0] - target_voltage[selected]
            physics_evaluation[partition_name] = {
                "rows": len(selected_np),
                "success_rate": float(valid.double().mean()),
                "voltage_rmse_v": float(voltage_error[valid].square().mean().sqrt())
                if bool(valid.any())
                else None,
                "mean_runtime_s": float(runtime_s.mean()),
            }
    report = {
        "data": str(args.data),
        "physics_backend": args.physics_backend,
        "split_policy": "grouped by source DOI",
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(validation_idx),
            "test": len(test_idx),
        },
        "physics_train_rows": len(train_physics),
        "online_physics_enabled": use_physics,
        "physics_weight": args.physics_weight,
        "model_seed": args.seed,
        "split_seed": split_seed,
        "target_standardization": {
            "mean_log_ms_cm": float(target_mean),
            "std_log_ms_cm": float(target_std),
        },
        "best_validation_property_loss": best_validation,
        "test_metrics": metrics(test_prediction, target_log_k[test_idx]),
        "physics_evaluation": physics_evaluation,
    }
    test_error_analysis = error_analysis(
        test_prediction,
        target_log_k[test_idx],
        groups[test_idx_np],
        group_names,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {
                "input_dim": features.shape[1],
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
            },
            "normalization": {
                "x_mean": x_mean,
                "x_std": x_std,
                "target_mean_log_ms_cm": target_mean,
                "target_std_log_ms_cm": target_std,
            },
            "feature_names": feature_names,
            "reference_conductivity_ms_cm": metadata["reference_conductivity_ms_cm"],
            "split_indices": {
                "train": train_idx.tolist(),
                "validation": validation_idx.tolist(),
                "test": test_idx.tolist(),
            },
        },
        args.output_dir / "best_model.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps(test_error_analysis, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        indices=test_idx.numpy(),
        target_log_conductivity=target_log_k[test_idx].numpy(),
        predicted_log_conductivity=test_prediction.detach().numpy(),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
