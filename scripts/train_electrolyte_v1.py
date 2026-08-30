from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gradcell.models import ElectrolytePropertyNetwork
from gradcell.physics import (
    AnalyticElectrolyteBackend,
    DifferentiablePhysicsLayer,
    PyBaMMElectrolyteDFNBackend,
)


def group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = max(1, int(0.7 * len(unique)))
    validation_end = max(train_end + 1, int(0.85 * len(unique)))
    train_groups = unique[:train_end]
    validation_groups = unique[train_end:validation_end]
    test_groups = unique[validation_end:]
    return (
        np.flatnonzero(np.isin(groups, train_groups)),
        np.flatnonzero(np.isin(groups, validation_groups)),
        np.flatnonzero(np.isin(groups, test_groups)),
    )


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    ss_total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return {
        "mae_log_ms_cm": float(error.abs().mean()),
        "rmse_log_ms_cm": float(error.square().mean().sqrt()),
        "r2_log_ms_cm": float(1.0 - error.square().sum() / ss_total),
        "median_relative_error": float(
            ((prediction.exp() - target.exp()).abs() / target.exp()).median()
        ),
    }


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
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--voltage-scale-v", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    with np.load(args.data, allow_pickle=False) as arrays:
        features = torch.from_numpy(arrays["features"].copy()).double()
        target = torch.from_numpy(arrays["target_log_conductivity"].copy()).double()
        groups = arrays["group_id"].copy()
        physics_mask = arrays["physics_mask"].copy()
        target_voltage = torch.from_numpy(arrays["target_voltage"].copy()).double()
        current_a = torch.from_numpy(arrays["current_a"].copy()).double()
        feature_names = arrays["feature_names"].astype(str).tolist()
        metadata = json.loads(str(arrays["metadata"]))
    train_idx_np, validation_idx_np, test_idx_np = group_split(groups, args.seed)
    if min(len(train_idx_np), len(validation_idx_np), len(test_idx_np)) == 0:
        raise RuntimeError("DOI-group split produced an empty partition")
    train_idx = torch.from_numpy(train_idx_np)
    validation_idx = torch.from_numpy(validation_idx_np)
    test_idx = torch.from_numpy(test_idx_np)
    x_mean = features[train_idx].mean(0)
    x_std = features[train_idx].std(0).clamp_min(1e-8)
    normalized = (features - x_mean) / x_std
    property_loader = DataLoader(
        TensorDataset(normalized[train_idx], target[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_physics_np = train_idx_np[physics_mask[train_idx_np]]
    if len(train_physics_np) == 0:
        raise RuntimeError("No physics-supervised rows fell in the training split")
    train_physics = torch.from_numpy(train_physics_np)
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
        physics_iterator = iter(physics_loader)
        property_sum = physics_sum = total_sum = 0.0
        updates = 0
        for property_x, property_y in property_loader:
            try:
                physics_x, voltage_y, physics_current = next(physics_iterator)
            except StopIteration:
                physics_iterator = iter(physics_loader)
                physics_x, voltage_y, physics_current = next(physics_iterator)
            predicted_log_k = model(property_x)
            property_loss = torch.nn.functional.huber_loss(predicted_log_k, property_y)
            predicted_physics_log_k = model(physics_x)
            physics_inputs = torch.stack(
                [predicted_physics_log_k - reference_log_k, physics_current], dim=-1
            )
            predicted_voltage, status, _ = physics_layer(physics_inputs)
            if not bool((status == 1).all()):
                raise RuntimeError(f"Online physics batch failed: {backend.last_solve_diagnostics}")
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
                    model(normalized[validation_idx]), target[validation_idx]
                )
            )
        record = {
            "epoch": epoch,
            "property_loss": property_sum / updates,
            "physics_voltage_loss_scaled": physics_sum / updates,
            "total_loss": total_sum / updates,
            "validation_property_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
        test_prediction = model(normalized[test_idx])
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
        "best_validation_property_loss": best_validation,
        "test_metrics": metrics(test_prediction, target[test_idx]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {
                "input_dim": features.shape[1],
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
            },
            "normalization": {"x_mean": x_mean, "x_std": x_std},
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
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        indices=test_idx.numpy(),
        target_log_conductivity=target[test_idx].numpy(),
        predicted_log_conductivity=test_prediction.detach().numpy(),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
