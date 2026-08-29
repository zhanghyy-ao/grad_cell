from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gradcell.experiments import ExperimentRun
from gradcell.models import Chen2020Surrogate


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    error = prediction - target
    ss_res = error.square().sum(dim=0)
    ss_tot = (target - target.mean(dim=0)).square().sum(dim=0).clamp_min(1e-12)
    return {
        "mae": error.abs().mean(dim=0).tolist(),
        "rmse": error.square().mean(dim=0).sqrt().tolist(),
        "r2": (1.0 - ss_res / ss_tot).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test supervised learnability of Chen2020 labels.")
    parser.add_argument("--data", type=Path, default=Path("data/chen2020_supervised.npz"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/supervised"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    with ExperimentRun("train_supervised_surrogate", args, run_dir=args.run_dir) as run:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(args.seed)
        with np.load(args.data, allow_pickle=False) as arrays:
            required = {"latent", "targets", "metadata"}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
            x = torch.from_numpy(arrays["latent"].copy()).to(torch.float64)
            y = torch.from_numpy(arrays["targets"].copy()).to(torch.float64)
            metadata = json.loads(str(arrays["metadata"]))
        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError(f"Expected latent with shape [N, 7], got {tuple(x.shape)}")
        if y.ndim != 2 or len(y) != len(x):
            raise ValueError(
                f"Expected targets with shape [N, K] matching latent, got {tuple(y.shape)}"
            )
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Dataset contains non-finite latent values or targets")
        target_fields = metadata.get("target_fields")
        if not isinstance(target_fields, list) or len(target_fields) != y.shape[1]:
            raise ValueError(
                "metadata.target_fields must be a list matching the target dimension"
            )
        if len(x) < 20:
            raise ValueError("At least 20 valid samples are required for train/validation/test splits")
        run.log(f"loaded {len(x)} samples from {args.data}")

        permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed))
        train_end, validation_end = int(0.7 * len(x)), int(0.85 * len(x))
        train_idx = permutation[:train_end]
        validation_idx = permutation[train_end:validation_end]
        test_idx = permutation[validation_end:]
        x_mean, x_std = x[train_idx].mean(0), x[train_idx].std(0).clamp_min(1e-8)
        y_mean, y_std = y[train_idx].mean(0), y[train_idx].std(0).clamp_min(1e-8)
        x_norm, y_norm = (x - x_mean) / x_std, (y - y_mean) / y_std
        run.event(
            "dataset_split",
            train=len(train_idx),
            validation=len(validation_idx),
            test=len(test_idx),
        )

        loader = DataLoader(
            TensorDataset(x_norm[train_idx], y_norm[train_idx]),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        model = Chen2020Surrogate(
            output_dim=y.shape[1], hidden_dim=args.hidden_dim, depth=args.depth
        ).double()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-5
        )
        best_state = None
        best_validation = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for features, targets in loader:
                loss = torch.nn.functional.mse_loss(model(features), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(features)
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    torch.nn.functional.mse_loss(
                        model(x_norm[validation_idx]), y_norm[validation_idx]
                    )
                )
            train_loss = train_loss_sum / len(train_idx)
            epoch_record = {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
            history.append(epoch_record)
            run.event("epoch_finished", **epoch_record)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
                run.event("best_model_updated", epoch=epoch, validation_mse=validation_loss)
            else:
                epochs_without_improvement += 1
            if epoch == 1 or epoch % 10 == 0:
                run.log(
                    f"epoch={epoch} train_mse={train_loss:.6g} "
                    f"val_mse={validation_loss:.6g}"
                )
            if epochs_without_improvement >= args.patience:
                run.event("early_stopping", epoch=epoch, patience=args.patience)
                break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(x_norm[test_idx]) * y_std + y_mean
        metrics = regression_metrics(prediction, y[test_idx])
        baseline = y_mean.expand_as(y[test_idx])
        baseline_metrics = regression_metrics(baseline, y[test_idx])
        report = {
            "data": str(args.data),
            "samples": len(x),
            "split_sizes": {
                "train": len(train_idx),
                "validation": len(validation_idx),
                "test": len(test_idx),
            },
            "split_indices": {
                "train": train_idx.tolist(),
                "validation": validation_idx.tolist(),
                "test": test_idx.tolist(),
            },
            "best_validation_standardized_mse": best_validation,
            "epochs_trained": len(history),
            "target_fields": target_fields,
            "test_metrics": {
                name: dict(zip(target_fields, values)) for name, values in metrics.items()
            },
            "train_mean_baseline_test_metrics": {
                name: dict(zip(target_fields, values))
                for name, values in baseline_metrics.items()
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.output_dir / "best_model.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "model_kwargs": {
                    "output_dim": y.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                },
                "normalization": {
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                "target_fields": target_fields,
                "split_indices": report["split_indices"],
            },
            model_path,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        reloaded_model = Chen2020Surrogate(**checkpoint["model_kwargs"]).double()
        reloaded_model.load_state_dict(checkpoint["model"])
        reloaded_model.eval()
        with torch.no_grad():
            reloaded_prediction = reloaded_model(x_norm[test_idx]) * y_std + y_mean
        if not torch.equal(prediction, reloaded_prediction):
            raise RuntimeError("Reloaded checkpoint predictions do not match saved model")
        np.savez_compressed(
            args.output_dir / "test_predictions.npz",
            indices=test_idx.numpy(),
            targets=y[test_idx].numpy(),
            predictions=prediction.numpy(),
            target_fields=np.asarray(target_fields),
        )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        run.save_summary(
            {
                "result": report,
                "artifacts": {
                    "model": str(model_path),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "history": str(args.output_dir / "history.json"),
                    "test_predictions": str(args.output_dir / "test_predictions.npz"),
                },
            }
        )
        run.log(json.dumps(report["test_metrics"]))


if __name__ == "__main__":
    main()
