from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gradcell.language import SingleDesignPhysicsMLP
from train_battery_description_direct_dfn import prepare_data


def make_loader(
    tensors: dict[str, torch.Tensor], indices: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    index = torch.from_numpy(indices.astype(np.int64))
    return DataLoader(
        TensorDataset(tensors["embeddings"][index], tensors["design"][index]),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


@torch.no_grad()
def evaluate(model: SingleDesignPhysicsMLP, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    smooth_l1 = 0.0
    seen = 0
    for embedding, target in loader:
        prediction = model(embedding.to(device))
        target = target.to(device)
        squared_error += float((prediction - target).square().sum())
        smooth_l1 += float(torch.nn.functional.smooth_l1_loss(prediction, target, reduction="sum"))
        seen += target.numel()
    return {
        "normalized_design_mse": squared_error / max(seen, 1),
        "normalized_design_smooth_l1": smooth_l1 / max(seen, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: supervised language-embedding to design MLP training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
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
    tensors, metadata, _ = prepare_data(args.data, args.embeddings)
    split_indices = {
        name: np.flatnonzero(metadata["splits"] == name)
        for name in ("train", "validation", "test")
    }
    if any(len(index) == 0 for index in split_indices.values()):
        raise ValueError("train, validation, and test must all be non-empty")
    loaders = {
        name: make_loader(tensors, index, args.batch_size, name == "train", args.seed)
        for name, index in split_indices.items()
    }
    model_config = {
        "input_dim": tensors["embeddings"].shape[1],
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "design_lower": metadata["design_lower_standardized"].tolist(),
        "design_upper": metadata["design_upper_standardized"].tolist(),
    }
    model = SingleDesignPhysicsMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for embedding, target in loaders["train"]:
            embedding, target = embedding.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(embedding), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        validation = evaluate(model, loaders["validation"], device)
        record = {
            "epoch": epoch,
            "train_design_smooth_l1": total / max(seen, 1),
            "validation_design_mse": validation["normalized_design_mse"],
            "validation_design_smooth_l1": validation["normalized_design_smooth_l1"],
        }
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        score = validation["normalized_design_smooth_l1"]
        if score < best:
            best = score
            patience = 0
            torch.save(
                {
                    "schema": "gradcell.language_design_three_stage.v1",
                    "stage": 1,
                    "stage_name": "design_parameter_supervision",
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
                    "dataset": str(args.data),
                    "dataset_sha256": metadata["dataset_sha256"],
                    "training_args": vars(args),
                },
                args.output_dir / "best_model.pt",
            )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    report = {
        "schema": "gradcell.language_design_stage1_report.v1",
        "best_validation_design_smooth_l1": best,
        "validation": evaluate(model, loaders["validation"], device),
        "test": evaluate(model, loaders["test"], device),
        "split_sizes": {name: len(index) for name, index in split_indices.items()},
        "qwen_frozen_embedding": True,
        "physics_used": False,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
