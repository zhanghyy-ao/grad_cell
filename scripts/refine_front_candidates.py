from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.evaluation import hard_cutoff_metrics
from gradcell.models import GradCell
from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend


def build_model(capacity_formula: str, capacity_multiplier: float) -> GradCell:
    backends = [
        PyBaMMBackend(model_name="SPMe", horizon_s=horizon)
        for horizon in (3600.0, 720.0, 600.0)
    ]
    model = GradCell(
        *(DifferentiablePhysicsLayer(backend) for backend in backends),
        design_space=DesignSpace(
            capacity_formula=capacity_formula,
            capacity_multiplier=capacity_multiplier,
        ),
    ).double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Locally refine representative 3D front points.")
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representatives", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-step-norm", type=float, default=0.15)
    parser.add_argument("--latent-limit", type=float, default=6.0)
    parser.add_argument("--retention-5c-min", type=float, default=0.0)
    parser.add_argument("--retention-6c-min", type=float, default=0.0)
    parser.add_argument("--constraint-weight", type=float, default=20.0)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.representatives, args.steps, args.batch_size) < 1:
        parser.error("representatives, steps, and batch-size must be positive")

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError("representative direction sampling requires scipy") from exc
    with np.load(args.front, allow_pickle=False) as arrays:
        front_latent = arrays["latent"].copy()
        objectives = np.column_stack(
            [
                arrays["energy_wh_kg"],
                arrays["energy_retention_5c"],
                arrays["energy_retention_6c"],
            ]
        )
        source_metadata = json.loads(str(arrays["metadata"]))
    lower, upper = objectives.min(axis=0), objectives.max(axis=0)
    scale = np.maximum(upper - lower, 1e-12)
    normalized = (objectives - lower) / scale
    directions = qmc.Sobol(d=3, scramble=True, seed=args.seed).random(args.representatives)
    directions = np.maximum(directions, 1e-6)
    directions /= directions.sum(axis=1, keepdims=True)
    initial_indices = np.argmax(normalized @ directions.T, axis=0)
    initial_latent = front_latent[initial_indices]
    model = build_model(args.capacity_formula, args.capacity_multiplier)
    refined_parts, direction_parts, source_parts = [], [], []

    for start in range(0, len(initial_latent), args.batch_size):
        stop = min(start + args.batch_size, len(initial_latent))
        latent = torch.from_numpy(initial_latent[start:stop]).double().requires_grad_(True)
        weights = torch.from_numpy(directions[start:stop]).double()
        optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
        best = latent.detach().clone()
        best_loss = torch.full((len(latent),), float("inf"), dtype=torch.float64)
        preference = torch.full((len(latent),), 0.5, dtype=torch.float64)
        for step in range(args.steps + 1):
            result = model.evaluate(latent, preference)
            if not bool(result.status.bool().all()):
                raise RuntimeError(
                    f"PyBaMM failed for refinement batch {start}:{stop}, step {step}"
                )
            normalized_metrics = torch.stack(
                [
                    (result.energy - lower[0]) / scale[0],
                    (result.retention_5c - lower[1]) / scale[1],
                    (result.retention_6c - lower[2]) / scale[2],
                ],
                dim=-1,
            )
            constraint = (
                torch.relu(args.retention_5c_min - result.retention_5c)
                + torch.relu(args.retention_6c_min - result.retention_6c)
            )
            loss = -(weights * normalized_metrics).sum(dim=-1) + args.constraint_weight * constraint
            improved = loss.detach() < best_loss
            best_loss = torch.where(improved, loss.detach(), best_loss)
            best[improved] = latent.detach()[improved]
            if step == args.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
                raise FloatingPointError(f"non-finite refinement gradient at step {step}")
            previous = latent.detach().clone()
            optimizer.step()
            with torch.no_grad():
                update = latent - previous
                norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                latent.copy_(
                    (
                        previous
                        + torch.clamp(args.max_step_norm / norm, max=1.0) * update
                    ).clamp(-args.latent_limit, args.latent_limit)
                )
        refined_parts.append(best.numpy())
        direction_parts.append(directions[start:stop])
        source_parts.append(initial_indices[start:stop])
        print(json.dumps({"refined": stop, "total": len(initial_latent)}))

    refined = np.concatenate(refined_parts)
    metrics = hard_cutoff_metrics(
        torch.from_numpy(refined).double(),
        "SPMe",
        args.capacity_formula,
        capacity_multiplier=args.capacity_multiplier,
    )
    valid = metrics["status"] == 1
    metadata = {
        "source_front": str(args.front),
        "source_metadata": source_metadata,
        "algorithm": "Sobol simplex directions plus differentiable Adam refinement",
        "representatives": args.representatives,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "valid_count": int(valid.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=refined[valid],
        energy_wh_kg=metrics["energy_wh_kg"][valid],
        energy_retention_5c=metrics["energy_retention_5c"][valid],
        energy_retention_6c=metrics["energy_retention_6c"][valid],
        direction=np.concatenate(direction_parts)[valid],
        source_front_index=np.concatenate(source_parts)[valid],
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
