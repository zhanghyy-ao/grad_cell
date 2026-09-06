from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.evaluation import hard_cutoff_metrics, scalarized_loss
from gradcell.losses import SmoothTchebycheff
from gradcell.models import GradCell
from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend


def load_front(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        return {
            "energy": arrays["energy_wh_kg"].copy(),
            "retention_5c": arrays["energy_retention_5c"].copy(),
            "retention_6c": arrays["energy_retention_6c"].copy(),
            "bounds": metadata["bounds"],
            "metadata": metadata,
        }


def checkpoint_config(checkpoint: dict) -> dict:
    config = checkpoint.get("model_config", {})
    return {
        "capacity_formula": config.get("capacity_formula", "chen2020_scaled"),
        "capacity_multiplier": float(config.get("capacity_multiplier", 1.0)),
        "physics_model": config.get("physics_model", "SPMe"),
        "max_refinement_update_norm": float(
            config.get("max_refinement_update_norm", 0.25)
        ),
        "refinement_steps": int(config.get("refinement_steps", 0)),
    }


def build_model(checkpoint: dict, bounds: dict[str, float]) -> GradCell:
    config = checkpoint_config(checkpoint)
    backends = [
        PyBaMMBackend(
            model_name=config["physics_model"],
            horizon_s=horizon,
            current_ramp_time_s=0.0,
        )
        for horizon in (3600.0, 720.0, 600.0)
    ]
    model = GradCell(
        *(DifferentiablePhysicsLayer(backend) for backend in backends),
        design_space=DesignSpace(
            capacity_formula=config["capacity_formula"],
            capacity_multiplier=config["capacity_multiplier"],
        ),
        objective=SmoothTchebycheff(**bounds),
        max_refinement_update_norm=config["max_refinement_update_norm"],
    ).double()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def optimize_latent_with_physics(
    model: GradCell,
    initial_latent: torch.Tensor,
    preferences: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    max_step_norm: float,
    latent_limit: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Optimize one independent latent per preference through PyBaMM sensitivities."""
    latent = initial_latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    trace: list[dict] = []

    for iteration in range(steps + 1):
        step = model.evaluate(latent, preferences)
        if not bool(step.status.bool().all()):
            failed = (~step.status.bool()).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Differentiable PyBaMM solve failed at optimization step {iteration} "
                f"for preference indices {failed}"
            )
        trace.append(
            {
                "step": iteration,
                "mean_soft_loss": float(step.loss.detach().mean()),
                "max_soft_loss": float(step.loss.detach().max()),
                "mean_energy_wh_kg": float(step.energy.detach().mean()),
                "mean_retention_5c": float(step.retention_5c.detach().mean()),
                "mean_retention_6c": float(step.retention_6c.detach().mean()),
            }
        )
        if iteration == steps:
            break

        optimizer.zero_grad(set_to_none=True)
        # A sum keeps each preference's latent gradient independent of batch size.
        step.loss.sum().backward()
        if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
            raise FloatingPointError(
                f"Non-finite latent gradient at optimization step {iteration}"
            )
        previous = latent.detach().clone()
        optimizer.step()
        with torch.no_grad():
            update = latent - previous
            update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(max_step_norm / update_norm, max=1.0)
            latent.copy_((previous + scale * update).clamp(-latent_limit, latent_limit))

    return latent.detach(), trace


def oracle_losses(
    front: dict, preferences: np.ndarray
) -> np.ndarray:
    values = []
    for preference in preferences:
        repeated = np.full(len(front["energy"]), preference, dtype=np.float64)
        loss = scalarized_loss(
            front["energy"],
            front["retention_5c"],
            front["retention_6c"],
            repeated,
            front["bounds"],
        )
        values.append(float(loss.min()))
    return np.asarray(values)


def evaluate_method(
    name: str,
    latent: torch.Tensor,
    preferences: np.ndarray,
    *,
    evaluation_model: str,
    capacity_formula: str,
    capacity_multiplier: float,
    time_points: int,
    calibration_rate: float,
    calibration_iterations: int,
    bounds: dict[str, float],
    oracle: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    metrics = hard_cutoff_metrics(
        latent,
        evaluation_model,
        capacity_formula,
        time_points,
        calibration_rate,
        calibration_iterations,
        capacity_multiplier,
    )
    loss = scalarized_loss(
        metrics["energy_wh_kg"],
        metrics["energy_retention_5c"],
        metrics["energy_retention_6c"],
        preferences,
        bounds,
    )
    valid = metrics["status"] == 1
    feasible = (
        valid
        & (metrics["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (metrics["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    report = {
        "method": name,
        "success_rate": float(valid.mean()),
        "constraint_satisfaction_rate": float(feasible[valid].mean())
        if valid.any()
        else None,
        "mean_scalarized_loss": float(loss[valid].mean()) if valid.any() else None,
        "median_scalarized_loss": float(np.median(loss[valid])) if valid.any() else None,
        "mean_scalarized_regret": float((loss[valid] - oracle[valid]).mean())
        if valid.any()
        else None,
        "median_scalarized_regret": float(np.median(loss[valid] - oracle[valid]))
        if valid.any()
        else None,
        "mean_energy_wh_kg": float(metrics["energy_wh_kg"][valid].mean())
        if valid.any()
        else None,
        "mean_retention_5c": float(metrics["energy_retention_5c"][valid].mean())
        if valid.any()
        else None,
        "mean_retention_6c": float(metrics["energy_retention_6c"][valid].mean())
        if valid.any()
        else None,
    }
    arrays = {**metrics, "latent": latent.detach().cpu().numpy(), "loss": loss}
    return report, arrays


def pairwise_report(
    left_name: str,
    left: dict[str, np.ndarray],
    right_name: str,
    right: dict[str, np.ndarray],
) -> dict:
    valid = (left["status"] == 1) & (right["status"] == 1)
    delta = left["loss"] - right["loss"]
    return {
        "comparison": f"{right_name}_versus_{left_name}",
        "joint_valid_samples": int(valid.sum()),
        "right_better_loss_fraction": float((delta[valid] > 0.0).mean())
        if valid.any()
        else None,
        "mean_loss_reduction": float(delta[valid].mean()) if valid.any() else None,
        "median_loss_reduction": float(np.median(delta[valid])) if valid.any() else None,
        "mean_energy_change_wh_kg": float(
            (right["energy_wh_kg"] - left["energy_wh_kg"])[valid].mean()
        )
        if valid.any()
        else None,
        "mean_retention_5c_change": float(
            (right["energy_retention_5c"] - left["energy_retention_5c"])[valid].mean()
        )
        if valid.any()
        else None,
        "mean_retention_6c_change": float(
            (right["energy_retention_6c"] - left["energy_retention_6c"])[valid].mean()
        )
        if valid.any()
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare K=0 initializer, direct differentiable-PyBaMM latent "
            "optimization, and a learned K=3 refiner under one hard-cutoff metric."
        )
    )
    parser.add_argument("--k0-checkpoint", type=Path, required=True)
    parser.add_argument("--k3-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path, required=True)
    parser.add_argument("--preference-points", type=int, default=21)
    parser.add_argument("--preference-values", type=float, nargs="+")
    parser.add_argument("--physics-optimization-steps", type=int, default=10)
    parser.add_argument("--physics-learning-rate", type=float, default=0.05)
    parser.add_argument("--physics-max-step-norm", type=float, default=0.25)
    parser.add_argument("--latent-limit", type=float, default=2.0)
    parser.add_argument("--k3-refinement-steps", type=int)
    parser.add_argument("--evaluation-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.preference_values is None and args.preference_points < 2:
        parser.error("--preference-points must be at least 2")
    if args.preference_values is not None and (
        not args.preference_values
        or any(value < 0.0 or value > 1.0 for value in args.preference_values)
    ):
        parser.error("--preference-values must contain values in [0,1]")
    if args.physics_optimization_steps < 1:
        parser.error("--physics-optimization-steps must be positive")
    if args.physics_learning_rate <= 0.0 or args.physics_max_step_norm <= 0.0:
        parser.error("physics learning rate and max step norm must be positive")
    if args.latent_limit <= 0.0:
        parser.error("--latent-limit must be positive")

    k0_checkpoint = torch.load(args.k0_checkpoint, map_location="cpu", weights_only=False)
    k3_checkpoint = torch.load(args.k3_checkpoint, map_location="cpu", weights_only=False)
    k0_config = checkpoint_config(k0_checkpoint)
    k3_config = checkpoint_config(k3_checkpoint)
    for field in ("capacity_formula", "capacity_multiplier", "physics_model"):
        if k0_config[field] != k3_config[field]:
            raise ValueError(
                f"K=0 and K=3 checkpoints disagree on {field}: "
                f"{k0_config[field]!r} != {k3_config[field]!r}"
            )

    front = load_front(args.reference_front)
    preferences = (
        np.asarray(args.preference_values, dtype=np.float64)
        if args.preference_values is not None
        else np.linspace(0.0, 1.0, args.preference_points)
    )
    preference_tensor = torch.from_numpy(preferences).double()
    k0_model = build_model(k0_checkpoint, front["bounds"])
    k3_model = build_model(k3_checkpoint, front["bounds"])

    with torch.no_grad():
        embedding = k0_model.task_encoder(preference_tensor)
        initial_latent = k0_model.initializer(embedding)
    physics_latent, trace = optimize_latent_with_physics(
        k0_model,
        initial_latent,
        preference_tensor,
        steps=args.physics_optimization_steps,
        learning_rate=args.physics_learning_rate,
        max_step_norm=args.physics_max_step_norm,
        latent_limit=args.latent_limit,
    )

    k3_steps = (
        k3_config["refinement_steps"]
        if args.k3_refinement_steps is None
        else args.k3_refinement_steps
    )
    if k3_steps < 1:
        parser.error(
            "K=3 comparison needs a positive refinement count; pass --k3-refinement-steps"
        )
    with torch.enable_grad():
        k3_latent = k3_model(preference_tensor, num_steps=k3_steps).final.latent.detach()

    oracle = oracle_losses(front, preferences)
    common = {
        "preferences": preferences,
        "evaluation_model": args.evaluation_model,
        "capacity_formula": k0_config["capacity_formula"],
        "capacity_multiplier": k0_config["capacity_multiplier"],
        "time_points": args.time_points,
        "calibration_rate": args.calibration_rate,
        "calibration_iterations": args.calibration_iterations,
        "bounds": front["bounds"],
        "oracle": oracle,
    }
    method_reports = []
    method_arrays = {}
    for name, latent in (
        ("initializer_k0", initial_latent),
        ("pybamm_gradient_optimized", physics_latent),
        (f"learned_refiner_k{k3_steps}", k3_latent),
    ):
        report, arrays = evaluate_method(name, latent, **common)
        method_reports.append(report)
        method_arrays[name] = arrays

    k3_name = f"learned_refiner_k{k3_steps}"
    comparisons = [
        pairwise_report(
            "initializer_k0",
            method_arrays["initializer_k0"],
            "pybamm_gradient_optimized",
            method_arrays["pybamm_gradient_optimized"],
        ),
        pairwise_report(
            "initializer_k0",
            method_arrays["initializer_k0"],
            k3_name,
            method_arrays[k3_name],
        ),
        pairwise_report(
            k3_name,
            method_arrays[k3_name],
            "pybamm_gradient_optimized",
            method_arrays["pybamm_gradient_optimized"],
        ),
    ]
    report = {
        "k0_checkpoint": str(args.k0_checkpoint),
        "k3_checkpoint": str(args.k3_checkpoint),
        "reference_front": str(args.reference_front),
        "optimization_physics_model": k0_config["physics_model"],
        "evaluation_model": args.evaluation_model,
        "preferences": preferences.tolist(),
        "physics_optimization": {
            "steps": args.physics_optimization_steps,
            "learning_rate": args.physics_learning_rate,
            "max_step_norm": args.physics_max_step_norm,
            "latent_limit": args.latent_limit,
            "objective": "differentiable fixed-horizon SmoothTchebycheff",
        },
        "k3_refinement_steps": k3_steps,
        "methods": method_reports,
        "comparisons": comparisons,
        "interpretation_note": (
            "Optimization uses differentiable fixed-horizon PyBaMM sensitivities; "
            "all reported method metrics use fresh physical hard-cutoff simulations."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "optimization_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "candidates.npz",
        preferences=preferences,
        oracle_scalarized_loss=oracle,
        initializer_latent=method_arrays["initializer_k0"]["latent"],
        initializer_status=method_arrays["initializer_k0"]["status"],
        initializer_energy_wh_kg=method_arrays["initializer_k0"]["energy_wh_kg"],
        initializer_retention_5c=method_arrays["initializer_k0"]["energy_retention_5c"],
        initializer_retention_6c=method_arrays["initializer_k0"]["energy_retention_6c"],
        initializer_scalarized_loss=method_arrays["initializer_k0"]["loss"],
        physics_optimized_latent=method_arrays["pybamm_gradient_optimized"]["latent"],
        physics_optimized_status=method_arrays["pybamm_gradient_optimized"]["status"],
        physics_optimized_energy_wh_kg=method_arrays["pybamm_gradient_optimized"]["energy_wh_kg"],
        physics_optimized_retention_5c=method_arrays["pybamm_gradient_optimized"]["energy_retention_5c"],
        physics_optimized_retention_6c=method_arrays["pybamm_gradient_optimized"]["energy_retention_6c"],
        physics_optimized_scalarized_loss=method_arrays["pybamm_gradient_optimized"]["loss"],
        k3_latent=method_arrays[k3_name]["latent"],
        k3_status=method_arrays[k3_name]["status"],
        k3_energy_wh_kg=method_arrays[k3_name]["energy_wh_kg"],
        k3_retention_5c=method_arrays[k3_name]["energy_retention_5c"],
        k3_retention_6c=method_arrays[k3_name]["energy_retention_6c"],
        k3_scalarized_loss=method_arrays[k3_name]["loss"],
    )
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda",
                "method",
                "status",
                "energy_wh_kg",
                "retention_5c",
                "retention_6c",
                "scalarized_loss",
                "scalarized_regret",
            ]
        )
        for name, arrays in method_arrays.items():
            for index, preference in enumerate(preferences):
                writer.writerow(
                    [
                        preference,
                        name,
                        int(arrays["status"][index]),
                        arrays["energy_wh_kg"][index],
                        arrays["energy_retention_5c"][index],
                        arrays["energy_retention_6c"][index],
                        arrays["loss"][index],
                        arrays["loss"][index] - oracle[index],
                    ]
                )

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
