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


def load_reference_front(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        return {
            "energy": arrays["energy_wh_kg"].copy(),
            "retention_5c": arrays["energy_retention_5c"].copy(),
            "retention_6c": arrays["energy_retention_6c"].copy(),
            "bounds": metadata["bounds"],
            "metadata": metadata,
        }


def read_checkpoint_config(checkpoint: dict) -> dict:
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


def validate_provenance(front: dict, config: dict, tolerance: float = 1e-10) -> None:
    metadata = front["metadata"]
    source_model = metadata.get("source_model")
    if source_model is not None and source_model != config["physics_model"]:
        raise ValueError(
            f"Reference front model {source_model!r} does not match checkpoint "
            f"model {config['physics_model']!r}"
        )
    front_formula = metadata.get("capacity_formula")
    if front_formula is not None and front_formula != config["capacity_formula"]:
        raise ValueError(
            f"Reference capacity formula {front_formula!r} does not match "
            f"checkpoint formula {config['capacity_formula']!r}"
        )
    front_multiplier = metadata.get("capacity_multiplier")
    if front_multiplier is not None and not np.isclose(
        float(front_multiplier),
        config["capacity_multiplier"],
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(
            f"Reference capacity multiplier {front_multiplier!r} does not match "
            f"checkpoint multiplier {config['capacity_multiplier']!r}"
        )


def build_k0_model(checkpoint: dict, bounds: dict[str, float]) -> GradCell:
    config = read_checkpoint_config(checkpoint)
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


def optimize_from_k0(
    model: GradCell,
    initial_latent: torch.Tensor,
    preferences: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    max_step_norm: float,
    latent_limit: float,
) -> tuple[torch.Tensor, np.ndarray, list[dict]]:
    """Use PyBaMM sensitivities to optimize one latent independently per lambda.

    The best soft-loss latent observed for each preference is returned, rather
    than blindly returning the final Adam iterate.
    """
    latent = initial_latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    best_latent = latent.detach().clone()
    best_loss = torch.full(
        (latent.shape[0],), float("inf"), dtype=latent.dtype, device=latent.device
    )
    best_step = torch.zeros(latent.shape[0], dtype=torch.int64, device=latent.device)
    trace: list[dict] = []

    for iteration in range(steps + 1):
        result = model.evaluate(latent, preferences)
        valid = result.status.bool() & torch.isfinite(result.loss)
        if not bool(valid.all()):
            failed = (~valid).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Differentiable PyBaMM failed at step {iteration} for indices {failed}"
            )

        detached_loss = result.loss.detach()
        improved = detached_loss < best_loss
        best_loss = torch.where(improved, detached_loss, best_loss)
        best_step = torch.where(
            improved,
            torch.full_like(best_step, iteration),
            best_step,
        )
        best_latent[improved] = latent.detach()[improved]
        trace.append(
            {
                "step": iteration,
                "soft_loss": detached_loss.cpu().tolist(),
                "energy_wh_kg": result.energy.detach().cpu().tolist(),
                "retention_5c": result.retention_5c.detach().cpu().tolist(),
                "retention_6c": result.retention_6c.detach().cpu().tolist(),
            }
        )
        if iteration == steps:
            break

        optimizer.zero_grad(set_to_none=True)
        result.loss.sum().backward()
        if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
            raise FloatingPointError(f"Non-finite latent gradient at step {iteration}")
        previous = latent.detach().clone()
        optimizer.step()
        with torch.no_grad():
            update = latent - previous
            norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(max_step_norm / norm, max=1.0)
            latent.copy_((previous + scale * update).clamp(-latent_limit, latent_limit))

    return best_latent.cpu(), best_step.cpu().numpy(), trace


def reference_oracle_loss(front: dict, preferences: np.ndarray) -> np.ndarray:
    losses = []
    for preference in preferences:
        values = scalarized_loss(
            front["energy"],
            front["retention_5c"],
            front["retention_6c"],
            np.full(len(front["energy"]), preference, dtype=np.float64),
            front["bounds"],
        )
        losses.append(float(values.min()))
    return np.asarray(losses)


def hard_evaluate(
    latent: torch.Tensor,
    preferences: np.ndarray,
    *,
    config: dict,
    bounds: dict[str, float],
    time_points: int,
    calibration_rate: float,
    calibration_iterations: int,
) -> dict[str, np.ndarray]:
    metrics = hard_cutoff_metrics(
        latent,
        config["physics_model"],
        config["capacity_formula"],
        time_points,
        calibration_rate,
        calibration_iterations,
        config["capacity_multiplier"],
    )
    loss = scalarized_loss(
        metrics["energy_wh_kg"],
        metrics["energy_retention_5c"],
        metrics["energy_retention_6c"],
        preferences,
        bounds,
    )
    return {**metrics, "latent": latent.detach().cpu().numpy(), "loss": loss}


def summarize(
    initial: dict[str, np.ndarray],
    optimized: dict[str, np.ndarray],
    oracle: np.ndarray,
    bounds: dict[str, float],
    tolerance: float,
    regret_threshold: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    joint_valid = (initial["status"] == 1) & (optimized["status"] == 1)
    initial_feasible = (
        joint_valid
        & (initial["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (initial["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    optimized_feasible = (
        joint_valid
        & (optimized["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (optimized["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    hard_loss_improved = joint_valid & (
        optimized["loss"] <= initial["loss"] + tolerance
    )
    regret = optimized["loss"] - oracle
    near_reference = joint_valid & (regret <= regret_threshold)
    # Lambda is a preference, not a physical equality constraint. In this
    # report it is considered respected when its scalarized hard-cutoff loss
    # does not worsen and the physical retention constraints remain feasible.
    preference_satisfied = hard_loss_improved & optimized_feasible
    diagnostics = {
        "joint_valid": joint_valid,
        "initial_feasible": initial_feasible,
        "optimized_feasible": optimized_feasible,
        "hard_loss_improved": hard_loss_improved,
        "optimized_regret": regret,
        "near_reference": near_reference,
        "preference_satisfied": preference_satisfied,
    }
    report = {
        "joint_success_rate": float(joint_valid.mean()),
        "initial_constraint_satisfaction_rate": float(initial_feasible[joint_valid].mean())
        if joint_valid.any()
        else None,
        "optimized_constraint_satisfaction_rate": float(
            optimized_feasible[joint_valid].mean()
        )
        if joint_valid.any()
        else None,
        "hard_loss_nonworsening_fraction": float(hard_loss_improved[joint_valid].mean())
        if joint_valid.any()
        else None,
        "preference_satisfaction_rate": float(preference_satisfied[joint_valid].mean())
        if joint_valid.any()
        else None,
        "near_reference_fraction": float(near_reference[joint_valid].mean())
        if joint_valid.any()
        else None,
        "mean_hard_loss_reduction": float(
            (initial["loss"] - optimized["loss"])[joint_valid].mean()
        )
        if joint_valid.any()
        else None,
        "median_hard_loss_reduction": float(
            np.median((initial["loss"] - optimized["loss"])[joint_valid])
        )
        if joint_valid.any()
        else None,
        "mean_optimized_regret": float(regret[joint_valid].mean())
        if joint_valid.any()
        else None,
        "median_optimized_regret": float(np.median(regret[joint_valid]))
        if joint_valid.any()
        else None,
    }
    return report, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate K=0 designs, refine their latents with differentiable PyBaMM, "
            "and compare both using fresh hard-cutoff PyBaMM simulations."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path, required=True)
    parser.add_argument("--preference-points", type=int, default=21)
    parser.add_argument("--preference-values", type=float, nargs="+")
    parser.add_argument("--optimization-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-step-norm", type=float, default=0.25)
    parser.add_argument(
        "--latent-limit",
        type=float,
        default=8.0,
        help="Numerical safety bound; physical bounds are enforced by DesignSpace.",
    )
    parser.add_argument("--loss-tolerance", type=float, default=1e-8)
    parser.add_argument("--regret-threshold", type=float, default=0.02)
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
    if args.optimization_steps < 1:
        parser.error("--optimization-steps must be positive")
    if args.learning_rate <= 0.0 or args.max_step_norm <= 0.0:
        parser.error("--learning-rate and --max-step-norm must be positive")
    if args.latent_limit <= 0.0 or args.regret_threshold < 0.0:
        parser.error("--latent-limit must be positive and --regret-threshold non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = read_checkpoint_config(checkpoint)
    if config["refinement_steps"] != 0:
        raise ValueError(
            f"Expected a K=0 checkpoint, but model_config.refinement_steps="
            f"{config['refinement_steps']}"
        )
    front = load_reference_front(args.reference_front)
    validate_provenance(front, config)
    preferences = (
        np.asarray(args.preference_values, dtype=np.float64)
        if args.preference_values is not None
        else np.linspace(0.0, 1.0, args.preference_points)
    )
    preference_tensor = torch.from_numpy(preferences).double()
    model = build_k0_model(checkpoint, front["bounds"])

    with torch.no_grad():
        initial_latent = model.initializer(model.task_encoder(preference_tensor)).cpu()
    optimized_latent, best_step, trace = optimize_from_k0(
        model,
        initial_latent,
        preference_tensor,
        steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        max_step_norm=args.max_step_norm,
        latent_limit=args.latent_limit,
    )

    hard_kwargs = {
        "preferences": preferences,
        "config": config,
        "bounds": front["bounds"],
        "time_points": args.time_points,
        "calibration_rate": args.calibration_rate,
        "calibration_iterations": args.calibration_iterations,
    }
    initial = hard_evaluate(initial_latent, **hard_kwargs)
    optimized = hard_evaluate(optimized_latent, **hard_kwargs)
    oracle = reference_oracle_loss(front, preferences)
    summary, diagnostics = summarize(
        initial,
        optimized,
        oracle,
        front["bounds"],
        args.loss_tolerance,
        args.regret_threshold,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "reference_front": str(args.reference_front),
        "physics_model": config["physics_model"],
        "preferences": preferences.tolist(),
        "optimization": {
            "steps": args.optimization_steps,
            "learning_rate": args.learning_rate,
            "max_step_norm": args.max_step_norm,
            "latent_limit": args.latent_limit,
            "returned_iterate": "best differentiable soft-loss step per lambda",
        },
        "preference_satisfaction_definition": (
            "Both hard-cutoff evaluations succeed, optimized scalarized loss does not "
            "exceed initializer loss by more than loss_tolerance, and R5/R6 satisfy "
            "their physical minimum constraints. Lambda is a preference, not a direct "
            "PyBaMM constraint."
        ),
        "loss_tolerance": args.loss_tolerance,
        "regret_threshold": args.regret_threshold,
        **summary,
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
        best_soft_step=best_step,
        oracle_scalarized_loss=oracle,
        initial_latent=initial["latent"],
        initial_status=initial["status"],
        initial_energy_wh_kg=initial["energy_wh_kg"],
        initial_retention_5c=initial["energy_retention_5c"],
        initial_retention_6c=initial["energy_retention_6c"],
        initial_scalarized_loss=initial["loss"],
        optimized_latent=optimized["latent"],
        optimized_status=optimized["status"],
        optimized_energy_wh_kg=optimized["energy_wh_kg"],
        optimized_retention_5c=optimized["energy_retention_5c"],
        optimized_retention_6c=optimized["energy_retention_6c"],
        optimized_scalarized_loss=optimized["loss"],
        optimized_regret=diagnostics["optimized_regret"],
        preference_satisfied=diagnostics["preference_satisfied"],
        near_reference=diagnostics["near_reference"],
    )
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda",
                "best_soft_step",
                "initial_status",
                "optimized_status",
                "initial_energy_wh_kg",
                "optimized_energy_wh_kg",
                "energy_change_wh_kg",
                "initial_retention_5c",
                "optimized_retention_5c",
                "retention_5c_change",
                "initial_retention_6c",
                "optimized_retention_6c",
                "retention_6c_change",
                "initial_loss",
                "optimized_loss",
                "hard_loss_reduction",
                "optimized_regret",
                "preference_satisfied",
                "near_reference",
            ]
        )
        for index, preference in enumerate(preferences):
            writer.writerow(
                [
                    preference,
                    best_step[index],
                    int(initial["status"][index]),
                    int(optimized["status"][index]),
                    initial["energy_wh_kg"][index],
                    optimized["energy_wh_kg"][index],
                    optimized["energy_wh_kg"][index] - initial["energy_wh_kg"][index],
                    initial["energy_retention_5c"][index],
                    optimized["energy_retention_5c"][index],
                    optimized["energy_retention_5c"][index]
                    - initial["energy_retention_5c"][index],
                    initial["energy_retention_6c"][index],
                    optimized["energy_retention_6c"][index],
                    optimized["energy_retention_6c"][index]
                    - initial["energy_retention_6c"][index],
                    initial["loss"][index],
                    optimized["loss"][index],
                    initial["loss"][index] - optimized["loss"][index],
                    diagnostics["optimized_regret"][index],
                    bool(diagnostics["preference_satisfied"][index]),
                    bool(diagnostics["near_reference"][index]),
                ]
            )

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
