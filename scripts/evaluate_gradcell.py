from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.evaluation import hard_cutoff_metrics, scalarized_loss
from gradcell.losses import SmoothTchebycheff
from gradcell.models import GradCell
from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a GradCell checkpoint with physical hard-cutoff simulations."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path, required=True)
    parser.add_argument("--evaluation-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--refinement-steps", type=int)
    parser.add_argument("--preference-points", type=int, default=11)
    parser.add_argument(
        "--preference-values",
        type=float,
        nargs="+",
        help="Explicit lambda values in [0,1]; overrides --preference-points.",
    )
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.preference_values is None and args.preference_points < 2:
        parser.error("--preference-points must be at least 2")
    if args.preference_values is not None and (
        len(args.preference_values) < 2
        or any(value < 0.0 or value > 1.0 for value in args.preference_values)
    ):
        parser.error("--preference-values requires at least two values in [0,1]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    capacity_formula = config.get("capacity_formula", "chen2020_scaled")
    capacity_multiplier = float(config.get("capacity_multiplier", 1.0))
    training_model = config.get("physics_model", "SPMe")
    refinement_steps = (
        int(config.get("refinement_steps", 0))
        if args.refinement_steps is None
        else args.refinement_steps
    )
    with np.load(args.reference_front, allow_pickle=False) as arrays:
        front_energy = arrays["energy_wh_kg"].copy()
        front_retention_5c = arrays["energy_retention_5c"].copy()
        front_retention_6c = arrays["energy_retention_6c"].copy()
        front_latent = arrays["latent"].copy()
        front_metadata = json.loads(str(arrays["metadata"]))
    bounds = front_metadata["bounds"]
    objective = SmoothTchebycheff(**bounds)
    backend_1c = PyBaMMBackend(
        model_name=training_model,
        horizon_s=3600.0,
        current_ramp_time_s=0.0,
    )
    backend_5c = PyBaMMBackend(
        model_name=training_model,
        horizon_s=720.0,
        current_ramp_time_s=0.0,
    )
    backend_6c = PyBaMMBackend(
        model_name=training_model,
        horizon_s=600.0,
        current_ramp_time_s=0.0,
    )
    model = GradCell(
        DifferentiablePhysicsLayer(backend_1c),
        DifferentiablePhysicsLayer(backend_5c),
        DifferentiablePhysicsLayer(backend_6c),
        design_space=DesignSpace(
            capacity_formula=capacity_formula,
            capacity_multiplier=capacity_multiplier,
        ),
        objective=objective,
    ).double()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    preferences = (
        np.asarray(args.preference_values, dtype=np.float64)
        if args.preference_values is not None
        else np.linspace(0.0, 1.0, args.preference_points)
    )
    with torch.enable_grad():
        output = model(torch.from_numpy(preferences).double(), num_steps=refinement_steps)
    latent = output.final.latent.detach()
    candidate = hard_cutoff_metrics(
        latent,
        args.evaluation_model,
        capacity_formula,
        args.time_points,
        args.calibration_rate,
        args.calibration_iterations,
        capacity_multiplier,
    )
    nominal_latent = torch.zeros((1, latent.shape[1]), dtype=latent.dtype)
    nominal = hard_cutoff_metrics(
        nominal_latent,
        args.evaluation_model,
        capacity_formula,
        args.time_points,
        args.calibration_rate,
        args.calibration_iterations,
        capacity_multiplier,
    )
    candidate_loss = scalarized_loss(
        candidate["energy_wh_kg"], candidate["energy_retention_5c"], candidate["energy_retention_6c"], preferences, bounds
    )
    nominal_loss = scalarized_loss(
        np.repeat(nominal["energy_wh_kg"], len(preferences)),
        np.repeat(nominal["energy_retention_5c"], len(preferences)),
        np.repeat(nominal["energy_retention_6c"], len(preferences)),
        preferences,
        bounds,
    )
    oracle_indices, oracle_losses = [], []
    for preference in preferences:
        repeated = np.full(len(front_energy), preference)
        losses = scalarized_loss(front_energy, front_retention_5c, front_retention_6c, repeated, bounds)
        index = int(np.argmin(losses))
        oracle_indices.append(index)
        oracle_losses.append(float(losses[index]))
    oracle_indices = np.asarray(oracle_indices, dtype=np.int64)
    oracle_losses = np.asarray(oracle_losses)
    valid = candidate["status"] == 1
    constraint_satisfied = (
        (candidate["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (candidate["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "reference_front": str(args.reference_front),
        "training_model": training_model,
        "evaluation_model": args.evaluation_model,
        "capacity_formula": capacity_formula,
        "capacity_multiplier": capacity_multiplier,
        "refinement_steps": refinement_steps,
        "preference_points": len(preferences),
        "preference_values": preferences.tolist(),
        "success_rate": float(valid.mean()),
        "constraint_satisfaction_rate": float(constraint_satisfied[valid].mean())
        if valid.any()
        else None,
        "mean_scalarized_regret": float(np.mean(candidate_loss[valid] - oracle_losses[valid]))
        if valid.any()
        else None,
        "median_scalarized_regret": float(np.median(candidate_loss[valid] - oracle_losses[valid]))
        if valid.any()
        else None,
        "candidate_beats_nominal_fraction": float(
            np.mean(candidate_loss[valid] < nominal_loss[valid])
        )
        if valid.any()
        else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "candidates.npz",
        preferences=preferences,
        capacity_multiplier=np.asarray(capacity_multiplier),
        latent=latent.numpy(),
        status=candidate["status"],
        energy_wh_kg=candidate["energy_wh_kg"],
        energy_retention_5c=candidate["energy_retention_5c"],
        energy_retention_6c=candidate["energy_retention_6c"],
        scalarized_loss=candidate_loss,
        nominal_energy_wh_kg=np.repeat(nominal["energy_wh_kg"], len(preferences)),
        nominal_energy_retention_5c=np.repeat(nominal["energy_retention_5c"], len(preferences)),
        nominal_energy_retention_6c=np.repeat(nominal["energy_retention_6c"], len(preferences)),
        nominal_scalarized_loss=nominal_loss,
        oracle_latent=front_latent[oracle_indices],
        oracle_energy_wh_kg=front_energy[oracle_indices],
        oracle_energy_retention_5c=front_retention_5c[oracle_indices],
        oracle_energy_retention_6c=front_retention_6c[oracle_indices],
        oracle_scalarized_loss=oracle_losses,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
