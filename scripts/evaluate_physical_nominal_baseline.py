from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.benchmark import load_pybamm_nominal_design
from gradcell.design import DesignSpace
from gradcell.evaluation import (
    hard_cutoff_metrics,
    hard_cutoff_metrics_from_physical_inputs,
    scalarized_loss,
)
from gradcell.losses import SmoothTchebycheff
from gradcell.models import GradCell
from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend


def repeated(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError("A fixed baseline must contain exactly one design")
    return np.repeat(values, count)


def valid_better_fraction(
    candidate_loss: np.ndarray,
    candidate_status: np.ndarray,
    baseline_loss: np.ndarray,
    baseline_status: np.ndarray,
) -> float | None:
    valid = (candidate_status == 1) & (baseline_status == 1)
    return float(np.mean(candidate_loss[valid] < baseline_loss[valid])) if valid.any() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a GradCell checkpoint against the unmodified PyBaMM parameter-set "
            "baseline and the historical zero-latent center baseline."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path, required=True)
    parser.add_argument("--evaluation-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--parameter-set", default="Chen2020")
    parser.add_argument("--refinement-steps", type=int)
    parser.add_argument("--preference-points", type=int, default=21)
    parser.add_argument("--preference-values", type=float, nargs="+")
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.preference_values is None and args.preference_points < 2:
        parser.error("--preference-points must be at least 2")
    if args.preference_values is not None and (
        len(args.preference_values) < 1
        or any(value < 0.0 or value > 1.0 for value in args.preference_values)
    ):
        parser.error("--preference-values must contain values in [0,1]")
    if args.calibration_rate <= 0.0 or args.calibration_iterations < 1:
        parser.error("calibration rate and iterations must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    training_model = config.get("physics_model", "SPMe")
    capacity_formula = config.get("capacity_formula", "chen2020_scaled")
    capacity_multiplier = float(config.get("capacity_multiplier", 1.0))
    refinement_steps = (
        int(config.get("refinement_steps", 0))
        if args.refinement_steps is None
        else args.refinement_steps
    )
    if refinement_steps < 0:
        parser.error("--refinement-steps cannot be negative")

    with np.load(args.reference_front, allow_pickle=False) as arrays:
        front_energy = arrays["energy_wh_kg"].copy()
        front_r5 = arrays["energy_retention_5c"].copy()
        front_r6 = arrays["energy_retention_6c"].copy()
        front_metadata = json.loads(str(arrays["metadata"]))
    source_model = front_metadata.get("source_model")
    if source_model is not None and source_model != training_model:
        raise ValueError(
            f"Reference model {source_model!r} does not match checkpoint model "
            f"{training_model!r}"
        )
    bounds = front_metadata["bounds"]
    objective = SmoothTchebycheff(**bounds)
    training_backends = [
        PyBaMMBackend(model_name=training_model, horizon_s=horizon, current_ramp_time_s=0.0)
        for horizon in (3600.0, 720.0, 600.0)
    ]
    model = GradCell(
        *(DifferentiablePhysicsLayer(backend) for backend in training_backends),
        design_space=DesignSpace(
            capacity_formula=capacity_formula,
            capacity_multiplier=capacity_multiplier,
        ),
        objective=objective,
        max_refinement_update_norm=float(
            config.get("max_refinement_update_norm", 0.25)
        ),
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
    candidate_latent = output.final.latent.detach().cpu()
    center_latent = torch.zeros((1, model.design_space.latent_dim), dtype=torch.float64)

    hard_kwargs = {
        "model_name": args.evaluation_model,
        "capacity_formula": capacity_formula,
        "time_points": args.time_points,
        "calibration_rate": args.calibration_rate,
        "calibration_iterations": args.calibration_iterations,
        "capacity_multiplier": capacity_multiplier,
        "parameter_set": args.parameter_set,
    }
    candidate = hard_cutoff_metrics(candidate_latent, **hard_kwargs)
    center = hard_cutoff_metrics(center_latent, **hard_kwargs)

    physical_design = load_pybamm_nominal_design(args.parameter_set)
    physical = hard_cutoff_metrics_from_physical_inputs(
        physical_design.physics_inputs,
        physical_design.stack_mass_kg,
        physical_design.initial_capacity_ah,
        args.evaluation_model,
        args.time_points,
        args.calibration_rate,
        args.calibration_iterations,
        args.parameter_set,
    )
    count = len(preferences)
    center_arrays = {name: repeated(values, count) for name, values in center.items()}
    physical_arrays = {name: repeated(values, count) for name, values in physical.items()}
    candidate_loss = scalarized_loss(
        candidate["energy_wh_kg"],
        candidate["energy_retention_5c"],
        candidate["energy_retention_6c"],
        preferences,
        bounds,
    )
    center_loss = scalarized_loss(
        center_arrays["energy_wh_kg"],
        center_arrays["energy_retention_5c"],
        center_arrays["energy_retention_6c"],
        preferences,
        bounds,
    )
    physical_loss = scalarized_loss(
        physical_arrays["energy_wh_kg"],
        physical_arrays["energy_retention_5c"],
        physical_arrays["energy_retention_6c"],
        preferences,
        bounds,
    )
    candidate_feasible = (
        (candidate["status"] == 1)
        & (candidate["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (candidate["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    physical_feasible = (
        (physical_arrays["status"] == 1)
        & (physical_arrays["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (physical_arrays["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    center_feasible = (
        (center_arrays["status"] == 1)
        & (center_arrays["energy_retention_5c"] >= bounds["retention_5c_min"])
        & (center_arrays["energy_retention_6c"] >= bounds["retention_6c_min"])
    )
    nominal_parameters = physical_design.parameters
    eps_p = nominal_parameters["Positive electrode porosity"]
    eps_n = nominal_parameters["Negative electrode porosity"]
    eps_s = nominal_parameters["Separator porosity"]
    phi_p = nominal_parameters["Positive electrode active material volume fraction"]
    phi_n = nominal_parameters["Negative electrode active material volume fraction"]
    search_space_checks = {
        "eps_p_within_bounds": bool(
            model.design_space.eps_p_bounds[0]
            <= eps_p
            <= model.design_space.eps_p_bounds[1]
        ),
        "eps_n_within_bounds": bool(
            model.design_space.eps_n_bounds[0]
            <= eps_n
            <= model.design_space.eps_n_bounds[1]
        ),
        "eps_s_within_bounds": bool(
            model.design_space.eps_s_bounds[0]
            <= eps_s
            <= model.design_space.eps_s_bounds[1]
        ),
        "positive_inactive_fraction": float(1.0 - eps_p - phi_p),
        "negative_inactive_fraction": float(1.0 - eps_n - phi_n),
        "positive_inactive_min_satisfied": bool(
            1.0 - eps_p - phi_p >= model.design_space.inactive_p_min
        ),
        "negative_inactive_min_satisfied": bool(
            1.0 - eps_n - phi_n >= model.design_space.inactive_n_min
        ),
        "note": (
            "These are direct fraction checks only. Full representability also depends "
            "on the decoder's N/P coupling."
        ),
    }

    oracle_loss = None
    if args.evaluation_model == source_model:
        oracle_loss = np.asarray(
            [
                scalarized_loss(
                    front_energy,
                    front_r5,
                    front_r6,
                    np.full(len(front_energy), preference),
                    bounds,
                ).min()
                for preference in preferences
            ]
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "reference_front": str(args.reference_front),
        "training_model": training_model,
        "evaluation_model": args.evaluation_model,
        "refinement_steps": refinement_steps,
        "preferences": preferences.tolist(),
        "physical_baseline_definition": (
            f"Unmodified PyBaMM {args.parameter_set} structural parameters, bypassing "
            "the GradCell latent decoder; capacity is recalibrated at the requested "
            "low C-rate and specific energy uses the same GradCell stack-mass proxy."
        ),
        "physical_baseline_is_experimental_data": False,
        "physical_baseline_parameters": physical_design.parameters,
        "physical_baseline_search_space_checks": search_space_checks,
        "physical_baseline_stack_mass_kg": float(physical_design.stack_mass_kg[0]),
        "physical_baseline_metrics": {
            "status": int(physical["status"][0]),
            "reference_capacity_ah": float(physical["reference_capacity_ah"][0]),
            "energy_wh_kg": float(physical["energy_wh_kg"][0]),
            "energy_retention_5c": float(physical["energy_retention_5c"][0]),
            "energy_retention_6c": float(physical["energy_retention_6c"][0]),
            "constraint_satisfied": bool(physical_feasible[0]),
        },
        "latent_center_metrics": {
            "status": int(center["status"][0]),
            "energy_wh_kg": float(center["energy_wh_kg"][0]),
            "energy_retention_5c": float(center["energy_retention_5c"][0]),
            "energy_retention_6c": float(center["energy_retention_6c"][0]),
            "constraint_satisfied": bool(center_feasible[0]),
        },
        "candidate_success_rate": float((candidate["status"] == 1).mean()),
        "candidate_constraint_satisfaction_rate": float(candidate_feasible.mean()),
        "candidate_beats_physical_baseline_fraction": valid_better_fraction(
            candidate_loss,
            candidate["status"],
            physical_loss,
            physical_arrays["status"],
        ),
        "candidate_beats_latent_center_fraction": valid_better_fraction(
            candidate_loss,
            candidate["status"],
            center_loss,
            center_arrays["status"],
        ),
        "mean_loss_reduction_vs_physical_baseline": float(
            np.mean(physical_loss - candidate_loss)
        ),
        "oracle_regret_reported": oracle_loss is not None,
        "oracle_note": (
            "Regret is reported because the hard evaluation model matches the reference model."
            if oracle_loss is not None
            else (
                "Regret is omitted because a reference front from another physics "
                "model is not a valid oracle."
            )
        ),
        "mean_scalarized_regret": (
            float(np.mean(candidate_loss - oracle_loss)) if oracle_loss is not None else None
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_arrays = {
        "preferences": preferences,
        "candidate_latent": candidate_latent.numpy(),
        "candidate_status": candidate["status"],
        "candidate_energy_wh_kg": candidate["energy_wh_kg"],
        "candidate_retention_5c": candidate["energy_retention_5c"],
        "candidate_retention_6c": candidate["energy_retention_6c"],
        "candidate_scalarized_loss": candidate_loss,
        "candidate_constraint_satisfied": candidate_feasible,
        "physical_baseline_inputs": physical_design.physics_inputs,
        "physical_baseline_status": physical_arrays["status"],
        "physical_baseline_energy_wh_kg": physical_arrays["energy_wh_kg"],
        "physical_baseline_retention_5c": physical_arrays["energy_retention_5c"],
        "physical_baseline_retention_6c": physical_arrays["energy_retention_6c"],
        "physical_baseline_scalarized_loss": physical_loss,
        "physical_baseline_constraint_satisfied": physical_feasible,
        "latent_center_status": center_arrays["status"],
        "latent_center_energy_wh_kg": center_arrays["energy_wh_kg"],
        "latent_center_retention_5c": center_arrays["energy_retention_5c"],
        "latent_center_retention_6c": center_arrays["energy_retention_6c"],
        "latent_center_scalarized_loss": center_loss,
        "latent_center_constraint_satisfied": center_feasible,
    }
    if oracle_loss is not None:
        save_arrays["oracle_scalarized_loss"] = oracle_loss
    np.savez_compressed(args.output_dir / "comparison.npz", **save_arrays)

    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda",
                "candidate_status",
                "candidate_feasible",
                "candidate_energy_wh_kg",
                "candidate_r5",
                "candidate_r6",
                "candidate_loss",
                "physical_baseline_energy_wh_kg",
                "physical_baseline_r5",
                "physical_baseline_r6",
                "physical_baseline_loss",
                "candidate_beats_physical_baseline",
                "latent_center_loss",
                "candidate_beats_latent_center",
                "oracle_loss",
                "candidate_regret",
            ]
        )
        for index, preference in enumerate(preferences):
            writer.writerow(
                [
                    preference,
                    int(candidate["status"][index]),
                    bool(candidate_feasible[index]),
                    candidate["energy_wh_kg"][index],
                    candidate["energy_retention_5c"][index],
                    candidate["energy_retention_6c"][index],
                    candidate_loss[index],
                    physical_arrays["energy_wh_kg"][index],
                    physical_arrays["energy_retention_5c"][index],
                    physical_arrays["energy_retention_6c"][index],
                    physical_loss[index],
                    bool(candidate_loss[index] < physical_loss[index]),
                    center_loss[index],
                    bool(candidate_loss[index] < center_loss[index]),
                    oracle_loss[index] if oracle_loss is not None else "",
                    candidate_loss[index] - oracle_loss[index]
                    if oracle_loss is not None
                    else "",
                ]
            )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
