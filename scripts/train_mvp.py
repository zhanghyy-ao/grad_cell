from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.experiments import ExperimentRun
from gradcell.losses import SmoothTchebycheff
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="DFN")
    parser.add_argument(
        "--capacity-formula",
        choices=("electrode_theoretical", "chen2020_scaled"),
        default="chen2020_scaled",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--initializer-checkpoint",
        type=Path,
        help="K=0 checkpoint used to initialize the task encoder and initializer.",
    )
    parser.add_argument("--frozen-refiner-steps", type=int, default=0)
    parser.add_argument("--joint-finetune-steps", type=int, default=0)
    parser.add_argument("--joint-learning-rate-scale", type=float, default=0.2)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument("--monotonic-weight", type=float, default=0.1)
    parser.add_argument("--step-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--initial-loss-weight", type=float, default=0.3)
    parser.add_argument("--initializer-distillation-weight", type=float, default=1.0)
    parser.add_argument("--k0-guard-loss-tolerance", type=float, default=2e-3)
    parser.add_argument("--max-refinement-update-norm", type=float, default=0.25)
    parser.add_argument("--current-ramp-time-s", type=float, default=0.0)
    parser.add_argument(
        "--reference-front",
        type=Path,
        help="Reference-front NPZ used to calibrate energy/power ideal and nadir values.",
    )
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/checkpoints/mvp.pt"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.initializer_checkpoint is not None and args.refinement_steps < 1:
        parser.error("--initializer-checkpoint requires --refinement-steps >= 1")
    if args.initializer_checkpoint is not None and args.frozen_refiner_steps < 1:
        parser.error("staged refiner training requires --frozen-refiner-steps >= 1")
    if args.joint_finetune_steps < 0:
        parser.error("--joint-finetune-steps cannot be negative")
    if not 0.0 < args.joint_learning_rate_scale <= 1.0:
        parser.error("--joint-learning-rate-scale must be in (0,1]")
    if args.initial_loss_weight < 0.0 or args.initializer_distillation_weight < 0.0:
        parser.error("K=0 preservation loss weights cannot be negative")
    if args.k0_guard_loss_tolerance < 0.0:
        parser.error("--k0-guard-loss-tolerance cannot be negative")
    with ExperimentRun("train_mvp", args, run_dir=args.run_dir) as run:
        torch.manual_seed(args.seed)
        objective = None
        objective_bounds = None
        capacity_multiplier = 1.0
        if args.reference_front is not None:
            with np.load(args.reference_front, allow_pickle=False) as arrays:
                front_metadata = json.loads(str(arrays["metadata"]))
            objective_bounds = front_metadata["bounds"]
            capacity_multiplier = float(front_metadata.get("capacity_multiplier", 1.0))
            objective = SmoothTchebycheff(**objective_bounds)
            run.log(f"objective bounds loaded from {args.reference_front}: {objective_bounds}")
            run.log(f"capacity multiplier loaded from reference data: {capacity_multiplier}")
        run.log(
            f"building {args.backend}/{args.model} 1C/5C/6C backends; "
            f"capacity_formula={args.capacity_formula}"
        )
        if args.backend == "toy":
            backend1 = AnalyticToyBackend(horizon_s=3600.0)
            backend5 = AnalyticToyBackend(horizon_s=720.0)
            backend6 = AnalyticToyBackend(horizon_s=600.0)
        else:
            backend1 = PyBaMMBackend(
                model_name=args.model,
                horizon_s=3600.0,
                current_ramp_time_s=args.current_ramp_time_s,
            )
            backend5 = PyBaMMBackend(
                model_name=args.model,
                horizon_s=720.0,
                current_ramp_time_s=args.current_ramp_time_s,
            )
            backend6 = PyBaMMBackend(
                model_name=args.model,
                horizon_s=600.0,
                current_ramp_time_s=args.current_ramp_time_s,
            )
        model = GradCell(
            DifferentiablePhysicsLayer(backend1),
            DifferentiablePhysicsLayer(backend5),
            DifferentiablePhysicsLayer(backend6),
            design_space=DesignSpace(
                capacity_formula=args.capacity_formula,
                capacity_multiplier=capacity_multiplier,
            ),
            objective=objective,
            max_refinement_update_norm=args.max_refinement_update_norm,
        ).double()
        initializer_source = None
        initializer_teacher = None
        if args.initializer_checkpoint is not None:
            source_checkpoint = torch.load(
                args.initializer_checkpoint, map_location="cpu", weights_only=False
            )
            source_state = source_checkpoint["model"]
            prefixes = ("task_encoder.", "initializer.")
            transferred = {
                name: value for name, value in source_state.items() if name.startswith(prefixes)
            }
            if not transferred:
                raise RuntimeError("K=0 checkpoint contains no initializer-stack parameters")
            model_state = model.state_dict()
            model_state.update(transferred)
            model.load_state_dict(model_state)
            teacher_encoder = copy.deepcopy(model.task_encoder).eval()
            teacher_initializer = copy.deepcopy(model.initializer).eval()
            for parameter in teacher_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in teacher_initializer.parameters():
                parameter.requires_grad_(False)

            def initializer_teacher(preference: torch.Tensor) -> torch.Tensor:
                return teacher_initializer(teacher_encoder(preference))

            initializer_source = str(args.initializer_checkpoint)
            run.event(
                "initializer_loaded",
                checkpoint=initializer_source,
                transferred_tensors=len(transferred),
            )
        preflight_preferences = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        with torch.enable_grad():
            preflight = model(preflight_preferences, num_steps=0).final
        preflight_summary = {
            "preferences": preflight_preferences.tolist(),
            "status": preflight.status.tolist(),
            "loss": preflight.loss.detach().tolist(),
            "energy_wh_kg": preflight.energy.detach().tolist(),
            "energy_retention_5c": preflight.retention_5c.detach().tolist(),
            "energy_retention_6c": preflight.retention_6c.detach().tolist(),
        }
        run.event("physics_preflight", **preflight_summary)
        if not bool(preflight.status.bool().all()):
            raise RuntimeError(
                "GradCell physics preflight failed for one or more canonical preferences"
            )
        run.event("training_started", parameter_count=sum(p.numel() for p in model.parameters()))
        phase_summaries = []
        selected_phase = "joint"
        preservation_summary = None
        if args.initializer_checkpoint is None:
            result = train(
                model,
                steps=args.steps,
                batch_size=args.batch_size,
                refinement_steps=args.refinement_steps,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                early_stopping_patience=args.early_stopping_patience,
                checkpoint_path=args.checkpoint,
                resume_from=args.resume_from,
                log_path=run.path / "training_steps.jsonl",
                auxiliary_loss_weight=args.auxiliary_loss_weight,
                monotonic_weight=args.monotonic_weight,
                step_penalty_weight=args.step_penalty_weight,
            )
            combined_losses = result.losses
            combined_validation_losses = result.validation_losses
        else:
            guard_preferences = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)

            def validation_snapshot(num_steps: int) -> dict:
                model.eval()
                with torch.enable_grad():
                    snapshot = model(guard_preferences, num_steps=num_steps).final
                model.train()
                feasible = (
                    snapshot.status.bool()
                    & (snapshot.retention_5c >= model.objective.retention_5c_min)
                    & (snapshot.retention_6c >= model.objective.retention_6c_min)
                )
                return {
                    "losses": snapshot.loss.detach().cpu(),
                    "mean_loss": float(snapshot.loss.detach().mean()),
                    "constraint_satisfaction_rate": float(feasible.double().mean()),
                }

            pretrained_k0 = validation_snapshot(0)
            for parameter in model.task_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in model.initializer.parameters():
                parameter.requires_grad_(False)
            run.event(
                "training_phase_started",
                phase="frozen_refiner",
                steps=args.frozen_refiner_steps,
                learning_rate=args.learning_rate,
            )
            phase1 = train(
                model,
                steps=args.frozen_refiner_steps,
                batch_size=args.batch_size,
                refinement_steps=args.refinement_steps,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                early_stopping_patience=args.early_stopping_patience,
                checkpoint_path=args.checkpoint.with_name(
                    f"{args.checkpoint.stem}.phase1{args.checkpoint.suffix}"
                ),
                log_path=run.path / "training_steps_phase1.jsonl",
                auxiliary_loss_weight=args.auxiliary_loss_weight,
                monotonic_weight=args.monotonic_weight,
                step_penalty_weight=args.step_penalty_weight,
                phase="frozen_refiner",
            )
            phase_summaries.append(
                {"phase": "frozen_refiner", "steps": len(phase1.losses), "best_validation_loss": phase1.best_validation_loss}
            )
            phase1_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            phase1_final = validation_snapshot(args.refinement_steps)
            for parameter in model.task_encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in model.initializer.parameters():
                parameter.requires_grad_(True)
            result = phase1
            combined_losses = list(phase1.losses)
            combined_validation_losses = list(phase1.validation_losses)
            if args.joint_finetune_steps > 0:
                joint_lr = args.learning_rate * args.joint_learning_rate_scale
                run.event(
                    "training_phase_started",
                    phase="joint_finetune",
                    steps=args.joint_finetune_steps,
                    learning_rate=joint_lr,
                )
                phase2 = train(
                    model,
                    steps=args.joint_finetune_steps,
                    batch_size=args.batch_size,
                    refinement_steps=args.refinement_steps,
                    learning_rate=joint_lr,
                    validation_interval=args.validation_interval,
                    early_stopping_patience=args.early_stopping_patience,
                    checkpoint_path=args.checkpoint.with_name(
                        f"{args.checkpoint.stem}.phase2{args.checkpoint.suffix}"
                    ),
                    log_path=run.path / "training_steps_phase2.jsonl",
                    auxiliary_loss_weight=args.auxiliary_loss_weight,
                    monotonic_weight=args.monotonic_weight,
                    step_penalty_weight=args.step_penalty_weight,
                    initial_loss_weight=args.initial_loss_weight,
                    initializer_teacher=initializer_teacher,
                    initializer_distillation_weight=args.initializer_distillation_weight,
                    phase="joint_finetune",
                )
                result = phase2
                combined_losses.extend(phase2.losses)
                combined_validation_losses.extend(phase2.validation_losses)
                phase_summaries.append(
                    {"phase": "joint_finetune", "steps": len(phase2.losses), "best_validation_loss": phase2.best_validation_loss}
                )
                joint_final = validation_snapshot(args.refinement_steps)
                joint_k0 = validation_snapshot(0)
                max_k0_loss_increase = float(
                    (joint_k0["losses"] - pretrained_k0["losses"]).max()
                )
                k0_guard_passed = (
                    max_k0_loss_increase <= args.k0_guard_loss_tolerance
                    and joint_k0["constraint_satisfaction_rate"]
                    >= pretrained_k0["constraint_satisfaction_rate"]
                )
                joint_improves_final = joint_final["mean_loss"] < phase1_final["mean_loss"]
                preservation_summary = {
                    "guard_preferences": len(guard_preferences),
                    "loss_tolerance": args.k0_guard_loss_tolerance,
                    "pretrained_k0_mean_loss": pretrained_k0["mean_loss"],
                    "joint_k0_mean_loss": joint_k0["mean_loss"],
                    "max_k0_loss_increase": max_k0_loss_increase,
                    "k0_constraint_rate_before": pretrained_k0[
                        "constraint_satisfaction_rate"
                    ],
                    "k0_constraint_rate_after": joint_k0[
                        "constraint_satisfaction_rate"
                    ],
                    "k0_guard_passed": k0_guard_passed,
                    "frozen_final_mean_loss": phase1_final["mean_loss"],
                    "joint_final_mean_loss": joint_final["mean_loss"],
                    "joint_improves_final": joint_improves_final,
                }
                if not (k0_guard_passed and joint_improves_final):
                    model.load_state_dict(phase1_state)
                    result = phase1
                    selected_phase = "frozen_refiner"
                else:
                    selected_phase = "joint_finetune"
                run.event(
                    "training_phase_selected",
                    phase=selected_phase,
                    frozen_validation_loss=phase1.best_validation_loss,
                    joint_validation_loss=phase2.best_validation_loss,
                    frozen_final_mean_loss=phase1_final["mean_loss"],
                    joint_final_mean_loss=joint_final["mean_loss"],
                    pretrained_k0_mean_loss=pretrained_k0["mean_loss"],
                    joint_k0_mean_loss=joint_k0["mean_loss"],
                    max_k0_loss_increase=max_k0_loss_increase,
                    k0_constraint_rate_before=pretrained_k0["constraint_satisfaction_rate"],
                    k0_constraint_rate_after=joint_k0["constraint_satisfaction_rate"],
                    k0_guard_passed=k0_guard_passed,
                    joint_improves_final=joint_improves_final,
                )
            else:
                selected_phase = "frozen_refiner"
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "losses": combined_losses,
                "validation_losses": combined_validation_losses,
                "best_validation_loss": result.best_validation_loss,
                "best_step": result.best_step,
                "stopped_early": result.stopped_early,
                "model_config": {
                    "backend": args.backend,
                    "physics_model": args.model,
                    "capacity_formula": args.capacity_formula,
                    "capacity_multiplier": capacity_multiplier,
                    "current_ramp_time_s": args.current_ramp_time_s,
                    "reference_front": str(args.reference_front)
                    if args.reference_front is not None
                    else None,
                    "objective_bounds": objective_bounds,
                    "high_rate_objective": "min(energy_retention_5c, energy_retention_6c)",
                    "refinement_steps": args.refinement_steps,
                    "initializer_checkpoint": initializer_source,
                    "frozen_refiner_steps": args.frozen_refiner_steps,
                    "joint_finetune_steps": args.joint_finetune_steps,
                    "joint_learning_rate_scale": args.joint_learning_rate_scale,
                    "auxiliary_loss_weight": args.auxiliary_loss_weight,
                    "monotonic_weight": args.monotonic_weight,
                    "step_penalty_weight": args.step_penalty_weight,
                    "initial_loss_weight": args.initial_loss_weight,
                    "initializer_distillation_weight": args.initializer_distillation_weight,
                    "k0_guard_loss_tolerance": args.k0_guard_loss_tolerance,
                    "max_refinement_update_norm": args.max_refinement_update_norm,
                    "training_phases": phase_summaries,
                    "selected_phase": selected_phase,
                    "k0_preservation": preservation_summary,
                    "seed": args.seed,
                },
            },
            args.checkpoint,
        )
        summary = {
            "steps_completed": len(combined_losses),
            "final_train_loss": combined_losses[-1] if combined_losses else None,
            "best_validation_loss": result.best_validation_loss,
            "best_step": result.best_step,
            "stopped_early": result.stopped_early,
            "training_phases": phase_summaries,
            "selected_phase": selected_phase,
            "k0_preservation": preservation_summary,
        }
        run.event("training_finished", **summary)
        run.save_summary({"result": summary, "artifacts": {"checkpoint": str(args.checkpoint)}})
        run.log(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
