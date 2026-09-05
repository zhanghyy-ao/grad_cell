from __future__ import annotations

import argparse
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
                    phase="joint_finetune",
                )
                result = phase2
                combined_losses.extend(phase2.losses)
                combined_validation_losses.extend(phase2.validation_losses)
                phase_summaries.append(
                    {"phase": "joint_finetune", "steps": len(phase2.losses), "best_validation_loss": phase2.best_validation_loss}
                )
                if phase1.best_validation_loss <= phase2.best_validation_loss:
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
                    "max_refinement_update_norm": args.max_refinement_update_norm,
                    "training_phases": phase_summaries,
                    "selected_phase": selected_phase,
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
        }
        run.event("training_finished", **summary)
        run.save_summary({"result": summary, "artifacts": {"checkpoint": str(args.checkpoint)}})
        run.log(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
