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
            f"building {args.backend}/{args.model} 1C/3C backends; "
            f"capacity_formula={args.capacity_formula}"
        )
        if args.backend == "toy":
            backend1 = AnalyticToyBackend(horizon_s=3600.0)
            backend3 = AnalyticToyBackend(horizon_s=1200.0)
        else:
            backend1 = PyBaMMBackend(
                model_name=args.model,
                horizon_s=3600.0,
                current_ramp_time_s=args.current_ramp_time_s,
            )
            backend3 = PyBaMMBackend(
                model_name=args.model,
                horizon_s=1200.0,
                current_ramp_time_s=args.current_ramp_time_s,
            )
        model = GradCell(
            DifferentiablePhysicsLayer(backend1),
            DifferentiablePhysicsLayer(backend3),
            design_space=DesignSpace(
                capacity_formula=args.capacity_formula,
                capacity_multiplier=capacity_multiplier,
            ),
            objective=objective,
        ).double()
        preflight_preferences = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        with torch.enable_grad():
            preflight = model(preflight_preferences, num_steps=0).final
        preflight_summary = {
            "preferences": preflight_preferences.tolist(),
            "status": preflight.status.tolist(),
            "loss": preflight.loss.detach().tolist(),
            "energy_wh_kg": preflight.energy.detach().tolist(),
            "capacity_retention_3c": preflight.retention.detach().tolist(),
        }
        run.event("physics_preflight", **preflight_summary)
        if not bool(preflight.status.bool().all()):
            raise RuntimeError(
                "GradCell physics preflight failed for one or more canonical preferences"
            )
        run.event("training_started", parameter_count=sum(p.numel() for p in model.parameters()))
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
        )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "losses": result.losses,
                "validation_losses": result.validation_losses,
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
                    "refinement_steps": args.refinement_steps,
                    "seed": args.seed,
                },
            },
            args.checkpoint,
        )
        summary = {
            "steps_completed": len(result.losses),
            "final_train_loss": result.losses[-1] if result.losses else None,
            "best_validation_loss": result.best_validation_loss,
            "best_step": result.best_step,
            "stopped_early": result.stopped_early,
        }
        run.event("training_finished", **summary)
        run.save_summary({"result": summary, "artifacts": {"checkpoint": str(args.checkpoint)}})
        run.log(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
