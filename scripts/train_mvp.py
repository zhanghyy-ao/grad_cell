from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gradcell.experiments import ExperimentRun
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/checkpoints/mvp.pt"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    with ExperimentRun("train_mvp", args, run_dir=args.run_dir) as run:
        torch.manual_seed(args.seed)
        run.log(f"building {args.backend} 1C/3C backends")
        if args.backend == "toy":
            backend1 = AnalyticToyBackend(horizon_s=3600.0)
            backend3 = AnalyticToyBackend(horizon_s=1200.0)
        else:
            backend1 = PyBaMMBackend(horizon_s=3600.0)
            backend3 = PyBaMMBackend(horizon_s=1200.0)
        model = GradCell(
            DifferentiablePhysicsLayer(backend1),
            DifferentiablePhysicsLayer(backend3),
        ).double()
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
