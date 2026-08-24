from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/checkpoints/mvp.pt"))
    args = parser.parse_args()
    torch.manual_seed(7)
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
    result = train(
        model,
        steps=args.steps,
        batch_size=args.batch_size,
        refinement_steps=args.refinement_steps,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "losses": result.losses}, args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()

