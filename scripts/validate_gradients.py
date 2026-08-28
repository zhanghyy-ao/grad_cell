from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradcell.experiments import ExperimentRun
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.physics.gradient_validation import directional_derivative_check


def make_backend(name: str):
    if name == "toy":
        return AnalyticToyBackend()
    return PyBaMMBackend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("results/gradient_physics.json"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    with ExperimentRun("gradient_physics", args, run_dir=args.run_dir) as run:
        run.log(f"validating {args.backend} physics gradient with eps={args.eps:g}")
        layer = DifferentiablePhysicsLayer(make_backend(args.backend))
        point = torch.tensor(
            [[0.30, 0.30, 0.45, 0.55, 0.58, 1.0, 1.0, 2.0]],
            dtype=torch.float64,
        )

        def objective(value):
            trajectory, status, _ = layer(value)
            if not bool(status.all()):
                raise RuntimeError("Physics solve failed during gradient validation")
            return trajectory.square().mean()

        report = directional_derivative_check(objective, point, eps=args.eps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        run.event("gradient_check_finished", **report)
        run.save_summary({"result": report, "artifacts": {"result": str(args.output)}})
        run.log(json.dumps(report))


if __name__ == "__main__":
    main()
