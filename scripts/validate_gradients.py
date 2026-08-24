from __future__ import annotations

import argparse
import json

import torch

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
    args = parser.parse_args()
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

    print(json.dumps(directional_derivative_check(objective, point, eps=args.eps), indent=2))


if __name__ == "__main__":
    main()
