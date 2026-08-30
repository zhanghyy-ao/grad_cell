from __future__ import annotations

import argparse
import json

import torch

from gradcell.physics import (
    AnalyticElectrolyteBackend,
    DifferentiablePhysicsLayer,
    PyBaMMElectrolyteDFNBackend,
)
from gradcell.physics.gradient_validation import directional_derivative_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DFN-to-PyTorch electrolyte gradients.")
    parser.add_argument("--backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--time-points", type=int, default=21)
    parser.add_argument("--probe-horizon-s", type=float, default=300.0)
    parser.add_argument("--current-a", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    backend_cls = (
        PyBaMMElectrolyteDFNBackend
        if args.backend == "dfn"
        else AnalyticElectrolyteBackend
    )
    backend = backend_cls(
        time_points=args.time_points,
        probe_horizon_s=args.probe_horizon_s,
    )
    layer = DifferentiablePhysicsLayer(backend)
    point = torch.tensor([[0.0, args.current_a]], dtype=torch.float64)

    def objective(value: torch.Tensor) -> torch.Tensor:
        trajectory, status, _ = layer(value)
        if not bool((status == 1).all()):
            raise RuntimeError(f"Physics solve failed: {backend.last_solve_diagnostics}")
        weights = torch.linspace(0.5, 1.5, trajectory.shape[-1], dtype=value.dtype)
        return (trajectory[:, 0] * weights).mean()

    direction = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    result = directional_derivative_check(
        objective, point, direction=direction, eps=args.eps
    )
    result.update(
        {
            "backend": args.backend,
            "input": {"log_conductivity_scale": 0.0, "current_a": args.current_a},
            "diagnostics": backend.last_solve_diagnostics,
        }
    )
    print(json.dumps(result, indent=2), flush=True)
    tolerance = 1e-5 if args.backend == "analytic" else 2e-2
    if result["relative_directional_error"] > tolerance:
        raise SystemExit(
            f"Gradient check failed: {result['relative_directional_error']:.3g} > {tolerance}"
        )


if __name__ == "__main__":
    main()
