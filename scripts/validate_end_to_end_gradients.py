from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradcell.design import DesignSpace
from gradcell.losses import SmoothTchebycheff
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.physics.soft_metrics import discharge_metrics


def make_layer(backend: str, horizon_s: float) -> DifferentiablePhysicsLayer:
    physics = (
        AnalyticToyBackend(horizon_s=horizon_s)
        if backend == "toy"
        else PyBaMMBackend(horizon_s=horizon_s)
    )
    return DifferentiablePhysicsLayer(physics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate decoder -> physics -> metrics -> objective gradients."
    )
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--preference", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results/gradient_chain.json"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(args.seed)
    decoder = DesignSpace()
    layer_1c = make_layer(args.backend, 3600.0)
    layer_3c = make_layer(args.backend, 1200.0)
    objective = SmoothTchebycheff()

    def scalar_loss(latent: torch.Tensor) -> torch.Tensor:
        design = decoder(latent)
        voltage_1c, status_1c, _ = layer_1c(design.physics_tensor(1.0))
        voltage_3c, status_3c, _ = layer_3c(design.physics_tensor(3.0))
        if not bool((status_1c * status_3c).all()):
            raise RuntimeError("Physics solve failed; use another seed or narrower samples")
        metrics_1c = discharge_metrics(
            voltage_1c, design.nominal_capacity_ah, design.stack_mass_kg, 3600.0
        )
        metrics_3c = discharge_metrics(
            voltage_3c, 3.0 * design.nominal_capacity_ah, design.stack_mass_kg, 1200.0
        )
        preference = torch.full(
            (latent.shape[0],), args.preference, dtype=latent.dtype, device=latent.device
        )
        return objective(
            metrics_1c.specific_energy_wh_kg,
            metrics_3c.specific_power_w_kg,
            preference,
        ).mean()

    records = []
    for sample_index in range(args.samples):
        point = (0.7 * torch.randn(1, decoder.latent_dim, generator=generator)).requires_grad_()
        loss = scalar_loss(point)
        (gradient,) = torch.autograd.grad(loss, point)
        for direction_index in range(args.directions):
            direction = torch.randn(point.shape, generator=generator)
            direction = direction / direction.norm().clamp_min(1e-12)
            autodiff = float((gradient * direction).sum())
            with torch.no_grad():
                plus = float(scalar_loss(point + args.eps * direction))
                minus = float(scalar_loss(point - args.eps * direction))
            finite_difference = (plus - minus) / (2.0 * args.eps)
            absolute_error = abs(autodiff - finite_difference)
            relative_error = absolute_error / max(abs(autodiff), abs(finite_difference), 1e-12)
            records.append(
                {
                    "sample": sample_index,
                    "direction": direction_index,
                    "loss": float(loss),
                    "autodiff": autodiff,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

    relative_errors = torch.tensor([row["relative_error"] for row in records])
    report = {
        "backend": args.backend,
        "eps": args.eps,
        "samples": args.samples,
        "directions_per_sample": args.directions,
        "median_relative_error": float(relative_errors.median()),
        "max_relative_error": float(relative_errors.max()),
        "checks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
