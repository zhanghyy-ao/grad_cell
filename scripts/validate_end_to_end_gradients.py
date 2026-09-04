from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from gradcell.design import DesignSpace
from gradcell.experiments import ExperimentRun
from gradcell.losses import SmoothTchebycheff
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer, PyBaMMBackend
from gradcell.physics.soft_metrics import discharge_metrics

DESIGN_FIELDS = (
    "eps_p",
    "eps_n",
    "eps_s",
    "phi_p",
    "phi_n",
    "np_ratio",
    "diffusivity_p_multiplier",
    "diffusivity_n_multiplier",
    "nominal_capacity_ah",
    "stack_mass_kg",
)


def safe_float(value: torch.Tensor | float) -> float | None:
    number = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
    return number if math.isfinite(number) else None


def tensor_diagnostics(value: torch.Tensor) -> dict:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "all_finite": bool(finite.all()),
        "nan_count": int(torch.isnan(detached).sum()),
        "positive_inf_count": int(torch.isposinf(detached).sum()),
        "negative_inf_count": int(torch.isneginf(detached).sum()),
        "minimum": float(finite_values.min()) if finite_values.numel() else None,
        "maximum": float(finite_values.max()) if finite_values.numel() else None,
    }


def first_nonfinite_stage(diagnostics: dict) -> str | None:
    stages = (
        ("voltage_1c", diagnostics["voltage_1c"]["all_finite"]),
        ("voltage_5c", diagnostics["voltage_5c"]["all_finite"]),
        ("voltage_6c", diagnostics["voltage_6c"]["all_finite"]),
        ("energy_1c", diagnostics["metrics"]["energy_1c_wh_kg"] is not None),
        (
            "energy_retention_5c",
            diagnostics["metrics"]["energy_retention_5c"] is not None,
        ),
        ("energy_retention_6c", diagnostics["metrics"]["energy_retention_6c"] is not None),
        ("minimum_voltage_1c", diagnostics["metrics"]["minimum_voltage_1c_v"] is not None),
        ("minimum_voltage_5c", diagnostics["metrics"]["minimum_voltage_5c_v"] is not None),
        ("minimum_voltage_6c", diagnostics["metrics"]["minimum_voltage_6c_v"] is not None),
        ("loss", diagnostics["loss"] is not None),
    )
    return next((name for name, is_finite in stages if not is_finite), None)


def make_layer(
    backend: str,
    horizon_s: float,
    model: str,
    current_ramp_time_s: float,
) -> DifferentiablePhysicsLayer:
    physics = (
        AnalyticToyBackend(horizon_s=horizon_s)
        if backend == "toy"
        else PyBaMMBackend(
            model_name=model,
            horizon_s=horizon_s,
            current_ramp_time_s=current_ramp_time_s,
        )
    )
    return DifferentiablePhysicsLayer(physics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate decoder -> physics -> metrics -> objective gradients."
    )
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="toy")
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="DFN")
    parser.add_argument(
        "--capacity-formula",
        choices=("electrode_theoretical", "chen2020_scaled"),
        default="chen2020_scaled",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--current-ramp-time-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--preference", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results/gradient_chain.json"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    with ExperimentRun("gradient_end_to_end", args, run_dir=args.run_dir) as run:
        torch.set_default_dtype(torch.float64)
        generator = torch.Generator().manual_seed(args.seed)
        decoder = DesignSpace(capacity_formula=args.capacity_formula)
        run.log(
            f"building {args.backend}/{args.model} backends; "
            f"capacity_formula={args.capacity_formula}"
        )
        layer_1c = make_layer(args.backend, 3600.0, args.model, args.current_ramp_time_s)
        layer_5c = make_layer(args.backend, 720.0, args.model, args.current_ramp_time_s)
        layer_6c = make_layer(args.backend, 600.0, args.model, args.current_ramp_time_s)
        objective = SmoothTchebycheff()

        def evaluate(latent: torch.Tensor) -> tuple[torch.Tensor, dict]:
            design = decoder(latent)
            physics_inputs_1c = design.physics_tensor(1.0)
            physics_inputs_5c = design.physics_tensor(5.0)
            physics_inputs_6c = design.physics_tensor(6.0)
            voltage_1c, status_1c, runtime_1c = layer_1c(physics_inputs_1c)
            voltage_5c, status_5c, runtime_5c = layer_5c(physics_inputs_5c)
            voltage_6c, status_6c, runtime_6c = layer_6c(physics_inputs_6c)
            metrics_1c = discharge_metrics(
                voltage_1c, design.nominal_capacity_ah, design.stack_mass_kg, 3600.0
            )
            metrics_5c = discharge_metrics(
                voltage_5c, 5.0 * design.nominal_capacity_ah, design.stack_mass_kg, 720.0
            )
            metrics_6c = discharge_metrics(
                voltage_6c, 6.0 * design.nominal_capacity_ah, design.stack_mass_kg, 600.0
            )
            preference = torch.full(
                (latent.shape[0],), args.preference, dtype=latent.dtype, device=latent.device
            )
            loss = objective(
                metrics_1c.specific_energy_wh_kg,
                metrics_5c.specific_energy_wh_kg / metrics_1c.specific_energy_wh_kg.clamp_min(1e-8),
                metrics_6c.specific_energy_wh_kg / metrics_1c.specific_energy_wh_kg.clamp_min(1e-8),
                preference,
            ).mean()
            diagnostics = {
                "latent": latent.detach().reshape(-1).tolist(),
                "design": {
                    field: safe_float(getattr(design, field).reshape(-1)[0])
                    for field in DESIGN_FIELDS
                },
                "physics_inputs_1c": physics_inputs_1c.detach().reshape(-1).tolist(),
                "physics_inputs_5c": physics_inputs_5c.detach().reshape(-1).tolist(),
                "physics_inputs_6c": physics_inputs_6c.detach().reshape(-1).tolist(),
                "solver": {
                    "status_1c": int(status_1c.reshape(-1)[0]),
                    "status_5c": int(status_5c.reshape(-1)[0]),
                    "status_6c": int(status_6c.reshape(-1)[0]),
                    "runtime_1c_s": safe_float(runtime_1c.reshape(-1)[0]),
                    "runtime_5c_s": safe_float(runtime_5c.reshape(-1)[0]),
                    "runtime_6c_s": safe_float(runtime_6c.reshape(-1)[0]),
                    "details_1c": layer_1c.backend.last_solve_diagnostics[0],
                    "details_5c": layer_5c.backend.last_solve_diagnostics[0],
                    "details_6c": layer_6c.backend.last_solve_diagnostics[0],
                },
                "voltage_1c": tensor_diagnostics(voltage_1c),
                "voltage_5c": tensor_diagnostics(voltage_5c),
                "voltage_6c": tensor_diagnostics(voltage_6c),
                "metrics": {
                    "energy_1c_wh_kg": safe_float(metrics_1c.specific_energy_wh_kg[0]),
                    "energy_retention_5c": safe_float(
                        metrics_5c.specific_energy_wh_kg[0]
                        / metrics_1c.specific_energy_wh_kg[0].clamp_min(1e-8)
                    ),
                    "energy_retention_6c": safe_float(
                        metrics_6c.specific_energy_wh_kg[0]
                        / metrics_1c.specific_energy_wh_kg[0].clamp_min(1e-8)
                    ),
                    "minimum_voltage_1c_v": safe_float(metrics_1c.minimum_voltage_v[0]),
                    "minimum_voltage_5c_v": safe_float(metrics_5c.minimum_voltage_v[0]),
                    "minimum_voltage_6c_v": safe_float(metrics_6c.minimum_voltage_v[0]),
                },
                "loss": safe_float(loss),
            }
            diagnostics["first_nonfinite_stage"] = first_nonfinite_stage(diagnostics)
            return loss, diagnostics

        def scalar_loss(latent: torch.Tensor) -> torch.Tensor:
            loss, _ = evaluate(latent)
            return loss

        records = []
        sample_diagnostics = []
        invalid_samples = []
        for sample_index in range(args.samples):
            run.log(f"sample {sample_index + 1}/{args.samples}")
            point = (0.7 * torch.randn(1, decoder.latent_dim, generator=generator)).requires_grad_()
            loss, diagnostics = evaluate(point)
            diagnostics["sample"] = sample_index
            if diagnostics["loss"] is not None:
                (gradient,) = torch.autograd.grad(loss, point)
                diagnostics["latent_gradient"] = tensor_diagnostics(gradient)
                if not diagnostics["latent_gradient"]["all_finite"]:
                    diagnostics["first_nonfinite_stage"] = "latent_gradient"
            else:
                gradient = None
                diagnostics["latent_gradient"] = None
            sample_diagnostics.append(diagnostics)
            run.event("sample_diagnostics", **diagnostics)
            if diagnostics["first_nonfinite_stage"] is not None:
                invalid_samples.append(sample_index)
                run.log(
                    f"sample={sample_index} invalid at {diagnostics['first_nonfinite_stage']}",
                    level="WARNING",
                )
                continue
            assert gradient is not None
            for direction_index in range(args.directions):
                direction = torch.randn(point.shape, generator=generator)
                direction = direction / direction.norm().clamp_min(1e-12)
                autodiff = float((gradient * direction).sum())
                with torch.no_grad():
                    plus = float(scalar_loss(point + args.eps * direction))
                    minus = float(scalar_loss(point - args.eps * direction))
                direction_finite = all(math.isfinite(value) for value in (autodiff, plus, minus))
                if direction_finite:
                    finite_difference = (plus - minus) / (2.0 * args.eps)
                    absolute_error = abs(autodiff - finite_difference)
                    relative_error = absolute_error / max(
                        abs(autodiff), abs(finite_difference), 1e-12
                    )
                else:
                    finite_difference = absolute_error = relative_error = None
                record = {
                    "sample": sample_index,
                    "direction": direction_index,
                    "valid": direction_finite,
                    "loss": safe_float(loss),
                    "autodiff": autodiff if math.isfinite(autodiff) else None,
                    "loss_plus": plus if math.isfinite(plus) else None,
                    "loss_minus": minus if math.isfinite(minus) else None,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
                records.append(record)
                run.event("direction_checked", **record)

        relative_errors = [
            row["relative_error"] for row in records if row["relative_error"] is not None
        ]
        report = {
            "backend": args.backend,
            "model": args.model,
            "capacity_formula": args.capacity_formula,
            "eps": args.eps,
            "samples": args.samples,
            "directions_per_sample": args.directions,
            "valid_samples": args.samples - len(invalid_samples),
            "invalid_samples": invalid_samples,
            "valid_direction_checks": len(relative_errors),
            "median_relative_error": (
                float(torch.tensor(relative_errors).median()) if relative_errors else None
            ),
            "max_relative_error": max(relative_errors) if relative_errors else None,
            "passed": bool(relative_errors and not invalid_samples),
            "sample_diagnostics": sample_diagnostics,
            "checks": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        run.save_summary({"result": report, "artifacts": {"result": str(args.output)}})
        run.log(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("checks", "sample_diagnostics")
                }
            )
        )


if __name__ == "__main__":
    main()
