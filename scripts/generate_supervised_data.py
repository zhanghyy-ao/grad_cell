from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.experiments import ExperimentRun
from gradcell.physics import AnalyticToyBackend, PyBaMMBackend
from gradcell.physics.soft_metrics import discharge_metrics

DESIGN_FIELDS = (
    "eps_p", "eps_n", "eps_s", "phi_p", "phi_n", "np_ratio",
    "diffusivity_p_multiplier", "diffusivity_n_multiplier",
    "nominal_capacity_ah", "stack_mass_kg",
)
PHYSICAL_DESIGN_FIELDS = (*DESIGN_FIELDS[:-2], "reference_capacity_ah", "stack_mass_kg")
TOY_TARGET_FIELDS = (
    "specific_energy_1c_wh_kg", "specific_power_3c_w_kg",
    "minimum_voltage_1c_v", "minimum_voltage_3c_v",
)
PHYSICAL_TARGET_FIELDS = (
    "reference_capacity_ah",
    "delivered_capacity_1c_ah",
    "delivered_capacity_3c_ah",
    "specific_energy_1c_wh_kg",
    "specific_power_3c_w_kg",
    "average_voltage_1c_v",
    "average_voltage_3c_v",
    "discharge_time_1c_s",
    "discharge_time_3c_s",
)


def make_backend(
    name: str,
    model: str,
    horizon_s: float,
    time_points: int,
    current_ramp_time_s: float,
):
    if name == "toy":
        return AnalyticToyBackend(time_points=time_points, horizon_s=horizon_s)
    return PyBaMMBackend(
        model_name=model,
        time_points=time_points,
        horizon_s=horizon_s,
        calculate_sensitivities=False,
        current_ramp_time_s=current_ramp_time_s,
    )


def save_dataset(
    output: Path,
    latent_parts: list[np.ndarray],
    design_parts: list[np.ndarray],
    target_parts: list[np.ndarray],
    metadata: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            latent=np.concatenate(latent_parts),
            design=np.concatenate(design_parts),
            targets=np.concatenate(target_parts),
            metadata=np.asarray(json.dumps(metadata)),
        )
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random hard-feasible Chen2020 designs and physics labels."
    )
    parser.add_argument("--backend", choices=("pybamm", "toy"), default="pybamm")
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="DFN")
    parser.add_argument(
        "--capacity-formula",
        choices=("electrode_theoretical", "chen2020_scaled"),
        default="chen2020_scaled",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-std", type=float, default=1.25)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument(
        "--current-ramp-time-s",
        type=float,
        default=0.0,
        help=(
            "Current ramp for physical labeling. Constant current (0, the default) "
            "avoids DFN consistent-initialization failures."
        ),
    )
    parser.add_argument(
        "--capacity-calibration-rate",
        type=float,
        default=0.1,
        help="Low C-rate used to measure each design's reference capacity.",
    )
    parser.add_argument(
        "--capacity-calibration-iterations",
        type=int,
        default=2,
        help="Number of current/capacity consistency updates.",
    )
    parser.add_argument(
        "--maximum-duration-factor",
        type=float,
        default=1.5,
        help="Maximum simulated duration relative to the nominal C-rate duration.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/chen2020_supervised.npz"))
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.capacity_calibration_rate <= 0.0:
        parser.error("--capacity-calibration-rate must be positive")
    if args.capacity_calibration_iterations < 1:
        parser.error("--capacity-calibration-iterations must be at least 1")
    if args.maximum_duration_factor <= 1.0:
        parser.error("--maximum-duration-factor must be greater than 1")

    with ExperimentRun("generate_supervised_data", args, run_dir=args.run_dir) as run:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(args.seed)
        decoder = DesignSpace(capacity_formula=args.capacity_formula)
        run.log(
            f"building {args.backend}/{args.model} 1C/3C backends; "
            f"capacity_formula={args.capacity_formula}"
        )
        target_fields = TOY_TARGET_FIELDS
        design_fields = DESIGN_FIELDS
        calibration_backend = None
        if args.backend == "pybamm":
            target_fields = PHYSICAL_TARGET_FIELDS
            design_fields = PHYSICAL_DESIGN_FIELDS
            calibration_backend = PyBaMMBackend(
                model_name=args.model,
                horizon_s=(
                    args.maximum_duration_factor
                    * 3600.0
                    / args.capacity_calibration_rate
                ),
                time_points=args.time_points,
                calculate_sensitivities=False,
                current_ramp_time_s=args.current_ramp_time_s,
                physical_voltage_cutoffs=True,
            )
            backend_1c = PyBaMMBackend(
                model_name=args.model,
                horizon_s=args.maximum_duration_factor * 3600.0,
                time_points=args.time_points,
                calculate_sensitivities=False,
                current_ramp_time_s=args.current_ramp_time_s,
                physical_voltage_cutoffs=True,
            )
            backend_3c = PyBaMMBackend(
                model_name=args.model,
                horizon_s=args.maximum_duration_factor * 1200.0,
                time_points=args.time_points,
                calculate_sensitivities=False,
                current_ramp_time_s=args.current_ramp_time_s,
                physical_voltage_cutoffs=True,
            )
        else:
            backend_1c = make_backend(
                args.backend, args.model, 3600.0, args.time_points, args.current_ramp_time_s
            )
            backend_3c = make_backend(
                args.backend, args.model, 1200.0, args.time_points, args.current_ramp_time_s
            )
        latent = args.latent_std * torch.randn(args.samples, decoder.latent_dim)
        saved_latent, saved_design, saved_targets = [], [], []
        failed_indices: list[int] = []
        failed_diagnostics: list[dict] = []
        started = time.perf_counter()

        for batch_index, start in enumerate(range(0, args.samples, args.batch_size), start=1):
            stop = min(start + args.batch_size, args.samples)
            batch_started = time.perf_counter()
            latent_batch = latent[start:stop]
            design = decoder(latent_batch)
            if args.backend == "pybamm":
                base_inputs = design.physics_tensor(1.0).detach().numpy()
                reference_capacity = design.nominal_capacity_ah.detach().numpy().copy()
                calibration_result = None
                for _ in range(args.capacity_calibration_iterations):
                    calibration_inputs = base_inputs.copy()
                    calibration_inputs[:, -1] = (
                        args.capacity_calibration_rate * reference_capacity
                    )
                    calibration_result = calibration_backend.solve_discharge_batch(
                        calibration_inputs
                    )
                    positive_capacity = calibration_result.delivered_capacity_ah > 0.0
                    reference_capacity[positive_capacity] = (
                        calibration_result.delivered_capacity_ah[positive_capacity]
                    )
                calibration_cutoff = np.asarray(
                    [
                        row["reached_voltage_cutoff"]
                        for row in calibration_backend.last_solve_diagnostics
                    ],
                    dtype=bool,
                )
                inputs_1c = base_inputs.copy()
                inputs_3c = base_inputs.copy()
                inputs_1c[:, -1] = reference_capacity
                inputs_3c[:, -1] = 3.0 * reference_capacity
                result_1c = backend_1c.solve_discharge_batch(inputs_1c)
                diagnostics_1c = list(backend_1c.last_solve_diagnostics)
                result_3c = backend_3c.solve_discharge_batch(inputs_3c)
                diagnostics_3c = list(backend_3c.last_solve_diagnostics)
                cutoff_1c = np.asarray(
                    [row["reached_voltage_cutoff"] for row in diagnostics_1c], dtype=bool
                )
                cutoff_3c = np.asarray(
                    [row["reached_voltage_cutoff"] for row in diagnostics_3c], dtype=bool
                )
                ok = (
                    (calibration_result.status == 1)
                    & calibration_cutoff
                    & (result_1c.status == 1)
                    & cutoff_1c
                    & (result_3c.status == 1)
                    & cutoff_3c
                )
            else:
                result_1c = backend_1c.solve_batch(
                    design.physics_tensor(1.0).detach().numpy()
                )
                result_3c = backend_3c.solve_batch(
                    design.physics_tensor(3.0).detach().numpy()
                )
                ok = (result_1c.status == 1) & (result_3c.status == 1)
                ok &= np.isfinite(result_1c.trajectories).all(axis=(1, 2))
                ok &= np.isfinite(result_3c.trajectories).all(axis=(1, 2))

            if bool(ok.any()):
                mask = torch.from_numpy(ok)
                saved_latent.append(latent_batch[mask].numpy())
                if args.backend == "pybamm":
                    reference_tensor = torch.from_numpy(reference_capacity)
                    design_columns = [
                        reference_tensor[mask]
                        if field == "nominal_capacity_ah"
                        else getattr(design, field)[mask]
                        for field in DESIGN_FIELDS
                    ]
                    saved_design.append(torch.stack(design_columns, dim=-1).numpy())
                    mass = design.stack_mass_kg.detach().numpy()
                    specific_energy_1c = result_1c.delivered_energy_wh / mass
                    duration_3c_h = np.maximum(result_3c.discharge_time_s / 3600.0, 1e-12)
                    specific_power_3c = (
                        result_3c.delivered_energy_wh / duration_3c_h / mass
                    )
                    saved_targets.append(
                        np.stack(
                            [
                                reference_capacity,
                                result_1c.delivered_capacity_ah,
                                result_3c.delivered_capacity_ah,
                                specific_energy_1c,
                                specific_power_3c,
                                result_1c.average_voltage_v,
                                result_3c.average_voltage_v,
                                result_1c.discharge_time_s,
                                result_3c.discharge_time_s,
                            ],
                            axis=-1,
                        )[ok]
                    )
                else:
                    capacity = design.nominal_capacity_ah[mask]
                    mass = design.stack_mass_kg[mask]
                    metrics_1c = discharge_metrics(
                        torch.from_numpy(result_1c.trajectories[ok]),
                        capacity,
                        mass,
                        3600.0,
                    )
                    metrics_3c = discharge_metrics(
                        torch.from_numpy(result_3c.trajectories[ok]),
                        3.0 * capacity,
                        mass,
                        1200.0,
                    )
                    saved_design.append(
                        torch.stack(
                            [getattr(design, field)[mask] for field in DESIGN_FIELDS], dim=-1
                        ).detach().numpy()
                    )
                    saved_targets.append(
                        torch.stack(
                            [
                                metrics_1c.specific_energy_wh_kg,
                                metrics_3c.specific_power_w_kg,
                                metrics_1c.minimum_voltage_v,
                                metrics_3c.minimum_voltage_v,
                            ],
                            dim=-1,
                        ).detach().numpy()
                    )
            failed_indices.extend((start + np.flatnonzero(~ok)).tolist())
            if args.backend == "pybamm":
                for local_index in np.flatnonzero(~ok):
                    failed_diagnostics.append(
                        {
                            "sample_index": int(start + local_index),
                            "capacity_calibration": (
                                calibration_backend.last_solve_diagnostics[local_index]
                            ),
                            "discharge_1c": diagnostics_1c[local_index],
                            "discharge_3c": diagnostics_3c[local_index],
                        }
                    )
            valid_count = sum(values.shape[0] for values in saved_latent)
            progress = {
                "batch": batch_index,
                "processed": stop,
                "valid": valid_count,
                "failed": len(failed_indices),
                "batch_elapsed_s": time.perf_counter() - batch_started,
            }
            run.event("batch_finished", **progress)
            run.log(
                f"processed={stop}/{args.samples} valid={valid_count} "
                f"failed={len(failed_indices)}"
            )
            if saved_latent and args.snapshot_every > 0 and batch_index % args.snapshot_every == 0:
                snapshot_metadata = {
                    "status": "partial",
                    "processed_samples": stop,
                    "valid_samples": valid_count,
                    "failed_indices": failed_indices,
                    "failed_diagnostics": failed_diagnostics,
                    "design_fields": design_fields,
                    "target_fields": target_fields,
                }
                save_dataset(
                    args.output, saved_latent, saved_design, saved_targets, snapshot_metadata
                )
                run.event("dataset_snapshot_saved", output=str(args.output), valid=valid_count)

        if not saved_latent:
            raise RuntimeError("No successful simulations were generated")
        metadata = {
            "status": "completed",
            "backend": args.backend,
            "parameter_set": "Chen2020",
            "model": args.model if args.backend == "pybamm" else "analytic-toy",
            "capacity_formula": args.capacity_formula,
            "seed": args.seed,
            "requested_samples": args.samples,
            "valid_samples": sum(values.shape[0] for values in saved_latent),
            "failed_indices": failed_indices,
            "failed_diagnostics": failed_diagnostics,
            "failure_rate": len(failed_indices) / args.samples,
            "latent_std": args.latent_std,
            "time_points": args.time_points,
            "design_fields": design_fields,
            "target_fields": target_fields,
            "capacity_calibration_rate": (
                args.capacity_calibration_rate if args.backend == "pybamm" else None
            ),
            "capacity_calibration_iterations": (
                args.capacity_calibration_iterations if args.backend == "pybamm" else None
            ),
            "elapsed_s": time.perf_counter() - started,
        }
        save_dataset(args.output, saved_latent, saved_design, saved_targets, metadata)
        run.save_summary({"result": metadata, "artifacts": {"dataset": str(args.output)}})
        run.log(f"saved {metadata['valid_samples']} samples to {args.output}")


if __name__ == "__main__":
    main()
