from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gradcell.benchmark.dfn_parameter import (
    PARAMETER_FIELDS,
    BenchmarkFilter,
    apply_multipliers,
    sample_log_multipliers,
    structural_feasibility,
)
from gradcell.physics import PyBaMMBackend


def parse_rates(value: str) -> tuple[float, ...]:
    rates = tuple(float(item) for item in value.split(","))
    if not rates or any(rate <= 0.0 for rate in rates):
        raise argparse.ArgumentTypeError("C-rates must be comma-separated positive numbers")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an inverse-parameter DFN benchmark with physical cutoff curves."
    )
    parser.add_argument("--family", choices=("single", "multi"), default="single")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--candidate-factor", type=int, default=5)
    parser.add_argument("--multiplier-low", type=float, default=0.5)
    parser.add_argument("--multiplier-high", type=float, default=2.0)
    parser.add_argument("--c-rates", type=parse_rates, default=(0.5, 1.0, 2.0))
    parser.add_argument("--nominal-capacity-ah", type=float, default=5.0)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--maximum-duration-factor", type=float, default=1.5)
    parser.add_argument("--min-capacity-change-fraction", type=float, default=0.01)
    parser.add_argument("--min-voltage-rmse-v", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("data/dfn_parameter_benchmark_single_s7.npz")
    )
    args = parser.parse_args()
    if args.samples < 1 or args.candidate_factor < 1:
        parser.error("--samples and --candidate-factor must be positive")
    if args.nominal_capacity_ah <= 0.0 or args.maximum_duration_factor <= 1.0:
        parser.error("capacity must be positive and duration factor must be greater than 1")

    longest_horizon = args.maximum_duration_factor * 3600.0 / min(args.c_rates)
    backend = PyBaMMBackend(
        model_name="DFN",
        parameter_set="Chen2020",
        time_points=args.time_points,
        horizon_s=longest_horizon,
        calculate_sensitivities=False,
        current_ramp_time_s=0.0,
        physical_voltage_cutoffs=True,
    )
    nominal_parameters = backend.nominal_input_values
    nominal_by_rate: dict[float, tuple[np.ndarray, float]] = {}
    for rate in args.c_rates:
        nominal_input = np.concatenate(
            [nominal_parameters, [args.nominal_capacity_ah * rate]]
        )[None, :]
        nominal_result = backend.solve_normalized_discharge_batch(nominal_input)
        if nominal_result.status[0] != 1:
            raise RuntimeError(
                f"Nominal Chen2020 DFN failed at {rate:g}C: "
                f"{backend.last_solve_diagnostics[0]}"
            )
        nominal_by_rate[rate] = (
            nominal_result.voltage_v[0],
            float(nominal_result.delivered_capacity_ah[0]),
        )

    candidate_count = args.samples * args.candidate_factor
    multipliers = sample_log_multipliers(
        args.family,
        candidate_count,
        len(PARAMETER_FIELDS),
        args.multiplier_low,
        args.multiplier_high,
        args.seed,
    )
    values = apply_multipliers(nominal_parameters, multipliers)
    feasible = structural_feasibility(values)
    rates = np.resize(np.asarray(args.c_rates, dtype=np.float64), candidate_count)
    inputs = np.concatenate(
        [values, (args.nominal_capacity_ah * rates)[:, None]], axis=1
    )
    result = backend.solve_normalized_discharge_batch(inputs)
    diagnostics = list(backend.last_solve_diagnostics)
    criterion = BenchmarkFilter(
        min_capacity_change_fraction=args.min_capacity_change_fraction,
        min_voltage_rmse_v=args.min_voltage_rmse_v,
    )
    accepted, audit = [], []
    for index in range(candidate_count):
        nominal_voltage, nominal_capacity = nominal_by_rate[float(rates[index])]
        keep, reason, capacity_change, voltage_rmse = criterion.accepts(
            int(result.status[index]),
            float(result.delivered_capacity_ah[index]),
            nominal_capacity,
            result.voltage_v[index],
            nominal_voltage,
        )
        if not feasible[index]:
            keep, reason = False, "infeasible_volume_fractions"
        record = {
            "candidate_index": index,
            "accepted": keep,
            "reason": reason,
            "c_rate": float(rates[index]),
            "capacity_change_fraction": capacity_change,
            "voltage_rmse_v": voltage_rmse,
            "solver": diagnostics[index],
        }
        audit.append(record)
        if keep and len(accepted) < args.samples:
            accepted.append(index)
    if len(accepted) < args.samples:
        raise RuntimeError(
            f"Only {len(accepted)}/{args.samples} informative cases survived "
            f"{candidate_count} candidates; increase --candidate-factor or relax filters"
        )

    selected = np.asarray(accepted, dtype=np.int64)
    case_ids = np.asarray(
        [f"{args.family}-{args.seed}-{position:04d}" for position in range(len(selected))]
    )
    metadata = {
        "status": "completed",
        "benchmark": "DFN inverse parameter estimation",
        "model": "DFN",
        "parameter_set": "Chen2020",
        "family": args.family,
        "seed": args.seed,
        "requested_samples": args.samples,
        "candidate_samples": candidate_count,
        "accepted_samples": len(selected),
        "parameter_fields": PARAMETER_FIELDS,
        "multiplier_bounds": [args.multiplier_low, args.multiplier_high],
        "c_rates": args.c_rates,
        "curve_representation": (
            "Voltage is interpolated on normalized discharge time; physical cutoff time "
            "and delivered capacity are retained as separate labels."
        ),
        "filters": {
            "requires_solver_success": True,
            "requires_minimum_voltage_cutoff": True,
            "min_capacity_change_fraction": args.min_capacity_change_fraction,
            "min_voltage_rmse_v": args.min_voltage_rmse_v,
            "informative_rule": "capacity change OR normalized-voltage RMSE",
        },
        "design_reference": (
            "Inspired by Battery-Sim-Agent's simulated inverse-parameter benchmark; "
            "implemented independently for GradCell."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            case_id=case_ids,
            parameter_names=np.asarray(PARAMETER_FIELDS),
            nominal_parameter_values=nominal_parameters,
            parameter_values=values[selected],
            parameter_multipliers=multipliers[selected],
            c_rate=rates[selected],
            normalized_time=result.normalized_time,
            target_voltage_v=result.voltage_v[selected],
            discharge_time_s=result.discharge_time_s[selected],
            delivered_capacity_ah=result.delivered_capacity_ah[selected],
            metadata=np.asarray(json.dumps(metadata)),
        )
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "candidate_audit": audit}, indent=2),
        encoding="utf-8",
    )
    settings = []
    for case_id, index in zip(case_ids.tolist(), selected.tolist()):
        changed = {
            name: float(value)
            for name, value, multiplier in zip(
                PARAMETER_FIELDS, values[index], multipliers[index]
            )
            if not np.isclose(multiplier, 1.0)
        }
        settings.append(
            {
                "case_id": case_id,
                "model": "DFN",
                "parameter_set": "Chen2020",
                "c_rate": float(rates[index]),
                "parameter_change": changed,
            }
        )
    args.output.with_suffix(".yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
