from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gradcell.benchmark.dfn_parameter import PARAMETER_FIELDS, structural_feasibility
from gradcell.physics import PyBaMMBackend


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def cutoff_flags(backend: PyBaMMBackend) -> np.ndarray:
    return np.asarray(
        [bool(row.get("reached_voltage_cutoff")) for row in backend.last_solve_diagnostics]
    )


def make_backend(
    parameter_set: str,
    horizon_s: float,
    config: dict[str, Any],
) -> PyBaMMBackend:
    return PyBaMMBackend(
        model_name="DFN",
        parameter_set=parameter_set,
        horizon_s=horizon_s,
        time_points=int(config["time_points"]),
        rtol=float(config["rtol"]),
        atol=float(config["atol"]),
        calculate_sensitivities=False,
        current_ramp_time_s=0.0,
        physical_voltage_cutoffs=True,
    )


def sobol_multipliers(
    count: int,
    bounds: np.ndarray,
    seed: int,
) -> np.ndarray:
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError("Sobol sampling requires scipy; install the physics extra") from exc
    unit = qmc.Sobol(d=len(bounds), scramble=True, seed=seed).random(count)
    log_low = np.log(bounds[:, 0])
    log_high = np.log(bounds[:, 1])
    return np.exp(log_low + unit * (log_high - log_low))


def regular_multipliers(count: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    """Change one parameter per case, analogous to the source benchmark's single mode."""
    sampled = sobol_multipliers(count, bounds, seed)
    result = np.ones_like(sampled)
    selected = np.arange(count) % sampled.shape[1]
    result[np.arange(count), selected] = sampled[np.arange(count), selected]
    return result


def review_difference(
    metrics: dict[str, Any], nominal: dict[str, Any], thresholds: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    suffixes = ("1c", "5c", "6c")
    capacity_change = max(
        abs(metrics[f"capacity_{suffix}_ah"] - nominal[f"capacity_{suffix}_ah"])
        / max(abs(nominal[f"capacity_{suffix}_ah"]), 1e-12)
        for suffix in suffixes
    )
    energy_change = max(
        abs(metrics[f"energy_{suffix}_wh"] - nominal[f"energy_{suffix}_wh"])
        / max(abs(nominal[f"energy_{suffix}_wh"]), 1e-12)
        for suffix in suffixes
    )
    voltage_change = max(
        abs(
            metrics[f"average_voltage_{suffix}_v"]
            - nominal[f"average_voltage_{suffix}_v"]
        )
        for suffix in suffixes
    )
    informative = (
        capacity_change >= float(thresholds["min_capacity_change_fraction"])
        or energy_change >= float(thresholds["min_energy_change_fraction"])
        or voltage_change >= float(thresholds["min_average_voltage_change_v"])
    )
    return informative, {
        "maximum_capacity_change_fraction": float(capacity_change),
        "maximum_energy_change_fraction": float(energy_change),
        "maximum_average_voltage_change_v": float(voltage_change),
    }


def nominal_capacity(backend: PyBaMMBackend) -> float:
    try:
        value = float(backend.parameters["Nominal cell capacity [A.h]"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The parameter set must define a scalar 'Nominal cell capacity [A.h]'"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Nominal cell capacity must be finite and positive")
    return value


def scalar_parameter(backend: PyBaMMBackend, name: str) -> float:
    try:
        value = float(backend.parameters[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The parameter set must define a scalar {name!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Parameter {name!r} must be finite")
    return value


def simulate_batch(
    values: np.ndarray,
    nominal_capacity_ah: float,
    calibration_backend: PyBaMMBackend,
    rate_backends: dict[float, PyBaMMBackend],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capacity = np.full(len(values), nominal_capacity_ah, dtype=np.float64)
    calibration_rate = float(config["calibration_rate"])
    calibration_result = None
    calibration_cutoff = None
    for _ in range(int(config["calibration_iterations"])):
        inputs = np.column_stack([values, calibration_rate * capacity])
        calibration_result = calibration_backend.solve_discharge_batch(inputs)
        calibration_cutoff = cutoff_flags(calibration_backend)
        valid = (calibration_result.status == 1) & calibration_cutoff
        capacity[valid] = calibration_result.delivered_capacity_ah[valid]

    assert calibration_result is not None and calibration_cutoff is not None
    results = {}
    rate_cutoffs = {}
    diagnostics = {}
    for rate, backend in rate_backends.items():
        inputs = np.column_stack([values, rate * capacity])
        results[rate] = backend.solve_discharge_batch(inputs)
        rate_cutoffs[rate] = cutoff_flags(backend)
        diagnostics[rate] = list(backend.last_solve_diagnostics)

    required_rates = sorted(rate_backends)
    reference_rate = 1.0
    if reference_rate not in results:
        raise ValueError("physics.c_rates must include 1.0C")
    reference_energy = np.maximum(results[reference_rate].delivered_energy_wh, 1e-12)
    reference_capacity = np.maximum(
        results[reference_rate].delivered_capacity_ah, 1e-12
    )
    metrics, audits = [], []
    for index in range(len(values)):
        success = bool(calibration_result.status[index] == 1 and calibration_cutoff[index])
        success = success and all(
            results[rate].status[index] == 1 and rate_cutoffs[rate][index]
            for rate in required_rates
        )
        metric = {
            "reference_capacity_ah": float(capacity[index]),
            "energy_1c_wh": float(results[1.0].delivered_energy_wh[index]),
            "capacity_1c_ah": float(results[1.0].delivered_capacity_ah[index]),
            "average_voltage_1c_v": float(results[1.0].average_voltage_v[index]),
            "minimum_voltage_1c_v": float(results[1.0].minimum_voltage_v[index]),
            "discharge_time_1c_s": float(results[1.0].discharge_time_s[index]),
        }
        for rate in required_rates:
            suffix = f"{rate:g}c"
            metric[f"energy_{suffix}_wh"] = float(results[rate].delivered_energy_wh[index])
            metric[f"capacity_{suffix}_ah"] = float(
                results[rate].delivered_capacity_ah[index]
            )
            metric[f"average_voltage_{suffix}_v"] = float(
                results[rate].average_voltage_v[index]
            )
            metric[f"minimum_voltage_{suffix}_v"] = float(
                results[rate].minimum_voltage_v[index]
            )
            metric[f"discharge_time_{suffix}_s"] = float(
                results[rate].discharge_time_s[index]
            )
            metric[f"energy_retention_{suffix}"] = float(
                results[rate].delivered_energy_wh[index] / reference_energy[index]
            )
            metric[f"capacity_retention_{suffix}"] = float(
                results[rate].delivered_capacity_ah[index] / reference_capacity[index]
            )
        metrics.append(metric)
        audits.append(
            {
                "solver_success": success,
                "calibration": calibration_backend.last_solve_diagnostics[index],
                "rates": {
                    f"{rate:g}C": diagnostics[rate][index] for rate in required_rates
                },
            }
        )
    return metrics, audits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a five-parameter-set DFN perturbation archive."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multiset_dfn_language.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--parameter-set", action="append", dest="parameter_sets")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    physics = config["physics"]
    perturbation = config["perturbation"]
    review = config["review"]
    output = args.output or Path(physics["output"])
    audit_path = args.audit or Path(physics["audit"])
    parameter_sets = args.parameter_sets or list(physics["parameter_sets"])
    mode_bounds = {}
    for mode in ("regular", "extreme"):
        fields = perturbation[f"{mode}_fields"]
        unknown_fields = set(fields) - set(PARAMETER_FIELDS)
        missing_fields = set(PARAMETER_FIELDS) - set(fields)
        if unknown_fields or missing_fields:
            raise ValueError(
                f"{mode} field mismatch; unknown={sorted(unknown_fields)}, "
                f"missing={sorted(missing_fields)}"
            )
        mode_bounds[mode] = np.asarray(
            [fields[name] for name in PARAMETER_FIELDS], dtype=np.float64
        )
        bounds = mode_bounds[mode]
        if np.any(bounds <= 0.0) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(f"Every {mode} multiplier interval must be positive")

    existing = load_jsonl(output)
    accepted_by_set_mode = {
        name: {
            mode: sum(
                row["parameter_set"] == name and row.get("mode") == mode
                for row in existing
            )
            for mode in ("regular", "extreme")
        }
        for name in parameter_sets
    }
    processed = {
        (row["parameter_set"], int(row["candidate_index"]))
        for row in load_jsonl(audit_path)
    }
    processed.update(
        (row["parameter_set"], int(row["candidate_index"])) for row in existing
    )
    requested_per_mode = int(physics["samples_per_mode"])
    candidate_count_per_mode = requested_per_mode * int(physics["candidate_factor"])
    candidate_count = 2 * candidate_count_per_mode
    batch_size = int(physics["batch_size"])
    rates = tuple(float(value) for value in physics["c_rates"])
    seed = int(experiment["seed"])

    for set_index, parameter_set in enumerate(parameter_sets):
        if all(
            accepted_by_set_mode[parameter_set][mode] >= requested_per_mode
            for mode in ("regular", "extreme")
        ):
            print(f"skip completed parameter set: {parameter_set}", flush=True)
            continue
        calibration_backend = make_backend(
            parameter_set,
            float(physics["maximum_duration_factor"])
            * 3600.0
            / float(physics["calibration_rate"]),
            physics,
        )
        rate_backends = {
            rate: make_backend(
                parameter_set,
                float(physics["maximum_duration_factor"]) * 3600.0 / rate,
                physics,
            )
            for rate in rates
        }
        nominal_values = calibration_backend.nominal_input_values.copy()
        capacity_ah = nominal_capacity(calibration_backend)
        temperature_k = scalar_parameter(calibration_backend, "Initial temperature [K]")
        regular = regular_multipliers(
            candidate_count_per_mode,
            mode_bounds["regular"],
            seed + 1009 * set_index,
        )
        extreme = sobol_multipliers(
            candidate_count_per_mode,
            mode_bounds["extreme"],
            seed + 1009 * set_index + 503,
        )
        multipliers = np.concatenate([regular, extreme])
        modes = np.asarray(
            ["regular"] * candidate_count_per_mode
            + ["extreme"] * candidate_count_per_mode
        )
        values = nominal_values[None, :] * multipliers
        feasible = structural_feasibility(values)
        nominal_metrics, nominal_audit = simulate_batch(
            nominal_values[None, :],
            capacity_ah,
            calibration_backend,
            rate_backends,
            physics,
        )
        if not nominal_audit[0]["solver_success"]:
            raise RuntimeError(f"Nominal DFN review failed for {parameter_set}")
        nominal_metric = nominal_metrics[0]

        for start in range(0, candidate_count, batch_size):
            if all(
                accepted_by_set_mode[parameter_set][mode] >= requested_per_mode
                for mode in ("regular", "extreme")
            ):
                break
            candidate_indices = [
                index
                for index in range(start, min(start + batch_size, candidate_count))
                if (parameter_set, index) not in processed
            ]
            if not candidate_indices:
                continue
            physics_indices = [index for index in candidate_indices if feasible[index]]
            metrics_by_index: dict[int, dict[str, Any]] = {}
            solver_by_index: dict[int, dict[str, Any]] = {}
            if physics_indices:
                batch_metrics, batch_audits = simulate_batch(
                    values[np.asarray(physics_indices)],
                    capacity_ah,
                    calibration_backend,
                    rate_backends,
                    physics,
                )
                metrics_by_index = dict(zip(physics_indices, batch_metrics, strict=True))
                solver_by_index = dict(zip(physics_indices, batch_audits, strict=True))

            accepted_rows, audit_rows = [], []
            batch_selected = {"regular": 0, "extreme": 0}
            for candidate_index in candidate_indices:
                mode = str(modes[candidate_index])
                case_id = f"{parameter_set}-{mode}-{seed}-{candidate_index:07d}"
                solver = solver_by_index.get(candidate_index)
                success = bool(feasible[candidate_index] and solver and solver["solver_success"])
                informative, review_metrics = (
                    review_difference(
                        metrics_by_index[candidate_index], nominal_metric, review
                    )
                    if success
                    else (False, {})
                )
                selected = (
                    success
                    and informative
                    and accepted_by_set_mode[parameter_set][mode]
                    + batch_selected[mode]
                    < requested_per_mode
                )
                if selected:
                    reason = "accepted"
                elif success and not informative:
                    reason = "indistinguishable_from_nominal"
                elif success:
                    reason = "successful_but_quota_full"
                elif feasible[candidate_index]:
                    reason = "solver_or_cutoff_failure"
                else:
                    reason = "infeasible_volume_fractions"
                audit_rows.append(
                    {
                        "case_id": case_id,
                        "parameter_set": parameter_set,
                        "mode": mode,
                        "candidate_index": candidate_index,
                        "accepted": selected,
                        "solver_success": success,
                        "reason": reason,
                        "review": review_metrics,
                        "solver": solver,
                    }
                )
                if selected:
                    accepted_rows.append(
                        {
                            "schema": "gradcell.multiset_dfn_physics.v3",
                            "case_id": case_id,
                            "physical_design_id": hashlib.sha256(
                                case_id.encode("utf-8")
                            ).hexdigest()[:20],
                            "model": "DFN",
                            "parameter_set": parameter_set,
                            "mode": mode,
                            "candidate_index": candidate_index,
                            "parameter_names": list(PARAMETER_FIELDS),
                            "nominal_parameter_values": nominal_values.tolist(),
                            "parameter_multipliers": multipliers[candidate_index].tolist(),
                            "parameter_values": values[candidate_index].tolist(),
                            "performance": metrics_by_index[candidate_index],
                            "review": review_metrics,
                            "simulation": {
                                "c_rates": list(rates),
                                "calibration_rate": float(physics["calibration_rate"]),
                                "calibration_iterations": int(
                                    physics["calibration_iterations"]
                                ),
                                "time_points": int(physics["time_points"]),
                                "rtol": float(physics["rtol"]),
                                "atol": float(physics["atol"]),
                                "temperature_k": temperature_k,
                            },
                        }
                    )
                    batch_selected[mode] += 1
            append_jsonl(audit_path, audit_rows)
            append_jsonl(output, accepted_rows)
            for mode in ("regular", "extreme"):
                accepted_by_set_mode[parameter_set][mode] += batch_selected[mode]
            processed.update((parameter_set, index) for index in candidate_indices)
            print(
                json.dumps(
                    {
                        "parameter_set": parameter_set,
                        "accepted": accepted_by_set_mode[parameter_set],
                        "requested_per_mode": requested_per_mode,
                        "processed_candidates": min(start + batch_size, candidate_count),
                    }
                ),
                flush=True,
            )
        incomplete = {
            mode: count
            for mode, count in accepted_by_set_mode[parameter_set].items()
            if count < requested_per_mode
        }
        if incomplete:
            raise RuntimeError(
                f"{parameter_set}: incomplete modes {incomplete}; increase "
                "physics.candidate_factor or revise bounds/review thresholds"
            )

    manifest = {
        "schema": "gradcell.multiset_dfn_physics_manifest.v3",
        "config": str(args.config),
        "output": str(output),
        "audit": str(audit_path),
        "parameter_sets": parameter_sets,
        "accepted_by_parameter_set_and_mode": accepted_by_set_mode,
        "parameter_fields": list(PARAMETER_FIELDS),
        "physics": physics,
        "perturbation": perturbation,
        "review": review,
    }
    atomic_json(Path(physics["manifest"]), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
