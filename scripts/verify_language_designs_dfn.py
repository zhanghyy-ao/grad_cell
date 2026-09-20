from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from generate_multiset_dfn_archive import (
    make_backend,
    nominal_capacity,
    simulate_batch,
)
from gradcell.benchmark.dfn_parameter import PARAMETER_FIELDS, structural_feasibility


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly replay language-generated single designs with PyBaMM DFN."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multiset_dfn_language.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.relative_tolerance <= 0.0:
        parser.error("--relative-tolerance must be positive")

    rows = read_jsonl(args.predictions)
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("Prediction file is empty")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    physics = config["physics"]
    rates = tuple(float(value) for value in physics["c_rates"])
    if rates != (1.0, 5.0, 6.0):
        raise ValueError("DFN replay requires physics.c_rates=[1.0, 5.0, 6.0]")
    by_parameter_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parameter_set[str(row["base_parameter_set"])].append(row)

    verified_rows = []
    for parameter_set, parameter_rows in by_parameter_set.items():
        calibration_backend = make_backend(
            parameter_set,
            float(physics["maximum_duration_factor"]) * 3600.0 / float(physics["calibration_rate"]),
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
        nominal_capacity_ah = nominal_capacity(calibration_backend)
        for start in range(0, len(parameter_rows), args.batch_size):
            batch = parameter_rows[start : start + args.batch_size]
            multipliers = np.asarray(
                [
                    [
                        float(row["predicted_parameter_multipliers"][name])
                        for name in PARAMETER_FIELDS
                    ]
                    for row in batch
                ],
                dtype=np.float64,
            )
            values = nominal_values[None, :] * multipliers
            feasible = structural_feasibility(values)
            feasible_indices = np.flatnonzero(feasible)
            metrics_by_index = {}
            audits_by_index = {}
            if len(feasible_indices):
                metrics, audits = simulate_batch(
                    values[feasible_indices],
                    nominal_capacity_ah,
                    calibration_backend,
                    rate_backends,
                    physics,
                )
                metrics_by_index = dict(zip(feasible_indices.tolist(), metrics, strict=True))
                audits_by_index = dict(zip(feasible_indices.tolist(), audits, strict=True))
            for index, row in enumerate(batch):
                audit = audits_by_index.get(index)
                success = bool(feasible[index] and audit and audit["solver_success"])
                dfn_performance = metrics_by_index.get(index)
                target = row["target_verified_performance"]
                relative_errors = (
                    {
                        name: abs(float(dfn_performance[name]) - float(value))
                        / max(abs(float(value)), 1e-12)
                        for name, value in target.items()
                    }
                    if success
                    else {}
                )
                verified_rows.append(
                    {
                        **row,
                        "dfn_replay": {
                            "structurally_feasible": bool(feasible[index]),
                            "solver_success": success,
                            "performance": dfn_performance,
                            "relative_errors": relative_errors,
                            "all_metrics_within_tolerance": bool(
                                success
                                and all(
                                    value <= args.relative_tolerance
                                    for value in relative_errors.values()
                                )
                            ),
                            "solver_audit": audit,
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        "parameter_set": parameter_set,
                        "completed": min(start + len(batch), len(parameter_rows)),
                        "total": len(parameter_rows),
                    }
                ),
                flush=True,
            )

    write_jsonl(args.output, verified_rows)
    successful = [row for row in verified_rows if row["dfn_replay"]["solver_success"]]
    fields = list(rows[0]["target_verified_performance"])
    per_field = {}
    for name in fields:
        errors = [row["dfn_replay"]["relative_errors"][name] for row in successful]
        per_field[name] = {
            "mean_absolute_percentage_error": float(np.mean(errors)) if errors else None,
            "median_absolute_percentage_error": float(np.median(errors)) if errors else None,
            "p90_absolute_percentage_error": float(np.quantile(errors, 0.9)) if errors else None,
        }
    report = {
        "schema": "gradcell.language_design_dfn_replay_report.v1",
        "predictions": str(args.predictions),
        "records": len(verified_rows),
        "structural_feasibility_rate": float(
            np.mean([row["dfn_replay"]["structurally_feasible"] for row in verified_rows])
        ),
        "dfn_success_rate": len(successful) / len(verified_rows),
        "all_metrics_within_tolerance_rate": float(
            np.mean([row["dfn_replay"]["all_metrics_within_tolerance"] for row in verified_rows])
        ),
        "relative_tolerance": args.relative_tolerance,
        "per_field": per_field,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
