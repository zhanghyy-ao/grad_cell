from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.language import DirectPhysicsPerformanceLayer, SingleDesignPhysicsMLP
from train_battery_description_direct_dfn import decode_design, prepare_data, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 final acceptance: independent sampled DFN replay without gradients."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--minimum-dfn-success-rate", type=float, default=0.95)
    parser.add_argument("--minimum-all-metrics-within-tolerance-rate", type=float, default=0.20)
    parser.add_argument("--time-points", type=int, default=301)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=7007)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("max records must be positive")
    if not 0.0 <= args.relative_tolerance <= 1.0:
        parser.error("relative tolerance must be between zero and one")

    device = torch.device(args.device)
    tensors, metadata, ordered = prepare_data(args.data, args.embeddings)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset_sha256") != metadata["dataset_sha256"]:
        raise ValueError("Checkpoint and audit dataset content do not match")
    model = SingleDesignPhysicsMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    test_indices = np.flatnonzero(metadata["splits"] == "test")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(
        rng.choice(test_indices, size=min(args.max_records, len(test_indices)), replace=False)
    )
    physics = DirectPhysicsPerformanceLayer(
        parameter_set=metadata["parameter_sets"][0],
        model_name="DFN",
        time_points=args.time_points,
        rtol=args.rtol,
        atol=args.atol,
        cutoff_v=float(checkpoint.get("physics_config", {}).get("cutoff_v", 2.5)),
        gate_temperature_v=float(
            checkpoint.get("physics_config", {}).get("gate_temperature_v", 0.02)
        ),
        current_ramp_time_s=float(
            checkpoint.get("physics_config", {}).get("current_ramp_time_s", 1.0)
        ),
        training_voltage_floor_v=float(
            checkpoint.get("physics_config", {}).get("training_voltage_floor_v", 2.0)
        ),
        calculate_sensitivities=False,
    ).to(device)
    design_mean = metadata["design_log_mean"].to(device)
    design_std = metadata["design_log_std"].to(device)
    nominal = physics.nominal_parameter_values.to(device)
    field_names = metadata["performance_fields"]
    rows = []
    absolute_percentage_errors = []
    successes = 0
    with torch.no_grad():
        for count, row_index in enumerate(selected, start=1):
            embedding = tensors["embeddings"][row_index : row_index + 1].to(device)
            normalized_design = model(embedding)
            _, multipliers, values = decode_design(
                normalized_design, design_mean, design_std, nominal
            )
            reference_capacity = tensors["reference_capacity"][row_index : row_index + 1].to(device)
            simulation = physics(values, reference_capacity)
            target = torch.exp(
                tensors["performance"][row_index].to(device)
                * metadata["performance_log_std"].to(device)
                + metadata["performance_log_mean"].to(device)
            )
            predicted = simulation.performance[0]
            ape = (predicted - target).abs() / target.abs().clamp_min(1e-8)
            success = bool(simulation.status[0])
            successes += int(success)
            absolute_percentage_errors.append(ape.cpu().numpy())
            source = ordered[int(row_index)]
            rows.append(
                {
                    "schema": "gradcell.language_design_stage3_dfn_audit_record.v1",
                    "task_id": source["task_id"],
                    "physical_design_id": source["physical_design_id"],
                    "split": source["split"],
                    "solver_success": success,
                    "runtime_s": float(simulation.runtime_s[0]),
                    "predicted_parameter_multipliers": dict(
                        zip(metadata["parameter_names"], multipliers[0].cpu().tolist(), strict=True)
                    ),
                    "dfn_predicted_performance": dict(
                        zip(field_names, predicted.cpu().tolist(), strict=True)
                    ),
                    "target_dfn_performance": dict(
                        zip(field_names, target.cpu().tolist(), strict=True)
                    ),
                    "absolute_percentage_error": dict(
                        zip(field_names, ape.cpu().tolist(), strict=True)
                    ),
                    "all_metrics_within_tolerance": bool((ape <= args.relative_tolerance).all()),
                }
            )
            print(json.dumps({"stage3_dfn_audit": count, "total": len(selected)}), flush=True)

    error = np.stack(absolute_percentage_errors)
    success_rate = successes / len(rows)
    all_within = float(np.mean(np.all(error <= args.relative_tolerance, axis=1)))
    accepted = (
        success_rate >= args.minimum_dfn_success_rate
        and all_within >= args.minimum_all_metrics_within_tolerance_rate
    )
    report = {
        "schema": "gradcell.language_design_stage3_dfn_acceptance.v1",
        "checkpoint": str(args.checkpoint),
        "selection": "seeded sample from untouched test split",
        "records": len(rows),
        "dfn_calculate_sensitivities": False,
        "dfn_success_rate": success_rate,
        "relative_tolerance": args.relative_tolerance,
        "all_metrics_within_tolerance_rate": all_within,
        "per_field": {
            field: {
                "mape": float(error[:, position].mean()),
                "median_ape": float(np.median(error[:, position])),
                "p90_ape": float(np.quantile(error[:, position], 0.9)),
            }
            for position, field in enumerate(field_names)
        },
        "acceptance_thresholds": {
            "minimum_dfn_success_rate": args.minimum_dfn_success_rate,
            "minimum_all_metrics_within_tolerance_rate": args.minimum_all_metrics_within_tolerance_rate,
        },
        "accepted": accepted,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "dfn_audit_predictions.jsonl", rows)
    (args.output_dir / "dfn_acceptance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        raise RuntimeError("Stage 3 DFN acceptance thresholds were not met; see report")


if __name__ == "__main__":
    main()
