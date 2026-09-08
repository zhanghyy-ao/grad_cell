"""Strictly re-evaluate quantized Stage-2 oracle labels with PyBaMM.

The front is normally optimized in continuous latent space, while Stage-2 learns
discrete ``<LEVEL_*>`` tokens.  Quantization changes the physical design, so this
script evaluates the exact quantized latent vectors, removes failed/infeasible
labels, re-ranks Top-K candidates, and rewrites the canonical target JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradcell.evaluation import hard_cutoff_metrics
from gradcell.design import DesignSpace
from gradcell.language import GradCellLanguageCodec, StructuredPreferenceTask


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-points", type=int, default=301)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--capacity-formula",
        choices=("electrode_theoretical", "chen2020_scaled"),
        default="chen2020_scaled",
    )
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--require-feasible", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latent_key(values: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in values)


def requirement_score(task: StructuredPreferenceTask, metrics: dict[str, float]) -> float:
    energy = metrics["energy_wh_kg"]
    r5 = metrics["energy_retention_5c"]
    r6 = metrics["energy_retention_6c"]
    violation = (
        max(task.target_energy - energy, 0.0)
        / max(task.target_energy, 1.0)
        + max(task.min_retention_5c - r5, 0.0)
        + max(task.min_retention_6c - r6, 0.0)
    )
    reward = task.preference * energy / 200.0 + (1.0 - task.preference) * min(r5, r6)
    return 100.0 * violation - reward


def evaluate_unique_latents(
    latents: list[tuple[float, ...]], args: argparse.Namespace
) -> dict[tuple[float, ...], dict[str, float | bool]]:
    results: dict[tuple[float, ...], dict[str, float | bool]] = {}
    for start in range(0, len(latents), args.batch_size):
        batch_keys = latents[start : start + args.batch_size]
        batch = torch.from_numpy(np.asarray(batch_keys, dtype=np.float64)).double()
        hard = hard_cutoff_metrics(
            batch,
            model_name=args.model,
            time_points=args.time_points,
            capacity_formula=args.capacity_formula,
            capacity_multiplier=args.capacity_multiplier,
            rtol=args.rtol,
            atol=args.atol,
        )
        for index, key in enumerate(batch_keys):
            results[key] = {
                "energy_wh_kg": float(hard["energy_wh_kg"][index]),
                "energy_retention_5c": float(hard["energy_retention_5c"][index]),
                "energy_retention_6c": float(hard["energy_retention_6c"][index]),
                "solver_success": bool(hard["status"][index]),
            }
    return results


def main() -> None:
    args = parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "--output-dir must differ from --input-dir so unverified labels are preserved"
        )

    records_by_split = {
        split: read_jsonl(args.input_dir / f"{split}.jsonl") for split in SPLITS
    }
    keys = {
        latent_key(candidate["quantized_latent"])
        for records in records_by_split.values()
        for record in records
        for candidate in record["oracle_designs"]
    }
    evaluations = evaluate_unique_latents(sorted(keys), args)

    design_space = DesignSpace(
        capacity_formula=args.capacity_formula,
        capacity_multiplier=args.capacity_multiplier,
    )
    provenance = json.loads((args.input_dir / "provenance.json").read_text(encoding="utf-8"))
    codec = GradCellLanguageCodec(
        bins=int(provenance.get("bins", 256)),
        latent_limit=float(provenance.get("latent_limit", 4.0)),
    )
    retained = 0
    dropped = 0
    feasible_candidates = 0
    total_candidates = 0
    output_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    for split, records in records_by_split.items():
        for record in records:
            task = StructuredPreferenceTask(
                preference=float(record["preference"]),
                target_energy=float(record["targets"][0]),
                min_retention_5c=float(record["targets"][1]),
                min_retention_6c=float(record["targets"][2]),
            )
            verified: list[dict[str, Any]] = []
            for candidate in record["oracle_designs"]:
                total_candidates += 1
                metrics = evaluations[latent_key(candidate["quantized_latent"])]
                feasible = bool(metrics["solver_success"]) and (
                    float(metrics["energy_wh_kg"]) >= task.target_energy
                    and float(metrics["energy_retention_5c"]) >= task.min_retention_5c
                    and float(metrics["energy_retention_6c"]) >= task.min_retention_6c
                )
                feasible_candidates += int(feasible)
                enriched = dict(candidate)
                enriched["quantized_performance"] = metrics
                enriched["requirements_satisfied"] = feasible
                enriched["verified_score"] = (
                    requirement_score(task, metrics)
                    if metrics["solver_success"]
                    else float("inf")
                )
                if metrics["solver_success"] and (feasible or not args.require_feasible):
                    verified.append(enriched)

            verified.sort(key=lambda item: item["verified_score"])
            if not verified:
                dropped += 1
                continue

            primary = verified[0]
            quantized_latent = np.asarray(primary["quantized_latent"], dtype=np.float64)[None, :]
            decoded = design_space(torch.from_numpy(quantized_latent).double())
            record["oracle_designs"] = verified
            record["teacher_latent"] = primary["quantized_latent"]
            record["teacher_levels"] = primary["teacher_levels"]
            record["design"] = {
                "positive_electrode_porosity": float(decoded.eps_p[0]),
                "negative_electrode_porosity": float(decoded.eps_n[0]),
                "separator_porosity": float(decoded.eps_s[0]),
                "positive_active_material_fraction": float(decoded.phi_p[0]),
                "negative_to_positive_capacity_ratio": float(decoded.np_ratio[0]),
            }
            record["task_text"] = codec.serialize_task(task)
            record["teacher_design_text"] = codec.serialize_design(
                torch.from_numpy(quantized_latent[0]).double()
            )
            record["verification"] = {
                "backend": "pybamm",
                "model": args.model,
                "time_points": args.time_points,
                "rtol": args.rtol,
                "atol": args.atol,
                "quantized_latent_recomputed": True,
            }
            output_records[split].append(record)
            retained += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in output_records.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    summary = {
        "input_dir": str(args.input_dir),
        "unique_quantized_designs": len(keys),
        "records_retained": retained,
        "records_dropped": dropped,
        "candidate_feasible_rate": feasible_candidates / max(total_candidates, 1),
        "split_sizes": {split: len(records) for split, records in output_records.items()},
        "settings": vars(args)
        | {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir)},
    }
    with (args.output_dir / "verification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
