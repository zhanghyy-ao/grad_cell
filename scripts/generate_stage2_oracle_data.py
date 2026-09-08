from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.language import GradCellLanguageCodec, StructuredPreferenceTask


def split_name(anchor: int, seed: int) -> str:
    bucket = (anchor * 2654435761 + seed) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-conditioned Top-K Stage-2 oracles.")
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--energy-margin-max", type=float, default=5.0)
    parser.add_argument("--retention-margin-min", type=float, default=0.002)
    parser.add_argument("--retention-margin-max", type=float, default=0.04)
    parser.add_argument("--capacity-formula", default=None)
    parser.add_argument("--capacity-multiplier", type=float, default=None)
    args = parser.parse_args()
    if args.tasks < 3 or args.top_k < 1:
        parser.error("--tasks must be at least 3 and --top-k must be positive")

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError("oracle task sampling requires scipy") from exc
    with np.load(args.front, allow_pickle=False) as arrays:
        latent = arrays["latent"].copy()
        energy = arrays["energy_wh_kg"].copy()
        retention_5c = arrays["energy_retention_5c"].copy()
        retention_6c = arrays["energy_retention_6c"].copy()
        front_metadata = json.loads(str(arrays["metadata"]))
    if len(latent) < args.top_k:
        raise ValueError("front contains fewer candidates than --top-k")

    sampler = qmc.Sobol(d=5, scramble=True, seed=args.seed)
    unit = sampler.random(args.tasks)
    anchors = np.minimum((unit[:, 0] * len(latent)).astype(int), len(latent) - 1)
    preferences = unit[:, 1]
    target_energy = np.maximum(energy[anchors] - args.energy_margin_max * unit[:, 2], 1e-6)
    margin_span = args.retention_margin_max - args.retention_margin_min
    target_r5 = np.clip(retention_5c[anchors] - (
        args.retention_margin_min + margin_span * unit[:, 3]
    ), 0.0, 1.0)
    target_r6 = np.clip(retention_6c[anchors] - (
        args.retention_margin_min + margin_span * unit[:, 4]
    ), 0.0, 1.0)
    energy_scale = max(float(energy.max() - energy.min()), 1e-12)
    r5_scale = max(float(retention_5c.max() - retention_5c.min()), 1e-12)
    r6_scale = max(float(retention_6c.max() - retention_6c.min()), 1e-12)
    codec = GradCellLanguageCodec(bins=args.bins)
    capacity_formula = args.capacity_formula or front_metadata.get(
        "capacity_formula", "chen2020_scaled"
    )
    capacity_multiplier = args.capacity_multiplier
    if capacity_multiplier is None:
        capacity_multiplier = float(front_metadata.get("capacity_multiplier", 1.0))
    decoder = DesignSpace(
        capacity_formula=capacity_formula,
        capacity_multiplier=capacity_multiplier,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    counts = {name: 0 for name in outputs}
    try:
        for index in range(args.tasks):
            violation = (
                np.maximum(target_energy[index] - energy, 0.0) / energy_scale
                + np.maximum(target_r5[index] - retention_5c, 0.0) / r5_scale
                + np.maximum(target_r6[index] - retention_6c, 0.0) / r6_scale
            )
            reward = (
                preferences[index] * (energy - energy.min()) / energy_scale
                + (1.0 - preferences[index])
                * np.minimum(
                    (retention_5c - retention_5c.min()) / r5_scale,
                    (retention_6c - retention_6c.min()) / r6_scale,
                )
            )
            rank = np.argsort(100.0 * violation - reward)[: args.top_k]
            task = StructuredPreferenceTask(
                float(preferences[index]),
                target_energy=float(target_energy[index]),
                min_retention_5c=float(target_r5[index]),
                min_retention_6c=float(target_r6[index]),
            )
            oracle_designs = []
            for candidate in rank:
                candidate_latent = torch.from_numpy(latent[candidate]).double()
                levels = codec.quantize(candidate_latent)
                quantized = codec.dequantize(levels, dtype=torch.float64)
                design = decoder(quantized.unsqueeze(0))
                oracle_designs.append(
                    {
                        "source_front_index": int(candidate),
                        "continuous_latent": latent[candidate].tolist(),
                        "teacher_levels": levels.tolist(),
                        "quantized_latent": quantized.tolist(),
                        "teacher_design_text": codec.serialize_design(quantized),
                        "design": {
                            "positive_electrode_porosity": float(design.eps_p[0]),
                            "negative_electrode_porosity": float(design.eps_n[0]),
                            "separator_porosity": float(design.eps_s[0]),
                            "positive_active_material_fraction": float(design.phi_p[0]),
                            "negative_to_positive_capacity_ratio": float(design.np_ratio[0]),
                        },
                        "source_performance": {
                            "specific_energy_1c_wh_kg": float(energy[candidate]),
                            "energy_retention_5c": float(retention_5c[candidate]),
                            "energy_retention_6c": float(retention_6c[candidate]),
                        },
                    }
                )
            primary = oracle_designs[0]
            record = {
                "id": f"stage2_oracle_{index:06d}",
                "split": split_name(int(anchors[index]), args.seed),
                "anchor_front_index": int(anchors[index]),
                "preference": task.preference,
                "targets": [task.target_energy, task.min_retention_5c, task.min_retention_6c],
                "task_text": codec.serialize_task(task),
                "teacher_latent": primary["quantized_latent"],
                "teacher_levels": primary["teacher_levels"],
                "teacher_design_text": primary["teacher_design_text"],
                "oracle_designs": oracle_designs,
            }
            destination = record["split"]
            outputs[destination].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[destination] += 1
    finally:
        for handle in outputs.values():
            handle.close()

    metadata = {
        "source_front": str(args.front),
        "source_front_metadata": front_metadata,
        "tasks": args.tasks,
        "top_k": args.top_k,
        "bins": args.bins,
        "latent_limit": codec.latent_limit,
        "seed": args.seed,
        "capacity_formula": capacity_formula,
        "capacity_multiplier": capacity_multiplier,
        "split_counts": counts,
        "warning": (
            "source_performance belongs to continuous archive points; quantized latents must "
            "be strictly re-evaluated before this dataset is marked physics-verified"
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
