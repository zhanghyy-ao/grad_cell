from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace


def pareto_mask(energy: np.ndarray, high_rate: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for two objectives that are both maximized."""
    order = np.lexsort((-high_rate, -energy))
    keep = np.zeros(len(energy), dtype=bool)
    best_retention = -np.inf
    for index in order:
        if high_rate[index] > best_retention:
            keep[index] = True
            best_retention = high_rate[index]
    return keep


def scalarized_loss(
    energy: np.ndarray,
    retention_5c: np.ndarray,
    retention_6c: np.ndarray,
    preference: float,
    bounds: dict[str, float],
    temperature: float = 0.05,
    augmented_weight: float = 0.05,
) -> np.ndarray:
    distances = np.stack(
        [
            (bounds["energy_ideal"] - energy)
            / max(bounds["energy_ideal"] - bounds["energy_nadir"], 1e-12),
            (bounds["high_rate_ideal"] - np.minimum(retention_5c, retention_6c))
            / max(bounds["high_rate_ideal"] - bounds["high_rate_nadir"], 1e-12),
        ],
        axis=-1,
    )
    weighted = distances * np.asarray([preference, 1.0 - preference])
    maximum = weighted.max(axis=-1, keepdims=True)
    smooth_max = maximum[:, 0] + temperature * np.log(
        np.exp((weighted - maximum) / temperature).sum(axis=-1)
    )
    constraint = (
        np.maximum(bounds["retention_5c_min"] - retention_5c, 0.0) ** 2
        + np.maximum(bounds["retention_6c_min"] - retention_6c, 0.0) ** 2
    )
    return smooth_max + augmented_weight * weighted.sum(axis=-1) + bounds["constraint_weight"] * constraint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the feasible 1C-energy versus 5C/6C-energy-retention Pareto front."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preference-points", type=int, default=21)
    parser.add_argument("--retention-5c-min", type=float, default=0.55)
    parser.add_argument("--retention-6c-min", type=float, default=0.45)
    parser.add_argument("--constraint-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.preference_points < 2:
        parser.error("--preference-points must be at least 2")

    with np.load(args.data, allow_pickle=False) as arrays:
        latent = arrays["latent"].copy()
        design = arrays["design"].copy()
        targets = arrays["targets"].copy()
        metadata = json.loads(str(arrays["metadata"]))
    fields = list(metadata["target_fields"])
    design_fields = list(metadata["design_fields"])
    energy_name = "specific_energy_1c_wh_kg"
    energy_5c_name = "delivered_energy_5c_wh"
    energy_6c_name = "delivered_energy_6c_wh"
    energy_1c_name = "delivered_energy_1c_wh"
    required = (energy_name, energy_1c_name, energy_5c_name, energy_6c_name)
    if any(name not in fields for name in required):
        raise ValueError("Dataset lacks the required 1C/5C/6C delivered-energy targets; regenerate it")
    energy = targets[:, fields.index(energy_name)]
    energy_1c = targets[:, fields.index(energy_1c_name)]
    retention_5c = targets[:, fields.index(energy_5c_name)] / np.maximum(energy_1c, 1e-12)
    retention_6c = targets[:, fields.index(energy_6c_name)] / np.maximum(energy_1c, 1e-12)
    high_rate = np.minimum(retention_5c, retention_6c)
    reference_capacity = design[:, design_fields.index("reference_capacity_ah")]
    capacity_formula = metadata.get("capacity_formula", "chen2020_scaled")
    with torch.no_grad():
        analytic_capacity = DesignSpace(capacity_formula=capacity_formula)(
            torch.from_numpy(latent).double()
        ).nominal_capacity_ah.numpy()
    capacity_ratios = reference_capacity / np.maximum(analytic_capacity, 1e-12)
    capacity_multiplier = float(np.median(capacity_ratios))
    finite = np.isfinite(energy) & np.isfinite(high_rate) & (energy_1c > 0.0)
    feasible = finite & (retention_5c >= args.retention_5c_min) & (retention_6c >= args.retention_6c_min)
    if feasible.sum() < 2:
        raise RuntimeError(f"Only {int(feasible.sum())} samples satisfy the 5C/6C constraints; lower thresholds or generate more data")
    source_indices = np.flatnonzero(feasible)
    mask = pareto_mask(energy[feasible], high_rate[feasible])
    front_indices = source_indices[mask]
    front_order = np.argsort(energy[front_indices])
    front_indices = front_indices[front_order]
    bounds = {
        "energy_ideal": float(energy[feasible].max()),
        "energy_nadir": float(energy[feasible].min()),
        "high_rate_ideal": float(high_rate[feasible].max()),
        "high_rate_nadir": float(high_rate[feasible].min()),
        "retention_5c_min": args.retention_5c_min,
        "retention_6c_min": args.retention_6c_min,
        "constraint_weight": args.constraint_weight,
    }
    preferences = np.linspace(0.0, 1.0, args.preference_points)
    best_indices = []
    best_losses = []
    for preference in preferences:
        loss = scalarized_loss(energy[feasible], retention_5c[feasible], retention_6c[feasible], float(preference), bounds)
        local_index = int(np.argmin(loss))
        best_indices.append(int(source_indices[local_index]))
        best_losses.append(float(loss[local_index]))

    output_metadata = {
        "source_data": str(args.data),
        "source_model": metadata.get("model"),
        "source_seed": metadata.get("seed"),
        "valid_samples": int(finite.sum()),
        "feasible_samples": int(feasible.sum()),
        "pareto_samples": len(front_indices),
        "bounds": bounds,
        "preference_points": args.preference_points,
        "primary_objective": "specific_energy_1c_wh_kg",
        "secondary_objective": "worst_case_energy_retention_5c_6c",
        "energy_retention_definition": "min(delivered_energy_5c, delivered_energy_6c) / delivered_energy_1c",
        "objective_correlation": float(np.corrcoef(energy[feasible], high_rate[feasible])[0, 1]),
        "capacity_formula": capacity_formula,
        "capacity_multiplier": capacity_multiplier,
        "capacity_multiplier_relative_spread": float(
            np.std(capacity_ratios) / max(abs(capacity_multiplier), 1e-12)
        ),
        "selection": "maximize hard-cutoff 1C specific energy and worst-case 5C/6C energy retention under minimum-retention constraints",
    }
    if len(front_indices) < 3:
        raise RuntimeError(
            f"Reference front has only {len(front_indices)} non-dominated points; "
            "the objectives or design space do not provide a usable trade-off"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_indices=front_indices,
        latent=latent[front_indices],
        design=design[front_indices],
        energy_wh_kg=energy[front_indices],
        energy_retention_5c=retention_5c[front_indices],
        energy_retention_6c=retention_6c[front_indices],
        high_rate_retention=high_rate[front_indices],
        preferences=preferences,
        scalarized_best_source_indices=np.asarray(best_indices, dtype=np.int64),
        scalarized_best_losses=np.asarray(best_losses),
        metadata=np.asarray(json.dumps(output_metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(output_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(output_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
