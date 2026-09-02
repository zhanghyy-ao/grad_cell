from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace


def pareto_mask(energy: np.ndarray, retention: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for two objectives that are both maximized."""
    order = np.lexsort((-retention, -energy))
    keep = np.zeros(len(energy), dtype=bool)
    best_retention = -np.inf
    for index in order:
        if retention[index] > best_retention:
            keep[index] = True
            best_retention = retention[index]
    return keep


def scalarized_loss(
    energy: np.ndarray,
    retention: np.ndarray,
    preference: float,
    bounds: dict[str, float],
    temperature: float = 0.05,
    augmented_weight: float = 0.05,
) -> np.ndarray:
    distances = np.stack(
        [
            (bounds["energy_ideal"] - energy)
            / max(bounds["energy_ideal"] - bounds["energy_nadir"], 1e-12),
            (bounds["retention_ideal"] - retention)
            / max(bounds["retention_ideal"] - bounds["retention_nadir"], 1e-12),
        ],
        axis=-1,
    )
    weighted = distances * np.asarray([preference, 1.0 - preference])
    maximum = weighted.max(axis=-1, keepdims=True)
    smooth_max = maximum[:, 0] + temperature * np.log(
        np.exp((weighted - maximum) / temperature).sum(axis=-1)
    )
    return smooth_max + augmented_weight * weighted.sum(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hard-cutoff energy-power reference Pareto front."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preference-points", type=int, default=21)
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
    capacity_1c_name = "delivered_capacity_1c_ah"
    capacity_3c_name = "delivered_capacity_3c_ah"
    if any(name not in fields for name in (energy_name, capacity_1c_name, capacity_3c_name)):
        raise ValueError("Dataset lacks energy or 1C/3C delivered-capacity targets")
    energy = targets[:, fields.index(energy_name)]
    capacity_1c = targets[:, fields.index(capacity_1c_name)]
    capacity_3c = targets[:, fields.index(capacity_3c_name)]
    retention = capacity_3c / np.maximum(capacity_1c, 1e-12)
    reference_capacity = design[:, design_fields.index("reference_capacity_ah")]
    capacity_formula = metadata.get("capacity_formula", "chen2020_scaled")
    with torch.no_grad():
        analytic_capacity = DesignSpace(capacity_formula=capacity_formula)(
            torch.from_numpy(latent).double()
        ).nominal_capacity_ah.numpy()
    capacity_ratios = reference_capacity / np.maximum(analytic_capacity, 1e-12)
    capacity_multiplier = float(np.median(capacity_ratios))
    finite = np.isfinite(energy) & np.isfinite(retention) & (capacity_1c > 0.0)
    if finite.sum() < 2:
        raise RuntimeError("Fewer than two finite energy-power samples")
    source_indices = np.flatnonzero(finite)
    mask = pareto_mask(energy[finite], retention[finite])
    front_indices = source_indices[mask]
    front_order = np.argsort(energy[front_indices])
    front_indices = front_indices[front_order]
    bounds = {
        "energy_ideal": float(energy[finite].max()),
        "energy_nadir": float(energy[finite].min()),
        "retention_ideal": float(retention[finite].max()),
        "retention_nadir": float(retention[finite].min()),
    }
    preferences = np.linspace(0.0, 1.0, args.preference_points)
    best_indices = []
    best_losses = []
    for preference in preferences:
        loss = scalarized_loss(energy[finite], retention[finite], float(preference), bounds)
        local_index = int(np.argmin(loss))
        best_indices.append(int(source_indices[local_index]))
        best_losses.append(float(loss[local_index]))

    output_metadata = {
        "source_data": str(args.data),
        "source_model": metadata.get("model"),
        "source_seed": metadata.get("seed"),
        "valid_samples": int(finite.sum()),
        "pareto_samples": len(front_indices),
        "bounds": bounds,
        "preference_points": args.preference_points,
        "primary_objective": "specific_energy_1c_wh_kg",
        "secondary_objective": "capacity_retention_3c",
        "capacity_retention_definition": "delivered_capacity_3c_ah / delivered_capacity_1c_ah",
        "objective_correlation": float(np.corrcoef(energy[finite], retention[finite])[0, 1]),
        "capacity_formula": capacity_formula,
        "capacity_multiplier": capacity_multiplier,
        "capacity_multiplier_relative_spread": float(
            np.std(capacity_ratios) / max(abs(capacity_multiplier), 1e-12)
        ),
        "selection": "maximize hard-cutoff 1C specific energy and 3C capacity retention",
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
        capacity_retention_3c=retention[front_indices],
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
