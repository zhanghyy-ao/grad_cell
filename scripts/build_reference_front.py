from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pareto_mask(energy: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for two objectives that are both maximized."""
    order = np.lexsort((-power, -energy))
    keep = np.zeros(len(energy), dtype=bool)
    best_power = -np.inf
    for index in order:
        if power[index] > best_power:
            keep[index] = True
            best_power = power[index]
    return keep


def scalarized_loss(
    energy: np.ndarray,
    power: np.ndarray,
    preference: float,
    bounds: dict[str, float],
    temperature: float = 0.05,
    augmented_weight: float = 0.05,
) -> np.ndarray:
    distances = np.stack(
        [
            (bounds["energy_ideal"] - energy)
            / max(bounds["energy_ideal"] - bounds["energy_nadir"], 1e-12),
            (bounds["power_ideal"] - power)
            / max(bounds["power_ideal"] - bounds["power_nadir"], 1e-12),
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
    energy_name = "specific_energy_1c_wh_kg"
    power_name = "specific_power_3c_w_kg"
    if energy_name not in fields or power_name not in fields:
        raise ValueError("Dataset lacks hard-cutoff energy or power targets")
    energy = targets[:, fields.index(energy_name)]
    power = targets[:, fields.index(power_name)]
    finite = np.isfinite(energy) & np.isfinite(power)
    if finite.sum() < 2:
        raise RuntimeError("Fewer than two finite energy-power samples")
    source_indices = np.flatnonzero(finite)
    mask = pareto_mask(energy[finite], power[finite])
    front_indices = source_indices[mask]
    front_order = np.argsort(energy[front_indices])
    front_indices = front_indices[front_order]
    bounds = {
        "energy_ideal": float(energy[finite].max()),
        "energy_nadir": float(energy[finite].min()),
        "power_ideal": float(power[finite].max()),
        "power_nadir": float(power[finite].min()),
    }
    preferences = np.linspace(0.0, 1.0, args.preference_points)
    best_indices = []
    best_losses = []
    for preference in preferences:
        loss = scalarized_loss(energy[finite], power[finite], float(preference), bounds)
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
        "selection": "maximize hard-cutoff 1C specific energy and 3C specific power",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_indices=front_indices,
        latent=latent[front_indices],
        design=design[front_indices],
        energy_wh_kg=energy[front_indices],
        power_w_kg=power[front_indices],
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
