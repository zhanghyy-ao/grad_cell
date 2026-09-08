from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace


OBJECTIVE_NAMES = ("energy_wh_kg", "energy_retention_5c", "energy_retention_6c")


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    """Return an empirical maximization Pareto mask for three objectives."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("values must have shape (N, 3)")
    order = np.lexsort((-values[:, 2], -values[:, 1], -values[:, 0]))
    archive: list[int] = []
    for index in order:
        if archive:
            incumbent = values[np.asarray(archive)]
            if np.any(np.all(incumbent >= values[index], axis=1)):
                continue
            keep = ~np.all(values[index] >= incumbent, axis=1)
            archive = list(np.asarray(archive)[keep])
        archive.append(int(index))
    mask = np.zeros(len(values), dtype=bool)
    mask[archive] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a three-objective SPMe Pareto archive.")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-retention-5c", type=float, default=0.0)
    parser.add_argument("--min-retention-6c", type=float, default=0.0)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    latent_parts, objective_parts, source_parts = [], [], []
    provenance = []
    for source_id, path in enumerate(args.data):
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            if all(name in arrays.files for name in OBJECTIVE_NAMES):
                objectives = np.column_stack([arrays[name] for name in OBJECTIVE_NAMES])
            else:
                fields = list(metadata["target_fields"])
                required = (
                    "delivered_energy_1c_wh",
                    "delivered_energy_5c_wh",
                    "delivered_energy_6c_wh",
                    "specific_energy_1c_wh_kg",
                )
                if any(name not in fields for name in required):
                    raise ValueError(f"{path} lacks required 1C/5C/6C energy fields")
                targets = arrays["targets"]
                energy_1c = targets[:, fields.index("delivered_energy_1c_wh")]
                objectives = np.column_stack(
                    [
                        targets[:, fields.index("specific_energy_1c_wh_kg")],
                        targets[:, fields.index("delivered_energy_5c_wh")]
                        / np.maximum(energy_1c, 1e-12),
                        targets[:, fields.index("delivered_energy_6c_wh")]
                        / np.maximum(energy_1c, 1e-12),
                    ]
                )
            latent_parts.append(arrays["latent"].copy())
            objective_parts.append(objectives)
            source_parts.append(np.full(len(objectives), source_id, dtype=np.int64))
            provenance.append({"path": str(path), "metadata": metadata})

    latent = np.concatenate(latent_parts)
    decoded = DesignSpace(
        capacity_formula=args.capacity_formula,
        capacity_multiplier=args.capacity_multiplier,
    )(torch.from_numpy(latent).double())
    design = torch.stack(
        [decoded.eps_p, decoded.eps_n, decoded.eps_s, decoded.phi_p, decoded.np_ratio],
        dim=-1,
    ).numpy()
    objectives = np.concatenate(objective_parts)
    source_id = np.concatenate(source_parts)
    finite = np.isfinite(objectives).all(axis=1)
    feasible = (
        finite
        & (objectives[:, 1] >= args.min_retention_5c)
        & (objectives[:, 2] >= args.min_retention_6c)
    )
    if not feasible.any():
        raise RuntimeError("no finite candidates satisfy the requested retention thresholds")
    feasible_indices = np.flatnonzero(feasible)
    mask = nondominated_mask(objectives[feasible])
    indices = feasible_indices[mask]
    order = np.argsort(objectives[indices, 0])
    indices = indices[order]
    metadata = {
        "front_kind": "empirical_three_objective_maximization",
        "objective_names": list(OBJECTIVE_NAMES),
        "source_archives": provenance,
        "candidate_count": int(len(objectives)),
        "finite_count": int(finite.sum()),
        "feasible_count": int(feasible.sum()),
        "pareto_count": int(len(indices)),
        "minimum_retention_5c": args.min_retention_5c,
        "minimum_retention_6c": args.min_retention_6c,
        "capacity_formula": args.capacity_formula,
        "capacity_multiplier": args.capacity_multiplier,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=latent[indices],
        design=design[indices],
        energy_wh_kg=objectives[indices, 0],
        energy_retention_5c=objectives[indices, 1],
        energy_retention_6c=objectives[indices, 2],
        source_archive_id=source_id[indices],
        source_candidate_index=indices,
        metadata=np.asarray(json.dumps(metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
