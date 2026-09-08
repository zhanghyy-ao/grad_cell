from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_objectives(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        return np.column_stack(
            [
                arrays["energy_wh_kg"],
                arrays["energy_retention_5c"],
                arrays["energy_retention_6c"],
            ]
        )


def dominated_area_2d(points: np.ndarray) -> float:
    area = 0.0
    levels = np.unique(points[:, 0])
    levels = np.sort(levels[levels > 0.0])[::-1]
    for index, current in enumerate(levels):
        following = levels[index + 1] if index + 1 < len(levels) else 0.0
        height = float(points[points[:, 0] >= current, 1].max(initial=0.0))
        area += float(current - following) * height
    return area


def hypervolume_3d(points: np.ndarray) -> float:
    volume = 0.0
    levels = np.unique(points[:, 0])
    levels = np.sort(levels[levels > 0.0])[::-1]
    for index, current in enumerate(levels):
        following = levels[index + 1] if index + 1 < len(levels) else 0.0
        active = points[points[:, 0] >= current, 1:]
        volume += float(current - following) * dominated_area_2d(active)
    return volume


def directed_distances(source: np.ndarray, target: np.ndarray, chunk: int = 512) -> np.ndarray:
    nearest = []
    for start in range(0, len(source), chunk):
        values = source[start : start + chunk]
        distances = np.linalg.norm(values[:, None, :] - target[None, :, :], axis=-1)
        nearest.append(distances.min(axis=1))
    return np.concatenate(nearest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare convergence of 3D Pareto fronts.")
    parser.add_argument("--fronts", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fronts) < 2:
        parser.error("at least two --fronts are required")
    if args.labels is not None and len(args.labels) != len(args.fronts):
        parser.error("--labels must have the same length as --fronts")

    labels = args.labels or [path.stem for path in args.fronts]
    raw = [load_objectives(path) for path in args.fronts]
    union = np.concatenate(raw)
    lower, upper = union.min(axis=0), union.max(axis=0)
    scale = np.maximum(upper - lower, 1e-12)
    normalized = [np.clip((values - lower) / scale, 0.0, 1.0) for values in raw]
    hypervolumes = [hypervolume_3d(values) for values in normalized]
    comparisons = []
    for index in range(len(normalized) - 1):
        left, right = normalized[index], normalized[index + 1]
        left_to_right = directed_distances(left, right)
        right_to_left = directed_distances(right, left)
        comparisons.append(
            {
                "from": labels[index],
                "to": labels[index + 1],
                "relative_hypervolume_change": abs(
                    hypervolumes[index + 1] - hypervolumes[index]
                )
                / max(hypervolumes[index + 1], 1e-12),
                "generational_distance": float(left_to_right.mean()),
                "inverted_generational_distance": float(right_to_left.mean()),
                "hausdorff_distance": float(
                    max(left_to_right.max(), right_to_left.max())
                ),
                "objective_extreme_change": np.abs(
                    right.max(axis=0) - left.max(axis=0)
                ).tolist(),
            }
        )
    report = {
        "normalization_lower": lower.tolist(),
        "normalization_upper": upper.tolist(),
        "fronts": [
            {"label": label, "path": str(path), "points": len(values), "hypervolume": hv}
            for label, path, values, hv in zip(labels, args.fronts, raw, hypervolumes)
        ],
        "comparisons": comparisons,
        "recommended_hv_threshold": 0.01,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
