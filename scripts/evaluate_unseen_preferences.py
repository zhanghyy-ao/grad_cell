from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def midpoint_preferences(grid_points: int) -> np.ndarray:
    """Return midpoints between an evenly spaced endpoint-inclusive lambda grid."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    grid = np.linspace(0.0, 1.0, grid_points)
    return 0.5 * (grid[:-1] + grid[1:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GradCell checkpoint at off-grid lambda midpoints and report "
            "hard-cutoff generalization, diversity, and smoothness diagnostics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument(
        "--base-grid-points",
        type=int,
        default=21,
        help="Use the midpoints between this many standard evaluation-grid points.",
    )
    parser.add_argument("--round-decimals", type=int, default=5)
    args = parser.parse_args()
    if args.base_grid_points < 2:
        parser.error("--base-grid-points must be at least 2")

    root = Path(__file__).resolve().parents[1]
    preferences = midpoint_preferences(args.base_grid_points)
    command = [
        sys.executable,
        "scripts/evaluate_gradcell.py",
        "--checkpoint",
        str(args.checkpoint),
        "--reference-front",
        str(args.reference_front),
        "--evaluation-model",
        args.evaluation_model,
        "--refinement-steps",
        str(args.refinement_steps),
        "--output-dir",
        str(args.output_dir),
        "--preference-values",
        *(f"{value:.17g}" for value in preferences),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)

    with np.load(args.output_dir / "candidates.npz", allow_pickle=False) as arrays:
        saved_preferences = arrays["preferences"].copy()
        latent = arrays["latent"].copy()
        status = arrays["status"].copy()
        energy = arrays["energy_wh_kg"].copy()
        retention_5c = arrays["energy_retention_5c"].copy()
        retention_6c = arrays["energy_retention_6c"].copy()
        regret = arrays["scalarized_loss"] - arrays["oracle_scalarized_loss"]

    adjacent_distance = np.linalg.norm(np.diff(latent, axis=0), axis=1)
    median_adjacent = float(np.median(adjacent_distance))
    report = {
        "test_type": "off_grid_midpoints",
        "claim_scope": (
            "Exact values are off the standard evaluation grid. Because historical "
            "training lambdas were not recorded, this is an interpolation test, not "
            "a certified held-out-lambda experiment."
        ),
        "preferences": saved_preferences.tolist(),
        "samples": len(saved_preferences),
        "success_rate": float(np.mean(status == 1)),
        "unique_latent_designs": int(
            len(np.unique(np.round(latent, args.round_decimals), axis=0))
        ),
        "latent_endpoint_distance": float(np.linalg.norm(latent[-1] - latent[0])),
        "mean_adjacent_latent_distance": float(np.mean(adjacent_distance)),
        "median_adjacent_latent_distance": median_adjacent,
        "max_adjacent_latent_distance": float(np.max(adjacent_distance)),
        "max_to_median_adjacent_distance": float(
            np.max(adjacent_distance) / max(median_adjacent, 1e-12)
        ),
        "energy_range_wh_kg": [float(energy.min()), float(energy.max())],
        "retention_5c_range": [float(retention_5c.min()), float(retention_5c.max())],
        "retention_6c_range": [float(retention_6c.min()), float(retention_6c.max())],
        "mean_scalarized_regret": float(np.mean(regret)),
        "median_scalarized_regret": float(np.median(regret)),
        "maximum_scalarized_regret": float(np.max(regret)),
        "energy_monotonicity_violations": int(np.sum(np.diff(energy) < -1e-6)),
        "retention_6c_monotonicity_violations": int(
            np.sum(np.diff(retention_6c) > 1e-6)
        ),
    }
    output = args.output_dir / "generalization_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
