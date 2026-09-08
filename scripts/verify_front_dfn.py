from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.evaluation import hard_cutoff_metrics


def select_indices(objectives: np.ndarray, maximum: int) -> np.ndarray:
    selected = {int(objectives[:, column].argmax()) for column in range(3)}
    scale = np.maximum(objectives.max(axis=0) - objectives.min(axis=0), 1e-12)
    normalized = (objectives - objectives.min(axis=0)) / scale
    selected.add(int(np.linalg.norm(1.0 - normalized, axis=1).argmin()))
    order = np.argsort(objectives[:, 0])
    for position in np.linspace(0, len(order) - 1, maximum).round().astype(int):
        selected.add(int(order[position]))
        if len(selected) >= maximum:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify representative 3D SPMe-front points with DFN."
    )
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--time-points", type=int, default=301)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--retention-5c-min", type=float, default=0.50)
    parser.add_argument("--retention-6c-min", type=float, default=0.44)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        raise ImportError("DFN rank verification requires scipy") from exc
    with np.load(args.front, allow_pickle=False) as arrays:
        latent = arrays["latent"].copy()
        spme = np.column_stack(
            [
                arrays["energy_wh_kg"],
                arrays["energy_retention_5c"],
                arrays["energy_retention_6c"],
            ]
        )
        source_metadata = json.loads(str(arrays["metadata"]))
    indices = select_indices(spme, min(args.max_candidates, len(latent)))
    dfn_metrics = hard_cutoff_metrics(
        torch.from_numpy(latent[indices]).double(),
        "DFN",
        args.capacity_formula,
        args.time_points,
        capacity_multiplier=args.capacity_multiplier,
        rtol=args.rtol,
        atol=args.atol,
    )
    dfn = np.column_stack(
        [
            dfn_metrics["energy_wh_kg"],
            dfn_metrics["energy_retention_5c"],
            dfn_metrics["energy_retention_6c"],
        ]
    )
    valid = dfn_metrics["status"] == 1
    errors = relative_error(dfn, spme[indices])
    spme_feasible = (
        (spme[indices, 1] >= args.retention_5c_min)
        & (spme[indices, 2] >= args.retention_6c_min)
    )
    dfn_feasible = (
        valid
        & (dfn[:, 1] >= args.retention_5c_min)
        & (dfn[:, 2] >= args.retention_6c_min)
    )
    objective_names = ("energy", "retention_5c", "retention_6c")
    rank_correlation = {
        name: float(spearmanr(spme[indices, column][valid], dfn[:, column][valid]).statistic)
        if valid.sum() >= 3
        else None
        for column, name in enumerate(objective_names)
    }
    report = {
        "source_front": str(args.front),
        "selected_candidates": len(indices),
        "joint_success_rate": float(valid.mean()),
        "mean_relative_errors": {
            name: float(errors[valid, column].mean()) if valid.any() else None
            for column, name in enumerate(objective_names)
        },
        "median_relative_errors": {
            name: float(np.median(errors[valid, column])) if valid.any() else None
            for column, name in enumerate(objective_names)
        },
        "rank_correlation": rank_correlation,
        "constraint_agreement_rate": float((spme_feasible == dfn_feasible).mean()),
        "source_metadata": source_metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dfn_verification.npz",
        source_indices=indices,
        latent=latent[indices],
        spme_objectives=spme[indices],
        dfn_objectives=dfn,
        dfn_status=dfn_metrics["status"],
        relative_errors=errors,
        spme_feasible=spme_feasible,
        dfn_feasible=dfn_feasible,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
