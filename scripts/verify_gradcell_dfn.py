from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.evaluation import hard_cutoff_metrics


def evenly_spaced_indices(count: int, maximum: int) -> np.ndarray:
    if maximum >= count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64))


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate selected SPMe GradCell candidates with hard-cutoff DFN."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=11)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")

    with np.load(args.candidates, allow_pickle=False) as arrays:
        preferences = arrays["preferences"].copy()
        latent = arrays["latent"].copy()
        spme_status = arrays["status"].copy()
        spme_energy = arrays["energy_wh_kg"].copy()
        spme_power = arrays["power_w_kg"].copy()
    indices = evenly_spaced_indices(len(preferences), args.max_candidates)
    dfn = hard_cutoff_metrics(
        torch.from_numpy(latent[indices]).double(),
        "DFN",
        args.capacity_formula,
        args.time_points,
        args.calibration_rate,
        args.calibration_iterations,
    )
    valid = (spme_status[indices] == 1) & (dfn["status"] == 1)
    energy_error = relative_error(dfn["energy_wh_kg"], spme_energy[indices])
    power_error = relative_error(dfn["power_w_kg"], spme_power[indices])
    report = {
        "source_candidates": str(args.candidates),
        "selected_candidates": len(indices),
        "joint_success_rate": float(valid.mean()),
        "mean_relative_energy_error": float(energy_error[valid].mean()) if valid.any() else None,
        "median_relative_energy_error": float(np.median(energy_error[valid]))
        if valid.any()
        else None,
        "mean_relative_power_error": float(power_error[valid].mean()) if valid.any() else None,
        "median_relative_power_error": float(np.median(power_error[valid]))
        if valid.any()
        else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dfn_verification.npz",
        source_indices=indices,
        preferences=preferences[indices],
        latent=latent[indices],
        spme_status=spme_status[indices],
        dfn_status=dfn["status"],
        spme_energy_wh_kg=spme_energy[indices],
        dfn_energy_wh_kg=dfn["energy_wh_kg"],
        spme_power_w_kg=spme_power[indices],
        dfn_power_w_kg=dfn["power_w_kg"],
        relative_energy_error=energy_error,
        relative_power_error=power_error,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
