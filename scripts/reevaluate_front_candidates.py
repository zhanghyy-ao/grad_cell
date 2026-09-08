from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.evaluation import hard_cutoff_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly re-evaluate a latent archive.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--time-points", type=int, default=301)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--capacity-formula", default="chen2020_scaled")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    with np.load(args.input, allow_pickle=False) as arrays:
        latent = arrays["latent"].copy()
        source_metadata = json.loads(str(arrays["metadata"])) if "metadata" in arrays else {}
    metric_parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(latent), args.batch_size):
        metrics = hard_cutoff_metrics(
            torch.from_numpy(latent[start : start + args.batch_size]).double(),
            args.model,
            args.capacity_formula,
            args.time_points,
            args.calibration_rate,
            args.calibration_iterations,
            args.capacity_multiplier,
            rtol=args.rtol,
            atol=args.atol,
        )
        for name, values in metrics.items():
            metric_parts.setdefault(name, []).append(values)
        progress = {"processed": min(start + args.batch_size, len(latent)), "total": len(latent)}
        print(json.dumps(progress))
    metrics = {name: np.concatenate(parts) for name, parts in metric_parts.items()}
    valid = metrics["status"] == 1
    decoder = DesignSpace(
        capacity_formula=args.capacity_formula,
        capacity_multiplier=args.capacity_multiplier,
    )
    design = decoder(torch.from_numpy(latent).double())
    design_matrix = torch.stack(
        [design.eps_p, design.eps_n, design.eps_s, design.phi_p, design.np_ratio], dim=-1
    ).numpy()
    metadata = {
        "source": str(args.input),
        "source_metadata": source_metadata,
        "model": args.model,
        "parameter_set": "Chen2020",
        "time_points": args.time_points,
        "rtol": args.rtol,
        "atol": args.atol,
        "capacity_formula": args.capacity_formula,
        "capacity_multiplier": args.capacity_multiplier,
        "calibration_rate": args.calibration_rate,
        "calibration_iterations": args.calibration_iterations,
        "candidate_count": len(latent),
        "valid_count": int(valid.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=latent[valid],
        design=design_matrix[valid],
        energy_wh_kg=metrics["energy_wh_kg"][valid],
        energy_retention_5c=metrics["energy_retention_5c"][valid],
        energy_retention_6c=metrics["energy_retention_6c"][valid],
        status=metrics["status"][valid],
        metadata=np.asarray(json.dumps(metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
