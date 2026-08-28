from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.experiments import ExperimentRun
from gradcell.physics import PyBaMMBackend

FORMULAS = ("electrode_theoretical", "chen2020_scaled")


def trajectory_summary(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "all_finite": bool(finite.all()),
        "nan_count": int(np.isnan(values).sum()),
        "minimum_finite_voltage_v": (
            float(finite_values.min()) if finite_values.size else None
        ),
        "maximum_finite_voltage_v": (
            float(finite_values.max()) if finite_values.size else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare capacity formulas with Chen2020 DFN termination behavior."
    )
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="DFN")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latent-scale", type=float, default=0.7)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--current-ramp-time-s", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/capacity_formula_comparison.json")
    )
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    with ExperimentRun("compare_capacity_formulas", args, run_dir=args.run_dir) as run:
        torch.set_default_dtype(torch.float64)
        generator = torch.Generator().manual_seed(args.seed)
        latent = args.latent_scale * torch.randn(args.samples, 7, generator=generator)
        run.log(f"building Chen2020 {args.model} backends")
        backend_1c = PyBaMMBackend(
            model_name=args.model,
            horizon_s=3600.0,
            time_points=args.time_points,
            calculate_sensitivities=False,
            current_ramp_time_s=args.current_ramp_time_s,
        )
        backend_3c = PyBaMMBackend(
            model_name=args.model,
            horizon_s=1200.0,
            time_points=args.time_points,
            calculate_sensitivities=False,
            current_ramp_time_s=args.current_ramp_time_s,
        )
        records = []
        for sample_index in range(args.samples):
            for formula in FORMULAS:
                design = DesignSpace(capacity_formula=formula)(
                    latent[sample_index : sample_index + 1]
                )
                result_1c = backend_1c.solve_batch(design.physics_tensor(1.0).numpy())
                details_1c = backend_1c.last_solve_diagnostics[0]
                result_3c = backend_3c.solve_batch(design.physics_tensor(3.0).numpy())
                details_3c = backend_3c.last_solve_diagnostics[0]
                record = {
                    "sample": sample_index,
                    "capacity_formula": formula,
                    "latent": latent[sample_index].tolist(),
                    "phi_p": float(design.phi_p[0]),
                    "phi_n": float(design.phi_n[0]),
                    "nominal_capacity_ah": float(design.nominal_capacity_ah[0]),
                    "current_1c_a": float(design.nominal_capacity_ah[0]),
                    "current_3c_a": float(3.0 * design.nominal_capacity_ah[0]),
                    "solve_1c": {
                        **details_1c,
                        **trajectory_summary(result_1c.trajectories[0, 0]),
                    },
                    "solve_3c": {
                        **details_3c,
                        **trajectory_summary(result_3c.trajectories[0, 0]),
                    },
                }
                records.append(record)
                run.event("formula_case_finished", **record)
                run.log(
                    f"sample={sample_index} formula={formula} "
                    f"capacity={record['nominal_capacity_ah']:.3f}Ah "
                    f"end_1c={details_1c['actual_end_time_s']}s "
                    f"end_3c={details_3c['actual_end_time_s']}s"
                )

        formula_summary = {}
        for formula in FORMULAS:
            selected = [row for row in records if row["capacity_formula"] == formula]
            complete_1c = sum(row["solve_1c"]["completed_requested_horizon"] for row in selected)
            complete_3c = sum(row["solve_3c"]["completed_requested_horizon"] for row in selected)
            formula_summary[formula] = {
                "samples": len(selected),
                "complete_1c": complete_1c,
                "complete_3c": complete_3c,
                "complete_1c_rate": complete_1c / len(selected),
                "complete_3c_rate": complete_3c / len(selected),
                "mean_capacity_ah": float(
                    np.mean([row["nominal_capacity_ah"] for row in selected])
                ),
            }
        report = {
            "model": args.model,
            "parameter_set": "Chen2020",
            "seed": args.seed,
            "summary": formula_summary,
            "cases": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        run.save_summary({"result": report, "artifacts": {"result": str(args.output)}})
        run.log(json.dumps(formula_summary))


if __name__ == "__main__":
    main()
