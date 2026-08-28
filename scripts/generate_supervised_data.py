from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gradcell.design import DesignSpace
from gradcell.physics import AnalyticToyBackend, PyBaMMBackend
from gradcell.physics.soft_metrics import discharge_metrics

DESIGN_FIELDS = (
    "eps_p", "eps_n", "eps_s", "phi_p", "phi_n", "np_ratio",
    "diffusivity_p_multiplier", "diffusivity_n_multiplier",
    "nominal_capacity_ah", "stack_mass_kg",
)
TARGET_FIELDS = (
    "specific_energy_1c_wh_kg", "specific_power_3c_w_kg",
    "minimum_voltage_1c_v", "minimum_voltage_3c_v",
)


def make_backend(name: str, horizon_s: float, time_points: int):
    if name == "toy":
        return AnalyticToyBackend(time_points=time_points, horizon_s=horizon_s)
    return PyBaMMBackend(
        time_points=time_points,
        horizon_s=horizon_s,
        calculate_sensitivities=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random hard-feasible Chen2020 designs and SPMe labels."
    )
    parser.add_argument("--backend", choices=("pybamm", "toy"), default="pybamm")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-std", type=float, default=1.25)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/chen2020_supervised.npz"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    decoder = DesignSpace()
    backend_1c = make_backend(args.backend, 3600.0, args.time_points)
    backend_3c = make_backend(args.backend, 1200.0, args.time_points)
    latent = args.latent_std * torch.randn(args.samples, decoder.latent_dim)
    saved_latent, saved_design, saved_targets = [], [], []
    failed_indices: list[int] = []
    started = time.perf_counter()

    for start in range(0, args.samples, args.batch_size):
        stop = min(start + args.batch_size, args.samples)
        latent_batch = latent[start:stop]
        design = decoder(latent_batch)
        result_1c = backend_1c.solve_batch(design.physics_tensor(1.0).detach().numpy())
        result_3c = backend_3c.solve_batch(design.physics_tensor(3.0).detach().numpy())
        ok = (result_1c.status == 1) & (result_3c.status == 1)
        ok &= np.isfinite(result_1c.trajectories).all(axis=(1, 2))
        ok &= np.isfinite(result_3c.trajectories).all(axis=(1, 2))

        if bool(ok.any()):
            mask = torch.from_numpy(ok)
            capacity = design.nominal_capacity_ah[mask]
            mass = design.stack_mass_kg[mask]
            metrics_1c = discharge_metrics(
                torch.from_numpy(result_1c.trajectories[ok]), capacity, mass, 3600.0
            )
            metrics_3c = discharge_metrics(
                torch.from_numpy(result_3c.trajectories[ok]), 3.0 * capacity, mass, 1200.0
            )
            saved_latent.append(latent_batch[mask].numpy())
            saved_design.append(
                torch.stack([getattr(design, field)[mask] for field in DESIGN_FIELDS], dim=-1)
                .detach().numpy()
            )
            saved_targets.append(
                torch.stack(
                    [
                        metrics_1c.specific_energy_wh_kg,
                        metrics_3c.specific_power_w_kg,
                        metrics_1c.minimum_voltage_v,
                        metrics_3c.minimum_voltage_v,
                    ], dim=-1,
                ).detach().numpy()
            )
        failed_indices.extend((start + np.flatnonzero(~ok)).tolist())
        valid_count = sum(values.shape[0] for values in saved_latent)
        print(f"generated {stop}/{args.samples}; valid={valid_count}")

    if not saved_latent:
        raise RuntimeError("No successful simulations were generated")
    metadata = {
        "backend": args.backend,
        "parameter_set": "Chen2020",
        "model": "SPMe" if args.backend == "pybamm" else "analytic-toy",
        "seed": args.seed,
        "requested_samples": args.samples,
        "valid_samples": sum(values.shape[0] for values in saved_latent),
        "failed_indices": failed_indices,
        "latent_std": args.latent_std,
        "time_points": args.time_points,
        "design_fields": DESIGN_FIELDS,
        "target_fields": TARGET_FIELDS,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=np.concatenate(saved_latent),
        design=np.concatenate(saved_design),
        targets=np.concatenate(saved_targets),
        metadata=np.asarray(json.dumps(metadata)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved {metadata['valid_samples']} samples to {args.output}")


if __name__ == "__main__":
    main()
