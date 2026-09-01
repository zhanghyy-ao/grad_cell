from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from gradcell.data import (
    MOLE_FRACTION_COLUMNS,
    SOLVENT_COLUMNS,
    clean_calisol23_frame,
    group_split,
)
from gradcell.physics import (
    AnalyticElectrolyteBackend,
    PyBaMMElectrolyteDFNBackend,
)

CALISOL23_URL = "https://ndownloader.figshare.com/files/43151344"


def build_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    continuous = frame.loc[:, ["T", "c", *MOLE_FRACTION_COLUMNS]].astype(np.float64)
    categories = pd.get_dummies(
        frame.loc[:, ["salt_canonical", "c units"]].fillna("unknown"),
        prefix=("salt", "concentration_unit"),
        dtype=np.float64,
    )
    matrix = np.concatenate([continuous.to_numpy(), categories.to_numpy()], axis=1)
    names = [
        "temperature_k",
        "salt_concentration",
        *[f"mole_fraction_{name}" for name in SOLVENT_COLUMNS],
        *categories.columns,
    ]
    return matrix, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DOI-isolated PyBaMM DFN voltage targets for physics-only training."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/calisol23.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/electrolyte_v1.npz"))
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--physics-samples", type=int, default=32)
    parser.add_argument("--physics-validation-samples", type=int, default=8)
    parser.add_argument("--physics-test-samples", type=int, default=8)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--probe-horizon-s", type=float, default=600.0)
    parser.add_argument("--reference-conductivity-ms-cm", type=float, default=10.0)
    parser.add_argument("--current-a", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="DOI split seed; defaults to --seed for backward compatibility.",
    )
    args = parser.parse_args()
    if min(
        args.physics_samples,
        args.physics_validation_samples,
        args.physics_test_samples,
    ) < 0:
        parser.error("physics sample counts must be non-negative")

    if not args.csv.exists():
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CALiSol-23 to {args.csv}", flush=True)
        urlretrieve(CALISOL23_URL, args.csv)
    cleaning = clean_calisol23_frame(pd.read_csv(args.csv))
    frame = cleaning.model_v1
    features, feature_names = build_features(frame)
    target_log_k = np.log(frame["k"].to_numpy(dtype=np.float64))
    groups, group_names = pd.factorize(frame["doi"].astype(str), sort=True)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_indices, validation_indices, test_indices = group_split(groups, split_seed)

    backend_cls = (
        PyBaMMElectrolyteDFNBackend
        if args.physics_backend == "dfn"
        else AnalyticElectrolyteBackend
    )
    backend = backend_cls(
        time_points=args.time_points,
        probe_horizon_s=args.probe_horizon_s,
        **({"calculate_sensitivities": False} if args.physics_backend == "dfn" else {}),
    )
    target_voltage = np.zeros((len(frame), args.time_points), dtype=np.float64)
    physics_mask = np.zeros(len(frame), dtype=bool)
    physics_partition = np.full(len(frame), -1, dtype=np.int8)
    physics_rank = np.full(len(frame), -1, dtype=np.int64)
    reference_log_k = np.log(args.reference_conductivity_ms_cm)
    log_scale = target_log_k - reference_log_k
    stable = (log_scale >= np.log(0.5)) & (log_scale <= np.log(2.0))
    rng = np.random.default_rng(args.seed)
    requested_by_partition = {
        "train": args.physics_samples,
        "validation": args.physics_validation_samples,
        "test": args.physics_test_samples,
    }
    indices_by_partition = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    partition_codes = {"train": 0, "validation": 1, "test": 2}
    valid_by_partition = {name: 0 for name in requested_by_partition}
    for partition_name, requested_count in requested_by_partition.items():
        stable_pool = indices_by_partition[partition_name][
            stable[indices_by_partition[partition_name]]
        ].copy()
        rng.shuffle(stable_pool)
        requested_count = min(requested_count, len(stable_pool))
        for rank, index in enumerate(stable_pool[:requested_count]):
            physics_input = np.array([[log_scale[index], args.current_a]], dtype=np.float64)
            result = backend.solve_batch(physics_input)
            if result.status[0] == 1:
                target_voltage[index] = result.trajectories[0, 0]
                physics_mask[index] = True
                physics_partition[index] = partition_codes[partition_name]
                physics_rank[index] = rank
                valid_by_partition[partition_name] += 1
            else:
                print(
                    f"DFN target failed for {partition_name} row {index}: "
                    f"{backend.last_solve_diagnostics[0]}"
                )
    if physics_mask.sum() == 0:
        raise RuntimeError("No valid Chen2020 physics targets were generated")

    metadata = {
        "dataset": "CALiSol-23",
        "dataset_doi": "10.11583/DTU.24559960",
        "conductivity_unit": "mS/cm",
        "rows": len(frame),
        "feature_count": features.shape[1],
        "physics_backend": args.physics_backend,
        "physics_samples_requested": requested_by_partition,
        "physics_samples_valid": int(physics_mask.sum()),
        "physics_samples_valid_by_partition": valid_by_partition,
        "physics_sampling_partition": "nested subsets within DOI-grouped partitions",
        "physics_interpretation": (
            "Observed conductivity is used only offline to define the synthetic target "
            "log_conductivity_scale and generate a PyBaMM Chen2020 DFN voltage trajectory. "
            "Training consumes voltage targets only; conductivity remains an evaluation-only "
            "diagnostic. These are synthetic model-domain targets, not experimental cell voltage."
        ),
        "reference_conductivity_ms_cm": args.reference_conductivity_ms_cm,
        "current_a": args.current_a,
        "probe_horizon_s": args.probe_horizon_s,
        "time_points": args.time_points,
        "seed": args.seed,
        "split_seed": split_seed,
        "cleaning": cleaning.report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        target_log_conductivity=target_log_k,
        group_id=groups.astype(np.int64),
        group_names=np.asarray(group_names, dtype=str),
        physics_mask=physics_mask,
        physics_partition=physics_partition,
        physics_rank=physics_rank,
        target_voltage=target_voltage,
        current_a=np.full(len(frame), args.current_a, dtype=np.float64),
        feature_names=np.asarray(feature_names, dtype=str),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
