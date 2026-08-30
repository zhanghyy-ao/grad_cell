from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from gradcell.physics import (
    AnalyticElectrolyteBackend,
    PyBaMMElectrolyteDFNBackend,
)

CALISOL23_URL = "https://ndownloader.figshare.com/files/43151344"
SOLVENT_COLUMNS = (
    "EC", "PC", "DMC", "EMC", "DEC", "DME", "DMSO", "AN", "MOEMC", "TFP",
    "EA", "MA", "FEC", "DOL", "2-MeTHF", "DMM", "Freon 11",
    "Methylene chloride", "THF", "Toluene", "Sulfolane", "2-Glyme", "3-Glyme",
    "4-Glyme", "3-Me-2-Oxazolidinone", "3-MeSulfolane", "Ethyldiglyme", "DMF",
    "Ethylbenzene", "Ethylmonoglyme", "Benzene", "g-Butyrolactone", "Cumene",
    "Propylsulfone", "Pseudocumeme", "TEOS", "m-Xylene", "o-Xylene",
)


def build_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    continuous = frame.loc[:, ["T", "c", *SOLVENT_COLUMNS]].astype(np.float64)
    categories = pd.get_dummies(
        frame.loc[:, ["salt", "c units", "solvent ratio type"]].fillna("unknown"),
        prefix=("salt", "concentration_unit", "ratio_type"),
        dtype=np.float64,
    )
    matrix = np.concatenate([continuous.to_numpy(), categories.to_numpy()], axis=1)
    names = ["temperature_k", "salt_concentration", *SOLVENT_COLUMNS, *categories.columns]
    return matrix, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CALiSol-23 plus a small Chen2020 DFN physics subset."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/calisol23.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/electrolyte_v1.npz"))
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--physics-samples", type=int, default=32)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--probe-horizon-s", type=float, default=600.0)
    parser.add_argument("--reference-conductivity-ms-cm", type=float, default=10.0)
    parser.add_argument("--current-a", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.csv.exists():
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CALiSol-23 to {args.csv}", flush=True)
        urlretrieve(CALISOL23_URL, args.csv)
    frame = pd.read_csv(args.csv, index_col=0)
    required = {"doi", "k", "T", "c", "salt", "c units", "solvent ratio type", *SOLVENT_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CALiSol-23 file is missing columns: {sorted(missing)}")
    finite = np.isfinite(frame["k"]) & np.isfinite(frame["T"]) & np.isfinite(frame["c"])
    frame = frame.loc[finite & (frame["k"] > 0.0)].reset_index(drop=True)
    features, feature_names = build_features(frame)
    target_log_k = np.log(frame["k"].to_numpy(dtype=np.float64))
    groups, group_names = pd.factorize(frame["doi"].astype(str), sort=True)

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
    reference_log_k = np.log(args.reference_conductivity_ms_cm)
    log_scale = target_log_k - reference_log_k
    stable_pool = np.flatnonzero((log_scale >= np.log(0.5)) & (log_scale <= np.log(2.0)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(stable_pool)
    requested = min(args.physics_samples, len(stable_pool))
    for index in stable_pool[:requested]:
        physics_input = np.array([[log_scale[index], args.current_a]], dtype=np.float64)
        result = backend.solve_batch(physics_input)
        if result.status[0] == 1:
            target_voltage[index] = result.trajectories[0, 0]
            physics_mask[index] = True
        else:
            print(f"DFN target failed for row {index}: {backend.last_solve_diagnostics[0]}")
    if physics_mask.sum() == 0:
        raise RuntimeError("No valid Chen2020 physics targets were generated")

    metadata = {
        "dataset": "CALiSol-23",
        "dataset_doi": "10.11583/DTU.24559960",
        "conductivity_unit": "mS/cm",
        "rows": len(frame),
        "feature_count": features.shape[1],
        "physics_backend": args.physics_backend,
        "physics_samples_requested": requested,
        "physics_samples_valid": int(physics_mask.sum()),
        "physics_interpretation": (
            "Observed conductivity is converted to a bounded multiplicative correction "
            "of the Chen2020 electrolyte-conductivity function. The resulting DFN voltage "
            "trajectory is a model-domain auxiliary label, not experimental cell voltage."
        ),
        "reference_conductivity_ms_cm": args.reference_conductivity_ms_cm,
        "current_a": args.current_a,
        "probe_horizon_s": args.probe_horizon_s,
        "time_points": args.time_points,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        target_log_conductivity=target_log_k,
        group_id=groups.astype(np.int64),
        group_names=np.asarray(group_names, dtype=str),
        physics_mask=physics_mask,
        target_voltage=target_voltage,
        current_a=np.full(len(frame), args.current_a, dtype=np.float64),
        feature_names=np.asarray(feature_names, dtype=str),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
