from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gradcell.physics import PyBaMMBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a nominal Chen2020 SPMe discharge")
    parser.add_argument("--c-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()
    backend = PyBaMMBackend(horizon_s=3600.0 / args.c_rate)
    nominal_capacity_ah = 5.0
    inputs = np.array(
        [[0.335, 0.25, 0.47, 0.665, 0.75, 1.0, 1.0, args.c_rate * nominal_capacity_ah]]
    )
    batch = backend.solve_batch(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": int(batch.status[0]),
        "runtime_s": float(batch.runtime_s[0]),
        "minimum_voltage_v": float(np.nanmin(batch.trajectories[0, 0])),
        "maximum_voltage_v": float(np.nanmax(batch.trajectories[0, 0])),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
