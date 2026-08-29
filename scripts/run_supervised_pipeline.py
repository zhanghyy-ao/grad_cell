from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradcell.models import Chen2020Surrogate


def run_stage(name: str, command: list[str], root: Path, env: dict[str, str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def verify_dataset(path: Path, expected_targets: int) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        latent = arrays["latent"]
        design = arrays["design"]
        targets = arrays["targets"]
        metadata = json.loads(str(arrays["metadata"]))
    if latent.ndim != 2 or latent.shape[1] != 5:
        raise RuntimeError(f"Expected latent [N,5], got {latent.shape}")
    if design.shape != (len(latent), 10):
        raise RuntimeError(f"Expected design [N,10], got {design.shape}")
    if targets.shape != (len(latent), expected_targets):
        raise RuntimeError(
            f"Expected targets [N,{expected_targets}], got {targets.shape}"
        )
    if not all(np.isfinite(values).all() for values in (latent, design, targets)):
        raise RuntimeError("Dataset contains NaN or infinite values")
    if not np.array_equal(design[:, 6], np.ones(len(design))):
        raise RuntimeError("Positive diffusivity multiplier is not fixed at 1.0")
    if not np.array_equal(design[:, 7], np.ones(len(design))):
        raise RuntimeError("Negative diffusivity multiplier is not fixed at 1.0")
    if metadata["valid_samples"] != len(latent):
        raise RuntimeError("Metadata valid_samples does not match dataset length")
    return metadata


def verify_training(output_dir: Path, expected_targets: int) -> dict:
    required = (
        output_dir / "best_model.pt",
        output_dir / "metrics.json",
        output_dir / "history.json",
        output_dir / "test_predictions.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training artifacts are missing: {missing}")
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model = Chen2020Surrogate(**checkpoint["model_kwargs"]).double()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with np.load(output_dir / "test_predictions.npz", allow_pickle=False) as predictions:
        target_values = predictions["targets"]
        prediction_values = predictions["predictions"]
    if (
        target_values.shape != prediction_values.shape
        or target_values.shape[1] != expected_targets
    ):
        raise RuntimeError("Saved test predictions have inconsistent shapes")
    if not np.isfinite(prediction_values).all():
        raise RuntimeError("Saved test predictions contain non-finite values")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tests, Chen2020 data generation, surrogate training, and verification."
    )
    parser.add_argument("--backend", choices=("pybamm", "toy"), default="pybamm")
    parser.add_argument("--model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--simulation-batch-size", type=int, default=4)
    parser.add_argument("--time-points", type=int, default=101)
    parser.add_argument("--latent-std", type=float, default=1.25)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.samples < 20:
        parser.error("--samples must be at least 20 so train/validation/test are non-empty")
    root = Path(__file__).resolve().parents[1]
    data_path = args.data or Path(
        f"data/chen2020_{args.model.lower()}_{args.samples}_seed{args.seed}.npz"
    )
    output_dir = args.output_dir or Path(
        f"results/supervised_{args.model.lower()}_{args.samples}_seed{args.seed}"
    )
    data_path = data_path if data_path.is_absolute() else root / data_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    if not args.overwrite and (data_path.exists() or output_dir.exists()):
        parser.error("Output already exists; choose new paths or pass --overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(root / "src")
    started = time.perf_counter()
    python = sys.executable

    if not args.skip_checks:
        run_stage("unit tests", [python, "-m", "pytest", "-q"], root, env)
        run_stage("static checks", [python, "-m", "ruff", "check", "src", "scripts", "tests"], root, env)

    generation_command = [
        python,
        "scripts/generate_supervised_data.py",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--samples",
        str(args.samples),
        "--batch-size",
        str(args.simulation_batch_size),
        "--time-points",
        str(args.time_points),
        "--latent-std",
        str(args.latent_std),
        "--capacity-calibration-rate",
        str(args.calibration_rate),
        "--capacity-calibration-iterations",
        str(args.calibration_iterations),
        "--current-ramp-time-s",
        "0",
        "--seed",
        str(args.seed),
        "--output",
        str(data_path),
    ]
    run_stage("Chen2020 data generation", generation_command, root, env)
    expected_targets = 9 if args.backend == "pybamm" else 4
    dataset_metadata = verify_dataset(data_path, expected_targets)

    training_command = [
        python,
        "scripts/train_supervised_surrogate.py",
        "--data",
        str(data_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.training_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--depth",
        str(args.depth),
        "--learning-rate",
        str(args.learning_rate),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    run_stage("surrogate training", training_command, root, env)
    training_report = verify_training(output_dir, expected_targets)

    summary = {
        "status": "completed",
        "elapsed_s": time.perf_counter() - started,
        "dataset": str(data_path),
        "training_output": str(output_dir),
        "valid_samples": dataset_metadata["valid_samples"],
        "failure_rate": dataset_metadata["failure_rate"],
        "test_r2": training_report["test_metrics"]["r2"],
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== pipeline completed ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
