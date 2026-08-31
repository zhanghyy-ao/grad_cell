from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def weight_slug(weight: float) -> str:
    return format(weight, "g").replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-material electrolyte DFN v1 pipeline."
    )
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--physics-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--physics-weights",
        type=float,
        nargs="+",
        default=[1.0],
        help="One or more physics loss weights trained against the same prepared dataset.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1"))
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()
    if any(weight < 0.0 for weight in args.physics_weights):
        parser.error("--physics-weights values must be non-negative")
    if len(set(args.physics_weights)) != len(args.physics_weights):
        parser.error("--physics-weights values must be unique")
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    data_path = root / "data" / f"electrolyte_v1_{args.physics_backend}_s{args.seed}.npz"
    if not args.skip_checks:
        run([python, "-m", "pytest", "-q"], root)
        run([python, "-m", "ruff", "check", "src", "scripts", "tests"], root)
    run(
        [
            python,
            "scripts/prepare_electrolyte_v1_data.py",
            "--physics-backend",
            args.physics_backend,
            "--physics-samples",
            str(args.physics_samples),
            "--seed",
            str(args.seed),
            "--output",
            str(data_path),
        ],
        root,
    )
    run(
        [
            python,
            "scripts/validate_electrolyte_v1_gradients.py",
            "--backend",
            args.physics_backend,
        ],
        root,
    )
    runs = []
    multiple_weights = len(args.physics_weights) > 1
    for weight in args.physics_weights:
        output_dir = (
            args.output_dir / f"weight_{weight_slug(weight)}"
            if multiple_weights
            else args.output_dir
        )
        run(
            [
                python,
                "scripts/train_electrolyte_v1.py",
                "--data",
                str(data_path),
                "--physics-backend",
                args.physics_backend,
                "--physics-weight",
                str(weight),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(output_dir),
            ],
            root,
        )
        metrics_path = output_dir / "metrics.json"
        runs.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if multiple_weights:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "data": str(data_path),
            "physics_backend": args.physics_backend,
            "physics_samples": args.physics_samples,
            "seed": args.seed,
            "weights": args.physics_weights,
            "runs": runs,
        }
        (args.output_dir / "weight_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
