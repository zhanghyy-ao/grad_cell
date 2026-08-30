from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-material electrolyte DFN v1 pipeline.")
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--physics-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1"))
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()
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
    run(
        [
            python,
            "scripts/train_electrolyte_v1.py",
            "--data",
            str(data_path),
            "--physics-backend",
            args.physics_backend,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(args.output_dir),
        ],
        root,
    )


if __name__ == "__main__":
    main()
