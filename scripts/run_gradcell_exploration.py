from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGES = (
    "checks",
    "reference-data",
    "reference-front",
    "train-k0",
    "evaluate-k0",
    "train-refiner",
    "evaluate-refiner",
    "verify-dfn",
)


def run(command: list[str], root: Path, env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one reproducible stage of the direct GradCell exploration."
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--reference-samples", type=int, default=5000)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--preference-points", type=int, default=11)
    parser.add_argument("--dfn-candidates", type=int, default=11)
    parser.add_argument("--reference-seed", type=int, default=101)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--output-root", type=Path, default=Path("results/gradcell_exploration"))
    args = parser.parse_args()
    if args.reference_samples < 20:
        parser.error("--reference-samples must be at least 20")
    if args.refinement_steps < 1:
        parser.error("--refinement-steps must be positive")

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    data = (
        root
        / "data"
        / f"gradcell_reference_spme_{args.reference_samples}_s{args.reference_seed}.npz"
    )
    front = output_root / "reference" / "pareto_front.npz"
    k0_checkpoint = output_root / f"k0_s{args.model_seed}" / "model.pt"
    refiner_checkpoint = output_root / f"k{args.refinement_steps}_s{args.model_seed}" / "model.pt"
    k0_eval = output_root / f"k0_s{args.model_seed}" / "evaluation_spme"
    refiner_eval = output_root / f"k{args.refinement_steps}_s{args.model_seed}" / "evaluation_spme"

    commands = {
        "checks": [python, "-m", "pytest", "-q"],
        "reference-data": [
            python,
            "scripts/generate_supervised_data.py",
            "--backend",
            "pybamm",
            "--model",
            "SPMe",
            "--samples",
            str(args.reference_samples),
            "--seed",
            str(args.reference_seed),
            "--output",
            str(data),
        ],
        "reference-front": [
            python,
            "scripts/build_reference_front.py",
            "--data",
            str(data),
            "--output",
            str(front),
            "--preference-points",
            str(args.preference_points),
        ],
        "train-k0": [
            python,
            "scripts/train_mvp.py",
            "--backend",
            "pybamm",
            "--model",
            "SPMe",
            "--steps",
            str(args.training_steps),
            "--batch-size",
            str(args.batch_size),
            "--refinement-steps",
            "0",
            "--reference-front",
            str(front),
            "--seed",
            str(args.model_seed),
            "--checkpoint",
            str(k0_checkpoint),
            "--run-dir",
            str(k0_checkpoint.parent / "run"),
        ],
        "evaluate-k0": [
            python,
            "scripts/evaluate_gradcell.py",
            "--checkpoint",
            str(k0_checkpoint),
            "--reference-front",
            str(front),
            "--preference-points",
            str(args.preference_points),
            "--output-dir",
            str(k0_eval),
        ],
        "train-refiner": [
            python,
            "scripts/train_mvp.py",
            "--backend",
            "pybamm",
            "--model",
            "SPMe",
            "--steps",
            str(args.training_steps),
            "--batch-size",
            str(args.batch_size),
            "--refinement-steps",
            str(args.refinement_steps),
            "--reference-front",
            str(front),
            "--seed",
            str(args.model_seed),
            "--checkpoint",
            str(refiner_checkpoint),
            "--run-dir",
            str(refiner_checkpoint.parent / "run"),
        ],
        "evaluate-refiner": [
            python,
            "scripts/evaluate_gradcell.py",
            "--checkpoint",
            str(refiner_checkpoint),
            "--reference-front",
            str(front),
            "--preference-points",
            str(args.preference_points),
            "--output-dir",
            str(refiner_eval),
        ],
        "verify-dfn": [
            python,
            "scripts/verify_gradcell_dfn.py",
            "--candidates",
            str(refiner_eval / "candidates.npz"),
            "--max-candidates",
            str(args.dfn_candidates),
            "--output-dir",
            str(refiner_checkpoint.parent / "verification_dfn"),
        ],
    }
    required = {
        "reference-front": data,
        "train-k0": front,
        "evaluate-k0": k0_checkpoint,
        "train-refiner": front,
        "evaluate-refiner": refiner_checkpoint,
        "verify-dfn": refiner_eval / "candidates.npz",
    }.get(args.stage)
    if required is not None and not required.is_file():
        parser.error(f"required artifact is missing: {required}")
    run(commands[args.stage], root, env)


if __name__ == "__main__":
    main()
