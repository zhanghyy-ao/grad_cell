from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def run(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split_seed"], row["physics_samples"])].append(row)
    summaries = []
    metric_names = ("voltage_rmse_v", "voltage_rmse_scaled", "success_rate")
    for key, group in sorted(grouped.items()):
        summary = {
            "split_seed": key[0],
            "physics_samples": key[1],
            "model_seeds": [row["model_seed"] for row in group],
            "runs": len(group),
        }
        for name in metric_names:
            values = np.asarray([row[name] for row in group], dtype=np.float64)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare nested DFN subsets and run a resumable multi-seed experiment matrix."
    )
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--physics-backend", choices=("dfn", "analytic"), default="dfn")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--physics-samples", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--validation-physics-samples", type=int, default=32)
    parser.add_argument("--test-physics-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--physics-batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/electrolyte_v1_matrix"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.physics_samples):
        parser.error("--physics-samples must contain positive values")

    root = Path(__file__).resolve().parents[1]
    output_root = (root / args.output_dir).resolve()
    data_root = root / "data" / "electrolyte_v1_matrix"
    maximum_physics_samples = max(args.physics_samples)
    completed_rows = []
    manifest = {
        "split_seeds": args.split_seeds,
        "model_seeds": args.model_seeds,
        "physics_samples": args.physics_samples,
        "physics_backend": args.physics_backend,
        "training_objective": "physics_voltage_only",
        "dataset_policy": (
            "One maximum-size nested physics dataset per DOI split; smaller training subsets "
            "use saved physics_rank, while validation/test physics rows are evaluation-only."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split_seed in args.split_seeds:
        data_path = data_root / (
            f"{args.physics_backend}_split_{split_seed}_physics_{maximum_physics_samples}.npz"
        )
        if args.force or not data_path.exists():
            run(
                [
                    sys.executable,
                    "scripts/prepare_electrolyte_v1_data.py",
                    "--physics-backend",
                    args.physics_backend,
                    "--physics-samples",
                    str(maximum_physics_samples),
                    "--physics-validation-samples",
                    str(args.validation_physics_samples),
                    "--physics-test-samples",
                    str(args.test_physics_samples),
                    "--time-points",
                    "41",
                    "--probe-horizon-s",
                    "600",
                    "--seed",
                    str(split_seed),
                    "--split-seed",
                    str(split_seed),
                    "--output",
                    str(data_path),
                ],
                root,
                output_root / "logs" / f"prepare_split_{split_seed}.log",
            )
        if args.prepare_only:
            continue

        for model_seed in args.model_seeds:
            for sample_count in args.physics_samples:
                run_name = f"split_{split_seed}/model_{model_seed}/physics_{sample_count}"
                run_dir = output_root / run_name
                metrics_path = run_dir / "metrics.json"
                if args.force or not metrics_path.exists():
                    run(
                        [
                            sys.executable,
                            "scripts/train_electrolyte_v1.py",
                            "--data",
                            str(data_path),
                            "--physics-backend",
                            args.physics_backend,
                            "--physics-samples",
                            str(sample_count),
                            "--split-seed",
                            str(split_seed),
                            "--seed",
                            str(model_seed),
                            "--epochs",
                            str(args.epochs),
                            "--physics-batch-size",
                            str(args.physics_batch_size),
                            "--output-dir",
                            str(run_dir),
                        ],
                        root,
                        output_root / "logs" / f"{run_name.replace('/', '_')}.log",
                    )
                report = json.loads(metrics_path.read_text(encoding="utf-8"))
                completed_rows.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "physics_samples": sample_count,
                        **report["physics_evaluation"]["test"],
                    }
                )

    if args.prepare_only:
        return
    summaries = aggregate(completed_rows)
    (output_root / "all_runs.json").write_text(
        json.dumps(completed_rows, indent=2), encoding="utf-8"
    )
    (output_root / "aggregate.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if summaries:
        with (output_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
