"""Long-running forward/sensitivity/autograd QA for the GradCell physics layer.

This script deliberately does not import GradCell, create an optimizer, or train
any network. It validates only the physical backend and its VJP wrapper.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradcell.physics import DifferentiablePhysicsLayer, PyBaMMBackend

LOGGER = logging.getLogger("qa_physics")


@dataclass(frozen=True)
class Case:
    case_id: str
    c_rate: float
    variant: str
    inputs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable DFN/SPMe forward, sensitivity and autograd QA."
    )
    parser.add_argument("--model", choices=("DFN", "SPMe"), default="DFN")
    parser.add_argument("--parameter-set", default="Chen2020")
    parser.add_argument(
        "--mode",
        choices=("forward", "sensitivity", "autograd", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/physics_qa"))
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--random-directions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")
        handle.flush()


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed JSONL line in %s", path)
                continue
            if record.get("case_id"):
                completed[str(record["case_id"])] = record
    return completed


def make_cases(nominal_capacity_ah: float = 5.0) -> list[Case]:
    base = np.array([0.335, 0.25, 0.47, 0.665, 0.75, 1.0, 1.0, 0.0], dtype=np.float64)
    variants = {
        "nominal": np.ones(8, dtype=np.float64),
        "diffusion_low": np.array([1, 1, 1, 1, 1, 0.75, 0.75, 1], dtype=np.float64),
        "diffusion_high": np.array([1, 1, 1, 1, 1, 1.25, 1.25, 1], dtype=np.float64),
    }
    cases: list[Case] = []
    for c_rate in (0.5, 1.0, 2.0, 3.0):
        for variant, multiplier in variants.items():
            row = base * multiplier
            row[-1] = c_rate * nominal_capacity_ah
            case_id = f"{variant}_c{c_rate:g}"
            cases.append(
                Case(
                    case_id=case_id,
                    c_rate=c_rate,
                    variant=variant,
                    inputs=row.tolist(),
                )
            )
    return cases


def make_backend(args: argparse.Namespace, c_rate: float) -> PyBaMMBackend:
    return PyBaMMBackend(
        model_name=args.model,
        parameter_set=args.parameter_set,
        output_variables=("Voltage [V]",),
        time_points=args.time_points,
        horizon_s=3600.0 / c_rate,
        rtol=args.rtol,
        atol=args.atol,
    )


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.linalg.norm(a - b))
    denominator = float(np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)
    return numerator / denominator


def finite_difference_sensitivity(
    backend: PyBaMMBackend,
    x: np.ndarray,
    analytic_jacobian: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    names = list(backend.input_names)
    values: dict[str, Any] = {}
    errors: list[float] = []
    signs: list[bool] = []
    for index, name in enumerate(names):
        step = max(abs(float(x[index])) * eps, eps)
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        if minus[index] <= 0.0 and index >= 5:
            values[name] = {"ok": False, "error": "negative perturbation"}
            continue
        plus_batch = backend.solve_batch(plus[None, :])
        minus_batch = backend.solve_batch(minus[None, :])
        ok = bool(plus_batch.status[0] and minus_batch.status[0])
        if not ok:
            values[name] = {"ok": False, "error": "perturbed solve failed"}
            continue
        fd = (plus_batch.trajectories[0, 0] - minus_batch.trajectories[0, 0]) / (2.0 * step)
        analytic = analytic_jacobian[:, index]
        error = relative_error(fd, analytic)
        sign_consistent = bool(np.dot(fd, analytic) >= 0.0)
        errors.append(error)
        signs.append(sign_consistent)
        values[name] = {
            "ok": True,
            "step": step,
            "relative_error": error,
            "sign_consistent": sign_consistent,
            "fd_l2": float(np.linalg.norm(fd)),
            "analytic_l2": float(np.linalg.norm(analytic)),
        }
    return {
        "parameters": values,
        "n_checked": len(errors),
        "median_relative_error": float(np.median(errors)) if errors else math.inf,
        "max_relative_error": float(max(errors)) if errors else math.inf,
        "sign_consistency": float(np.mean(signs)) if signs else 0.0,
    }


def forward_check(batch: Any) -> dict[str, Any]:
    trajectory = np.asarray(batch.trajectories[0, 0], dtype=np.float64)
    finite = bool(np.isfinite(trajectory).all())
    status_ok = bool(batch.status[0] == 1)
    return {
        "passed": bool(status_ok and finite),
        "status": int(batch.status[0]),
        "finite": finite,
        "time_points": int(trajectory.size),
        "minimum_voltage_v": float(np.nanmin(trajectory)),
        "maximum_voltage_v": float(np.nanmax(trajectory)),
        "runtime_s": float(batch.runtime_s[0]),
    }


def autograd_check(
    backend: PyBaMMBackend,
    x: np.ndarray,
    fd_eps: float,
    random_directions: int,
    seed: int,
) -> dict[str, Any]:
    layer = DifferentiablePhysicsLayer(backend)
    point = torch.tensor(x[None, :], dtype=torch.float64)

    def objective(value: torch.Tensor) -> torch.Tensor:
        trajectory, status, _ = layer(value)
        if not bool(status.all()):
            raise RuntimeError("solve failed during autograd check")
        return trajectory.square().mean()

    def check_direction(direction: torch.Tensor, label: str) -> dict[str, Any]:
        direction = direction / direction.norm().clamp_min(1e-12)
        point_for_grad = point.detach().clone().requires_grad_(True)
        value = objective(point_for_grad)
        (gradient,) = torch.autograd.grad(value, point_for_grad)
        autodiff = torch.sum(gradient * direction)
        with torch.no_grad():
            plus = objective(point + fd_eps * direction)
            minus = objective(point - fd_eps * direction)
        finite_difference = (plus - minus) / (2.0 * fd_eps)
        error = torch.abs(autodiff - finite_difference) / (1.0 + torch.abs(autodiff))
        return {
            "label": label,
            "autodiff": float(autodiff),
            "finite_difference": float(finite_difference),
            "relative_directional_error": float(error),
            "passed": bool(float(error) < 0.05),
        }

    checks = []
    for index in range(point.shape[1]):
        direction = torch.zeros_like(point)
        direction[0, index] = 1.0
        checks.append(check_direction(direction, f"coordinate_{index}"))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    for index in range(random_directions):
        direction = torch.randn(point.shape, generator=generator, dtype=torch.float64)
        checks.append(check_direction(direction, f"random_{index}"))

    errors = [item["relative_directional_error"] for item in checks]
    return {
        "checks": checks,
        "n_checked": len(checks),
        "median_relative_error": float(np.median(errors)) if errors else math.inf,
        "max_relative_error": float(max(errors)) if errors else math.inf,
        "passed": bool(all(item["passed"] for item in checks)),
    }


def run_case(
    case: Case,
    backend: PyBaMMBackend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    x = np.asarray(case.inputs, dtype=np.float64)
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "c_rate": case.c_rate,
        "variant": case.variant,
        "inputs": case.inputs,
        "model": args.model,
        "parameter_set": args.parameter_set,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        batch = backend.solve_batch(x[None, :])
        checks: dict[str, Any] = {}
        checks["forward"] = forward_check(batch)
        if args.mode in ("sensitivity", "all") and checks["forward"]["passed"]:
            jac = np.asarray(batch.jacobian[0, 0], dtype=np.float64)
            checks["sensitivity"] = finite_difference_sensitivity(
                backend, x, jac, args.fd_eps
            )
            checks["sensitivity"]["passed"] = bool(
                checks["sensitivity"]["n_checked"] == len(backend.input_names)
                and checks["sensitivity"]["median_relative_error"] < 0.10
                and checks["sensitivity"]["sign_consistency"] >= 0.90
            )
        if args.mode in ("autograd", "all") and checks["forward"]["passed"]:
            checks["autograd"] = autograd_check(
                backend, x, args.fd_eps, args.random_directions, args.seed
            )
        record["checks"] = checks
        record["passed"] = bool(
            all(item.get("passed", False) for item in checks.values())
        )
    except Exception as exc:
        LOGGER.exception("Case %s failed", case.case_id)
        record["passed"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_s"] = time.perf_counter() - started
    return record


def summarize(records: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    passed = [record for record in records.values() if record.get("passed")]
    failed = [record for record in records.values() if not record.get("passed")]
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "parameter_set": args.parameter_set,
        "mode": args.mode,
        "n_cases": len(records),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pass_rate": len(passed) / len(records) if records else 0.0,
        "failed_case_ids": [record.get("case_id") for record in failed],
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)
    results_path = args.output_dir / "case_results.jsonl"
    summary_path = args.output_dir / "summary.json"
    completed = {} if args.no_resume else read_completed(results_path)
    cases = make_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "model": args.model,
        "parameter_set": args.parameter_set,
        "mode": args.mode,
        "time_points": args.time_points,
        "rtol": args.rtol,
        "atol": args.atol,
        "fd_eps": args.fd_eps,
        "seed": args.seed,
        "n_cases": len(cases),
    }
    write_json(args.output_dir / "metadata.json", metadata)
    LOGGER.info("Starting QA: model=%s cases=%d mode=%s", args.model, len(cases), args.mode)

    backends: dict[float, PyBaMMBackend] = {}
    for number, case in enumerate(cases, start=1):
        if case.case_id in completed:
            LOGGER.info("Skipping completed case %s (%d/%d)", case.case_id, number, len(cases))
            continue
        LOGGER.info("Running case %s (%d/%d)", case.case_id, number, len(cases))
        if case.c_rate not in backends:
            LOGGER.info("Building %s backend for %.1fC", args.model, case.c_rate)
            backends[case.c_rate] = make_backend(args, case.c_rate)
        record = run_case(case, backends[case.c_rate], args)
        append_jsonl(results_path, record)
        completed[case.case_id] = record
        write_json(summary_path, summarize(completed, args))
        LOGGER.info(
            "Finished %s: passed=%s elapsed=%.1fs",
            case.case_id,
            record.get("passed"),
            record.get("elapsed_s", float("nan")),
        )

    final_summary = summarize(completed, args)
    write_json(summary_path, final_summary)
    LOGGER.info("QA complete: %s", json.dumps(final_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
