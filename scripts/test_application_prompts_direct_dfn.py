from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradcell.language import DirectDFNPerformanceLayer, SingleDesignPhysicsMLP
from predict_battery_design_physics_guided import dtype_from_name, input_device, mean_pool


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Qwen + design MLP with real PyBaMM DFN solves."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = read_jsonl(args.data)
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("Prompt dataset is empty")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "gradcell.language_single_design_direct_dfn.v1":
        raise ValueError("Checkpoint is not a direct-DFN language design model")
    model = SingleDesignPhysicsMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    physics_config = checkpoint["physics_config"]
    physics = DirectDFNPerformanceLayer(
        parameter_set=checkpoint["parameter_sets"][0],
        time_points=int(physics_config["time_points"]),
        maximum_duration_factor=float(physics_config["maximum_duration_factor"]),
        rtol=float(physics_config["rtol"]),
        atol=float(physics_config["atol"]),
        cutoff_v=float(physics_config["cutoff_v"]),
        gate_temperature_v=float(physics_config["gate_temperature_v"]),
        current_ramp_time_s=float(physics_config["current_ramp_time_s"]),
        training_voltage_floor_v=float(physics_config.get("training_voltage_floor_v", 2.0)),
        calculate_sensitivities=False,
    ).to(device)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        args.model_name,
        dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    qwen.eval().requires_grad_(False)
    qwen_device = input_device(qwen)
    design_mean = checkpoint["design_log_mean"].to(device)
    design_std = checkpoint["design_log_std"].to(device)
    nominal = checkpoint["nominal_parameter_values"].to(device)
    results = []
    with torch.no_grad():
        for begin in range(0, len(rows), args.batch_size):
            batch = rows[begin : begin + args.batch_size]
            tokens = tokenizer(
                [row["battery_description"] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(qwen_device) for name, value in tokens.items()}
            embedding = mean_pool(
                qwen(**tokens, return_dict=True).last_hidden_state,
                tokens["attention_mask"],
            ).float()
            normalized_design = model(embedding.to(device))
            multipliers = torch.exp(normalized_design * design_std + design_mean)
            parameter_values = multipliers * nominal
            requested_capacity = torch.as_tensor(
                [float(row["requirements"]["capacity_ah"]) for row in batch],
                dtype=parameter_values.dtype,
                device=device,
            )
            dfn = physics(parameter_values, requested_capacity)
            for index, source in enumerate(batch):
                results.append(
                    {
                        **source,
                        "model_output": {
                            "base_parameter_set": checkpoint["parameter_sets"][0],
                            "parameter_multipliers": dict(
                                zip(
                                    checkpoint["parameter_names"],
                                    multipliers[index].cpu().tolist(),
                                    strict=True,
                                )
                            ),
                            "parameter_values": dict(
                                zip(
                                    checkpoint["parameter_names"],
                                    parameter_values[index].cpu().tolist(),
                                    strict=True,
                                )
                            ),
                            "direct_dfn": {
                                "solver_success": bool(dfn.status[index]),
                                "runtime_s": float(dfn.runtime_s[index]),
                                "performance": dict(
                                    zip(
                                        checkpoint["performance_fields"],
                                        dfn.performance[index].cpu().tolist(),
                                        strict=True,
                                    )
                                ),
                            },
                        },
                    }
                )
            print(
                json.dumps({"completed": min(begin + len(batch), len(rows)), "total": len(rows)}),
                flush=True,
            )
    write_jsonl(args.output, results)
    successful = [row for row in results if row["model_output"]["direct_dfn"]["solver_success"]]
    capacity_errors = []
    for row in successful:
        requested = float(row["requirements"]["capacity_ah"])
        predicted = float(row["model_output"]["direct_dfn"]["performance"]["capacity_1c_ah"])
        capacity_errors.append(abs(predicted - requested) / max(abs(requested), 1e-12))
    report = {
        "schema": "gradcell.application_prompt_direct_dfn_report.v1",
        "records": len(results),
        "dfn_success_rate": len(successful) / len(results),
        "capacity_mape": float(np.mean(capacity_errors)) if capacity_errors else None,
        "gradient_source_during_training": checkpoint["gradient_qa"]["source"],
        "uses_performance_surrogate": False,
        "output": str(args.output),
        "warning": (
            "DFN verification uses the checkpoint parameter set. Requests for another chemistry, "
            "product geometry, cycle life, safety, cost, or low-temperature behavior remain out of scope."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
