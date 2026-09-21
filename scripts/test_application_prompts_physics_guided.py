from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from predict_battery_design_physics_guided import (
    dtype_from_name,
    input_device,
    load_models,
    mean_pool,
)


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
        description="Batch-test the current Qwen + single-design MLP on application prompts."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch size and maximum length must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = read_jsonl(args.data)
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("Prompt dataset is empty")
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Prompt IDs must be unique")
    if any(not str(row.get("battery_description", "")).strip() for row in rows):
        raise ValueError("Every record must contain a non-empty battery_description")

    device = torch.device(args.device)
    design_model, surrogate, design_checkpoint, surrogate_checkpoint = load_models(
        args.design_checkpoint, args.surrogate_checkpoint, device
    )
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        args.model_name,
        dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    qwen.eval()
    qwen.requires_grad_(False)
    qwen_device = input_device(qwen)
    design_mean = design_checkpoint["design_log_mean"]
    design_std = design_checkpoint["design_log_std"]
    performance_mean = design_checkpoint["performance_log_mean"]
    performance_std = design_checkpoint["performance_log_std"]
    nominal = surrogate_checkpoint["nominal_parameter_values"].numpy()

    result_rows = []
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            batch = rows[begin : begin + args.batch_size]
            tokens = tokenizer(
                [row["battery_description"] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                pad_to_multiple_of=8 if qwen_device.type == "cuda" else None,
                return_tensors="pt",
            )
            tokens = {name: value.to(qwen_device) for name, value in tokens.items()}
            embedding = mean_pool(
                qwen(**tokens, return_dict=True).last_hidden_state,
                tokens["attention_mask"],
            ).float()
            if embedding.shape[1] != design_checkpoint["model_config"]["input_dim"]:
                raise ValueError("Qwen embedding dimension does not match the checkpoint")
            normalized_design = design_model(embedding.to(device))
            normalized_performance = surrogate(normalized_design)
            design_log = normalized_design.cpu() * design_std + design_mean
            performance_log = normalized_performance.cpu() * performance_std + performance_mean
            multipliers = torch.exp(design_log).numpy()
            performance = torch.exp(performance_log).numpy()
            values = multipliers * nominal[None, :]
            for index, source in enumerate(batch):
                structurally_feasible = bool(
                    np.isfinite(values[index]).all()
                    and np.all(values[index] > 0.0)
                    and values[index, 0] + values[index, 3] <= 1.0 + 1e-10
                    and values[index, 1] + values[index, 4] <= 1.0 + 1e-10
                )
                predicted_performance = dict(
                    zip(
                        design_checkpoint["performance_fields"],
                        performance[index].tolist(),
                        strict=True,
                    )
                )
                requested_capacity = float(source["requirements"]["capacity_ah"])
                predicted_capacity = float(predicted_performance["capacity_1c_ah"])
                scope = source["current_model_scope"]
                result_rows.append(
                    {
                        **source,
                        "model_output": {
                            "base_parameter_set": design_checkpoint["parameter_sets"][0],
                            "generation_mode": design_checkpoint["generation_modes"][0],
                            "parameter_multipliers": dict(
                                zip(
                                    design_checkpoint["parameter_names"],
                                    multipliers[index].tolist(),
                                    strict=True,
                                )
                            ),
                            "parameter_values": dict(
                                zip(
                                    design_checkpoint["parameter_names"],
                                    values[index].tolist(),
                                    strict=True,
                                )
                            ),
                            "surrogate_predicted_performance": predicted_performance,
                            "structurally_feasible": structurally_feasible,
                        },
                        "capability_audit": {
                            "scope_verdict": (
                                "out_of_training_scope"
                                if scope["unsupported"]
                                else "within_declared_scope"
                            ),
                            "unsupported_requirements": scope["unsupported"],
                            "requested_capacity_ah": requested_capacity,
                            "predicted_capacity_1c_ah": predicted_capacity,
                            "capacity_relative_error": abs(predicted_capacity - requested_capacity)
                            / max(abs(requested_capacity), 1e-12),
                            "warning": (
                                "A bounded output is not proof that chemistry, geometry, mass, "
                                "cycle life, safety, cost, or low-temperature requirements were met."
                            ),
                        },
                    }
                )
            print(
                json.dumps({"completed": min(begin + len(batch), len(rows)), "total": len(rows)}),
                flush=True,
            )

    write_jsonl(args.output, result_rows)
    capacity_errors = [row["capability_audit"]["capacity_relative_error"] for row in result_rows]
    structural = [row["model_output"]["structurally_feasible"] for row in result_rows]
    unsupported_counts = Counter(
        name for row in result_rows for name in row["capability_audit"]["unsupported_requirements"]
    )
    designs_by_spec: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in result_rows:
        multipliers = row["model_output"]["parameter_multipliers"]
        designs_by_spec[row["spec_id"]].append(
            np.log(
                np.asarray(
                    [multipliers[name] for name in design_checkpoint["parameter_names"]],
                    dtype=np.float64,
                )
            )
        )
    family_log_rms = []
    for designs in designs_by_spec.values():
        array = np.stack(designs)
        center = array.mean(axis=0)
        family_log_rms.append(float(np.sqrt(np.mean((array - center) ** 2))))
    report = {
        "schema": "gradcell.application_prompt_model_test_report.v1",
        "data": str(args.data),
        "output": str(args.output),
        "records": len(result_rows),
        "specifications": len(designs_by_spec),
        "model_scope": {
            "parameter_sets": design_checkpoint["parameter_sets"],
            "generation_modes": design_checkpoint["generation_modes"],
            "all_prompts_out_of_training_scope": all(
                row["capability_audit"]["scope_verdict"] == "out_of_training_scope"
                for row in result_rows
            ),
        },
        "structural_feasibility_rate": float(np.mean(structural)),
        "requested_vs_predicted_capacity": {
            "mean_absolute_percentage_error": float(np.mean(capacity_errors)),
            "median_absolute_percentage_error": float(np.median(capacity_errors)),
            "p90_absolute_percentage_error": float(np.quantile(capacity_errors, 0.9)),
        },
        "paraphrase_design_consistency": {
            "mean_within_spec_log_rms": float(np.mean(family_log_rms)),
            "p90_within_spec_log_rms": float(np.quantile(family_log_rms, 0.9)),
            "interpretation": "Lower is more stable across paraphrases of the same specification.",
        },
        "unsupported_requirement_counts": dict(unsupported_counts),
        "interpretation": (
            "This is a capability-boundary test. The current Chen2020 Regular model cannot "
            "validate chemistry choice, product geometry, mass, cycle life, safety, cost, or "
            "low-temperature requirements. Surrogate performance is not strict DFN verification."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
