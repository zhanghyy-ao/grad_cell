from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradcell.language import DirectPhysicsPerformanceLayer, SingleDesignPhysicsMLP
from predict_battery_design_physics_guided import dtype_from_name, input_device, mean_pool


VERDICTS = {"satisfied", "partially_satisfied", "not_satisfied", "insufficient_evidence"}
REQUIREMENT_STATUSES = {"satisfied", "not_satisfied", "not_evaluable"}


class RetryableAPIError(RuntimeError):
    pass


class FatalAPIError(RuntimeError):
    pass


class InvalidJudgeResponse(ValueError):
    pass


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def structural_feasibility(values: np.ndarray, tolerance: float = 1e-10) -> bool:
    return bool(
        np.isfinite(values).all()
        and np.all(values > 0.0)
        and values[0] + values[3] <= 1.0 + tolerance
        and values[1] + values[4] <= 1.0 + tolerance
    )


def project_structural_feasibility(
    values: np.ndarray, margin: float = 1e-4
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project porosity/active-fraction pairs onto their physical simplex.

    Only the two coupled volume-fraction constraints are changed. Positive finite
    values are otherwise preserved, so the correction is transparent and small.
    """
    if not 0.0 < margin < 1.0:
        raise ValueError("projection margin must be between zero and one")
    projected = np.asarray(values, dtype=np.float64).copy()
    if projected.shape != (7,) or not np.isfinite(projected).all() or np.any(projected <= 0):
        raise ValueError("MLP parameter values must be seven finite positive numbers")
    pair_reports = []
    for electrode, porosity_index, active_index in (
        ("positive", 0, 3),
        ("negative", 1, 4),
    ):
        before = float(projected[porosity_index] + projected[active_index])
        scale = min(1.0, (1.0 - margin) / before)
        projected[[porosity_index, active_index]] *= scale
        pair_reports.append(
            {
                "electrode": electrode,
                "sum_before": before,
                "sum_after": float(projected[porosity_index] + projected[active_index]),
                "scale": float(scale),
                "corrected": bool(scale < 1.0),
            }
        )
    relative_change = np.abs(projected - values) / np.maximum(np.abs(values), 1e-12)
    return projected, {
        "method": "proportional_electrode_volume_fraction_simplex_projection",
        "margin": margin,
        "pairs": pair_reports,
        "applied": any(item["corrected"] for item in pair_reports),
        "maximum_relative_parameter_change": float(relative_change.max()),
        "rms_relative_parameter_change": float(np.sqrt(np.mean(relative_change**2))),
    }


def extract_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    if not isinstance(content, str) or not content.strip():
        raise InvalidJudgeResponse("API returned empty judge content")
    text = re.sub(r"^<think>.*?</think>\s*", "", content.strip(), flags=re.DOTALL)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        begin, end = text.find("{"), text.rfind("}")
        if begin < 0 or end <= begin:
            raise InvalidJudgeResponse(f"Judge did not return JSON: {text[:240]!r}")
        try:
            parsed = json.loads(text[begin : end + 1])
        except json.JSONDecodeError as exc:
            raise InvalidJudgeResponse(f"Malformed judge JSON: {text[:240]!r}") from exc
    if not isinstance(parsed, dict):
        raise InvalidJudgeResponse("Judge output must be a JSON object")
    return parsed


def validate_judgement(value: dict[str, Any]) -> dict[str, Any]:
    verdict = value.get("overall_verdict")
    if verdict not in VERDICTS:
        raise InvalidJudgeResponse(f"Invalid overall_verdict: {verdict!r}")
    assessments = value.get("requirement_assessments")
    if not isinstance(assessments, list) or not assessments:
        raise InvalidJudgeResponse("requirement_assessments must be a non-empty list")
    for index, item in enumerate(assessments):
        if not isinstance(item, dict):
            raise InvalidJudgeResponse(f"Assessment {index} is not an object")
        if item.get("status") not in REQUIREMENT_STATUSES:
            raise InvalidJudgeResponse(f"Assessment {index} has an invalid status")
        if not str(item.get("requirement", "")).strip():
            raise InvalidJudgeResponse(f"Assessment {index} has no requirement")
        if not str(item.get("evidence", "")).strip():
            raise InvalidJudgeResponse(f"Assessment {index} has no evidence")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise InvalidJudgeResponse("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise InvalidJudgeResponse("confidence must be between zero and one")
    if not str(value.get("summary", "")).strip():
        raise InvalidJudgeResponse("summary must be non-empty")
    value["confidence"] = float(confidence)
    value["counts"] = dict(Counter(item["status"] for item in assessments))
    return value


def judge_instruction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "判断候选电芯设计在现有证据下是否满足用户需求",
        "role": "你是严格的电池设计验收员，不是设计生成器。不得补造或推断未提供的仿真结果。",
        "mandatory_scope_rules": [
            "SPMe结果仅验证Chen2020基准单体在25摄氏度下的1C/5C/6C短时放电性能。",
            "SPMe的reference_capacity_ah是仿真电流标定值，不等于已经实现用户要求的产品级额定容量。",
            "化学体系选择、标称电压、封装形式、尺寸、质量、循环寿命、安全、成本、低温性能若无直接证据，必须标记not_evaluable。",
            "若SPMe求解失败，与SPMe性能有关的要求必须标记not_evaluable，不能标记satisfied。",
            "设计在仿真前若发生可行性投影，必须在总结中说明；不得把投影前的越界输出称为直接可行。",
            "只有提供的数值证据可以用于satisfied或not_satisfied判断。",
        ],
        "allowed_values": {
            "overall_verdict": sorted(VERDICTS),
            "requirement_status": sorted(REQUIREMENT_STATUSES),
        },
        "verdict_policy": {
            "satisfied": "所有硬性需求均有证据且满足",
            "partially_satisfied": "至少一项有证据且满足，但仍有不满足或不可验证项",
            "not_satisfied": "至少一项有证据的硬性需求明确不满足",
            "insufficient_evidence": "没有足够证据判断任何核心硬性需求",
        },
        "required_output": {
            "overall_verdict": "allowed value",
            "confidence": "0到1",
            "summary": "简洁中文结论",
            "requirement_assessments": [
                {
                    "requirement": "逐项需求",
                    "category": "需求类别",
                    "status": "satisfied|not_satisfied|not_evaluable",
                    "evidence": "引用输入中的具体证据或说明缺少什么证据",
                }
            ],
            "critical_gaps": ["仍缺少的关键验证"],
            "recommended_next_tests": ["建议的后续仿真或实验"],
        },
        "case": {
            "user_description": row["battery_description"],
            "structured_requirements": row.get("requirements", {}),
            "model_scope": row.get("current_model_scope", {}),
            "candidate_design": row["candidate_design"],
            "spme_evaluation": row["spme_evaluation"],
        },
    }


def request_judgement(
    row: dict[str, Any], api: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    payload: dict[str, Any] = {
        "model": api["model"],
        "messages": [
            {
                "role": "system",
                "content": "你负责基于明确证据严格验收电池设计，并输出JSON。",
            },
            {
                "role": "user",
                "content": json.dumps(judge_instruction(row), ensure_ascii=False),
            },
        ],
        "temperature": api["temperature"],
    }
    if api["max_tokens"] is not None:
        payload["max_tokens"] = api["max_tokens"]
    if api["json_mode"]:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        api["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=api["timeout_s"]) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        message = f"HTTP {exc.code}: {detail or exc.reason}"
        if exc.code in (401, 402, 403):
            raise FatalAPIError(message) from exc
        if exc.code in (408, 409, 425, 429) or exc.code >= 500:
            raise RetryableAPIError(message) from exc
        raise InvalidJudgeResponse(message) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableAPIError(f"{type(exc).__name__}: {exc}") from exc
    try:
        result = json.loads(raw)
        message = result["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise InvalidJudgeResponse(f"Invalid API envelope: {raw[:240]!r}") from exc
    content = message.get("content")
    if not content:
        content = message.get("reasoning_content")
    usage = {
        str(name): float(value)
        for name, value in result.get("usage", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return validate_judgement(extract_json_object(content)), usage


def load_design_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[SingleDesignPhysicsMLP, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "gradcell.language_design_three_stage.v1":
        raise ValueError("Expected a three-stage language-design checkpoint")
    model = SingleDesignPhysicsMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen + MLP design, SPMe replay, and DeepSeek requirement judgement."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--projection-margin", type=float, default=1e-4)
    parser.add_argument("--spme-reference-capacity-ah", type=float, default=5.0)
    parser.add_argument("--time-points", type=int, default=151)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--current-ramp-time-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--deepseek-max-tokens", type=int)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--skip-deepseek", action="store_true")
    parser.add_argument("--require-deepseek-success", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.time_points, args.retries) < 1:
        parser.error("batch size, lengths, time points, and retries must be positive")
    if args.spme_reference_capacity_ah <= 0 or args.current_ramp_time_s < 0:
        parser.error("SPMe reference capacity must be positive and ramp time non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_rows = read_jsonl(args.data)
    if args.max_records is not None:
        source_rows = source_rows[: args.max_records]
    if not source_rows:
        raise ValueError("Application prompt dataset is empty")
    ids = [str(row.get("prompt_id", index)) for index, row in enumerate(source_rows)]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs must be unique")
    if any(not str(row.get("battery_description", "")).strip() for row in source_rows):
        raise ValueError("Every row must contain battery_description")
    previous_by_id = (
        {str(row.get("prompt_id")): row for row in read_jsonl(args.output)}
        if args.output.exists()
        else {}
    )

    device = torch.device(args.device)
    design_model, checkpoint = load_design_model(args.checkpoint, device)
    parameter_set = str(checkpoint.get("parameter_sets", ["Chen2020"])[0])
    physics = DirectPhysicsPerformanceLayer(
        parameter_set=parameter_set,
        model_name="SPMe",
        time_points=args.time_points,
        rtol=args.rtol,
        atol=args.atol,
        current_ramp_time_s=args.current_ramp_time_s,
        training_voltage_floor_v=2.0,
        calculate_sensitivities=False,
    ).double().cpu()
    nominal = physics.nominal_parameter_values.detach().cpu().double().numpy()
    checkpoint_nominal = checkpoint.get("nominal_parameter_values")
    if checkpoint_nominal is not None and not np.allclose(
        nominal, torch.as_tensor(checkpoint_nominal).double().numpy(), rtol=1e-5, atol=1e-8
    ):
        raise ValueError("SPMe nominal parameter values do not match the training checkpoint")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        args.model_name,
        dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    qwen.eval().requires_grad_(False)
    qwen_device = input_device(qwen)

    names = list(checkpoint["parameter_names"])
    performance_fields = list(checkpoint["performance_fields"])
    design_mean = torch.as_tensor(checkpoint["design_log_mean"]).float()
    design_std = torch.as_tensor(checkpoint["design_log_std"]).float()
    result_rows: list[dict[str, Any]] = []
    # Use no_grad rather than inference_mode: the PyBaMM wrapper is implemented as
    # a custom autograd.Function and still creates an explicit (empty) Jacobian.
    with torch.no_grad():
        for begin in range(0, len(source_rows), args.batch_size):
            batch = source_rows[begin : begin + args.batch_size]
            tokens = tokenizer(
                [str(row["battery_description"]) for row in batch],
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
            if embedding.shape[1] != checkpoint["model_config"]["input_dim"]:
                raise ValueError("Qwen embedding dimension does not match the MLP checkpoint")
            normalized = design_model(embedding.to(device)).cpu()
            raw_values_batch = torch.exp(normalized * design_std + design_mean).numpy() * nominal
            for offset, (source, raw_values) in enumerate(zip(batch, raw_values_batch, strict=True)):
                projected, projection = project_structural_feasibility(
                    raw_values, args.projection_margin
                )
                simulation = physics(
                    torch.from_numpy(projected).double().unsqueeze(0),
                    torch.tensor([args.spme_reference_capacity_ah], dtype=torch.float64),
                )
                success = bool(simulation.status.item())
                performance = simulation.performance[0].detach().cpu().numpy()
                record = {
                    **source,
                    "schema": "gradcell.application_requirement_spme_deepseek_evaluation.v1",
                    "prompt_id": str(source.get("prompt_id", begin + offset)),
                    "candidate_design": {
                        "model": "frozen_qwen_embedding_plus_trained_mlp",
                        "checkpoint": str(args.checkpoint),
                        "base_parameter_set": parameter_set,
                        "parameter_names": names,
                        "raw_parameter_values": dict(zip(names, raw_values.tolist(), strict=True)),
                        "raw_parameter_multipliers": dict(
                            zip(names, (raw_values / nominal).tolist(), strict=True)
                        ),
                        "raw_structurally_feasible": structural_feasibility(raw_values),
                        "spme_parameter_values": dict(zip(names, projected.tolist(), strict=True)),
                        "spme_parameter_multipliers": dict(
                            zip(names, (projected / nominal).tolist(), strict=True)
                        ),
                        "spme_structurally_feasible": structural_feasibility(projected),
                        "feasibility_projection": projection,
                    },
                    "spme_evaluation": {
                        "model": "PyBaMM SPMe",
                        "parameter_set": parameter_set,
                        "reference_capacity_ah": args.spme_reference_capacity_ah,
                        "reference_capacity_interpretation": (
                            "Current-scaling reference for the Chen2020 base-cell simulation; "
                            "it is not proof of the requested product-level rated capacity."
                        ),
                        "rates_c": [1.0, 5.0, 6.0],
                        "success": success,
                        "runtime_s": float(simulation.runtime_s.item()),
                        "performance": (
                            dict(zip(performance_fields, performance.tolist(), strict=True))
                            if success
                            else None
                        ),
                        "configuration": {
                            "time_points": args.time_points,
                            "rtol": args.rtol,
                            "atol": args.atol,
                            "current_ramp_time_s": args.current_ramp_time_s,
                        },
                    },
                    "deepseek_judgement": {"status": "pending"},
                }
                previous = previous_by_id.get(record["prompt_id"], {})
                if previous.get("deepseek_judgement", {}).get("status") == "success":
                    record["deepseek_judgement"] = previous["deepseek_judgement"]
                result_rows.append(record)
            write_jsonl(args.output, result_rows)
            print(
                json.dumps(
                    {"stage": "qwen_mlp_spme", "completed": len(result_rows), "total": len(source_rows)}
                ),
                flush=True,
            )

    usage_totals: dict[str, float] = {}
    if not args.skip_deepseek:
        load_env(args.env_file)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"DEEPSEEK_API_KEY is not set; checked the process environment and {args.env_file}"
            )
        api = {
            "api_key": api_key,
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "temperature": args.temperature,
            "max_tokens": args.deepseek_max_tokens,
            "json_mode": args.json_mode,
            "timeout_s": args.timeout_s,
        }
        failures = 0
        for index, row in enumerate(result_rows, start=1):
            if row.get("deepseek_judgement", {}).get("status") == "success":
                continue
            error = None
            for attempt in range(1, args.retries + 1):
                try:
                    judgement, usage = request_judgement(row, api)
                    for name, value in usage.items():
                        usage_totals[name] = usage_totals.get(name, 0.0) + value
                    row["deepseek_judgement"] = {
                        "status": "success",
                        "model": api["model"],
                        "result": judgement,
                        "usage": usage,
                    }
                    error = None
                    break
                except FatalAPIError:
                    write_jsonl(args.output, result_rows)
                    raise
                except (RetryableAPIError, InvalidJudgeResponse) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if error is not None:
                failures += 1
                row["deepseek_judgement"] = {
                    "status": "failed",
                    "model": api["model"],
                    "error": error,
                }
            write_jsonl(args.output, result_rows)
            print(
                json.dumps(
                    {
                        "stage": "deepseek_judge",
                        "completed": index,
                        "total": len(result_rows),
                        "failed": failures,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    successful_judgements = [
        row["deepseek_judgement"]["result"]
        for row in result_rows
        if row["deepseek_judgement"].get("status") == "success"
    ]
    usage_totals = {}
    for row in result_rows:
        for name, value in row.get("deepseek_judgement", {}).get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[name] = usage_totals.get(name, 0.0) + float(value)
    report = {
        "schema": "gradcell.application_requirement_spme_deepseek_report.v1",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "records": len(result_rows),
        "raw_structural_feasibility_rate": float(
            np.mean([row["candidate_design"]["raw_structurally_feasible"] for row in result_rows])
        ),
        "projected_structural_feasibility_rate": float(
            np.mean([row["candidate_design"]["spme_structurally_feasible"] for row in result_rows])
        ),
        "projection_rate": float(
            np.mean([row["candidate_design"]["feasibility_projection"]["applied"] for row in result_rows])
        ),
        "spme_success_rate": float(
            np.mean([row["spme_evaluation"]["success"] for row in result_rows])
        ),
        "deepseek_success_rate": len(successful_judgements) / len(result_rows),
        "verdict_counts": dict(Counter(item["overall_verdict"] for item in successful_judgements)),
        "requirement_status_counts": dict(
            Counter(
                assessment["status"]
                for item in successful_judgements
                for assessment in item["requirement_assessments"]
            )
        ),
        "reported_api_usage": usage_totals,
        "interpretation": (
            "SPMe validates only the supplied short-term Chen2020 discharge metrics. Product "
            "capacity, chemistry, geometry, mass, lifetime, safety, cost, and low-temperature "
            "claims remain not evaluable unless independent evidence is supplied."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.require_deepseek_success and len(successful_judgements) != len(result_rows):
        raise RuntimeError("At least one DeepSeek judgement failed; checkpoints were preserved")


if __name__ == "__main__":
    main()
