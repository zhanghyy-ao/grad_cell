from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


CORE_METRICS = (
    "reference_capacity_ah",
    "energy_1c_wh",
    "capacity_1c_ah",
    "average_voltage_1c_v",
    "minimum_voltage_1c_v",
    "discharge_time_1c_s",
    "energy_5c_wh",
    "capacity_5c_ah",
    "average_voltage_5c_v",
    "discharge_time_5c_s",
    "energy_retention_5c",
    "energy_6c_wh",
    "capacity_6c_ah",
    "average_voltage_6c_v",
    "discharge_time_6c_s",
    "energy_retention_6c",
)


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def split_for_design(design_id: str, seed: int, ratios: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{design_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    boundary = 0.0
    for split in ("train", "validation", "test"):
        boundary += ratios[split]
        if value < boundary:
            return split
    return "test"


def design_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "gradcell.multiset_parameter_design.v1",
        "base_parameter_set": row["parameter_set"],
        "generation_mode": row["mode"],
        "parameter_updates": {
            name: {"multiplier": float(multiplier), "value": float(value)}
            for name, multiplier, value in zip(
                row["parameter_names"],
                row["parameter_multipliers"],
                row["parameter_values"],
                strict=True,
            )
        },
    }


def observation_payload(row: dict[str, Any]) -> dict[str, Any]:
    performance = row["performance"]
    missing = sorted(set(CORE_METRICS) - set(performance))
    if missing:
        raise ValueError(
            f"Physics row {row['case_id']} lacks description metrics: {missing}. "
            "Regenerate the physics archive with generate_multiset_dfn_archive.py."
        )
    return {
        "parameter_set": row["parameter_set"],
        "temperature_k": float(row["simulation"]["temperature_k"]),
        "discharge_protocol": "constant_current_1C_5C_6C_to_voltage_cutoff",
        "performance": {name: float(performance[name]) for name in CORE_METRICS},
    }


def deterministic_description(observation: dict[str, Any], variant: int) -> str:
    values = {
        "parameter_set": observation["parameter_set"],
        "temperature": observation["temperature_k"],
        **observation["performance"],
    }
    templates = (
        (
            "这是一块采用{parameter_set}参数体系的电池，在{temperature:.2f} K下进行恒流"
            "放电。其参考容量为{reference_capacity_ah:.6f} Ah；1C时能量为"
            "{energy_1c_wh:.6f} Wh、容量为{capacity_1c_ah:.6f} Ah、平均电压为"
            "{average_voltage_1c_v:.6f} V、最低电压{minimum_voltage_1c_v:.6f} V，"
            "放电持续{discharge_time_1c_s:.6f} s。5C时容量{capacity_5c_ah:.6f} Ah、"
            "能量{energy_5c_wh:.6f} Wh、平均电压{average_voltage_5c_v:.6f} V、"
            "持续{discharge_time_5c_s:.6f} s、能量保持率{energy_retention_5c:.6f}；"
            "6C时容量{capacity_6c_ah:.6f} Ah、能量{energy_6c_wh:.6f} Wh、平均电压"
            "{average_voltage_6c_v:.6f} V、持续{discharge_time_6c_s:.6f} s、"
            "能量保持率{energy_retention_6c:.6f}。"
        ),
        (
            "在{temperature:.2f} K测试了一块{parameter_set}体系电池。1C测试得到"
            "{capacity_1c_ah:.6f} Ah和{energy_1c_wh:.6f} Wh，平均工作电压"
            "{average_voltage_1c_v:.6f} V，截止前运行{discharge_time_1c_s:.6f} s。"
            "最低电压{minimum_voltage_1c_v:.6f} V，参考容量为"
            "{reference_capacity_ah:.6f} Ah。5C下输出{capacity_5c_ah:.6f} Ah和"
            "{energy_5c_wh:.6f} Wh，平均电压{average_voltage_5c_v:.6f} V，"
            "运行{discharge_time_5c_s:.6f} s，能量保持率{energy_retention_5c:.6f}。"
            "6C下输出{capacity_6c_ah:.6f} Ah和{energy_6c_wh:.6f} Wh，平均电压"
            "{average_voltage_6c_v:.6f} V，运行{discharge_time_6c_s:.6f} s，"
            "能量保持率{energy_retention_6c:.6f}。"
        ),
        (
            "该电芯属于{parameter_set}体系，环境温度{temperature:.2f} K。DFN放电表征显示："
            "参考容量{reference_capacity_ah:.6f} Ah；1C、5C、6C能量依次为"
            "{energy_1c_wh:.6f}、{energy_5c_wh:.6f}、{energy_6c_wh:.6f} Wh，"
            "相应容量为{capacity_1c_ah:.6f}、{capacity_5c_ah:.6f}、"
            "{capacity_6c_ah:.6f} Ah，平均电压为{average_voltage_1c_v:.6f}、"
            "{average_voltage_5c_v:.6f}、{average_voltage_6c_v:.6f} V，放电时间为"
            "{discharge_time_1c_s:.6f}、{discharge_time_5c_s:.6f}、"
            "{discharge_time_6c_s:.6f} s。1C最低电压为{minimum_voltage_1c_v:.6f} V，"
            "5C/6C能量保持率为{energy_retention_5c:.6f}/"
            "{energy_retention_6c:.6f}。"
        ),
    )
    return templates[variant % len(templates)].format(**values)


def build_records(
    physics_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    variants = int(config["dataset"]["variants_per_design"])
    ratios = {
        name: float(config["split"][name])
        for name in ("train", "validation", "test")
    }
    if not abs(sum(ratios.values()) - 1.0) < 1e-9:
        raise ValueError("split ratios must sum to 1")
    records: list[dict[str, Any]] = []
    for row in physics_rows:
        observation = observation_payload(row)
        target = design_payload(row)
        split = split_for_design(row["physical_design_id"], seed, ratios)
        for variant in range(variants):
            description = deterministic_description(observation, variant)
            task_id = f"battery-description-{row['physical_design_id']}-v{variant:02d}"
            records.append(
                {
                    "schema": "gradcell.battery_description_training.v1",
                    "task_id": task_id,
                    "description_family_id": row["physical_design_id"],
                    "physical_design_id": row["physical_design_id"],
                    "generation_mode": row["mode"],
                    "split": split,
                    "battery_description": description,
                    "observation_canonical": observation,
                    "teacher_design": target,
                    "verified_performance": row["performance"],
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "根据电池体系、工况和性能描述还原严格JSON参数设计；"
                                "不得输出JSON之外的内容。"
                            ),
                        },
                        {"role": "user", "content": description},
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                target,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    "provenance": {
                        "physics_model": "DFN",
                        "physics_source": row["case_id"],
                        "language_source": "deterministic_template",
                    },
                }
            )
    return records


def language_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config["language"]
    return {
        **value,
        "api_key": os.environ.get(value["api_key_env"], ""),
        "base_url": os.environ.get(value["base_url_env"], value["default_base_url"]),
        "model": os.environ.get(value["model_env"], value["default_model"]),
        "concurrency": int(
            os.environ.get(value["concurrency_env"], value["default_concurrency"])
        ),
    }


def protected_anchors(observation: dict[str, Any]) -> list[str]:
    performance = observation["performance"]
    return [
        observation["parameter_set"],
        f"{observation['temperature_k']:.2f}",
        *[f"{performance[name]:.6f}" for name in CORE_METRICS],
    ]


def deepseek_description(record: dict[str, Any], language: dict[str, Any]) -> str:
    if not language["api_key"]:
        raise RuntimeError(f"Environment variable {language['api_key_env']} is not set")
    anchors = protected_anchors(record["observation_canonical"])
    prompt = {
        "task": "把结构化DFN观测改写成一段自然、专业且连贯的中文电池描述",
        "rules": [
            "这是对一块已有电池的客观描述，不是用户需求或设计请求",
            "不得使用希望、要求、目标、至少、不低于、优化等需求措辞",
            "不得推测寿命、安全、成本、材料成分或未提供的性能",
            "不得出现内部参数、multiplier、teacher、答案或设计建议",
            "必须原样保留所有protected_anchors",
            "只返回含battery_description字段的JSON对象",
        ],
        "protected_anchors": anchors,
        "observation": record["observation_canonical"],
        "variant_id": record["task_id"],
    }
    payload = {
        "model": language["model"],
        "messages": [
            {
                "role": "system",
                "content": "你只负责忠实描述已有电池，不负责提出需求或泄露设计参数。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": float(language["temperature"]),
        "max_tokens": int(language["max_tokens"]),
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        language["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {language['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(language["timeout_s"])) as response:
        result = json.loads(response.read().decode("utf-8"))
    parsed = json.loads(result["choices"][0]["message"]["content"])
    text = str(parsed["battery_description"]).strip()
    missing = [anchor for anchor in anchors if anchor not in text]
    forbidden = ("希望", "要求", "目标", "不低于", "至少", "优化")
    found_forbidden = [word for word in forbidden if word in text]
    if missing or found_forbidden:
        raise ValueError(
            f"Invalid description; missing={missing}, forbidden={found_forbidden}"
        )
    return text


def apply_deepseek(
    records: list[dict[str, Any]],
    language: dict[str, Any],
    checkpoint: Path,
    require_success: bool,
) -> None:
    prior = (
        {row["task_id"]: row for row in read_jsonl(checkpoint)}
        if checkpoint.exists()
        else {}
    )
    for record in records:
        old = prior.get(record["task_id"])
        if old and old.get("provenance", {}).get("language_source") == language["model"]:
            record["battery_description"] = old["battery_description"]
            record["messages"][1]["content"] = old["battery_description"]
            record["provenance"] = old["provenance"]
    pending = [
        record
        for record in records
        if record["provenance"]["language_source"] != language["model"]
    ]

    def rewrite(record: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
        last_error = None
        for attempt in range(1, int(language["retries"]) + 1):
            try:
                return record, deepseek_description(record, language), None
            except (
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                json.JSONDecodeError,
                urllib.error.URLError,
                TimeoutError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** (attempt - 1), 8))
        return record, None, last_error

    failures = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=int(language["concurrency"])) as executor:
        futures = [executor.submit(rewrite, record) for record in pending]
        for future in as_completed(futures):
            record, text, error = future.result()
            if text is None:
                failures += 1
                record.setdefault("quality_flags", []).append("deepseek_description_failed")
                record["provenance"]["language_error"] = error
            else:
                record["battery_description"] = text
                record["messages"][1]["content"] = text
                record["provenance"]["language_source"] = language["model"]
                record["provenance"].pop("language_error", None)
                record.pop("quality_flags", None)
            completed += 1
            if completed % 25 == 0:
                write_jsonl(checkpoint, records)
                print(f"DeepSeek descriptions: {completed}/{len(pending)}", flush=True)
    write_jsonl(checkpoint, records)
    if failures and require_success:
        raise RuntimeError(f"DeepSeek failed for {failures} records; checkpoint was preserved")


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [row["task_id"] for row in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values are not unique")
    design_splits: dict[str, set[str]] = {}
    for row in records:
        design_splits.setdefault(row["physical_design_id"], set()).add(row["split"])
        json.loads(row["messages"][2]["content"])
    leaked = [design_id for design_id, splits in design_splits.items() if len(splits) > 1]
    if leaked:
        raise RuntimeError(f"physical-design split leakage detected for {len(leaked)} designs")
    parameter_sets = sorted(
        {row["observation_canonical"]["parameter_set"] for row in records}
    )
    return {
        "records": len(records),
        "physical_designs": len(design_splits),
        "split_counts": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "validation", "test")
        },
        "parameter_set_counts": {
            name: sum(
                row["observation_canonical"]["parameter_set"] == name for row in records
            )
            for name in parameter_sets
        },
        "deepseek_records": sum(
            row["provenance"]["language_source"] != "deterministic_template"
            for row in records
        ),
        "quality_flagged_records": sum(bool(row.get("quality_flags")) for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe simulated batteries in natural language for inverse design training."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multiset_dfn_language.yaml")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-deepseek", action="store_true")
    parser.add_argument("--require-deepseek-success", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    archive = args.archive or Path(config["physics"]["output"])
    output = args.output or Path(config["dataset"]["output"])
    physics_rows = read_jsonl(archive)
    if not physics_rows:
        raise ValueError(f"Physics archive is empty: {archive}")
    records = build_records(physics_rows, config)
    if args.with_deepseek:
        apply_deepseek(
            records,
            language_config(config),
            output,
            require_success=args.require_deepseek_success,
        )
    write_jsonl(output, records)
    validation = validate_dataset(records)
    manifest = {
        "schema": "gradcell.battery_description_manifest.v1",
        "task_definition": "battery_description_to_parameter_design",
        "config": str(args.config),
        "physics_archive": str(archive),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "deepseek_requested": bool(args.with_deepseek),
        "validation": validation,
        "identifiability_note": (
            "The mapping from finite performance summaries to seven parameters may be "
            "one-to-many; teacher labels are simulator-consistent reconstructions, not "
            "proof of unique physical identifiability."
        ),
    }
    atomic_json(Path(config["dataset"]["manifest"]), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
