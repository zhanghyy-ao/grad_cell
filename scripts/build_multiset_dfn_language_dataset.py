from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
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


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


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


def performance_close(
    left: dict[str, float],
    right: dict[str, float],
    relative_tolerances: dict[str, float],
    absolute_tolerances: dict[str, float],
) -> tuple[bool, float]:
    normalized_errors = []
    for name, tolerance in relative_tolerances.items():
        denominator = max(abs(left[name]), abs(right[name]), 1e-12)
        normalized_errors.append(abs(left[name] - right[name]) / denominator / tolerance)
    for name, tolerance in absolute_tolerances.items():
        normalized_errors.append(abs(left[name] - right[name]) / tolerance)
    distance = max(normalized_errors, default=0.0)
    return distance <= 1.0, float(distance)


def design_separation(
    left: dict[str, Any], right: dict[str, Any], settings: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    left_values = np.asarray(left["parameter_multipliers"], dtype=np.float64)
    right_values = np.asarray(right["parameter_multipliers"], dtype=np.float64)
    if left_values.shape != right_values.shape or np.any(left_values <= 0) or np.any(
        right_values <= 0
    ):
        raise ValueError("Parameter multipliers must be matching positive vectors")
    log_rms = float(np.sqrt(np.mean(np.square(np.log(left_values / right_values)))))
    fractional = np.maximum(left_values, right_values) / np.minimum(
        left_values, right_values
    ) - 1.0
    maximum_fraction = float(np.max(fractional))
    changed_count = int(
        np.count_nonzero(fractional >= float(settings["changed_parameter_fraction"]))
    )
    separated = (
        log_rms >= float(settings["minimum_log_rms"])
        or maximum_fraction >= float(settings["minimum_maximum_fraction"])
        or changed_count >= int(settings["minimum_changed_parameters"])
    )
    return separated, {
        "log_multiplier_rms": log_rms,
        "maximum_parameter_fraction": maximum_fraction,
        "changed_parameter_count": changed_count,
    }


def build_ambiguity_index(
    physics_rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Find DFN-verified designs that are indistinguishable at configured precision."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "Inverse ambiguity analysis requires scipy; install gradcell[physics]"
        ) from exc

    settings = config["inverse_ambiguity"]
    relative = {
        name: float(value) for name, value in settings["relative_tolerances"].items()
    }
    absolute = {
        name: float(value) for name, value in settings["absolute_tolerances"].items()
    }
    unknown = (set(relative) | set(absolute)) - set(CORE_METRICS)
    overlap = set(relative) & set(absolute)
    if unknown or overlap or not relative or not absolute:
        raise ValueError(
            f"Invalid ambiguity metrics; unknown={sorted(unknown)}, "
            f"overlap={sorted(overlap)}"
        )
    if any(value <= 0 or value >= 1 for value in relative.values()) or any(
        value <= 0 for value in absolute.values()
    ):
        raise ValueError("Ambiguity tolerances must be positive; relative values must be < 1")

    by_condition: dict[tuple[str, float], list[int]] = {}
    required_physics = config["physics"]
    for index, row in enumerate(physics_rows):
        if row.get("model") != "DFN":
            raise ValueError(f"Ambiguity analysis requires DFN rows: {row.get('case_id')}")
        simulation = row["simulation"]
        strict_enough = (
            float(simulation["rtol"]) <= float(required_physics["rtol"])
            and float(simulation["atol"]) <= float(required_physics["atol"])
            and int(simulation["time_points"]) >= int(required_physics["time_points"])
        )
        if not strict_enough:
            raise ValueError(
                f"Physics row {row.get('case_id')} is less strict than the configured "
                "DFN verification settings; regenerate the archive before ambiguity analysis"
            )
        key = (
            row["parameter_set"],
            round(float(simulation["temperature_k"]), 8),
        )
        by_condition.setdefault(key, []).append(index)

    disjoint = DisjointSet(len(physics_rows))
    neighbors: list[list[dict[str, Any]]] = [[] for _ in physics_rows]
    performance_pair_count = 0
    separated_pair_count = 0
    relative_names = list(relative)
    absolute_names = list(absolute)
    for indices in by_condition.values():
        features = []
        for index in indices:
            performance = physics_rows[index]["performance"]
            row_features = [
                np.log(max(float(performance[name]), 1e-300))
                / -np.log1p(-relative[name])
                for name in relative_names
            ]
            row_features.extend(
                float(performance[name]) / absolute[name] for name in absolute_names
            )
            features.append(row_features)
        tree = cKDTree(np.asarray(features, dtype=np.float64))
        for local_left, local_right in sorted(tree.query_pairs(r=1.0, p=np.inf)):
            left_index = indices[local_left]
            right_index = indices[local_right]
            left = physics_rows[left_index]
            right = physics_rows[right_index]
            if left["physical_design_id"] == right["physical_design_id"]:
                continue
            close, performance_distance = performance_close(
                left["performance"], right["performance"], relative, absolute
            )
            if not close:
                continue
            performance_pair_count += 1
            # Performance-near designs share a split even when structures are also near.
            disjoint.union(left_index, right_index)
            separated, separation = design_separation(left, right, settings)
            if not separated:
                continue
            separated_pair_count += 1
            for source_index, target_index in (
                (left_index, right_index),
                (right_index, left_index),
            ):
                target = physics_rows[target_index]
                neighbors[source_index].append(
                    {
                        "physical_design_id": target["physical_design_id"],
                        "physics_source": target["case_id"],
                        "performance_distance": performance_distance,
                        **separation,
                        "teacher_design": design_payload(target),
                    }
                )

    components: dict[int, list[int]] = {}
    for index in range(len(physics_rows)):
        components.setdefault(disjoint.find(index), []).append(index)
    group_ids = {}
    for members in components.values():
        member_ids = sorted(physics_rows[index]["physical_design_id"] for index in members)
        digest = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:20]
        for index in members:
            group_ids[index] = f"performance-equivalence-{digest}"

    maximum_alternatives = int(settings["maximum_alternatives_per_design"])
    ambiguity_by_id = {}
    ambiguous_designs = 0
    alternative_counts = []
    audit_rows = []
    for index, row in enumerate(physics_rows):
        alternatives = sorted(
            neighbors[index],
            key=lambda item: (
                item["performance_distance"],
                -item["log_multiplier_rms"],
                item["physical_design_id"],
            ),
        )[:maximum_alternatives]
        group_size = len(components[disjoint.find(index)])
        if alternatives:
            ambiguous_designs += 1
        alternative_counts.append(len(alternatives))
        value = {
            "equivalence_group_id": group_ids[index],
            "performance_equivalence_group_size": group_size,
            "is_one_to_many": bool(alternatives),
            "alternative_design_count": len(alternatives),
            "alternative_teacher_designs": alternatives,
        }
        ambiguity_by_id[row["physical_design_id"]] = value
        audit_rows.append(
            {
                "physical_design_id": row["physical_design_id"],
                "physics_source": row["case_id"],
                "parameter_set": row["parameter_set"],
                **value,
            }
        )
    counts = np.asarray(alternative_counts, dtype=np.float64)
    grouped_rates = {}
    for field in ("parameter_set", "mode"):
        grouped_rates[f"by_{field}"] = {}
        for name in sorted({str(row[field]) for row in physics_rows}):
            member_indices = [
                index for index, row in enumerate(physics_rows) if str(row[field]) == name
            ]
            ambiguous = sum(bool(neighbors[index]) for index in member_indices)
            grouped_rates[f"by_{field}"][name] = {
                "physical_designs": len(member_indices),
                "ambiguous_designs": ambiguous,
                "ambiguity_rate": ambiguous / max(len(member_indices), 1),
            }
    summary = {
        "schema": "gradcell.inverse_ambiguity_report.v1",
        "verification_basis": (
            "Each source and alternative design is an independently simulated row "
            "from the strict DFN physics archive."
        ),
        "physical_designs": len(physics_rows),
        "performance_near_pair_count": performance_pair_count,
        "structurally_separated_pair_count": separated_pair_count,
        "equivalence_groups": len(components),
        "ambiguous_designs": ambiguous_designs,
        "ambiguity_rate": ambiguous_designs / max(len(physics_rows), 1),
        "alternatives_per_design_mean": float(counts.mean()) if len(counts) else 0.0,
        "alternatives_per_design_p50": float(np.quantile(counts, 0.5)) if len(counts) else 0.0,
        "alternatives_per_design_p90": float(np.quantile(counts, 0.9)) if len(counts) else 0.0,
        "relative_tolerances": relative,
        "absolute_tolerances": absolute,
        "design_separation": {
            name: settings[name]
            for name in (
                "minimum_log_rms",
                "minimum_maximum_fraction",
                "changed_parameter_fraction",
                "minimum_changed_parameters",
            )
        },
        **grouped_rates,
        "rows": audit_rows,
    }
    return ambiguity_by_id, summary


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
    physics_rows: list[dict[str, Any]],
    config: dict[str, Any],
    ambiguity_by_id: dict[str, dict[str, Any]],
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
        ambiguity = ambiguity_by_id[row["physical_design_id"]]
        split = split_for_design(ambiguity["equivalence_group_id"], seed, ratios)
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
                    "alternative_teacher_designs": ambiguity[
                        "alternative_teacher_designs"
                    ],
                    "inverse_ambiguity": {
                        name: ambiguity[name]
                        for name in (
                            "equivalence_group_id",
                            "performance_equivalence_group_size",
                            "is_one_to_many",
                            "alternative_design_count",
                        )
                    },
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


def protected_placeholders(observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    replacements = {
        "[[PARAMETER_SET]]": str(observation["parameter_set"]),
        "[[TEMPERATURE_K]]": f"{observation['temperature_k']:.2f}",
        **{
            f"[[{name.upper()}]]": f"{observation['performance'][name]:.6f}"
            for name in CORE_METRICS
        },
    }
    facts = {
        "parameter_set": "[[PARAMETER_SET]]",
        "temperature_k": "[[TEMPERATURE_K]]",
        "performance": {
            name: f"[[{name.upper()}]]" for name in CORE_METRICS
        },
    }
    return facts, replacements


def restore_placeholders(text: str, replacements: dict[str, str]) -> str:
    missing = [token for token in replacements if text.count(token) != 1]
    unknown = sorted(set(re.findall(r"\[\[[A-Z0-9_]+\]\]", text)) - set(replacements))
    if missing or unknown:
        raise InvalidLanguageResponse(
            "Placeholder validation failed; each required token must occur exactly once; "
            f"missing_or_repeated={missing}, unknown={unknown}; preview={text[:160]!r}"
        )
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


class FatalLanguageAPIError(RuntimeError):
    """An API error that must stop the batch immediately, such as exhausted credit."""


class RetryableLanguageAPIError(RuntimeError):
    """A transient API error for which a bounded retry is appropriate."""


class InvalidLanguageResponse(ValueError):
    """A completed, billable response that cannot be accepted as a description."""

    def __init__(self, message: str, usage: dict[str, float] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}


def response_text(content: Any) -> str:
    """Accept OpenAI-style text, JSON, fenced JSON, or multipart text content."""
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    if not isinstance(content, str) or not content.strip():
        raise InvalidLanguageResponse("API returned empty message.content")
    value = content.strip()
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Some OpenAI-compatible endpoints ignore response_format and return
        # the requested paragraph directly. Plain text is valid for this task.
        if value.startswith(("{", "[")):
            raise InvalidLanguageResponse(
                f"Malformed JSON response: {value[:160]!r}"
            ) from None
        return value
    if isinstance(parsed, dict):
        text = parsed.get("battery_description")
        if not isinstance(text, str) or not text.strip():
            raise InvalidLanguageResponse(
                "JSON response lacks a non-empty battery_description field"
            )
        return text.strip()
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    raise InvalidLanguageResponse(
        f"Unsupported response JSON type: {type(parsed).__name__}"
    )


def validate_generated_description(text: str, anchors: list[str]) -> str:
    missing = [anchor for anchor in anchors if anchor not in text]
    forbidden = (
        "希望",
        "要求",
        "目标",
        "不低于",
        "至少",
        "优化",
        "multiplier",
        "teacher",
        "设计建议",
    )
    found_forbidden = [word for word in forbidden if word in text]
    if missing or found_forbidden:
        raise InvalidLanguageResponse(
            f"Description validation failed; missing={missing}, "
            f"forbidden={found_forbidden}; preview={text[:160]!r}"
        )
    return text


def deepseek_description(
    record: dict[str, Any], language: dict[str, Any]
) -> tuple[str, dict[str, float]]:
    if not language["api_key"]:
        raise RuntimeError(f"Environment variable {language['api_key_env']} is not set")
    observation = record["observation_canonical"]
    anchors = protected_anchors(observation)
    placeholder_facts, replacements = protected_placeholders(observation)
    json_mode = bool(language.get("request_json_mode", False))
    prompt = {
        "task": "把占位符形式的DFN事实组织成一段自然、专业且连贯的中文电池描述",
        "rules": [
            "这是对一块已有电池的客观描述，不是用户需求或设计请求",
            "不得使用希望、要求、目标、至少、不低于、优化等需求措辞",
            "不得推测寿命、安全、成本、材料成分或未提供的性能",
            "不得出现内部参数、multiplier、teacher、答案或设计建议",
            "必须原样保留每个[[...]]占位符且每个占位符恰好出现一次",
            "不要自行编写、修改或猜测任何数值，真实数值将由程序填回",
            (
                "只返回含battery_description字段的JSON对象"
                if json_mode
                else "只返回描述正文，不要JSON、Markdown、标题或解释"
            ),
        ],
        "placeholder_facts": placeholder_facts,
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
    }
    # Omit max_tokens when it is null so reasoning-capable providers can use
    # their native output budget for both reasoning and the final answer.
    max_tokens = language.get("max_tokens")
    if max_tokens is not None:
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError("language.max_tokens must be positive or null")
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        language["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {language['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=float(language["timeout_s"])
        ) as response:
            raw_result = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            detail = ""
        message = f"HTTP {exc.code}: {detail or exc.reason}"
        if exc.code in (401, 402, 403):
            raise FatalLanguageAPIError(message) from exc
        if exc.code in (408, 409, 425, 429) or exc.code >= 500:
            raise RetryableLanguageAPIError(message) from exc
        raise InvalidLanguageResponse(message) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableLanguageAPIError(f"{type(exc).__name__}: {exc}") from exc
    try:
        result = json.loads(raw_result)
        content = result["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise InvalidLanguageResponse(
            f"Invalid API response envelope: {raw_result[:160]!r}"
        ) from exc
    raw_usage = result.get("usage", {})
    usage = {
        str(name): float(value)
        for name, value in raw_usage.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    try:
        text = response_text(content)
        text = restore_placeholders(text, replacements)
        return validate_generated_description(text, anchors), usage
    except InvalidLanguageResponse as exc:
        exc.usage = usage
        raise


def apply_deepseek(
    records: list[dict[str, Any]],
    language: dict[str, Any],
    checkpoint: Path,
    require_success: bool,
    maximum_records: int | None = None,
) -> dict[str, Any]:
    if maximum_records is not None and maximum_records <= 0:
        raise ValueError("maximum_records must be positive when provided")
    if int(language["concurrency"]) <= 0 or int(language["retries"]) <= 0:
        raise ValueError("DeepSeek concurrency and retries must be positive")
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
    all_pending = [
        record
        for record in records
        if record["provenance"]["language_source"] != language["model"]
    ]
    pending = all_pending[:maximum_records] if maximum_records is not None else all_pending

    def rewrite(
        record: dict[str, Any],
    ) -> tuple[
        dict[str, Any], str | None, dict[str, float], str | None, bool
    ]:
        last_error = None
        for attempt in range(1, int(language["retries"]) + 1):
            try:
                text, usage = deepseek_description(record, language)
                return record, text, usage, None, False
            except FatalLanguageAPIError as exc:
                return record, None, {}, f"{type(exc).__name__}: {exc}", True
            except InvalidLanguageResponse as exc:
                # The provider has already billed this completed generation.
                # Do not pay for repeated format retries.
                return (
                    record,
                    None,
                    exc.usage,
                    f"{type(exc).__name__}: {exc}",
                    False,
                )
            except RetryableLanguageAPIError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < int(language["retries"]):
                    time.sleep(min(2 ** (attempt - 1), 8))
        return record, None, {}, last_error, False

    failures = 0
    completed = 0
    consecutive_failures = 0
    abort_error = None
    usage_totals: dict[str, float] = {}
    concurrency = int(language["concurrency"])
    submission_batch_size = max(
        concurrency, int(language.get("submission_batch_size", concurrency * 2))
    )
    probe_records = int(language.get("failure_probe_records", 20))
    maximum_failure_rate = float(language.get("maximum_failure_rate", 0.25))
    maximum_consecutive_failures = int(
        language.get("maximum_consecutive_failures", 8)
    )
    with ThreadPoolExecutor(max_workers=int(language["concurrency"])) as executor:
        for start in range(0, len(pending), submission_batch_size):
            batch = pending[start : start + submission_batch_size]
            futures = [executor.submit(rewrite, record) for record in batch]
            fatal_in_batch = None
            for future in as_completed(futures):
                record, text, usage, error, fatal = future.result()
                for name, value in usage.items():
                    usage_totals[name] = usage_totals.get(name, 0.0) + value
                if usage:
                    record["provenance"]["language_usage"] = usage
                if text is None:
                    failures += 1
                    consecutive_failures += 1
                    flags = record.setdefault("quality_flags", [])
                    if "deepseek_description_failed" not in flags:
                        flags.append("deepseek_description_failed")
                    record["provenance"]["language_error"] = error
                    if fatal:
                        fatal_in_batch = error
                else:
                    consecutive_failures = 0
                    record["battery_description"] = text
                    record["messages"][1]["content"] = text
                    record["provenance"]["language_source"] = language["model"]
                    record["provenance"].pop("language_error", None)
                    remaining_flags = [
                        flag
                        for flag in record.get("quality_flags", [])
                        if flag != "deepseek_description_failed"
                    ]
                    if remaining_flags:
                        record["quality_flags"] = remaining_flags
                    else:
                        record.pop("quality_flags", None)
                completed += 1
            write_jsonl(checkpoint, records)
            print(f"DeepSeek descriptions: {completed}/{len(pending)}", flush=True)
            if fatal_in_batch:
                abort_error = (
                    "DeepSeek batch stopped after a fatal API error: " + fatal_in_batch
                )
                break
            observed_failure_rate = failures / max(completed, 1)
            if consecutive_failures >= maximum_consecutive_failures:
                abort_error = (
                    "DeepSeek batch stopped by consecutive-failure circuit breaker: "
                    f"{consecutive_failures} failures"
                )
                break
            if completed >= probe_records and observed_failure_rate > maximum_failure_rate:
                abort_error = (
                    "DeepSeek batch stopped by failure-rate circuit breaker: "
                    f"{failures}/{completed}={observed_failure_rate:.1%} > "
                    f"{maximum_failure_rate:.1%}"
                )
                break
    write_jsonl(checkpoint, records)
    result = {
        "model": language["model"],
        "pending_before_limit": len(all_pending),
        "selected": len(pending),
        "completed": completed,
        "succeeded": completed - failures,
        "failed": failures,
        "unattempted": len(all_pending) - completed,
        "aborted": bool(abort_error),
        "reported_usage": usage_totals,
    }
    if abort_error:
        raise RuntimeError(f"{abort_error}; checkpoint was preserved; summary={result}")
    if failures and require_success:
        raise RuntimeError(f"DeepSeek failed for {failures} records; checkpoint was preserved")
    return result


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [row["task_id"] for row in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values are not unique")
    design_splits: dict[str, set[str]] = {}
    equivalence_splits: dict[str, set[str]] = {}
    for row in records:
        design_splits.setdefault(row["physical_design_id"], set()).add(row["split"])
        group_id = row["inverse_ambiguity"]["equivalence_group_id"]
        equivalence_splits.setdefault(group_id, set()).add(row["split"])
        json.loads(row["messages"][2]["content"])
    leaked = [design_id for design_id, splits in design_splits.items() if len(splits) > 1]
    if leaked:
        raise RuntimeError(f"physical-design split leakage detected for {len(leaked)} designs")
    equivalence_leaks = [
        group_id for group_id, splits in equivalence_splits.items() if len(splits) > 1
    ]
    if equivalence_leaks:
        raise RuntimeError(
            "performance-equivalence split leakage detected for "
            f"{len(equivalence_leaks)} groups"
        )
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
        "one_to_many_records": sum(
            bool(row["inverse_ambiguity"]["is_one_to_many"]) for row in records
        ),
        "performance_equivalence_groups": len(equivalence_splits),
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
    parser.add_argument(
        "--deepseek-max-records",
        type=int,
        help="Only rewrite this many pending rows, for a low-cost smoke test.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    archive = args.archive or Path(config["physics"]["output"])
    output = args.output or Path(config["dataset"]["output"])
    physics_rows = read_jsonl(archive)
    if not physics_rows:
        raise ValueError(f"Physics archive is empty: {archive}")
    ambiguity_by_id, ambiguity_summary = build_ambiguity_index(physics_rows, config)
    records = build_records(physics_rows, config, ambiguity_by_id)
    deepseek_run = None
    if args.with_deepseek:
        deepseek_run = apply_deepseek(
            records,
            language_config(config),
            output,
            require_success=args.require_deepseek_success,
            maximum_records=args.deepseek_max_records,
        )
    write_jsonl(output, records)
    ambiguity_audit = Path(config["inverse_ambiguity"]["audit"])
    write_jsonl(ambiguity_audit, ambiguity_summary.pop("rows"))
    ambiguity_report = Path(config["inverse_ambiguity"]["report"])
    atomic_json(ambiguity_report, ambiguity_summary)
    validation = validate_dataset(records)
    manifest = {
        "schema": "gradcell.battery_description_manifest.v1",
        "task_definition": "battery_description_to_parameter_design",
        "config": str(args.config),
        "physics_archive": str(archive),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "deepseek_requested": bool(args.with_deepseek),
        "deepseek_run": deepseek_run,
        "validation": validation,
        "inverse_ambiguity_audit": str(ambiguity_audit),
        "inverse_ambiguity_report": str(ambiguity_report),
        "inverse_ambiguity": ambiguity_summary,
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
