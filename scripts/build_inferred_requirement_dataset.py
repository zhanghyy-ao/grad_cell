from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


DEMAND_WORDS = ("希望", "需要", "要求", "目标", "优先", "兼顾", "倾向")
FORBIDDEN_OUTPUT_WORDS = (
    "porosity",
    "diffusivity",
    "volume fraction",
    "multiplier",
    "teacher",
    "孔隙率",
    "扩散率",
    "体积分数",
    "Chen2020",
    "ORegan2022",
    "Prada2013",
    "Ecker2015",
    "Marquis2019",
)


class FatalAPIError(RuntimeError):
    pass


class RetryableAPIError(RuntimeError):
    pass


class InvalidLanguageResponse(ValueError):
    def __init__(self, message: str, usage: dict[str, float] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_rank(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.size <= 1:
        return 0.5
    less = float(np.count_nonzero(reference < value))
    equal = float(np.count_nonzero(reference == value))
    return (less + 0.5 * equal) / reference.size


def level_for_percentile(value: float) -> str:
    if value < 0.10:
        return "基础"
    if value < 0.35:
        return "中低"
    if value < 0.65:
        return "中等"
    if value < 0.90:
        return "中高"
    return "高"


def design_emphasis(row: dict[str, Any]) -> str:
    updates = row["teacher_design"]["parameter_updates"]
    changes = {
        name: abs(float(value["multiplier"]) - 1.0) for name, value in updates.items()
    }
    strongest = max(changes, key=changes.get)
    if "diffusivity" in strongest.lower():
        return "动力学与倍率响应"
    if "active material" in strongest.lower():
        return "可用容量与能量输出"
    if "porosity" in strongest.lower():
        return "传输能力与材料利用率折中"
    return "综合性能平衡"


def build_references(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["physical_design_id"]), row)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        teacher = row["teacher_design"]
        grouped[(str(teacher["base_parameter_set"]), str(teacher["generation_mode"]))].append(
            row
        )
    fields = (
        "capacity_1c_ah",
        "energy_1c_wh",
        "average_voltage_1c_v",
        "minimum_voltage_1c_v",
        "energy_retention_5c",
        "energy_retention_6c",
        "energy_5c_wh",
        "energy_6c_wh",
    )
    return {
        key: {
            field: np.asarray(
                [float(row["verified_performance"][field]) for row in members],
                dtype=np.float64,
            )
            for field in fields
        }
        for key, members in grouped.items()
    }


def derive_requirement_profile(
    row: dict[str, Any], references: dict[tuple[str, str], dict[str, np.ndarray]]
) -> dict[str, Any]:
    teacher = row["teacher_design"]
    key = (str(teacher["base_parameter_set"]), str(teacher["generation_mode"]))
    reference = references[key]
    performance = row["verified_performance"]
    percentiles = {
        name: percentile_rank(float(performance[name]), values)
        for name, values in reference.items()
    }
    energy_score = float(
        np.mean([percentiles["capacity_1c_ah"], percentiles["energy_1c_wh"]])
    )
    rate_score = float(
        np.mean(
            [
                percentiles["energy_retention_5c"],
                percentiles["energy_retention_6c"],
                percentiles["energy_5c_wh"],
                percentiles["energy_6c_wh"],
            ]
        )
    )
    voltage_score = float(
        np.mean(
            [percentiles["average_voltage_1c_v"], percentiles["minimum_voltage_1c_v"]]
        )
    )
    if abs(energy_score - rate_score) <= 0.12:
        primary_goal = "能量与高倍率均衡"
    elif energy_score > rate_score:
        primary_goal = "常规倍率能量与容量优先"
    else:
        primary_goal = "高倍率输出优先"
    strictness = level_for_percentile(max(energy_score, rate_score, voltage_score))
    profile = {
        "schema": "gradcell.inferred_requirement_profile.v1",
        "basis": "relative_rank_within_parameter_set_and_generation_mode",
        "primary_goal": primary_goal,
        "energy_requirement_level": level_for_percentile(energy_score),
        "high_rate_requirement_level": level_for_percentile(rate_score),
        "voltage_stability_level": level_for_percentile(voltage_score),
        "request_strictness": strictness,
        "design_emphasis": design_emphasis(row),
        "operating_context": "常温下1C基础表现，并关注5C与6C短时放电",
        "allowed_claims": [
            "1C容量与能量的相对要求",
            "5C与6C短时放电的相对要求",
            "工作电压稳定性的相对要求",
            "能量与倍率性能之间的取舍",
        ],
        "unsupported_claims": [
            "循环寿命",
            "安全性",
            "成本",
            "低温性能",
            "产品尺寸与质量",
            "材料化学成分",
        ],
        "diagnostic_percentiles": {
            "energy": energy_score,
            "high_rate": rate_score,
            "voltage_stability": voltage_score,
        },
    }
    profile["profile_id"] = hashlib.sha256(
        json.dumps(
            {
                name: profile[name]
                for name in (
                    "primary_goal",
                    "energy_requirement_level",
                    "high_rate_requirement_level",
                    "voltage_stability_level",
                    "request_strictness",
                    "design_emphasis",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return profile


def profile_tags(profile: dict[str, Any]) -> str:
    fields = (
        ("goal", "primary_goal"),
        ("energy_level", "energy_requirement_level"),
        ("rate_level", "high_rate_requirement_level"),
        ("voltage_level", "voltage_stability_level"),
        ("strictness", "request_strictness"),
    )
    return "".join(f"<{tag}>{profile[name]}</{tag}>" for tag, name in fields)


def deterministic_request(profile: dict[str, Any], variant: int) -> str:
    templates = (
        (
            "我希望设计一款以{goal}为主要方向的电芯。在常温使用时，1C容量与能量表现达到"
            "{energy}水平，同时5C和6C短时放电能力达到{rate}水平，工作电压稳定性希望处于"
            "{voltage}水平。设计时请重点考虑{emphasis}，并给出一组可仿真验证的参数方案。"
        ),
        (
            "请给出一个偏向{goal}的电池参数设计。需求严格程度为{strictness}，希望常温1C"
            "能量和容量达到{energy}水平，高倍率短时输出达到{rate}水平，并兼顾{voltage}水平"
            "的电压稳定性。方案应体现{emphasis}方面的取舍。"
        ),
        (
            "我的目标是在常温工况下获得{goal}的电芯：基础能量需求为{energy}，5C与6C短时"
            "放电需求为{rate}，电压稳定性需求为{voltage}。希望参数选择侧重{emphasis}，"
            "不需要延伸到寿命、安全、成本或产品外形结论。"
        ),
    )
    return templates[variant % len(templates)].format(
        goal=profile["primary_goal"],
        energy=profile["energy_requirement_level"],
        rate=profile["high_rate_requirement_level"],
        voltage=profile["voltage_stability_level"],
        strictness=profile["request_strictness"],
        emphasis=profile["design_emphasis"],
    )


def parse_descriptions(content: Any, expected: int) -> list[str]:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    if not isinstance(content, str) or not content.strip():
        raise InvalidLanguageResponse("API returned empty message.content")
    value = re.sub(r"^<think>.*?</think>\s*", "", content.strip(), flags=re.DOTALL)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidLanguageResponse(f"Malformed JSON response: {value[:240]!r}") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("descriptions")
    if not isinstance(parsed, list) or len(parsed) != expected:
        actual = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise InvalidLanguageResponse(f"Expected {expected} descriptions, received {actual}")
    return [str(item).strip() for item in parsed]


def validate_description(text: str) -> str:
    if not 50 <= len(text) <= 500:
        raise InvalidLanguageResponse(f"Description length is outside [50, 500]: {len(text)}")
    if not any(word in text for word in DEMAND_WORDS):
        raise InvalidLanguageResponse("Description is not phrased as a user requirement")
    forbidden = [word for word in FORBIDDEN_OUTPUT_WORDS if word.lower() in text.lower()]
    if forbidden:
        raise InvalidLanguageResponse(f"Description leaked internal design facts: {forbidden}")
    if re.search(r"\d+\.\d{3,}", text):
        raise InvalidLanguageResponse("Description contains a high-precision numeric fact")
    return text


def request_descriptions(
    profile: dict[str, Any], variants: int, api: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    public_profile = {
        name: value
        for name, value in profile.items()
        if name not in {"diagnostic_percentiles", "profile_id", "schema", "basis"}
    }
    prompt = {
        "task": "根据抽象性能画像，反推出一个合理的中文用户设计需求",
        "rules": [
            f"生成恰好{variants}条语义一致但表达明显不同的用户需求",
            "使用希望、需要、目标、优先或兼顾等需求措辞，不要描述成已有电池的测试报告",
            "只能讨论画像允许的1C能量容量、高倍率短时输出、电压稳定性和性能取舍",
            "不得加入循环寿命、安全、成本、低温、尺寸、质量、封装或材料化学结论",
            "不得出现参数集名称、孔隙率、活性材料体积分数、扩散率、倍率值或教师设计",
            "不得复述精确仿真数值，不得编造阈值；只使用基础、中低、中等、中高、高等相对等级",
            "每条需求应要求输出一组可仿真验证的设计参数，但不得给出设计答案",
            "返回JSON对象，唯一字段descriptions是字符串数组",
        ],
        "requirement_profile": public_profile,
    }
    payload: dict[str, Any] = {
        "model": api["model"],
        "messages": [
            {
                "role": "system",
                "content": "你把仿真归纳出的抽象性能画像改写成用户需求，不复述原始事实。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
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
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        message = f"HTTP {exc.code}: {detail or exc.reason}"
        if exc.code in (401, 402, 403):
            raise FatalAPIError(message) from exc
        if exc.code in (408, 409, 425, 429) or exc.code >= 500:
            raise RetryableAPIError(message) from exc
        raise InvalidLanguageResponse(message) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableAPIError(f"{type(exc).__name__}: {exc}") from exc
    try:
        result = json.loads(raw)
        message = result["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise InvalidLanguageResponse(f"Invalid API response envelope: {raw[:240]!r}") from exc
    content = message.get("content") or message.get("reasoning_content")
    usage = {
        str(name): float(value)
        for name, value in result.get("usage", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    try:
        descriptions = [validate_description(text) for text in parse_descriptions(content, variants)]
    except InvalidLanguageResponse as exc:
        exc.usage = usage
        raise
    if len(set(descriptions)) != len(descriptions):
        raise InvalidLanguageResponse("DeepSeek returned duplicate descriptions", usage)
    return descriptions, usage


def transform_family(
    family_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    descriptions: list[str],
    language_source: str,
    usage: dict[str, float],
) -> list[dict[str, Any]]:
    ordered = sorted(family_rows, key=lambda row: int(row.get("variant_index", 0)))
    transformed = []
    tags = profile_tags(profile)
    for variant, (source, description) in enumerate(zip(ordered, descriptions, strict=True)):
        row = json.loads(json.dumps(source, ensure_ascii=False))
        row["schema"] = "gradcell.inferred_requirement_training.v1"
        row["task_id"] = f"battery-requirement-{source['physical_design_id']}-v{variant:02d}"
        row["description_family_id"] = source["physical_design_id"]
        row["variant_index"] = variant
        row["battery_description"] = f"{tags}\n{description}"
        row["requirement_profile"] = profile
        row["messages"] = [
            {
                "role": "system",
                "content": "根据用户性能需求输出一组严格JSON电池参数设计。",
            },
            {"role": "user", "content": row["battery_description"]},
            source["messages"][2],
        ]
        row["provenance"] = {
            **source.get("provenance", {}),
            "language_source": language_source,
            "generation_method": "performance_and_design_to_inferred_user_requirement",
            "source_task_id": source["task_id"],
            "api_usage": usage if variant == 0 else {},
        }
        transformed.append(row)
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fact descriptions into inferred user-requirement supervision data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--with-deepseek", action="store_true")
    parser.add_argument("--require-deepseek-success", action="store_true")
    parser.add_argument("--max-new-families", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--json-mode", action="store_true")
    args = parser.parse_args()
    if args.retries < 1 or args.timeout_s <= 0:
        parser.error("retries and timeout must be positive")
    if args.max_new_families is not None and args.max_new_families < 1:
        parser.error("maximum new families must be positive")

    source_rows = read_jsonl(args.input)
    if not source_rows:
        raise ValueError("Input dataset is empty")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        families[str(row["physical_design_id"])].append(row)
    for design_id, rows in families.items():
        splits = {str(row["split"]) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Physical design {design_id} crosses splits: {splits}")
    references = build_references(source_rows)
    profiles = {
        design_id: derive_requirement_profile(rows[0], references)
        for design_id, rows in families.items()
    }

    load_env(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    concurrency = args.concurrency or int(os.environ.get("DEEPSEEK_CONCURRENCY", "4"))
    if concurrency < 1:
        parser.error("concurrency must be positive")
    api = {
        "api_key": api_key,
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "json_mode": args.json_mode,
        "timeout_s": args.timeout_s,
    }
    if args.with_deepseek and not api_key:
        raise RuntimeError(
            f"DEEPSEEK_API_KEY is not set; checked the process environment and {args.env_file}"
        )

    prior_rows = read_jsonl(args.output) if args.output.exists() else []
    prior_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        prior_by_family[str(row["physical_design_id"])].append(row)
    completed: dict[str, list[dict[str, Any]]] = {}
    pending = []
    for design_id, rows in families.items():
        old = prior_by_family.get(design_id, [])
        valid_old = (
            len(old) == len(rows)
            and all(
                row.get("requirement_profile", {}).get("profile_id")
                == profiles[design_id]["profile_id"]
                for row in old
            )
            and all(
                row.get("provenance", {}).get("language_source")
                == (api["model"] if args.with_deepseek else "deterministic_requirement_template")
                for row in old
            )
        )
        if valid_old:
            completed[design_id] = sorted(old, key=lambda row: row["variant_index"])
        else:
            pending.append(design_id)
    selected = pending[: args.max_new_families] if args.max_new_families else pending
    usage_totals: Counter[str] = Counter()
    failures: dict[str, str] = {}

    def generate(design_id: str) -> tuple[str, list[str] | None, dict[str, float], str | None]:
        profile = profiles[design_id]
        count = len(families[design_id])
        if not args.with_deepseek:
            return (
                design_id,
                [deterministic_request(profile, variant) for variant in range(count)],
                {},
                None,
            )
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                descriptions, usage = request_descriptions(profile, count, api)
                return design_id, descriptions, usage, None
            except FatalAPIError:
                raise
            except InvalidLanguageResponse as exc:
                return design_id, None, exc.usage, f"{type(exc).__name__}: {exc}"
            except RetryableAPIError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return design_id, None, {}, last_error or "unknown API failure"

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate, design_id) for design_id in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            design_id, descriptions, usage, error = future.result()
            usage_totals.update(usage)
            if descriptions is None:
                failures[design_id] = error or "unknown error"
            else:
                source = api["model"] if args.with_deepseek else "deterministic_requirement_template"
                completed[design_id] = transform_family(
                    families[design_id], profiles[design_id], descriptions, source, usage
                )
            ordered_output = [
                row for family_id in families for row in completed.get(family_id, [])
            ]
            write_jsonl(args.output, ordered_output)
            print(
                json.dumps(
                    {
                        "inferred_requirement_families": index,
                        "selected": len(selected),
                        "completed_total": len(completed),
                        "failed": len(failures),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    output_rows = read_jsonl(args.output) if args.output.exists() else []
    profile_counts = Counter(
        row["requirement_profile"]["profile_id"] for row in output_rows
    )
    split_counts = Counter(str(row["split"]) for row in output_rows)
    manifest = {
        "schema": "gradcell.inferred_requirement_manifest.v1",
        "task_definition": "inferred_user_requirement_to_parameter_design",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output) if args.output.exists() else None,
        "records": len(output_rows),
        "physical_designs": len({row["physical_design_id"] for row in output_rows}),
        "requirement_profiles": len(profile_counts),
        "largest_profile_records": max(profile_counts.values(), default=0),
        "split_counts": dict(split_counts),
        "deepseek_requested": args.with_deepseek,
        "language_source": api["model"] if args.with_deepseek else "deterministic_requirement_template",
        "selected_families_this_run": len(selected),
        "failed_families": failures,
        "reported_usage_this_run": dict(usage_totals),
        "information_policy": {
            "exact_performance_values_in_text": False,
            "internal_design_parameters_in_text": False,
            "relative_requirement_tags": True,
            "teacher_design_retained_as_supervision": True,
        },
        "warning": (
            "Requirement inference is many-to-one by construction. Compare profile counts and "
            "prediction variance before interpreting single-output MLP accuracy."
        ),
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if failures and args.require_deepseek_success:
        raise RuntimeError(
            f"DeepSeek failed for {len(failures)} families; successful checkpoints were preserved"
        )


if __name__ == "__main__":
    main()
