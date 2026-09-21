from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CHEMISTRIES = (
    ("LFP/石墨", 3.2),
    ("NMC/石墨", 3.65),
    ("NCA/石墨", 3.65),
    ("LFP/硅碳", 3.2),
)
FORM_FACTORS = ("软包堆叠", "方形卷绕", "方形叠片", "圆柱卷绕")
SCENARIOS = ("储能", "乘用车", "商用车", "通信备电", "低速交通工具")
PRIORITIES = ("安全性", "成本", "低温可用性", "快充能力", "循环寿命")


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
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def reference_spec() -> dict[str, Any]:
    return {
        "chemistry": "LFP/石墨",
        "nominal_voltage_v": 3.2,
        "capacity_ah": 70.0,
        "form_factor": "软包堆叠",
        "thickness_max_mm": 20.0,
        "width_max_mm": 150.0,
        "length_max_mm": 220.0,
        "mass_max_g": 1500.0,
        "scenario": "储能",
        "cycle_temperature_c": 25.0,
        "cycle_rate_c": 0.5,
        "cycle_count": 3000,
        "cycle_retention_min": 0.80,
        "priorities": ["安全性", "成本", "低温可用性"],
    }


def random_spec(rng: random.Random) -> dict[str, Any]:
    chemistry, voltage = rng.choice(CHEMISTRIES)
    capacity = rng.choice((20, 40, 50, 70, 100, 120, 150, 280))
    form_factor = rng.choice(FORM_FACTORS)
    if "圆柱" in form_factor:
        thickness = rng.choice((18, 21, 26, 32, 46))
        width = thickness
        length = rng.choice((65, 70, 80, 120))
    else:
        thickness = rng.choice((8, 12, 16, 20, 25, 35))
        width = rng.choice((90, 120, 150, 180, 220))
        length = rng.choice((120, 180, 220, 280, 350))
    specific_energy_guess = rng.uniform(120.0, 230.0)
    mass = round(capacity * voltage / specific_energy_guess * 1000.0 / 50.0) * 50.0
    return {
        "chemistry": chemistry,
        "nominal_voltage_v": voltage,
        "capacity_ah": float(capacity),
        "form_factor": form_factor,
        "thickness_max_mm": float(thickness),
        "width_max_mm": float(width),
        "length_max_mm": float(length),
        "mass_max_g": max(200.0, mass),
        "scenario": rng.choice(SCENARIOS),
        "cycle_temperature_c": float(rng.choice((-10, 0, 25, 45))),
        "cycle_rate_c": float(rng.choice((0.5, 1.0, 1.5, 2.0))),
        "cycle_count": int(rng.choice((1000, 2000, 3000, 5000, 8000))),
        "cycle_retention_min": float(rng.choice((0.70, 0.75, 0.80, 0.85, 0.90))),
        "priorities": rng.sample(list(PRIORITIES), rng.choice((2, 3))),
    }


def format_number(value: float) -> str:
    return f"{value:g}"


def protected_spec(spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    replacements = {
        "[[CHEMISTRY]]": str(spec["chemistry"]),
        "[[NOMINAL_VOLTAGE_V]]": format_number(spec["nominal_voltage_v"]),
        "[[CAPACITY_AH]]": format_number(spec["capacity_ah"]),
        "[[FORM_FACTOR]]": str(spec["form_factor"]),
        "[[THICKNESS_MAX_MM]]": format_number(spec["thickness_max_mm"]),
        "[[WIDTH_MAX_MM]]": format_number(spec["width_max_mm"]),
        "[[LENGTH_MAX_MM]]": format_number(spec["length_max_mm"]),
        "[[MASS_MAX_G]]": format_number(spec["mass_max_g"]),
        "[[SCENARIO]]": str(spec["scenario"]),
        "[[CYCLE_TEMPERATURE_C]]": format_number(spec["cycle_temperature_c"]),
        "[[CYCLE_RATE_C]]": format_number(spec["cycle_rate_c"]),
        "[[CYCLE_COUNT]]": str(spec["cycle_count"]),
        "[[CYCLE_RETENTION_PCT]]": format_number(100.0 * spec["cycle_retention_min"]),
        "[[PRIORITIES]]": "、".join(spec["priorities"]),
    }
    facts = {
        "chemistry": "[[CHEMISTRY]]",
        "nominal_voltage_v": "[[NOMINAL_VOLTAGE_V]]",
        "capacity_ah": "[[CAPACITY_AH]]",
        "form_factor": "[[FORM_FACTOR]]",
        "thickness_max_mm": "[[THICKNESS_MAX_MM]]",
        "width_max_mm": "[[WIDTH_MAX_MM]]",
        "length_max_mm": "[[LENGTH_MAX_MM]]",
        "mass_max_g": "[[MASS_MAX_G]]",
        "scenario": "[[SCENARIO]]",
        "cycle_temperature_c": "[[CYCLE_TEMPERATURE_C]]",
        "cycle_rate_c": "[[CYCLE_RATE_C]]",
        "cycle_count": "[[CYCLE_COUNT]]",
        "cycle_retention_pct": "[[CYCLE_RETENTION_PCT]]",
        "priorities": "[[PRIORITIES]]",
    }
    return facts, replacements


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
        raise InvalidLanguageResponse(f"Malformed JSON response: {value[:200]!r}") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("descriptions")
    if not isinstance(parsed, list) or len(parsed) != expected:
        actual = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise InvalidLanguageResponse(f"Expected {expected} descriptions, received {actual}")
    descriptions = [str(item).strip() for item in parsed]
    if any(not item for item in descriptions):
        raise InvalidLanguageResponse("DeepSeek returned an empty description")
    if len(set(descriptions)) != len(descriptions):
        raise InvalidLanguageResponse("DeepSeek returned duplicate descriptions")
    return descriptions


def restore_and_validate(text: str, replacements: dict[str, str]) -> str:
    missing = [token for token in replacements if text.count(token) != 1]
    unknown = sorted(set(re.findall(r"\[\[[A-Z0-9_]+\]\]", text)) - set(replacements))
    if missing or unknown:
        raise InvalidLanguageResponse(
            "Every protected fact must occur exactly once; "
            f"missing_or_repeated={missing}, unknown={unknown}; preview={text[:180]!r}"
        )
    for token, value in replacements.items():
        text = text.replace(token, value)
    if len(text) < 80 or len(text) > 600:
        raise InvalidLanguageResponse(f"Description length is outside [80, 600]: {len(text)}")
    return text


def request_descriptions(
    spec: dict[str, Any], variants: int, api: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    facts, replacements = protected_spec(spec)
    instruction = {
        "task": "根据给定事实生成真实用户口吻的中文电芯设计需求",
        "rules": [
            f"生成恰好{variants}条语义完全一致、表达方式明显不同的自然语言需求",
            "可以调整语序、句式和工程表达，但不得增加、删除、推导或修改任何事实",
            "每条描述必须原样包含下方每个[[...]]占位符，而且每个占位符恰好一次",
            "不要给出设计答案、分析、免责声明、标题或Markdown",
            "不要提及数据集、占位符、模型或本指令",
            "返回JSON对象，唯一字段descriptions是字符串数组",
        ],
        "protected_facts": facts,
    }
    payload: dict[str, Any] = {
        "model": api["model"],
        "messages": [
            {
                "role": "system",
                "content": "你负责把结构化电池需求写成自然、专业的中文用户提问。",
            },
            {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)},
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
            raw_result = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        message = f"HTTP {exc.code}: {detail or exc.reason}"
        if exc.code in (401, 402, 403):
            raise FatalAPIError(message) from exc
        if exc.code in (408, 409, 425, 429) or exc.code >= 500:
            raise RetryableAPIError(message) from exc
        raise InvalidLanguageResponse(message) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableAPIError(f"{type(exc).__name__}: {exc}") from exc
    try:
        result = json.loads(raw_result)
        content = result["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise InvalidLanguageResponse(
            f"Invalid API response envelope: {raw_result[:180]!r}"
        ) from exc
    raw_usage = result.get("usage", {})
    usage = {
        str(name): float(value)
        for name, value in raw_usage.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    try:
        descriptions = parse_descriptions(content, variants)
        return [restore_and_validate(text, replacements) for text in descriptions], usage
    except InvalidLanguageResponse as exc:
        exc.usage = usage
        raise


def scope_annotation(spec: dict[str, Any]) -> dict[str, Any]:
    unsupported = [
        "chemistry_selection",
        "nominal_voltage_target",
        "form_factor",
        "cell_dimensions",
        "cell_mass",
        "cycle_life",
    ]
    priority_fields = {
        "安全性": "safety",
        "成本": "cost",
        "低温可用性": "low_temperature_performance",
        "快充能力": "fast_charging",
    }
    unsupported.extend(
        priority_fields[name] for name in spec["priorities"] if name in priority_fields
    )
    return {
        "parameter_set": "Chen2020",
        "generation_mode": "regular",
        "partially_comparable": ["capacity_ah"],
        "unsupported": unsupported,
        "chemistry_compatible": spec["chemistry"] == "NMC/石墨",
        "out_of_domain_reason": (
            f"The request specifies {spec['chemistry']} and product-level geometry/lifetime, "
            "while the current model was trained only on Chen2020 Regular Mode DFN summaries."
        ),
    }


def rows_for_spec(
    spec_id: str,
    spec: dict[str, Any],
    descriptions: list[str],
    model: str,
    usage: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "schema": "gradcell.application_battery_prompt.v2",
            "prompt_id": f"{spec_id}-v{variant}",
            "spec_id": spec_id,
            "variant_index": variant,
            "battery_description": description,
            "requirements": spec,
            "current_model_scope": scope_annotation(spec),
            "provenance": {
                "language_source": model,
                "generation_method": "deepseek_structured_facts_to_natural_language",
                "api_usage": usage if variant == 0 else {},
            },
        }
        for variant, description in enumerate(descriptions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate application-level battery prompts with DeepSeek."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--spec-count", type=int, default=32)
    parser.add_argument("--variants-per-spec", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--max-new-specs", type=int)
    args = parser.parse_args()
    if args.spec_count < 1 or args.variants_per_spec < 1:
        parser.error("spec count and variants per spec must be positive")
    if args.retries < 1 or args.timeout_s <= 0:
        parser.error("retries and timeout must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("max tokens must be positive when provided")
    if args.max_new_specs is not None and args.max_new_specs <= 0:
        parser.error("max new specs must be positive when provided")

    load_env(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            f"DEEPSEEK_API_KEY is not set; checked process environment and {args.env_file}"
        )
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

    rng = random.Random(args.seed)
    specs = [reference_spec()]
    specs.extend(random_spec(rng) for _ in range(args.spec_count - 1))
    prior_rows = read_jsonl(args.output)
    prior_by_spec: dict[str, list[dict[str, Any]]] = {}
    for row in prior_rows:
        prior_by_spec.setdefault(str(row["spec_id"]), []).append(row)
    completed_rows: dict[str, list[dict[str, Any]]] = {}
    jobs = []
    for index, spec in enumerate(specs):
        spec_id = f"application-spec-{args.seed}-{index:05d}"
        old = sorted(prior_by_spec.get(spec_id, []), key=lambda row: row["variant_index"])
        valid_old = (
            len(old) == args.variants_per_spec
            and all(row.get("requirements") == spec for row in old)
            and all(row.get("provenance", {}).get("language_source") == api["model"] for row in old)
        )
        if valid_old:
            completed_rows[spec_id] = old
        else:
            jobs.append((spec_id, spec))
    selected_jobs = jobs[: args.max_new_specs] if args.max_new_specs is not None else jobs

    usage_totals: dict[str, float] = {}
    failures: dict[str, str] = {}
    fatal_error: str | None = None

    def generate_one(
        item: tuple[str, dict[str, Any]],
    ) -> tuple[str, dict[str, Any], list[str] | None, dict[str, float], str | None, bool]:
        spec_id, spec = item
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                descriptions, usage = request_descriptions(spec, args.variants_per_spec, api)
                return spec_id, spec, descriptions, usage, None, False
            except FatalAPIError as exc:
                return spec_id, spec, None, {}, f"{type(exc).__name__}: {exc}", True
            except InvalidLanguageResponse as exc:
                return (
                    spec_id,
                    spec,
                    None,
                    exc.usage,
                    f"{type(exc).__name__}: {exc}",
                    False,
                )
            except RetryableAPIError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return spec_id, spec, None, {}, last_error, False

    completed_now = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_one, job) for job in selected_jobs]
        for future in as_completed(futures):
            spec_id, spec, descriptions, usage, error, fatal = future.result()
            for name, value in usage.items():
                usage_totals[name] = usage_totals.get(name, 0.0) + value
            if descriptions is None:
                failures[spec_id] = error or "unknown error"
                if fatal:
                    fatal_error = failures[spec_id]
            else:
                completed_rows[spec_id] = rows_for_spec(
                    spec_id, spec, descriptions, api["model"], usage
                )
            completed_now += 1
            ordered_rows = [
                row
                for index in range(len(specs))
                for row in completed_rows.get(f"application-spec-{args.seed}-{index:05d}", [])
            ]
            write_jsonl(args.output, ordered_rows)
            print(
                f"DeepSeek specifications: {completed_now}/{len(selected_jobs)}; "
                f"successful={len(completed_rows)}; failed={len(failures)}",
                flush=True,
            )
            if fatal_error:
                for pending in futures:
                    pending.cancel()
                break

    final_rows = read_jsonl(args.output)
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(
            "\n".join(row["battery_description"] for row in final_rows) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema": "gradcell.application_battery_prompt_manifest.v2",
        "output": str(args.output),
        "text_output": str(args.text_output) if args.text_output else None,
        "requested_specifications": len(specs),
        "completed_specifications": len(completed_rows),
        "variants_per_spec": args.variants_per_spec,
        "records": len(final_rows),
        "seed": args.seed,
        "language_source": api["model"],
        "specifications_attempted_this_run": completed_now,
        "api_successes_this_run": completed_now - len(failures),
        "api_failures_this_run": failures,
        "reported_usage_this_run": usage_totals,
        "unattempted_specifications": len(specs) - len(completed_rows) - len(failures),
        "fatal_error": fatal_error,
        "purpose": "Out-of-domain language robustness and capability-boundary evaluation.",
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if fatal_error:
        raise RuntimeError(f"DeepSeek stopped after a fatal API error: {fatal_error}")
    if failures and args.require_success:
        raise RuntimeError(
            f"DeepSeek failed for {len(failures)} specifications; successful output was preserved"
        )


if __name__ == "__main__":
    main()
