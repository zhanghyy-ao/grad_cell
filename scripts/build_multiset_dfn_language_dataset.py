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

import numpy as np
import yaml


OBJECTIVES = ("energy_1c_wh", "energy_retention_5c", "energy_retention_6c")


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


def nondominated_indices(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((-values[:, 2], -values[:, 1], -values[:, 0]))
    archive: list[int] = []
    for index in order:
        if archive:
            incumbent = values[np.asarray(archive)]
            if np.any(np.all(incumbent >= values[index], axis=1)):
                continue
            archive = list(np.asarray(archive)[~np.all(values[index] >= incumbent, axis=1)])
        archive.append(int(index))
    return np.asarray(archive, dtype=np.int64)


def objective_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["performance"][name]) for name in OBJECTIVES] for row in rows],
        dtype=np.float64,
    )


def deterministic_text(requirements: dict[str, Any], variant: int) -> str:
    templates = (
        (
            "请在{temperature:.2f} K下基于{parameter_set}参数体系调整电芯，"
            "使1C放电能量至少达到"
            "{energy:.6f} Wh，并确保5C、6C能量保持率分别不低于"
            "{r5:.6f}和{r6:.6f}。能量优先权重为{preference:.6f}。"
        ),
        (
            "我需要一套{parameter_set}电池参数设计，工作温度为{temperature:.2f} K："
            "1C能量目标不少于"
            "{energy:.6f} Wh，5C保持率不少于{r5:.6f}，6C保持率不少于"
            "{r6:.6f}；本次能量侧偏好系数为{preference:.6f}。"
        ),
        (
            "在{temperature:.2f} K下对{parameter_set}基准电池进行参数优化。"
            "约束为1C能量≥"
            "{energy:.6f} Wh、R5≥{r5:.6f}、R6≥{r6:.6f}，能量与倍率"
            "之间的能量权重取{preference:.6f}，请返回可验证的设计。"
        ),
    )
    return templates[variant % len(templates)].format(
        parameter_set=requirements["parameter_set"],
        energy=requirements["target_energy_1c_wh"],
        r5=requirements["min_energy_retention_5c"],
        r6=requirements["min_energy_retention_6c"],
        preference=requirements["preference_energy"],
        temperature=requirements["temperature_k"],
    )


def design_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "gradcell.multiset_parameter_design.v1",
        "base_parameter_set": row["parameter_set"],
        "parameter_updates": {
            name: {
                "multiplier": float(multiplier),
                "value": float(value),
            }
            for name, multiplier, value in zip(
                row["parameter_names"],
                row["parameter_multipliers"],
                row["parameter_values"],
                strict=True,
            )
        },
    }


def candidate_payload(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "physical_design_id": row["physical_design_id"],
        "source_case_id": row["case_id"],
        "design": design_payload(row),
        "verified_performance": row["performance"],
        "oracle_score": float(score),
    }


def build_tasks(
    physics_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    dataset = config["dataset"]
    split_cfg = {name: float(config["split"][name]) for name in ("train", "validation", "test")}
    if not np.isclose(sum(split_cfg.values()), 1.0):
        raise ValueError("split ratios must sum to 1")
    for row in physics_rows:
        row["split"] = split_for_design(row["physical_design_id"], seed, split_cfg)

    tasks: list[dict[str, Any]] = []
    parameter_sets = sorted({row["parameter_set"] for row in physics_rows})
    tasks_per_set = int(dataset["tasks_per_set"])
    variants = int(dataset["variants_per_task"])
    top_k = int(dataset["top_k"])
    regular_fraction = float(dataset["regular_fraction"])

    for set_index, parameter_set in enumerate(parameter_sets):
        set_rows = [row for row in physics_rows if row["parameter_set"] == parameter_set]
        for split_index, split in enumerate(("train", "validation", "test")):
            rows = [row for row in set_rows if row["split"] == split]
            if len(rows) < top_k:
                raise RuntimeError(
                    f"{parameter_set}/{split} has {len(rows)} designs, fewer than top_k={top_k}"
                )
            objectives = objective_matrix(rows)
            front = nondominated_indices(objectives)
            split_tasks = int(round(tasks_per_set * split_cfg[split]))
            rng = np.random.default_rng(seed + 1009 * set_index + 97 * split_index)
            anchors = rng.choice(front, size=split_tasks, replace=split_tasks > len(front))
            scale = np.maximum(objectives.max(axis=0) - objectives.min(axis=0), 1e-12)
            normalized = (objectives - objectives.min(axis=0)) / scale

            for local_task, anchor_index in enumerate(anchors):
                anchor = rows[int(anchor_index)]
                anchor_values = objectives[int(anchor_index)]
                boundary = bool(rng.random() >= regular_fraction)
                family = "boundary" if boundary else "regular"
                energy_bounds = dataset[f"energy_margin_{family}"]
                retention_bounds = dataset[f"retention_margin_{family}"]
                energy_margin = float(rng.uniform(*energy_bounds))
                r5_margin = float(rng.uniform(*retention_bounds))
                r6_margin = float(rng.uniform(*retention_bounds))
                preference = float(rng.uniform(0.0, 1.0))
                requirements = {
                    "parameter_set": parameter_set,
                    "target_energy_1c_wh": float(anchor_values[0] * (1.0 - energy_margin)),
                    "min_energy_retention_5c": float(
                        max(0.0, anchor_values[1] - r5_margin)
                    ),
                    "min_energy_retention_6c": float(
                        max(0.0, anchor_values[2] - r6_margin)
                    ),
                    "preference_energy": preference,
                    "preference_high_rate": 1.0 - preference,
                    "temperature_k": float(anchor["simulation"]["temperature_k"]),
                }
                feasible = (
                    (objectives[:, 0] >= requirements["target_energy_1c_wh"])
                    & (objectives[:, 1] >= requirements["min_energy_retention_5c"])
                    & (objectives[:, 2] >= requirements["min_energy_retention_6c"])
                )
                violation = (
                    np.maximum(requirements["target_energy_1c_wh"] - objectives[:, 0], 0.0)
                    / scale[0]
                    + np.maximum(
                        requirements["min_energy_retention_5c"] - objectives[:, 1], 0.0
                    )
                    / scale[1]
                    + np.maximum(
                        requirements["min_energy_retention_6c"] - objectives[:, 2], 0.0
                    )
                    / scale[2]
                )
                reward = preference * normalized[:, 0] + (1.0 - preference) * np.minimum(
                    normalized[:, 1], normalized[:, 2]
                )
                score = 100.0 * violation - reward
                eligible = np.flatnonzero(feasible)
                if len(eligible) == 0:
                    raise RuntimeError(
                        "Anchor-derived requirement unexpectedly has no feasible design"
                    )
                rank = eligible[np.argsort(score[eligible])[:top_k]]
                family_id = (
                    f"{parameter_set}-{split}-{local_task:06d}-"
                    f"{anchor['physical_design_id']}"
                )
                candidates = [
                    candidate_payload(rows[int(index)], score[int(index)]) for index in rank
                ]
                for variant in range(variants):
                    task_id = f"{family_id}-v{variant:02d}"
                    target = candidates[0]["design"]
                    record = {
                        "schema": "gradcell.multiset_language_training.v1",
                        "task_id": task_id,
                        "requirement_family_id": family_id,
                        "split": split,
                        "sample_kind": f"feasible_{family}",
                        "requirements_canonical": requirements,
                        "requirement_text": deterministic_text(requirements, variant),
                        "teacher_design": target,
                        "teacher_candidates": candidates,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "根据用户性能需求输出严格 JSON 电池参数设计；"
                                    "不得输出 JSON 之外的内容。"
                                ),
                            },
                            {
                                "role": "user",
                                "content": deterministic_text(requirements, variant),
                            },
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
                            "physics_source": anchor["case_id"],
                            "language_source": "deterministic_template",
                            "physical_split_group": anchor["physical_design_id"],
                        },
                    }
                    tasks.append(record)
    return tasks


def deepseek_config(config: dict[str, Any]) -> dict[str, Any]:
    language = config["language"]
    return {
        **language,
        "api_key": os.environ.get(language["api_key_env"], ""),
        "base_url": os.environ.get(
            language["base_url_env"], language["default_base_url"]
        ),
        "model": os.environ.get(language["model_env"], language["default_model"]),
        "concurrency": int(
            os.environ.get(
                language["concurrency_env"], language["default_concurrency"]
            )
        ),
    }


def deepseek_rewrite(record: dict[str, Any], language: dict[str, Any]) -> str:
    if not language["api_key"]:
        raise RuntimeError(f"Environment variable {language['api_key_env']} is not set")
    requirement = record["requirements_canonical"]
    anchors = [
        f"{requirement['target_energy_1c_wh']:.6f}",
        f"{requirement['min_energy_retention_5c']:.6f}",
        f"{requirement['min_energy_retention_6c']:.6f}",
        f"{requirement['preference_energy']:.6f}",
        f"{requirement['temperature_k']:.2f}",
    ]
    prompt = {
        "task": "将结构化电池性能需求改写为自然、专业但表达多样的中文用户请求",
        "rules": [
            "只能改写需求，不得给出设计答案",
            "不得提及隐藏的teacher设计或参数值",
            "不得增加DFN未验证的寿命、安全、成本或低温要求",
            "必须原样保留parameter_set和全部数值锚点",
            "仅返回包含requirement_text字段的JSON对象",
        ],
        "numeric_anchors": anchors,
        "requirements": requirement,
        "variant_id": record["task_id"],
    }
    payload = {
        "model": language["model"],
        "messages": [
            {
                "role": "system",
                "content": "你是电池工程需求改写器，不负责生成或评价设计。",
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
    text = str(parsed["requirement_text"]).strip()
    missing = [anchor for anchor in anchors if anchor not in text]
    if missing or requirement["parameter_set"] not in text:
        raise ValueError(f"DeepSeek changed or omitted protected anchors: {missing}")
    return text


def apply_deepseek(
    records: list[dict[str, Any]],
    language: dict[str, Any],
    checkpoint: Path,
    require_success: bool,
) -> None:
    prior = {}
    if checkpoint.exists():
        prior = {row["task_id"]: row for row in read_jsonl(checkpoint)}
    for record in records:
        old = prior.get(record["task_id"])
        if old and old.get("provenance", {}).get("language_source") == language["model"]:
            record["requirement_text"] = old["requirement_text"]
            record["messages"][1]["content"] = old["requirement_text"]
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
                return record, deepseek_rewrite(record, language), None
            except (
                ValueError,
                KeyError,
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
                record.setdefault("quality_flags", []).append("deepseek_rewrite_failed")
                record["provenance"]["language_error"] = error
            else:
                record["requirement_text"] = text
                record["messages"][1]["content"] = text
                record["provenance"]["language_source"] = language["model"]
                record["provenance"].pop("language_error", None)
                record.pop("quality_flags", None)
            completed += 1
            if completed % 25 == 0:
                write_jsonl(checkpoint, records)
                print(f"DeepSeek rewrites: {completed}/{len(pending)}", flush=True)
    write_jsonl(checkpoint, records)
    if failures and require_success:
        raise RuntimeError(f"DeepSeek failed for {failures} records; checkpoint was preserved")


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [row["task_id"] for row in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values are not unique")
    family_splits: dict[str, set[str]] = {}
    design_splits: dict[str, set[str]] = {}
    for row in records:
        family_splits.setdefault(row["requirement_family_id"], set()).add(row["split"])
        for candidate in row["teacher_candidates"]:
            design_splits.setdefault(candidate["physical_design_id"], set()).add(row["split"])
        json.loads(row["messages"][2]["content"])
    leaked_families = [key for key, splits in family_splits.items() if len(splits) > 1]
    leaked_designs = [key for key, splits in design_splits.items() if len(splits) > 1]
    if leaked_families or leaked_designs:
        raise RuntimeError(
            f"split leakage detected: families={len(leaked_families)}, "
            f"physical_designs={len(leaked_designs)}"
        )
    return {
        "records": len(records),
        "families": len(family_splits),
        "physical_teacher_designs": len(design_splits),
        "split_counts": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "validation", "test")
        },
        "parameter_set_counts": {
            name: sum(
                row["requirements_canonical"]["parameter_set"] == name for row in records
            )
            for name in sorted(
                {row["requirements_canonical"]["parameter_set"] for row in records}
            )
        },
        "deepseek_records": sum(
            row["provenance"]["language_source"] != "deterministic_template"
            for row in records
        ),
        "quality_flagged_records": sum(bool(row.get("quality_flags")) for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build grouped natural-language training data from a multi-set DFN archive."
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
    records = build_tasks(physics_rows, config)
    if args.with_deepseek:
        language = deepseek_config(config)
        apply_deepseek(
            records,
            language,
            output,
            require_success=args.require_deepseek_success,
        )
    write_jsonl(output, records)
    validation = validate_dataset(records)
    manifest = {
        "schema": "gradcell.multiset_language_manifest.v1",
        "config": str(args.config),
        "physics_archive": str(archive),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "deepseek_requested": bool(args.with_deepseek),
        "validation": validation,
    }
    atomic_json(Path(config["dataset"]["manifest"]), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
