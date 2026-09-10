from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REQUIRED_TARGETS = (
    "delivered_energy_1c_wh",
    "delivered_energy_5c_wh",
    "delivered_energy_6c_wh",
    "specific_energy_1c_wh_kg",
)


@dataclass(frozen=True)
class Archive:
    latent: np.ndarray
    design: np.ndarray
    energy: np.ndarray
    retention_5c: np.ndarray
    retention_6c: np.ndarray
    design_fields: tuple[str, ...]
    source_indices: np.ndarray
    metadata: dict[str, Any]


def _metadata(npz: Any) -> dict[str, Any]:
    raw = npz["metadata"]
    value = raw.item() if np.asarray(raw).shape == () else raw.tolist()
    return json.loads(value) if isinstance(value, str) else dict(value)


def load_archive(path: Path) -> Archive:
    with np.load(path, allow_pickle=False) as npz:
        metadata = _metadata(npz)
        if all(
            name in npz.files
            for name in ("energy_wh_kg", "energy_retention_5c", "energy_retention_6c")
        ):
            latent = np.asarray(npz["latent"], dtype=np.float64)
            design = np.asarray(npz["design"], dtype=np.float64)
            design_fields = tuple(
                metadata.get(
                    "design_fields",
                    ("eps_p", "eps_n", "eps_s", "phi_p", "np_ratio"),
                )
            )
            source_indices = (
                np.asarray(npz["source_candidate_index"], dtype=np.int64)
                if "source_candidate_index" in npz.files
                else np.arange(len(latent), dtype=np.int64)
            )
            return Archive(
                latent=latent,
                design=design,
                energy=np.asarray(npz["energy_wh_kg"], dtype=np.float64),
                retention_5c=np.asarray(npz["energy_retention_5c"], dtype=np.float64),
                retention_6c=np.asarray(npz["energy_retention_6c"], dtype=np.float64),
                design_fields=design_fields,
                source_indices=source_indices,
                metadata=metadata,
            )
        target_fields = tuple(metadata["target_fields"])
        design_fields = tuple(metadata["design_fields"])
        missing = sorted(set(REQUIRED_TARGETS) - set(target_fields))
        if missing:
            raise ValueError(f"Archive is missing required targets: {missing}")
        targets = np.asarray(npz["targets"], dtype=np.float64)
        target = {name: targets[:, target_fields.index(name)] for name in REQUIRED_TARGETS}
        energy_1c_wh = np.maximum(targets[:, target_fields.index("delivered_energy_1c_wh")], 1e-12)
        return Archive(
            latent=np.asarray(npz["latent"], dtype=np.float64),
            design=np.asarray(npz["design"], dtype=np.float64),
            energy=target["specific_energy_1c_wh_kg"],
            retention_5c=target["delivered_energy_5c_wh"] / energy_1c_wh,
            retention_6c=target["delivered_energy_6c_wh"] / energy_1c_wh,
            design_fields=design_fields,
            source_indices=np.arange(len(targets), dtype=np.int64),
            metadata=metadata,
        )


def _design_dict(archive: Archive, index: int) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(archive.design_fields, archive.design[index], strict=True)
    }


def _base_record(archive: Archive, index: int, sample_id: int) -> dict[str, Any]:
    return {
        "task_id": f"paper-explore-{sample_id:04d}",
        "requirement_family_id": f"family-{sample_id:04d}",
        "source_archive_index": int(archive.source_indices[index]),
        "teacher_latent": archive.latent[index].round(10).tolist(),
        "teacher_design": _design_dict(archive, index),
        "teacher_performance": {
            "energy_1c_wh_kg": float(archive.energy[index]),
            "retention_5c": float(archive.retention_5c[index]),
            "retention_6c": float(archive.retention_6c[index]),
        },
        "temperature_k": 298.15,
        "material_parameter_set": "Chen2020",
        "application_tag": "energy_storage",
        "provenance": {
            "physics_source": "GradCell hard-cutoff PyBaMM archive",
            "language_source": "deterministic_template",
            "evidence_level": "B",
        },
    }


def make_records(archive: Archive, counts: dict[str, int], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    energy_min, energy_max = float(archive.energy.min()), float(archive.energy.max())
    r5_max, r6_max = float(archive.retention_5c.max()), float(archive.retention_6c.max())
    energy_span = max(energy_max - energy_min, 1e-12)
    high_rate = np.minimum(archive.retention_5c, archive.retention_6c)
    rate_min, rate_max = float(high_rate.min()), float(high_rate.max())
    rate_span = max(rate_max - rate_min, 1e-12)
    energy_distance = (energy_max - archive.energy) / energy_span
    rate_distance = (rate_max - high_rate) / rate_span

    kinds = [kind for kind, count in counts.items() for _ in range(int(count))]
    rng.shuffle(kinds)
    for sample_id, kind in enumerate(kinds, start=1):
        preference_energy = float(rng.uniform(0.0, 1.0))
        losses = np.maximum(
            preference_energy * energy_distance,
            (1.0 - preference_energy) * rate_distance,
        ) + 0.05 * (
            preference_energy * energy_distance
            + (1.0 - preference_energy) * rate_distance
        )
        top_indices = np.argsort(losses)[: min(5, len(losses))]
        index = int(top_indices[0])
        row = _base_record(archive, index, sample_id)
        energy = float(archive.energy[index])
        r5 = float(archive.retention_5c[index])
        r6 = float(archive.retention_6c[index])
        unsupported: list[str] = []
        reasons: list[str] = []

        if kind == "feasible_regular":
            energy_margin = float(rng.uniform(0.03, 0.12))
            rate_margin = float(rng.uniform(0.01, 0.05))
            target_energy = max(energy_min, energy * (1.0 - energy_margin))
            target_r5 = max(0.0, r5 - rate_margin)
            target_r6 = max(0.0, r6 - rate_margin)
            feasible = True
        elif kind == "feasible_boundary":
            energy_margin = float(rng.uniform(0.001, 0.015))
            rate_margin = float(rng.uniform(0.001, 0.008))
            target_energy = energy * (1.0 - energy_margin)
            target_r5 = max(0.0, r5 - rate_margin)
            target_r6 = max(0.0, r6 - rate_margin)
            feasible = True
        elif kind == "infeasible_modeled":
            target_energy = energy_max * float(rng.uniform(1.01, 1.08))
            target_r5 = r5_max + float(rng.uniform(0.005, 0.04))
            target_r6 = r6_max + float(rng.uniform(0.005, 0.04))
            feasible = False
            reasons.append("targets_above_observed_archive_maxima")
        elif kind == "unsupported_or_ambiguous":
            target_energy = max(energy_min, energy * 0.95)
            target_r5 = max(0.0, r5 - 0.02)
            target_r6 = max(0.0, r6 - 0.02)
            feasible = False
            unsupported = random.Random(seed + sample_id).sample(
                ["cycle_life", "safety", "cost", "low_temperature", "finished_cell_geometry"],
                k=int(rng.integers(1, 4)),
            )
            reasons.append("contains_requirements_outside_current_model_scope")
        else:
            raise ValueError(f"Unknown sample kind: {kind}")

        row.update(
            {
                "sample_kind": kind,
                "requirements_canonical": {
                    "preference_energy": preference_energy,
                    "preference_high_rate": 1.0 - preference_energy,
                    "target_energy_wh_kg": round(target_energy, 4),
                    "min_retention_5c": round(target_r5, 6),
                    "min_retention_6c": round(target_r6, 6),
                    "temperature_k": 298.15,
                    "material_parameter_set": "Chen2020",
                    "application_tag": "energy_storage",
                    "unverified_requirements": unsupported,
                },
                "requirement_feasible": feasible,
                "infeasible_reasons": reasons,
                "teacher_candidates": [
                    {
                        "source_archive_index": int(archive.source_indices[candidate]),
                        "latent": archive.latent[candidate].round(10).tolist(),
                        "energy_1c_wh_kg": float(archive.energy[candidate]),
                        "retention_5c": float(archive.retention_5c[candidate]),
                        "retention_6c": float(archive.retention_6c[candidate]),
                        "oracle_loss": float(losses[candidate]),
                    }
                    for candidate in top_indices
                ],
            }
        )
        row["requirement_text"] = deterministic_text(row)
        records.append(row)
    return records


def deterministic_text(record: dict[str, Any]) -> str:
    req = record["requirements_canonical"]
    text = (
        "请在固定 Chen2020 材料体系和25℃条件下设计储能电芯结构，"
        f"希望1C比能量不低于{req['target_energy_wh_kg']:.4f} Wh/kg，"
        f"5C和6C能量保持率分别不低于{100 * req['min_retention_5c']:.4f}%和"
        f"{100 * req['min_retention_6c']:.4f}%。"
        f"能量目标权重为{req['preference_energy']:.4f}，其余权重用于高倍率性能。"
    )
    if req["unverified_requirements"]:
        text += "另外关注" + "、".join(req["unverified_requirements"]) + "，无法验证时请明确说明。"
    return text


def deepseek_rewrite(record: dict[str, Any], config: dict[str, Any]) -> str:
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Environment variable {config['api_key_env']} is not set")
    req = record["requirements_canonical"]
    numeric_anchors = [
        f"{req['target_energy_wh_kg']:.4f}",
        f"{100 * req['min_retention_5c']:.4f}%",
        f"{100 * req['min_retention_6c']:.4f}%",
        f"{req['preference_energy']:.4f}",
        f"{req['preference_high_rate']:.4f}",
    ]
    instruction = {
        "task": "将电芯需求改写成一段自然、非结构化的中文用户提问",
        "constraints": [
            "不得改变、推导或新增任何数值",
            "必须保留材料体系、温度、1C比能量、5C/6C保持率及两个偏好权重",
            "不得给出设计答案",
            "必须逐字保留下列数值锚点：" + "、".join(numeric_anchors),
            "只返回JSON对象，字段名为requirement_text",
        ],
        "canonical_requirements": req,
    }
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "你只负责忠实改写工程需求，不负责生成物理标签。"},
            {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)},
        ],
        "temperature": float(config["temperature"]),
        "max_tokens": int(config["max_tokens"]),
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(config["timeout_s"])) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = json.loads(result["choices"][0]["message"]["content"])
    text = str(content["requirement_text"]).strip()
    if not text:
        raise ValueError("DeepSeek returned an empty requirement_text")
    missing_anchors = [anchor for anchor in numeric_anchors if anchor not in text]
    if missing_anchors:
        raise ValueError(f"DeepSeek rewrite changed or omitted numeric anchors: {missing_anchors}")
    if "Chen2020" not in text or ("25℃" not in text and "298.15" not in text):
        raise ValueError("DeepSeek rewrite omitted the material set or temperature")
    return text


def apply_deepseek(
    records: list[dict[str, Any]], config: dict[str, Any], checkpoint_path: Path | None = None
) -> None:
    pending = [
        (index, record)
        for index, record in enumerate(records)
        if record["provenance"].get("language_source") != config["model"]
    ]

    def rewrite_one(index: int, record: dict[str, Any]) -> tuple[int, str | None, int, str | None]:
        last_error: Exception | None = None
        for attempt in range(int(config["retries"])):
            try:
                return index, deepseek_rewrite(record, config), attempt + 1, None
            except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        return index, None, int(config["retries"]), type(last_error).__name__

    completed = 0
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 1))) as executor:
        futures = [executor.submit(rewrite_one, index, record) for index, record in pending]
        for future in as_completed(futures):
            index, text, attempts, error_name = future.result()
            record = records[index]
            if text is not None:
                record["requirement_text"] = text
                record["provenance"]["language_source"] = config["model"]
                record["provenance"]["language_attempts"] = attempts
                record.pop("quality_flags", None)
            else:
                record.setdefault("quality_flags", []).append("deepseek_rewrite_failed")
                record["provenance"]["language_error"] = error_name
            completed += 1
            if checkpoint_path is not None and completed % 25 == 0:
                write_jsonl(checkpoint_path, records)
            if completed % 25 == 0:
                print(f"DeepSeek rewrites: {completed}/{len(pending)}")
    if checkpoint_path is not None:
        write_jsonl(checkpoint_path, records)


def assign_splits(records: list[dict[str, Any]], sizes: dict[str, int], seed: int) -> None:
    if sum(sizes.values()) != len(records):
        raise ValueError("Split sizes must sum to the number of records")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    cursor = 0
    for split, count in sizes.items():
        for index in indices[cursor : cursor + count]:
            records[index]["split"] = split
        cursor += count


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper_explore_v1 language dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_explore_v1.yaml"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--with-deepseek", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    counts = {name: int(value) for name, value in dataset_cfg["counts"].items()}
    requested_samples = int(args.samples or dataset_cfg["samples"])
    if args.samples is not None:
        if requested_samples != sum(counts.values()):
            raise ValueError("--samples override currently requires matching dataset.counts total")
    elif requested_samples != sum(counts.values()):
        raise ValueError("dataset.samples must equal the sum of dataset.counts")

    archive_path = args.archive or Path(config["source_archive"]["path"])
    output = args.output or Path(dataset_cfg["output"])
    archive = load_archive(archive_path)
    records = make_records(archive, counts, int(config["experiment"]["seed"]))
    split_cfg = config["split"]
    assign_splits(
        records,
        {"train": int(split_cfg["train"]), "validation": int(split_cfg["validation"]), "test": int(split_cfg["test"])},
        int(config["experiment"]["seed"]),
    )
    if args.with_deepseek:
        if output.exists():
            prior = {
                row["task_id"]: row
                for row in (
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            for record in records:
                old = prior.get(record["task_id"])
                if old and old.get("provenance", {}).get("language_source") == config["language"]["model"]:
                    record["requirement_text"] = old["requirement_text"]
                    record["provenance"] = old["provenance"]
                    if "quality_flags" in old:
                        record["quality_flags"] = old["quality_flags"]
        apply_deepseek(records, config["language"], output)
    write_jsonl(output, records)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "experiment": config["experiment"],
        "archive": str(archive_path),
        "archive_metadata": archive.metadata,
        "records": len(records),
        "counts": counts,
        "splits": {name: sum(row["split"] == name for row in records) for name in ("train", "validation", "test")},
        "deepseek_enabled": bool(args.with_deepseek),
        "output": str(output),
        "sha256": digest,
    }
    manifest_path = Path(dataset_cfg["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
