from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


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


def stable_tiebreak(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def row_stratum(row: dict[str, Any]) -> str:
    teacher = row["teacher_design"]
    return f"{teacher['base_parameter_set']}::{teacher['generation_mode']}"


def validate_selected(rows: list[dict[str, Any]], language_source: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No successful {language_source!r} records were found")
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Selected task_id values are not unique")
    failures = [
        row["task_id"]
        for row in rows
        if row.get("quality_flags")
        or row.get("provenance", {}).get("language_error")
        or row.get("provenance", {}).get("language_source") != language_source
    ]
    if failures:
        raise ValueError(f"Selected data contains {len(failures)} failed language rows")
    family_counts = Counter(row["physical_design_id"] for row in rows)
    invalid_family_sizes = {
        design_id: count for design_id, count in family_counts.items() if count not in (1, 2, 3)
    }
    if invalid_family_sizes:
        raise ValueError(f"Unexpected description-family sizes: {invalid_family_sizes}")
    for row in rows:
        inverse = row.get("inverse_ambiguity", {})
        if not inverse.get("equivalence_group_id"):
            raise ValueError(f"Missing equivalence group for {row['task_id']}")
        alternatives = row.get("alternative_teacher_designs", [])
        if inverse.get("alternative_design_count") != len(alternatives):
            raise ValueError(f"Alternative count mismatch for {row['task_id']}")
    return {
        "physical_designs": len(family_counts),
        "complete_three_variant_families": sum(value == 3 for value in family_counts.values()),
        "incomplete_families": sum(value != 3 for value in family_counts.values()),
    }


def assign_groups(
    rows: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["inverse_ambiguity"]["equivalence_group_id"]].append(row)
    strata = sorted({row_stratum(row) for row in rows})
    total_by_stratum = Counter(row_stratum(row) for row in rows)
    target_total = {split: ratios[split] * len(rows) for split in SPLITS}
    target_stratum = {
        split: {
            stratum: ratios[split] * total_by_stratum[stratum] for stratum in strata
        }
        for split in SPLITS
    }
    assigned_total = Counter()
    assigned_stratum = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), stable_tiebreak(seed, item[0])),
    )
    for group_id, members in ordered_groups:
        group_stratum = Counter(row_stratum(row) for row in members)
        scores = {}
        for candidate_split in SPLITS:
            total_error = 0.0
            stratum_error = 0.0
            overflow = 0.0
            for split in SPLITS:
                increment = len(members) if split == candidate_split else 0
                new_total = assigned_total[split] + increment
                total_error += (new_total - target_total[split]) ** 2 / max(
                    target_total[split] ** 2, 1.0
                )
                overflow += max(new_total - target_total[split], 0.0) / max(
                    target_total[split], 1.0
                )
                for stratum in strata:
                    stratum_increment = (
                        group_stratum[stratum] if split == candidate_split else 0
                    )
                    stratum_error += (
                        assigned_stratum[split][stratum]
                        + stratum_increment
                        - target_stratum[split][stratum]
                    ) ** 2 / max(target_stratum[split][stratum] ** 2, 1.0)
            scores[candidate_split] = total_error + stratum_error + 2.0 * overflow
        split = min(
            SPLITS,
            key=lambda name: (scores[name], SPLITS.index(name)),
        )
        assignment[group_id] = split
        assigned_total[split] += len(members)
        assigned_stratum[split].update(group_stratum)

    if any(assigned_total[split] == 0 for split in SPLITS):
        raise RuntimeError(
            "Strict group split produced an empty subset; equivalence groups are too large "
            f"for ratios {ratios}. Counts: {dict(assigned_total)}"
        )
    return assignment


def validate_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    design_splits: dict[str, set[str]] = defaultdict(set)
    task_ids = set()
    for row in rows:
        if row["task_id"] in task_ids:
            raise ValueError(f"Duplicate task ID after split: {row['task_id']}")
        task_ids.add(row["task_id"])
        group_splits[row["inverse_ambiguity"]["equivalence_group_id"]].add(row["split"])
        design_splits[row["physical_design_id"]].add(row["split"])
    leaked_groups = [key for key, values in group_splits.items() if len(values) != 1]
    leaked_designs = [key for key, values in design_splits.items() if len(values) != 1]
    if leaked_groups or leaked_designs:
        raise RuntimeError(
            f"Strict split leakage: groups={len(leaked_groups)}, "
            f"physical_designs={len(leaked_designs)}"
        )
    return {
        "records": len(rows),
        "physical_designs": len(design_splits),
        "equivalence_groups": len(group_splits),
        "split_records": Counter(row["split"] for row in rows),
        "split_physical_designs": Counter(
            next(iter(values)) for values in design_splits.values()
        ),
        "split_equivalence_groups": Counter(
            next(iter(values)) for values in group_splits.values()
        ),
        "strata_by_split": {
            split: Counter(row_stratum(row) for row in rows if row["split"] == split)
            for split in SPLITS
        },
        "group_leakage": 0,
        "physical_design_leakage": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter successful DeepSeek battery descriptions and create a strict "
            "group-aware train/validation/test split for Top-K inverse training."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language-source", default="deepseek-v4-flash")
    parser.add_argument("--expected-records", type=int, default=2158)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        parser.error("Split ratios must be positive and sum to 1")
    source_rows = read_jsonl(args.input)
    selected = [
        row
        for row in source_rows
        if row.get("provenance", {}).get("language_source") == args.language_source
        and not row.get("provenance", {}).get("language_error")
        and not row.get("quality_flags")
    ]
    if args.expected_records and len(selected) != args.expected_records:
        raise ValueError(
            f"Expected {args.expected_records} successful {args.language_source} rows, "
            f"found {len(selected)}. Refusing to silently change the experiment cohort."
        )
    family_summary = validate_selected(selected, args.language_source)
    assignment = assign_groups(selected, ratios, args.seed)
    output_rows = []
    for source_row in selected:
        row = json.loads(json.dumps(source_row, ensure_ascii=False))
        row["original_split"] = row["split"]
        row["split"] = assignment[row["inverse_ambiguity"]["equivalence_group_id"]]
        row["split_provenance"] = {
            "method": "deterministic_greedy_performance_equivalence_group",
            "seed": args.seed,
            "ratios": ratios,
        }
        output_rows.append(row)
    validation = validate_split(output_rows)
    write_jsonl(args.output, output_rows)
    manifest = {
        "schema": "gradcell.deepseek_topk_split_manifest.v1",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "language_source": args.language_source,
        "expected_records": args.expected_records,
        "seed": args.seed,
        "ratios": ratios,
        "family_summary": family_summary,
        "validation": validation,
        "limitations": [
            "The 2158-row checkpoint cohort contains Chen2020 Regular Mode only.",
            "Results do not establish Extreme Mode or cross-parameter-set generalization.",
            "One incomplete three-description family is retained to preserve all 2158 rows.",
        ],
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
