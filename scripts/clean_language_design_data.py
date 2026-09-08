from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DESIGN_FIELD_ALIASES = {
    "positive_electrode_porosity": ("positive_electrode_porosity",),
    "negative_electrode_porosity": ("negative_electrode_porosity",),
    "separator_porosity": ("separator_porosity",),
    "positive_active_material_fraction": (
        "positive_active_material_fraction",
        "positive_active_fraction",
        "positive_electrode_active_fraction",
    ),
    "negative_to_positive_capacity_ratio": (
        "negative_to_positive_capacity_ratio",
        "np_ratio",
    ),
}


def first_json_object(value: str | dict, line_number: int) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"target_json must be a string or object at line {line_number}")
    start = value.find("{")
    if start < 0:
        raise ValueError(f"target_json contains no JSON object at line {line_number}")
    try:
        payload, _ = json.JSONDecoder().raw_decode(value[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid target_json at line {line_number}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"target_json must decode to an object at line {line_number}")
    return payload


def canonical_design(payload: dict, line_number: int) -> dict[str, float]:
    source = payload.get("design", payload)
    if not isinstance(source, dict):
        raise ValueError(f"design must be an object at line {line_number}")
    cleaned = {}
    for canonical_name, aliases in DESIGN_FIELD_ALIASES.items():
        matches = [name for name in aliases if name in source]
        if not matches:
            raise ValueError(
                f"missing {canonical_name!r} (accepted aliases: {aliases}) at line {line_number}"
            )
        cleaned[canonical_name] = float(source[matches[0]])
    return cleaned


def clean_task_text(text: str) -> str:
    text = re.sub(
        r"<FORBIDDEN_OUTPUT_FIELDS>.*?</FORBIDDEN_OUTPUT_FIELDS>\s*",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.replace(
        "NP_RATIO_RANGE", "NEGATIVE_TO_POSITIVE_CAPACITY_RATIO_RANGE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean an existing GradCell JSONL into five-field Stage-1 supervision."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input so the source dataset is preserved")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    renamed_np_ratio = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            latent = record.get("teacher_latent")
            if not isinstance(latent, list) or len(latent) != 5:
                raise ValueError(f"teacher_latent must contain five values at line {line_number}")
            payload = first_json_object(record.get("target_json"), line_number)
            source_design = payload.get("design", payload)
            renamed_np_ratio += int(
                isinstance(source_design, dict)
                and "np_ratio" in source_design
                and "negative_to_positive_capacity_ratio" not in source_design
            )
            target = {
                "schema": "gradcell.material_design.v1",
                "design": canonical_design(payload, line_number),
            }
            cleaned_record = {
                key: record[key]
                for key in ("id", "preference", "targets")
                if key in record
            }
            cleaned_record.update(
                {
                    "task_text": clean_task_text(record["task_text"]),
                    "teacher_latent": [float(value) for value in latent],
                    "target_json": json.dumps(
                        target, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
            serialized = json.dumps(cleaned_record, ensure_ascii=False)
            if "np_ratio" in serialized or "physics_loss" in serialized:
                raise RuntimeError(f"forbidden field survived cleaning at line {line_number}")
            destination.write(serialized + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "records": written,
                "renamed_np_ratio": renamed_np_ratio,
                "target_design_fields": list(DESIGN_FIELD_ALIASES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
