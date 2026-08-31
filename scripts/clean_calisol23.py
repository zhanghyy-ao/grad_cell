from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gradcell.data import clean_calisol23_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize CALiSol-23 and create the model-v1 training table."
    )
    parser.add_argument("--input", type=Path, default=Path("data/calisol23.csv"))
    parser.add_argument(
        "--canonical-output", type=Path, default=Path("data/calisol23_canonical.csv")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("data/calisol23_model_v1.csv")
    )
    parser.add_argument(
        "--report-output", type=Path, default=Path("data/calisol23_cleaning_report.json")
    )
    parser.add_argument("--conductivity-floor-ms-cm", type=float, default=1e-12)
    parser.add_argument("--keep-tfsi-aliases-separate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = (args.canonical_output, args.model_output, args.report_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"Outputs already exist; use --overwrite: {existing}")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    result = clean_calisol23_frame(
        pd.read_csv(args.input),
        conductivity_floor_ms_cm=args.conductivity_floor_ms_cm,
        merge_tfsi_aliases=not args.keep_tfsi_aliases_separate,
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.canonical.to_csv(args.canonical_output, index=False)
    result.model_v1.to_csv(args.model_output, index=False)
    report = {
        **result.report,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "canonical_output": str(args.canonical_output),
        "model_output": str(args.model_output),
    }
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
