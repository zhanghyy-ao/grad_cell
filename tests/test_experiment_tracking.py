import json

import numpy as np

from gradcell.experiments import ExperimentRun


def test_experiment_run_records_success(tmp_path):
    with ExperimentRun("unit_test", {"seed": 7}, run_dir=tmp_path / "run") as run:
        run.event("progress", step=1)
        run.save_summary({"metric": 0.25})

    metadata = json.loads((run.path / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run.path / "summary.json").read_text(encoding="utf-8"))
    events = (run.path / "events.jsonl").read_text(encoding="utf-8")
    assert metadata["status"] == "completed"
    assert summary["metric"] == 0.25
    assert '"event": "progress"' in events


def test_experiment_run_serializes_numpy_arrays(tmp_path):
    with ExperimentRun("array_test", {}, run_dir=tmp_path / "array") as run:
        run.event("inputs", values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    records = [
        json.loads(line)
        for line in (run.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    input_record = next(record for record in records if record["event"] == "inputs")
    assert input_record["values"] == [[1.0, 2.0], [3.0, 4.0]]


def test_experiment_run_records_failure(tmp_path):
    try:
        with ExperimentRun("unit_test", {}, run_dir=tmp_path / "failed"):
            raise ValueError("expected")
    except ValueError:
        pass

    metadata = json.loads(
        (tmp_path / "failed" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert "ValueError" in metadata["error"]
