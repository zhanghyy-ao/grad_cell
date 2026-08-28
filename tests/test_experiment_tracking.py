import json

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
